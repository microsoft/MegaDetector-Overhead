"""
Modular DPT (Dense Prediction Transformer) decoder for animal localization
using a DINOv3 encoder, producing a 1-channel FIDT heatmap.

Key improvements over herdnet_dino.py (v1):
  - DinoV3DPTDecoder base class with modular _make_* factory methods
  - CLS token ReadProj fusion ("project" / "add" / "ignore")
  - Padding for arbitrary input resolutions (non-multiples of 16), crop after
  - forward_from_features() entry point for pre-extracted feature caching
  - HerdNetDINO_S / _B / _L / _H as clean, registered subclasses

Architecture (per-forward summary):
  Input [B, 3, H, W]
    → pad to multiple of 16
  DINOv3 backbone (frozen or partially unfrozen)
    → 4 × (patch_feat [B, C, H/16, W/16], cls_token [B, C])
  ReadProj (if readout_type="project"):
    → Linear(2C → C) + GELU per layer
  ReassembleBlocksV2:
    Conv1×1 + ConvTranspose2d/Identity/Conv2d(s=2)
    → strides [4, 8, 16, 32], all 256-ch
  FusionDecoderV2:
    4 × Conv3×3 + 4 × FeatureFusionBlock (×2 upsample each)
    → [B, 256, H/down_ratio, W/down_ratio]
  LocalizationHead:
    Conv3×3 → ReLU → Conv1×1 → Sigmoid
    → [B, 1, H/down_ratio, W/down_ratio]
    → crop to [B, 1, H//down_ratio, W//down_ratio]
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

from .register import MODELS


__all__ = [
    'DinoV3DPTDecoder',
    'HerdNetDINO_S',
    'HerdNetDINO_B',
    'HerdNetDINO_L',
    'HerdNetDINO_H',
]

_VALID_DOWN_RATIOS = (1, 2, 4)
_VALID_READOUT_TYPES = ("project", "add", "ignore")

# ---------------------------------------------------------------------------
# Backbone specifications: {name → (hub_function_name, embed_dim, depth)}
# ---------------------------------------------------------------------------
_BACKBONE_SPECS = {
    'vits16':     ('dinov3_vits16',     384,  12),
    'vitb16':     ('dinov3_vitb16',     768,  12),
    'vitl16':     ('dinov3_vitl16',     1024, 24),
    'vith16plus': ('dinov3_vith16plus', 1280, 32),
}

# DINOv3 DPT head channel configs for pretrained decoder weight loading.
# Format: {backbone_name: (reassembly_out_ch, fusion_ch)}
# These must match the DPT head checkpoint architecture.
_DPT_CHANNEL_CONFIGS = {
    'vitl16':     (1024, 512),
    'vith16plus': (2048, 512),
}


def _layer_indices_for_depth(depth: int) -> List[int]:
    """Return 4 evenly-spaced 0-indexed block indices for a ViT of given depth."""
    return [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]


# ---------------------------------------------------------------------------
# ReadProj: fuse CLS token into patch features via a learned linear projection
# ---------------------------------------------------------------------------

class ReadProj(nn.Module):
    """CLS token fusion via a learned linear projection (DPT "project" readout).

    For each spatial patch token, the CLS token (a global image summary) is
    concatenated and the pair is projected back down to the original embedding
    dimension via Linear(2C → C) + GELU.  This allows the model to inject
    global context into per-location features before the decoder.

    Args:
        embed_dim (int): ViT embedding dimension for this layer.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         Patch feature map  [B, C, h, w]
            cls_token: CLS token vector   [B, C]
        Returns:
            Tensor [B, C, h, w] with CLS context fused in.
        """
        B, C, h, w = x.shape
        # Flatten spatial dims: [B, C, h, w] → [B, h*w, C]
        x_flat = x.flatten(2).permute(0, 2, 1)
        # Expand CLS to every spatial position: [B, C] → [B, h*w, C]
        readout = cls_token.unsqueeze(1).expand_as(x_flat)
        # Concat and project: [B, h*w, 2C] → [B, h*w, C]
        fused = self.proj(torch.cat([x_flat, readout], dim=-1))
        # Reshape back: [B, h*w, C] → [B, C, h, w]
        return fused.permute(0, 2, 1).reshape(B, C, h, w)


# ---------------------------------------------------------------------------
# PreActResidualConvUnit
# ---------------------------------------------------------------------------

class PreActResidualConvUnit(nn.Module):
    """Pre-activation residual conv unit (ReLU → Conv → BN → ReLU → Conv → BN + skip).

    Pre-activation ordering improves gradient flow vs. post-activation blocks.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn1(self.conv1(self.relu1(x)))
        x = self.bn2(self.conv2(self.relu2(x)))
        return x + residual


# ---------------------------------------------------------------------------
# FeatureFusionBlock
# ---------------------------------------------------------------------------

class FeatureFusionBlock(nn.Module):
    """Merge a shallower feature map with a deeper one, refine, ×2 upsample, project.

    When called with one tensor (first / deepest block):
        ResConvUnit2(x) → ×2 bilinear upsample → Conv1×1
    When called with two tensors (x=shallow reassembled, skip=previous fusion output):
        x + ResConvUnit1(skip) → ResConvUnit2 → ×2 upsample → Conv1×1

    Args:
        ch (int): Channel count (same in and out).
        first_block (bool): If True, builds without res_conv1 (no skip expected).
    """

    def __init__(self, ch: int, first_block: bool = False) -> None:
        super().__init__()
        self.res_conv1 = None if first_block else PreActResidualConvUnit(ch)
        self.res_conv2 = PreActResidualConvUnit(ch)
        self.project   = nn.Conv2d(ch, ch, 1, bias=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            # Align skip to x's spatial dims if they differ (shouldn't in normal use)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(
                    skip, size=x.shape[2:], mode='bilinear', align_corners=False
                )
            x = x + self.res_conv1(skip)
        x = self.res_conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.project(x)
        return x


# ---------------------------------------------------------------------------
# UpConvHead: learned ×2 upsample for down_ratio=1
# ---------------------------------------------------------------------------

class UpConvHead(nn.Module):
    """Learned ×2 upsample: Conv3×3 → bilinear ×2 → Conv3×3 → ReLU → Conv1×1.

    Used as the final decoder stage when down_ratio=1.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch // 2, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# ReassembleBlocksV2: project + spatially rescale ViT features
# ---------------------------------------------------------------------------

class ReassembleBlocksV2(nn.Module):
    """Project and spatially rescale 4 ViT intermediate feature maps.

    DINOv3 ViT produces all intermediate features at H/16 × W/16.  This
    module builds a multi-scale pyramid at strides {4, 8, 16, 32} using
    learned spatial rescaling, optionally fusing the CLS token into patch
    features first.

    Readout types:
        "project" — CLS token fused via ReadProj (Linear(2C→C) + GELU)
        "add"     — CLS token added directly to patch features (simple)
        "ignore"  — CLS token discarded

    Spatial rescaling:
        Layer 0: Conv1×1 + ConvTranspose2d(k=4,s=4) → stride  4  (4× up)
        Layer 1: Conv1×1 + ConvTranspose2d(k=2,s=2) → stride  8  (2× up)
        Layer 2: Conv1×1 + Identity                 → stride 16  (no change)
        Layer 3: Conv1×1 + Conv2d(k=3,s=2,p=1)      → stride 32  (2× down)

    Args:
        in_channels  (List[int]): embed_dim for each of the 4 layers.
        out_channels (List[int]): output channel count per layer (all dec_ch=256).
        readout_type (str): One of "project", "add", "ignore".
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        readout_type: str = "project",
    ) -> None:
        super().__init__()
        assert readout_type in _VALID_READOUT_TYPES, \
            f"readout_type must be one of {_VALID_READOUT_TYPES}, got '{readout_type}'"
        assert len(in_channels) == 4 and len(out_channels) == 4

        self.readout_type = readout_type

        # ReadProj modules (one per layer) — only built for "project" mode
        if readout_type == "project":
            self.read_projs = nn.ModuleList([
                ReadProj(c) for c in in_channels
            ])
        else:
            self.read_projs = None

        # 1×1 channel projections: embed_dim → dec_ch
        self.projects = nn.ModuleList([
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)
            for in_c, out_c in zip(in_channels, out_channels)
        ])

        # Spatial rescaling
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

    def forward(
        self,
        inputs: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> List[torch.Tensor]:
        """
        Args:
            inputs: When readout_type is "project" or "add":
                        List of 4 (spatial [B,C,h,w], cls [B,C]) tuples.
                    When readout_type is "ignore":
                        List of 4 spatial tensors [B, C, h, w].
        Returns:
            List of 4 tensors at strides [4, 8, 16, 32].
        """
        out = []
        for i, inp in enumerate(inputs):
            if self.readout_type == "ignore":
                x = inp  # plain spatial tensor
            elif self.readout_type == "add":
                x, cls = inp
                B, C, h, w = x.shape
                x_flat = x.flatten(2).permute(0, 2, 1)              # [B, h*w, C]
                cls_exp = cls.unsqueeze(1).expand_as(x_flat)         # [B, h*w, C]
                x = (x_flat + cls_exp).permute(0, 2, 1).reshape(B, C, h, w)
            else:  # "project"
                x, cls = inp
                x = self.read_projs[i](x, cls)

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


# ---------------------------------------------------------------------------
# FusionDecoderV2: DPT-style FPN with per-level Conv3×3 projectors
# ---------------------------------------------------------------------------

class FusionDecoderV2(nn.Module):
    """DPT-style FPN decoder.

    Applies per-level Conv3×3 smoothing, then merges from the deepest
    (stride 32) to the shallowest level using FeatureFusionBlocks that
    each perform: merge → PreActResidual → ×2 upsample → project.

    Active fusion stages by down_ratio:
        down_ratio=4 → 3 stages  (stride 32→16→8→4)
        down_ratio=2 → 4 stages  (stride 32→16→8→4→2)
        down_ratio=1 → 4 stages + UpConvHead → stride 1

    Args:
        dec_ch (int): Decoder channel count. Default 256.
        in_ch (int | None): Input channel count from reassembly. When None,
            defaults to dec_ch. Set differently when loading pretrained DPT
            decoder weights (e.g., in_ch=1024, dec_ch=512 for ViT-L).
        down_ratio (int): Output stride relative to input. {1, 2, 4}. Default 2.
    """

    def __init__(self, dec_ch: int = 256, in_ch: Optional[int] = None,
                 down_ratio: int = 2) -> None:
        super().__init__()
        assert down_ratio in _VALID_DOWN_RATIOS, \
            f"down_ratio must be one of {_VALID_DOWN_RATIOS}, got {down_ratio}"
        self._down_ratio = down_ratio
        self._n_stages = 3 if down_ratio == 4 else 4
        _in_ch = in_ch if in_ch is not None else dec_ch

        # Per-level Conv3×3 projection (reassembly_ch → dec_ch)
        self.convs = nn.ModuleList([
            nn.Conv2d(_in_ch, dec_ch, kernel_size=3, padding=1, bias=False)
            for _ in range(4)
        ])

        # 4 FeatureFusionBlocks (index 0 = deepest = first_block)
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(dec_ch, first_block=(i == 0))
            for i in range(4)
        ])

        # Post-fusion projection
        self.project = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
        )

        # Optional UpConvHead for down_ratio=1
        self.up_conv_head = UpConvHead(dec_ch) if down_ratio == 1 else None

    def forward(self, reassembled: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            reassembled: [R0, R1, R2, R3] at strides [4, 8, 16, 32], 256-ch each.
        Returns:
            Tensor [B, dec_ch, H/down_ratio, W/down_ratio].
        """
        # Per-level smooth
        projected = [self.convs[i](reassembled[i]) for i in range(4)]

        # Bottom-up fusion: start at deepest (stride 32), upsample ×2 each stage
        # Matches reference DPT: res_conv1 refines the shallow reassembled feature
        # before adding it to the upsampled deep path.
        out = self.fusion_blocks[0](projected[3])           # stride 32 → ×2 → stride 16
        for i in range(1, self._n_stages):
            out = self.fusion_blocks[i](out, projected[3 - i])  # deep upsampled + shallow

        out = self.project(out)

        if self._down_ratio == 1 and self.up_conv_head is not None:
            out = self.up_conv_head(out)

        return out


# ---------------------------------------------------------------------------
# LocalizationHead
# ---------------------------------------------------------------------------

class LocalizationHead(nn.Module):
    """Localization head producing a 1-channel FIDT heatmap.

    Conv(in_ch → head_conv, 3×3) → ReLU → Conv(head_conv → 1, 1×1) → Sigmoid

    Weight init:
        Conv weights: normal_(std=0.001)
        All biases: constant_(0)
        Final conv bias: -2.19  (so sigmoid(-2.19) ≈ 0.10, low background prior)

    Args:
        in_ch (int): Input channel count. Default 256.
        head_conv (int): Intermediate channel count. Default 64.
    """

    def __init__(self, in_ch: int = 256, head_conv: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self._init_localization_weights()

    def _init_localization_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Low prior: sigmoid(-2.19) ≈ 0.10 — suppresses background at init
        self.head[-2].bias.data.fill_(-2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# DinoV3DPTDecoder — base class
# ---------------------------------------------------------------------------

class DinoV3DPTDecoder(nn.Module):
    """Base class for DINOv3-backbone + DPT-decoder animal localization models.

    Subclasses must define:
        _BACKBONE_NAME (str)             — key into _BACKBONE_SPECS
        _DEFAULT_WEIGHTS_FILENAME (str)  — path relative to dinov3_root

    Args:
        pretrained (bool): Load pretrained DINOv3 backbone weights. Default True.
        weights_path (str | None): Explicit local .pth path (overrides default).
        dinov3_root (str | None): Root directory of the dinov3 Python package.
            Falls back to environment variable DINOV3_ROOT or _DEFAULT_DINOV3_ROOT.
        head_conv (int): Intermediate channels in the localization head. Default 64.
        readout_type (str): CLS token handling mode. Default "ignore".
            "ignore"  — discard CLS token entirely (matches DINOv3 DPT default)
            "project" — inject CLS into patch features via Linear(2C→C)+GELU
            "add"     — add CLS token element-wise to patch features
        freeze_backbone (bool): Freeze the entire backbone. Default True.
            Set to False to enable full backbone fine-tuning (useful for ViT-S/B).
        unfreeze_last_n (int): When freeze_backbone=True, unfreeze the last N
            transformer blocks. Useful for domain adaptation of larger models.
        down_ratio (int): Output heatmap stride relative to input. {1, 2, 4}.
            2 → H/2 × W/2 output (default, best precision for small animals)
            4 → H/4 × W/4 output (faster, lower memory)
            1 → full resolution (highest memory, bilinear artefacts likely)
        dec_ch (int): Decoder channel width. Default 256.
            When decoder_weights_path is set, this is overridden to match the
            DINOv3 DPT head architecture (512 for ViT-L/H).
        decoder_weights_path (str | None): Path to a DINOv3 DPT head checkpoint
            (.pth) to initialize the decoder from pretrained depth weights.
            Only reassembly, fusion, and projection weights are loaded;
            the localization head always trains from scratch.
            Available checkpoints (download from DINOv3 release):
                ViT-L SYNTHMIX: dinov3_vitl16_synthmix_dpt_head-*.pth
            When set, dec_ch is automatically adjusted to match the checkpoint.

    Returns (forward):
        Tuple[heatmap [B, 1, H//down_ratio, W//down_ratio], None]
        The None placeholder maintains API compatibility with HerdNet.
    """

    _DEFAULT_DINOV3_ROOT: str = '/home/v-ichaconsil/azurefiles/dinov3'
    _BACKBONE_NAME: str = ''          # override in subclass
    _DEFAULT_WEIGHTS_FILENAME: str = ''  # override in subclass
    _DEC_CH: int = 256

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        dinov3_root: Optional[str] = None,
        head_conv: int = 64,
        readout_type: str = "ignore",
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
        down_ratio: int = 2,
        dec_ch: int = 256,
        decoder_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        assert self._BACKBONE_NAME in _BACKBONE_SPECS, (
            f"{self.__class__.__name__}._BACKBONE_NAME='{self._BACKBONE_NAME}' "
            f"is not in {list(_BACKBONE_SPECS)}."
        )
        assert readout_type in _VALID_READOUT_TYPES, \
            f"readout_type must be one of {_VALID_READOUT_TYPES}, got '{readout_type}'"
        assert down_ratio in _VALID_DOWN_RATIOS, \
            f"down_ratio must be one of {_VALID_DOWN_RATIOS}, got {down_ratio}"

        hub_fn_name, embed_dim, depth = _BACKBONE_SPECS[self._BACKBONE_NAME]
        self._embed_dim        = embed_dim
        self._depth            = depth
        self._layer_indices    = _layer_indices_for_depth(depth)
        self._readout_type     = readout_type
        self._freeze_backbone  = freeze_backbone
        self._unfreeze_last_n  = unfreeze_last_n
        self._down_ratio       = down_ratio

        # When loading pretrained DPT decoder weights, override channel dims
        # to match the DINOv3 DPT head architecture.
        reassembly_ch = dec_ch
        if decoder_weights_path is not None:
            assert self._BACKBONE_NAME in _DPT_CHANNEL_CONFIGS, (
                f"Pretrained DPT decoder weights not supported for backbone "
                f"'{self._BACKBONE_NAME}'. Supported: {list(_DPT_CHANNEL_CONFIGS)}"
            )
            reassembly_ch, dec_ch = _DPT_CHANNEL_CONFIGS[self._BACKBONE_NAME]
            print(f"[decoder_weights] Overriding dec_ch={dec_ch}, "
                  f"reassembly_ch={reassembly_ch} to match DPT checkpoint.")

        # Build all components
        self._load_backbone(hub_fn_name, dinov3_root, pretrained, weights_path)
        self.reassembly = self._make_reassemble_layer(embed_dim, reassembly_ch, dec_ch, readout_type)
        self.decoder    = self._make_fusion_decoder(
            dec_ch, down_ratio,
            in_ch=reassembly_ch if reassembly_ch != dec_ch else None,
        )
        self.loc_head   = self._make_localization_head(dec_ch, head_conv)

        self.head_conv = head_conv

        # Load pretrained DPT decoder weights (reassembly + fusion only)
        if decoder_weights_path is not None:
            self._load_decoder_weights(decoder_weights_path)

    # ------------------------------------------------------------------
    # Factory methods (subclasses may override to customise)
    # ------------------------------------------------------------------

    def _load_backbone(
        self,
        hub_fn_name: str,
        dinov3_root: Optional[str],
        pretrained: bool,
        weights_path: Optional[str],
    ) -> None:
        """Instantiate the DINOv3 backbone and apply freeze/unfreeze settings."""
        _dinov3_root = (
            dinov3_root
            or os.environ.get('DINOV3_ROOT', self._DEFAULT_DINOV3_ROOT)
        )
        if _dinov3_root not in sys.path:
            sys.path.insert(0, _dinov3_root)

        import dinov3.hub.backbones as _hub  # noqa: PLC0415
        backbone_ctor = getattr(_hub, hub_fn_name)

        # Resolve weight file
        _weights = weights_path
        if _weights is None and self._DEFAULT_WEIGHTS_FILENAME:
            _candidate = os.path.join(_dinov3_root, self._DEFAULT_WEIGHTS_FILENAME)
            if os.path.isfile(_candidate):
                _weights = _candidate

        if _weights is not None:
            self.backbone = backbone_ctor(pretrained=pretrained, weights=_weights)
        else:
            self.backbone = backbone_ctor(pretrained=pretrained)

        # Freeze / partial unfreeze
        if self._freeze_backbone:
            self.backbone.requires_grad_(False)
            if self._unfreeze_last_n > 0:
                for block in self.backbone.blocks[-self._unfreeze_last_n:]:
                    block.requires_grad_(True)

    def _make_reassemble_layer(
        self, embed_dim: int, reassembly_ch: int, dec_ch: int, readout_type: str
    ) -> ReassembleBlocksV2:
        # reassembly_ch: output channels of reassembly (spatial rescaling)
        # dec_ch: fusion channel width (convs project reassembly_ch -> dec_ch)
        # When no pretrained decoder weights: reassembly_ch == dec_ch
        return ReassembleBlocksV2(
            in_channels=[embed_dim] * 4,
            out_channels=[reassembly_ch] * 4,
            readout_type=readout_type,
        )

    def _make_fusion_decoder(self, dec_ch: int, down_ratio: int,
                             in_ch: Optional[int] = None) -> FusionDecoderV2:
        return FusionDecoderV2(dec_ch=dec_ch, in_ch=in_ch, down_ratio=down_ratio)

    def _make_localization_head(self, dec_ch: int, head_conv: int) -> LocalizationHead:
        return LocalizationHead(in_ch=dec_ch, head_conv=head_conv)

    # ------------------------------------------------------------------
    # Keep backbone in eval when fully frozen
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> 'DinoV3DPTDecoder':
        super().train(mode)
        if self._freeze_backbone and self._unfreeze_last_n == 0:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Padding / crop utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor, multiple: int = 16
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad x spatially to the nearest multiple of `multiple`.

        Uses reflect padding to minimise cold-edge artefacts at the border.

        Returns:
            (padded_x, (pad_h, pad_w))
        """
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            # F.pad order: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop_output(
        x: torch.Tensor, orig_h: int, orig_w: int, down_ratio: int
    ) -> torch.Tensor:
        """Crop heatmap to correspond to the original (pre-pad) input size."""
        return x[:, :, : orig_h // down_ratio, : orig_w // down_ratio]

    # ------------------------------------------------------------------
    # Core feature extraction (shared by forward and forward_from_features)
    # ------------------------------------------------------------------

    def _run_backbone(self, x: torch.Tensor):
        """Run backbone and return features in the format expected by reassembly.

        Returns:
            List of 4 elements, each either:
                - (spatial [B,C,H/16,W/16], cls [B,C]) tuple  if readout_type != "ignore"
                - spatial tensor [B, C, H/16, W/16]             if readout_type == "ignore"
        """
        return_cls = self._readout_type != "ignore"
        call_kwargs = dict(
            n=self._layer_indices,
            reshape=True,
            norm=True,
            return_class_token=return_cls,
        )
        backbone_fully_frozen = self._freeze_backbone and self._unfreeze_last_n == 0

        if backbone_fully_frozen:
            with torch.no_grad():
                raw = self.backbone.get_intermediate_layers(x, **call_kwargs)
        else:
            raw = self.backbone.get_intermediate_layers(x, **call_kwargs)

        # Cast to float32 (vith16plus may output bfloat16)
        if return_cls:
            return [(f.float(), c.float()) for f, c in raw]
        else:
            return [f.float() for f in raw]

    def _run_decoder(self, feats) -> torch.Tensor:
        """Reassemble → fuse → localization head."""
        reassembled = self.reassembly(feats)
        d_out = self.decoder(reassembled)
        return self.loc_head(d_out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            input: Image tensor [B, 3, H, W].
                   H and W need not be multiples of 16 — will be padded.
        Returns:
            heatmap [B, 1, H//down_ratio, W//down_ratio], None
        """
        _, _, H, W = input.shape
        x_padded, _ = self._pad_to_multiple(input, multiple=16)

        feats = self._run_backbone(x_padded)
        heatmap = self._run_decoder(feats)
        heatmap = self._crop_output(heatmap, H, W, self._down_ratio)

        return heatmap, None

    # ------------------------------------------------------------------
    # Feature-caching entry point
    # ------------------------------------------------------------------

    def forward_from_features(
        self,
        features: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, None]:
        """Run decoder + head on pre-extracted backbone features.

        Allows the "extract once, train decoder many times" workflow — especially
        valuable for ViT-L/H on large aerial datasets where backbone inference
        is the main compute bottleneck.  Features can also be augmented (spatial
        flips, channel noise) before being passed to this method.

        Args:
            features: List of 4 elements.  Each is either:
                - Tensor [B, C, H/16, W/16]                       (readout_type="ignore")
                - Tuple (Tensor [B,C,H/16,W/16], Tensor [B,C])    (readout_type="project"/"add")
                These are the direct output of backbone.get_intermediate_layers(
                    ..., reshape=True, return_class_token=True/False).

        Returns:
            heatmap [B, 1, H_feat*(16//down_ratio), W_feat*(16//down_ratio)], None

        Example::
            # Offline feature extraction
            with torch.no_grad():
                feats = model.backbone.get_intermediate_layers(
                    img, n=model._layer_indices, reshape=True,
                    norm=True, return_class_token=True,
                )
            torch.save(feats, f"cache/{img_id}.pt")

            # Online decoder training (backbone weights not needed on GPU)
            feats = torch.load(f"cache/{img_id}.pt")
            feats = [(f.to(device), c.to(device)) for f, c in feats]  # readout="project"
            heatmap, _ = model.forward_from_features(feats)
            loss = criterion(heatmap, target)
            loss.backward()
        """
        # Cast to float32 in case features were saved in another dtype
        if self._readout_type != "ignore":
            features = [(f.float(), c.float()) for f, c in features]
        else:
            features = [f.float() for f in features]

        heatmap = self._run_decoder(features)
        return heatmap, None

    # ------------------------------------------------------------------
    # Layer freezing utilities (API parity with HerdNet)
    # ------------------------------------------------------------------

    def freeze(self, layers: List[str]) -> None:
        """Freeze decoder/head submodules by attribute name."""
        for layer in layers:
            self._freeze_layer(layer)

    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Pretrained DPT decoder weight loading
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dpt_key_map() -> dict:
        """Build state_dict key mapping: DINOv3 DPT head → custom decoder.

        The DINOv3 DPT head (dpt_head.py) uses ConvModule wrappers that add
        a `.conv` suffix to weight keys.  The custom decoder uses plain
        nn.Conv2d / nn.BatchNorm2d, so suffixes differ.

        Returns:
            dict mapping DINOv3 DPT key prefixes → custom key prefixes.
        """
        m = {}

        # Reassembly: projects (1x1 conv) and resize_layers
        for i in range(4):
            # DINOv3 ConvModule wraps Conv2d as .conv
            m[f'reassemble_blocks.projects.{i}.conv.'] = f'reassembly.projects.{i}.'
            # Resize layers are plain Conv2d / ConvTranspose2d — same names
            m[f'reassemble_blocks.resize_layers.{i}.'] = f'reassembly.resize_layers.{i}.'

        # Decoder convs (reassembly_ch → dec_ch smoothing)
        for i in range(4):
            m[f'convs.{i}.conv.'] = f'decoder.convs.{i}.'

        # Fusion blocks
        for i in range(4):
            pfx_src = f'fusion_blocks.{i}'
            pfx_dst = f'decoder.fusion_blocks.{i}'

            # FeatureFusionBlock.project (1x1 conv, bias=True)
            m[f'{pfx_src}.project.conv.'] = f'{pfx_dst}.project.'

            # PreActResidualConvUnits: conv1/conv2 each wrapped in ConvModule
            for unit_src, unit_dst in [('res_conv_unit1', 'res_conv1'),
                                       ('res_conv_unit2', 'res_conv2')]:
                for j in [1, 2]:
                    # DINOv3: .conv{j}.conv.weight → custom: .conv{j}.weight
                    m[f'{pfx_src}.{unit_src}.conv{j}.conv.'] = \
                        f'{pfx_dst}.{unit_dst}.conv{j}.'
                    # DINOv3: .conv{j}.bn.* → custom: .bn{j}.*
                    m[f'{pfx_src}.{unit_src}.conv{j}.bn.'] = \
                        f'{pfx_dst}.{unit_dst}.bn{j}.'

        # Post-fusion projection
        m['project.conv.'] = 'decoder.project.0.'  # Conv2d in Sequential

        return m

    def _load_decoder_weights(self, path: str) -> None:
        """Load DINOv3 DPT head weights into the custom decoder.

        Loads all parameters where both key and shape match after remapping.
        The localization head is never loaded (it always trains from scratch).
        """
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(path, map_location=map_location)
        # DINOv3 DPT checkpoints are plain state_dicts (no 'model_state_dict' wrapper)
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']

        key_map = self._build_dpt_key_map()
        model_sd = self.state_dict()
        mapped = {}

        for src_key, src_val in checkpoint.items():
            # Try each prefix mapping
            dst_key = None
            for src_pfx, dst_pfx in key_map.items():
                if src_key.startswith(src_pfx):
                    dst_key = dst_pfx + src_key[len(src_pfx):]
                    break
            if dst_key is None:
                continue
            if dst_key in model_sd and model_sd[dst_key].shape == src_val.shape:
                mapped[dst_key] = src_val

        n_checkpoint = len(checkpoint)
        n_loaded = len(mapped)
        self.load_state_dict(mapped, strict=False)
        print(f"[decoder_weights] Loaded {n_loaded}/{n_checkpoint} params from "
              f"DPT checkpoint. Localization head initialised from scratch.")

        # Report any shape mismatches for debugging
        mismatched = []
        for src_key, src_val in checkpoint.items():
            for src_pfx, dst_pfx in key_map.items():
                if src_key.startswith(src_pfx):
                    dst_key = dst_pfx + src_key[len(src_pfx):]
                    if dst_key in model_sd and model_sd[dst_key].shape != src_val.shape:
                        mismatched.append(
                            f"  {src_key} {list(src_val.shape)} "
                            f"→ {dst_key} {list(model_sd[dst_key].shape)}"
                        )
                    break
        if mismatched:
            print(f"[decoder_weights] Shape mismatches (skipped):")
            for line in mismatched:
                print(line)


# ---------------------------------------------------------------------------
# Concrete subclasses — one per backbone variant
# ---------------------------------------------------------------------------

@MODELS.register()
class HerdNetDINO_S(DinoV3DPTDecoder):
    """DINOv3 ViT-S/16 backbone (embed=384, 12 blocks, ~22M backbone params).

    Lightest variant.  Benefits most from full backbone fine-tuning
    (freeze_backbone=False) since the small encoder may underfit aerial imagery.
    """
    _BACKBONE_NAME = 'vits16'
    _DEFAULT_WEIGHTS_FILENAME = 'weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@MODELS.register()
class HerdNetDINO_B(DinoV3DPTDecoder):
    """DINOv3 ViT-B/16 backbone (embed=768, 12 blocks, ~86M backbone params).

    Good balance of quality and speed.  Partial fine-tuning (unfreeze_last_n=3)
    often gives gains on domain-specific aerial imagery.
    """
    _BACKBONE_NAME = 'vitb16'
    _DEFAULT_WEIGHTS_FILENAME = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@MODELS.register()
class HerdNetDINO_L(DinoV3DPTDecoder):
    """DINOv3 ViT-L/16 backbone (embed=1024, 24 blocks, ~307M backbone params).

    High-quality features; recommended for frozen-backbone + feature-caching
    workflows on large datasets (see forward_from_features).
    Two pretrained weight files are available:
        LVD-1.6B: dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth  (default)
        SAT-493M: dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth    (satellite imagery)
    """
    _BACKBONE_NAME = 'vitl16'
    _DEFAULT_WEIGHTS_FILENAME = 'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@MODELS.register()
class HerdNetDINO_H(DinoV3DPTDecoder):
    """DINOv3 ViT-H+/16 backbone (embed=1280, 32 blocks, ~840M backbone params).

    Highest-quality features.  Strongly recommended to use frozen backbone with
    feature caching (forward_from_features) for training efficiency.
    """
    _BACKBONE_NAME = 'vith16plus'
    _DEFAULT_WEIGHTS_FILENAME = 'weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
