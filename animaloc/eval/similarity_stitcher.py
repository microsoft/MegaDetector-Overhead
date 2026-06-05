"""
SimilarityMapStitcher — stitch DINO cosine-similarity maps from overlapping
patches into a full-image refinement map, then run LMDS to extract refined
point detections.

Design decisions:
  - Composition over inheritance: the base Stitcher assumes model(patch) → map,
    but similarity refinement needs DINO features + detection coordinates per
    patch.  We compose ImageToPatches and HannWindow2D directly.
  - No per-patch normalization: raw aggregated maps are stitched first, then
    globally normalized to [0, 1] to avoid visible seams at patch boundaries.
  - Overlap must be divisible by down_ratio (ViT patch_size, default 16) so
    that token grids align across patches.
  - Hann windowing + mean reduction for smooth blending (same pattern as
    DensityMapStitcher).
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from typing import Dict, List, Optional, Tuple, Union

import PIL.Image

from ..data import ImageToPatches
from .utils import HannWindow2D
from .lmds import LMDS
from .similarity_utils import (
    points_to_token_indices,
    compute_similarity_maps,
    aggregate_similarity_raw,
    upsample_map,
    locs_to_pixel_coords,
    power_transform,
    log_sharpen,
    blob_score,
)

__all__ = ['SimilarityMapStitcher']

# ImageNet statistics used by DINOv3 preprocessing
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SimilarityMapStitcher:
    """Stitch DINO similarity maps from overlapping patches into a full-image
    map, then run LMDS peak detection to produce refined point detections.

    Usage::

        stitcher = SimilarityMapStitcher(dino_model, size=(512,512), overlap=128)
        result = stitcher(image, detections, image_name='IMG_001.jpg')
        # result['detections'] is a list of dicts ready for CSV output
        # result['stitched_map'] is the [1, 1, H_tok, W_tok] similarity map
    """

    def __init__(
        self,
        dino_model: torch.nn.Module,
        size: Tuple[int, int] = (512, 512),
        overlap: int = 128,
        down_ratio: int = 16,
        aggregation: str = 'weighted',
        use_hann: bool = True,
        device: str = 'cuda',
        n_layers: int = 12,
        sim_upsample: int = 1,
        lmds_kernel: int = 5,
        lmds_adapt_ts: float = 100.0 / 255.0,
        lmds_neg_ts: float = 0.1,
        # --- Postprocessing parameters ---
        pre_upsample: int = 1,
        gamma: Optional[float] = None,
        log_sigma: Optional[float] = None,
        log_alpha: float = 0.3,
        blob_threshold: float = 0.0,
    ) -> None:
        """
        Args:
            dino_model: DINOv3 ViT model (already loaded and in eval mode).
            size: Patch size (height, width) in pixels.
            overlap: Overlap between patches in pixels. Must be divisible by
                down_ratio.
            down_ratio: ViT patch_size (pixels per token). Default 16.
            aggregation: 'weighted' (by dscore) or 'unweighted' mean of
                per-point similarity maps.
            use_hann: Apply Hann windowing for smooth blending at boundaries.
            device: Device for computation ('cuda' or 'cpu').
            n_layers: Number of ViT layers for get_intermediate_layers.
            sim_upsample: Bilinear upsample factor for the stitched similarity
                map before LMDS (default 1 = token resolution).
            lmds_kernel: LMDS kernel size (must be odd).
            lmds_adapt_ts: LMDS adaptive threshold.
            lmds_neg_ts: LMDS negative sample threshold.
            pre_upsample: Bilinear upsample factor applied to each patch
                **before** feature extraction (1 = no upsample, 2 = double
                resolution).  Doubles token grid per patch and improves
                spatial precision of similarity maps.
            gamma: Power-transform exponent applied globally after stitching.
                None disables the power transform.
            log_sigma: Gaussian sigma for LoG sharpening applied globally
                after stitching.  None disables LoG.
            log_alpha: Weight of the LoG term (default 0.3).
            blob_threshold: Per-patch blob-score threshold.  Patches with
                ``blob_score < blob_threshold`` are zeroed before stitching
                (false-positive suppression).  0 disables filtering.
        """
        assert overlap % down_ratio == 0, (
            f"overlap ({overlap}) must be divisible by down_ratio ({down_ratio}) "
            f"so that token grids align across patches"
        )

        self.dino_model = dino_model
        self.size = size
        self.overlap = overlap
        self.down_ratio = down_ratio
        self.aggregation = aggregation
        self.use_hann = use_hann
        self.device = torch.device(device)
        self.n_layers = n_layers
        self.sim_upsample = sim_upsample
        self.lmds_kernel = (lmds_kernel, lmds_kernel)
        self.lmds_adapt_ts = lmds_adapt_ts
        self.lmds_neg_ts = lmds_neg_ts

        self.pre_upsample = pre_upsample
        self.gamma = gamma
        self.log_sigma = log_sigma
        self.log_alpha = log_alpha
        self.blob_threshold = blob_threshold

        self.dino_model.to(self.device)
        self.dino_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        detections: List[Dict],
        image_name: str = 'image',
    ) -> Dict:
        """Full pipeline: stitch similarity maps → LMDS → refined detections.

        Args:
            image: Full-size image as PIL Image or tensor [C, H, W].
            detections: List of dicts with keys 'x', 'y', 'dscore' in
                full-image pixel coordinates.
            image_name: Image filename for the output CSV rows.

        Returns:
            dict with keys:
              'stitched_map': [1, 1, H_tok, W_tok] normalized raw similarity map
              'postprocessed_map': [1, 1, H_tok, W_tok] after power+LoG (or
                  same as stitched_map when postprocessing is disabled)
              'detections': list of dicts {images, labels, dscores, x, y, count_1}
              'locs': raw LMDS locations in map coordinates
              'scores': LMDS similarity scores at peaks
              'blob_scores': list of per-patch blob scores (empty when
                  blob_threshold is 0)
        """
        stitched_map, (img_h, img_w), patch_blob_scores = self.stitch(
            image, detections,
        )

        # Global postprocessing (power transform + LoG)
        postprocessed = self.postprocess(stitched_map)

        # Optional bilinear upsample before LMDS
        sim_map = postprocessed.squeeze(0).squeeze(0)  # [H_tok, W_tok]
        sim_up = upsample_map(sim_map, self.sim_upsample)
        map_h, map_w = sim_up.shape

        # LMDS peak detection
        lmds = LMDS(
            kernel_size=self.lmds_kernel,
            adapt_ts=self.lmds_adapt_ts,
            neg_ts=self.lmds_neg_ts,
        )
        est_map = sim_up.unsqueeze(0).unsqueeze(0)    # [1, 1, H, W]
        _, b_locs, _, b_scores = lmds(est_map)
        locs = b_locs[0]
        scores = b_scores[0]

        # Convert to full-image pixel coordinates
        pixel_coords = locs_to_pixel_coords(locs, img_w, img_h, map_w, map_h)

        # Format output rows (same schema as input detections CSV)
        count = len(pixel_coords)
        det_rows = []
        for (x_px, y_px), score in zip(pixel_coords, scores):
            det_rows.append({
                "images": image_name,
                "labels": 1,
                "dscores": round(float(score), 6),
                "x": round(float(x_px), 1),
                "y": round(float(y_px), 1),
                "count_1": count,
            })

        return {
            'stitched_map': stitched_map,
            'postprocessed_map': postprocessed,
            'detections': det_rows,
            'locs': locs,
            'scores': scores,
            'blob_scores': patch_blob_scores,
        }

    def stitch(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        detections: List[Dict],
    ) -> Tuple[torch.Tensor, Tuple[int, int], List[float]]:
        """Stitch per-patch similarity maps into a full-image map.

        Args:
            image: Full-size image as PIL Image or tensor [C, H, W].
            detections: List of dicts with keys 'x', 'y', 'dscore'.

        Returns:
            stitched_map: [1, 1, H_tok, W_tok] globally normalized [0,1] map.
            (img_h, img_w): original image pixel dimensions.
            blob_scores: per-patch blob scores (empty when blob filtering
                is disabled).
        """
        # Ensure image is a tensor [C, H, W]
        if isinstance(image, PIL.Image.Image):
            image = TF.to_tensor(image)
        image = image.to(torch.device('cpu'))

        _, img_h, img_w = image.shape

        # Convert detections to tensors
        if len(detections) == 0:
            all_x = torch.zeros(0, dtype=torch.float32)
            all_y = torch.zeros(0, dtype=torch.float32)
            all_scores = torch.zeros(0, dtype=torch.float32)
        else:
            all_x = torch.tensor([d['x'] for d in detections], dtype=torch.float32)
            all_y = torch.tensor([d['y'] for d in detections], dtype=torch.float32)
            all_scores = torch.tensor([d['dscore'] for d in detections], dtype=torch.float32)

        # Effective token grid dimensions per patch (accounting for
        # pre_upsample: a 512px patch upsampled 2x → 1024px → 64 tokens)
        eff_h_tok = (self.size[0] * self.pre_upsample) // self.down_ratio
        eff_w_tok = (self.size[1] * self.pre_upsample) // self.down_ratio
        eff_overlap_tok = (self.overlap * self.pre_upsample) // self.down_ratio

        # Small image optimization: process as single patch
        if img_h <= self.size[0] and img_w <= self.size[1]:
            sim_map = self._process_single_patch(
                image, all_x, all_y, all_scores, img_w, img_h,
                eff_h_tok, eff_w_tok,
            )
            sim_map = self._global_normalize(sim_map)
            out = sim_map.unsqueeze(0).unsqueeze(0)
            return out, (img_h, img_w), []

        # Tile image into patches
        patcher = ImageToPatches(image, self.size, self.overlap)
        patches = patcher.make_patches()               # [N_patches, C, H_p, W_p]
        limits = patcher.get_limits()                  # {i: BoundingBox}
        n_patches = patches.shape[0]

        # Naming convention from ImageToPatches:
        # _nrow = number of columns (patches per row)
        # _ncol = number of rows (rows of patches)
        n_cols_grid = patcher._nrow
        n_rows_grid = patcher._ncol

        # Build Hann windows at effective token resolution
        if self.use_hann:
            hann_windows = self._build_hann_matrix(n_rows_grid, n_cols_grid)
        else:
            hann_windows = [torch.ones(1, eff_h_tok, eff_w_tok)] * n_patches

        # Process each patch
        maps = []
        patch_blob_scores: List[float] = []
        for i in range(n_patches):
            patch_tensor = patches[i]                  # [C, H_p, W_p]
            limit = limits[i]
            patch_h = limit.y_max - limit.y_min
            patch_w = limit.x_max - limit.x_min

            # Filter detections for this patch
            local_x, local_y, local_scores = self._filter_detections_for_patch(
                all_x, all_y, all_scores, limit,
            )

            if local_x.numel() == 0:
                sim_map = torch.zeros(eff_h_tok, eff_w_tok)
            else:
                # Optional pre-upsample
                if self.pre_upsample > 1:
                    patch_tensor = F.interpolate(
                        patch_tensor.unsqueeze(0),
                        scale_factor=self.pre_upsample,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)
                    up_patch_w = patch_w * self.pre_upsample
                    up_patch_h = patch_h * self.pre_upsample
                    up_local_x = local_x * self.pre_upsample
                    up_local_y = local_y * self.pre_upsample
                else:
                    up_patch_w = patch_w
                    up_patch_h = patch_h
                    up_local_x = local_x
                    up_local_y = local_y

                # Extract DINO features for this patch
                features_normed, (hp, wp) = self._extract_patch_features(
                    patch_tensor,
                )

                # Map detection coords to token indices
                flat_idx = points_to_token_indices(
                    up_local_x, up_local_y, up_patch_w, up_patch_h, wp, hp,
                )

                # Compute and aggregate similarity maps
                sim_maps = compute_similarity_maps(features_normed, flat_idx)
                weights = local_scores if self.aggregation == 'weighted' else None
                sim_map = aggregate_similarity_raw(sim_maps, hp, wp, weights=weights)

                # Per-patch blob score filtering.
                # Build a validity mask so that zero-padded border
                # pixels do not dilute the score.
                if self.blob_threshold > 0:
                    valid_h_tok = min(
                        (patch_h * self.pre_upsample) // self.down_ratio,
                        hp,
                    )
                    valid_w_tok = min(
                        (patch_w * self.pre_upsample) // self.down_ratio,
                        wp,
                    )
                    mask = np.zeros((hp, wp), dtype=np.float64)
                    mask[:valid_h_tok, :valid_w_tok] = 1.0

                    bs = blob_score(sim_map.numpy(), mask=mask)
                    patch_blob_scores.append(bs)
                    if bs > self.blob_threshold:
                        sim_map = torch.zeros_like(sim_map)

            # Apply Hann window
            sim_map = sim_map * hann_windows[i].squeeze(0).to(sim_map.device)
            maps.append(sim_map.unsqueeze(0).unsqueeze(0))

        # Stitch via F.fold
        kernel_size = np.array([eff_h_tok, eff_w_tok])
        stride = kernel_size - eff_overlap_tok

        output_size = (
            n_rows_grid * kernel_size[0] - (n_rows_grid - 1) * eff_overlap_tok,
            n_cols_grid * kernel_size[1] - (n_cols_grid - 1) * eff_overlap_tok,
        )

        maps_cat = torch.cat(maps, dim=0)             # [N, 1, h_tok, w_tok]
        n = maps_cat.shape[0]
        # Reshape for F.fold: [1, C*kH*kW, N]
        folded_input = maps_cat.permute(1, 2, 3, 0).contiguous().view(1, -1, n)
        stitched = F.fold(
            folded_input,
            output_size=tuple(output_size),
            kernel_size=tuple(kernel_size),
            stride=tuple(stride),
        )

        # Normalization map (count of contributions per pixel, accounting for
        # Hann weighting)
        if self.use_hann:
            hann_maps = torch.cat(
                [h.unsqueeze(0) for h in hann_windows], dim=0,
            )  # [N, 1, h_tok, w_tok]
            hann_folded = hann_maps.permute(1, 2, 3, 0).contiguous().view(1, -1, n)
            norm_map = F.fold(
                hann_folded,
                output_size=tuple(output_size),
                kernel_size=tuple(kernel_size),
                stride=tuple(stride),
            )
        else:
            ones = torch.ones_like(maps_cat)
            ones_folded = ones.permute(1, 2, 3, 0).contiguous().view(1, -1, n)
            norm_map = F.fold(
                ones_folded,
                output_size=tuple(output_size),
                kernel_size=tuple(kernel_size),
                stride=tuple(stride),
            )

        # Avoid division by zero
        norm_map = norm_map.clamp(min=1e-8)
        stitched = stitched / norm_map

        # Crop to actual image token dimensions (effective ratio accounts for
        # pre_upsample: each original pixel maps to pre_upsample/down_ratio
        # tokens)
        eff_ratio = self.down_ratio // self.pre_upsample
        dh = img_h // eff_ratio
        dw = img_w // eff_ratio
        stitched = stitched[:, :, :dh, :dw]

        # Global normalization to [0, 1]
        s_min = stitched.min()
        s_max = stitched.max()
        if s_max - s_min > 1e-8:
            stitched = (stitched - s_min) / (s_max - s_min)
        else:
            stitched = torch.zeros_like(stitched)

        return stitched, (img_h, img_w), patch_blob_scores

    def postprocess(self, stitched_map: torch.Tensor) -> torch.Tensor:
        """Apply global power transform + LoG sharpening to the stitched map.

        Called automatically by :meth:`__call__`.  The input is expected to
        be globally normalized to [0, 1] (output of :meth:`stitch`).

        When both *gamma* and *log_sigma* are ``None`` this is a no-op and
        returns the input unchanged.

        Returns:
            [1, 1, H, W] tensor in [0, 1].
        """
        if self.gamma is None and self.log_sigma is None:
            return stitched_map

        h = stitched_map.squeeze().cpu().numpy()       # [H, W]

        # Power transform (input already in [0, 1])
        if self.gamma is not None:
            h = power_transform(h, self.gamma)

        # LoG sharpening
        if self.log_sigma is not None:
            h = log_sharpen(h, self.log_sigma, self.log_alpha)

        # Clamp negatives (LoG can produce negative values)
        h = np.clip(h, 0, None).astype(np.float32)

        # Re-normalize to [0, 1]
        h_min, h_max = h.min(), h.max()
        if h_max - h_min > 1e-8:
            h = (h - h_min) / (h_max - h_min)
        else:
            h = np.zeros_like(h)

        return torch.from_numpy(h).float().unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_detections_for_patch(
        self,
        all_x: torch.Tensor,
        all_y: torch.Tensor,
        all_scores: torch.Tensor,
        limit,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter detections falling within the patch and shift to local coords.

        Uses inclusive lower bound, exclusive upper bound: [x_min, x_max).
        Detections in overlap regions will appear in multiple patches — this is
        intentional and handled by the mean reduction during stitching.
        """
        mask = (
            (all_x >= limit.x_min) & (all_x < limit.x_max) &
            (all_y >= limit.y_min) & (all_y < limit.y_max)
        )
        local_x = all_x[mask] - limit.x_min
        local_y = all_y[mask] - limit.y_min
        local_scores = all_scores[mask]
        return local_x, local_y, local_scores

    @torch.inference_mode()
    def _extract_patch_features(
        self, patch_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract L2-normalized spatial features from a single patch.

        Args:
            patch_tensor: [C, H, W] image tensor (unnormalized, [0, 1] range).

        Returns:
            features_normed: [N_tokens, embed_dim] L2-normalized features.
            (h_p, w_p): spatial grid dimensions.
        """
        # ImageNet normalization
        img_norm = TF.normalize(patch_tensor, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

        feats = self.dino_model.get_intermediate_layers(
            img_norm.unsqueeze(0).to(self.device),
            n=range(self.n_layers),
            reshape=True,
            norm=True,
        )
        # Last layer: [1, C, h_p, w_p]
        x = feats[-1].squeeze(0).detach().cpu()
        dim, h_p, w_p = x.shape
        x_flat = x.reshape(dim, -1).permute(1, 0)     # [N_tokens, C]
        features_normed = F.normalize(x_flat, dim=1)
        return features_normed, (h_p, w_p)

    def _process_single_patch(
        self,
        image: torch.Tensor,
        all_x: torch.Tensor,
        all_y: torch.Tensor,
        all_scores: torch.Tensor,
        img_w: int,
        img_h: int,
        h_tok: int,
        w_tok: int,
    ) -> torch.Tensor:
        """Process a single small image (no stitching needed)."""
        if all_x.numel() == 0:
            return torch.zeros(h_tok, w_tok)

        patch = image
        det_x, det_y = all_x, all_y
        pw, ph = img_w, img_h
        if self.pre_upsample > 1:
            patch = F.interpolate(
                patch.unsqueeze(0), scale_factor=self.pre_upsample,
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            det_x = all_x * self.pre_upsample
            det_y = all_y * self.pre_upsample
            pw = img_w * self.pre_upsample
            ph = img_h * self.pre_upsample

        features_normed, (hp, wp) = self._extract_patch_features(patch)
        flat_idx = points_to_token_indices(det_x, det_y, pw, ph, wp, hp)
        sim_maps = compute_similarity_maps(features_normed, flat_idx)
        weights = all_scores if self.aggregation == 'weighted' else None
        return aggregate_similarity_raw(sim_maps, hp, wp, weights=weights)

    def _build_hann_matrix(self, n_rows: int, n_cols: int) -> List[torch.Tensor]:
        """Build position-aware Hann windows for all patches.

        Mirrors DensityMapStitcher._make_hann_matrix() but at token-grid
        resolution (size // down_ratio).

        Naming follows ImageToPatches convention:
          n_rows = _ncol = number of rows of patches (vertical count)
          n_cols = _nrow = number of columns per row (horizontal count)

        Returns:
            List of [1, h_tok, w_tok] Hann window tensors, one per patch,
            ordered to match ImageToPatches patch ordering.
        """
        tok_size = (self.size[0] * self.pre_upsample) // self.down_ratio
        hann = HannWindow2D(size=tok_size)

        # First row of patches
        first_row = [hann.get_window('edge', 'up')] * n_cols
        first_row[0] = hann.get_window('corner', 'up_left')
        if n_cols > 1:
            first_row[-1] = hann.get_window('corner', 'up_right')

        # Middle rows
        middle_row = [hann.get_window('original', 'up')] * n_cols
        middle_row[0] = hann.get_window('edge', 'left')
        if n_cols > 1:
            middle_row[-1] = hann.get_window('edge', 'right')

        # Last row
        last_row = [hann.get_window('edge', 'down')] * n_cols
        last_row[0] = hann.get_window('corner', 'down_left')
        if n_cols > 1:
            last_row[-1] = hann.get_window('corner', 'down_right')

        if n_rows == 1:
            matrix = first_row
        elif n_rows == 2:
            matrix = [*first_row, *last_row]
        else:
            matrix = [*first_row, *middle_row * (n_rows - 2), *last_row]

        return matrix

    @staticmethod
    def _global_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Min-max normalize a tensor to [0, 1]."""
        t_min, t_max = tensor.min(), tensor.max()
        if t_max - t_min > 1e-8:
            return (tensor - t_min) / (t_max - t_min)
        return torch.zeros_like(tensor)
