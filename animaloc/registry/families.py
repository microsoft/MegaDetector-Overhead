"""Per-model deployment defaults: which Stitcher, Evaluator, model
constructor kwargs, image normalization stats, and downsample ratio to
use for each registered model in `animaloc.models.MODELS`.

This is a *deployment* concern (used by tools/infer.py and any future
prediction tool), NOT a property of the model itself. Models never
import from here.

## Design

A `FAMILIES[name]` entry has the shape of `ModelFamily`:

    stitcher:      str             # name of class in animaloc.eval.stitchers
    evaluator:     str             # name of class in animaloc.eval.evaluators
    model_kwargs:  dict[str, Any]  # constructor kwargs for the model class
    down_ratio:    int             # output stride; threaded into transforms + stitcher
    mean:          list[float]     # image normalization mean (RGB)
    std:           list[float]     # image normalization std (RGB)
    multi_class:   bool            # True if model outputs (heatmap, classmap), False if heatmap-only

## How tools should use it

The `resolve_family(name, *, checkpoint_meta=None, overrides=None)` helper
returns the effective config for a given model name. Resolution order
(later wins): family defaults -> checkpoint metadata -> explicit CLI
overrides.

## Extending

To register a new model family, add an entry to `FAMILIES` here, NOT in
`tools/infer.py`. The model itself only needs to be registered with
`@MODELS.register()` (in its own file under `animaloc.models`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional


# Normalization stats used by every config in this repo (HerdNet + all
# OWL variants). DINOv3 backbones happen to use these too in the OWLD_*
# training configs (verified against exp_dpt_vits_proj_r12_frozen.yaml,
# exp_dpt_vith_dinov3_overhead_generalized.yaml, etc.).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ModelFamily:
    """Deployment defaults for one model family."""

    stitcher: str
    evaluator: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    down_ratio: int = 2
    mean: list[float] = field(default_factory=lambda: list(_IMAGENET_MEAN))
    std: list[float] = field(default_factory=lambda: list(_IMAGENET_STD))
    multi_class: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "stitcher": self.stitcher,
            "evaluator": self.evaluator,
            "model_kwargs": copy.deepcopy(self.model_kwargs),
            "down_ratio": self.down_ratio,
            "mean": list(self.mean),
            "std": list(self.std),
            "multi_class": self.multi_class,
        }


FAMILIES: dict[str, ModelFamily] = {
    # Legacy HerdNet -- multi-class, outputs (heatmap, classmap).
    "HerdNet": ModelFamily(
        stitcher="HerdNetStitcher",
        evaluator="HerdNetEvaluator",
        model_kwargs=dict(
            num_layers=34,
            pretrained=False,  # inference loads from the .pth checkpoint
            down_ratio=2,
            head_conv=64,
        ),
        down_ratio=2,
        multi_class=True,
    ),
    # OWL-C: HerdNet detection branch, single-class FIDT heatmap, DLA-34.
    "OWLC": ModelFamily(
        stitcher="HerdNet_Detection_Branch_Stitcher",
        evaluator="HerdNet_Detection_Branch_Evaluator",
        model_kwargs=dict(
            num_layers=34,
            pretrained=False,
            down_ratio=2,
            head_conv=64,
        ),
        down_ratio=2,
        multi_class=False,
    ),
    # OWL-T: DLA-34 + Swin multiscale residual. Note kwarg `pretrained_cnn`,
    # not `pretrained`, on the DLA base.
    "OWLT": ModelFamily(
        stitcher="HerdNet_Detection_Branch_Stitcher",
        evaluator="HerdNet_Detection_Branch_Evaluator",
        model_kwargs=dict(
            num_layers=34,
            pretrained_cnn=False,
            down_ratio=2,
            head_conv=64,
        ),
        down_ratio=2,
        multi_class=False,
    ),
}

# OWL-D family: DINOv3 ViT (S/B/L/H) + DPT decoder. All four variants
# share the same stitcher / evaluator / kwargs (the variant is selected
# by the class name itself). pretrained=False to make sure the
# constructor does not try to fetch DINOv3 hub weights at inference --
# the checkpoint's state_dict supersedes them anyway.
_OWLD_DEFAULT_KWARGS = dict(down_ratio=2, freeze_backbone=True, pretrained=False)

for _owld_name in ("OWLD_S", "OWLD_B", "OWLD_L", "OWLD_H"):
    FAMILIES[_owld_name] = ModelFamily(
        stitcher="HerdNet_Detection_Branch_Stitcher",
        evaluator="HerdNet_Detection_Branch_Evaluator",
        model_kwargs=dict(_OWLD_DEFAULT_KWARGS),
        down_ratio=2,
        multi_class=False,
    )


def resolve_family(
    name: str,
    *,
    checkpoint_meta: Optional[dict[str, Any]] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return the effective deployment config for the named model.

    Resolution order (later wins):
        1. `FAMILIES[name]` defaults
        2. Values pulled from the checkpoint metadata (`mean`, `std`,
           `classes`, anything else stored by `tools/train.py`)
        3. Explicit CLI overrides

    Args:
        name: Registered model name (must be a key of `FAMILIES`).
        checkpoint_meta: Optional dict pulled from `torch.load(pth_file)`.
            Recognized keys: `mean`, `std`, `classes` (passes through),
            and any other key that matches a `ModelFamily` field.
        overrides: Optional dict of CLI-driven overrides. Same recognized
            keys as `checkpoint_meta`, plus `model_kwargs` (merged into
            family defaults, not replaced).

    Returns:
        Plain dict with the resolved config. Always has the keys:
        `stitcher`, `evaluator`, `model_kwargs`, `down_ratio`, `mean`,
        `std`, `multi_class`. Plus passthrough keys like `classes` when
        present in metadata.

    Raises:
        KeyError: if `name` is not in `FAMILIES`. The caller should
            catch this and report the available families to the user.
    """
    if name not in FAMILIES:
        raise KeyError(
            f"Unknown model family {name!r}. Known families: {sorted(FAMILIES.keys())}. "
            "Add an entry to animaloc/registry/families.py for new model classes."
        )

    resolved = FAMILIES[name].as_dict()

    # Pull supported keys from checkpoint metadata (mean, std, classes,
    # plus any direct field overrides).
    if checkpoint_meta:
        for key in ("mean", "std", "down_ratio"):
            if key in checkpoint_meta and checkpoint_meta[key] is not None:
                resolved[key] = checkpoint_meta[key]
        if "classes" in checkpoint_meta:
            resolved["classes"] = checkpoint_meta["classes"]

    # CLI overrides. `model_kwargs` is MERGED (not replaced) so users
    # can override one kwarg without listing every default.
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "model_kwargs" and isinstance(value, dict):
                merged = dict(resolved["model_kwargs"])
                merged.update(value)
                resolved["model_kwargs"] = merged
            else:
                resolved[key] = value

    return resolved
