---
description: "End-to-end training, evaluation, and inference workflow for the MegaDetector-Overhead OWL family (OWL-C, OWL-D, OWL-T) and the legacy HerdNet models."
tags:
  - MegaDetector-Overhead training
  - HerdNet training
  - DINOv3 training
  - aerial wildlife detection
  - drone imagery model training
  - PyTorch-Wildlife
---

# Training, Evaluation, and Inference

This page covers the day-to-day workflow for the training/eval stack
vendored from HerdNet and DINOv3.

## Architecture overview

The repository ships three model families, all derived from HerdNet's
detection branch with progressively richer encoders:

| Model | Backbone | Source file | Registry name |
|---|---|---|---|
| **OWL-C** | DLA-34 (HerdNet baseline) | `animaloc/models/owl_c.py` | `OWLC` |
| **OWL-T** | DLA-34 + Swin transformer multiscale residual | `animaloc/models/owl_t.py` | `OWLT` |
| **OWL-D** | DINOv3 ViT (S/B/L/H) + DPT decoder | `animaloc/models/owl_d.py` | `OWLD_S`, `OWLD_B`, `OWLD_L`, `OWLD_H` |

All three produce a single-channel FIDT heatmap; peak post-processing
yields per-instance point detections. The legacy `HerdNet` multi-class
model is also available (`HerdNet` registry entry).

### Shared post-processing infrastructure

OWL-C, OWL-D, and OWL-T all share the same evaluation, stitching, and
local-maximum-detection helpers from `animaloc/eval/`:

* `HerdNet_Detection_Branch_Evaluator` (`animaloc/eval/evaluators.py`)
* `HerdNet_Detection_Branch_Stitcher` (`animaloc/eval/stitchers.py`)
* `HerdNet_Detection_Branch_LMDS` (`animaloc/eval/lmds.py`)

These keep their historical `HerdNet_Detection_Branch_*` names — they
are infrastructure, not model classes, and renaming them would force
churn across every OWL config with no real benefit.

## Dataset format

The animaloc package expects point-annotated aerial imagery. See the
[HerdNet upstream README](https://github.com/Alexandre-Delplanque/HerdNet#dataset-format)
for the dataset layout (folder structure + per-image CSV with
`x,y,labels` columns). Configs under `configs/train/` and `configs/test/`
reference dataset paths under `dataset.root_dir`.

!!! tip "Caribou benchmark dataset"
    The [Caribou Aerial Survey Dataset](datasets.md) provides ready-to-use
    train/test splits in this format. Download from Zenodo and point
    `dataset.root_dir` at the extracted directory.

## Training

!!! note "Activate the venv first"
    All commands below assume the project venv is active:
    `source .venv/bin/activate` (after `uv sync`, or
    `uv sync --no-default-groups --group gpu` for a GPU — see
    [Installation](installation.md)). Using the activated venv runs the build you
    synced without reverting it; avoid bare `uv run` on a GPU.

Training is launched via Hydra:

```bash
python tools/train.py --config-path ../configs --config-name train/<config-name>
```

Examples:

```bash
# OWL-C on the terrestrial dataset
python tools/train.py --config-path ../configs \
    --config-name train/herdnet_loc_branch_terrestrial_datasets

# OWL-D ViT-B/16 frozen with proj read-token, r=12
python tools/train.py --config-path ../configs \
    --config-name train/exp_dpt_vitb_proj_r12_frozen

# OWL-D ViT-H+/16 on the overhead-generalized split
python tools/train.py --config-path ../configs \
    --config-name train/exp_dpt_vith_dinov3_overhead_generalized

# OWL-T hybrid multiscale residual on the Eikelboom dataset
python tools/train.py --config-path ../configs \
    --config-name train/herdnet_hybrid_multiscale_detection_branch_final_eikelboom
```

Each config sets the `model.name` field to one of the registry entries
listed above. Hydra writes outputs under `outputs/` (gitignored).

### W&B

`tools/train.py` logs metrics to Weights & Biases by default. Either
`wandb login` once, or set `WANDB_MODE=offline` to skip remote logging.

## Evaluation

```bash
python tools/test.py --config-path ../configs \
    --config-name test/<config-name>
```

The eval harness runs the trained model over a held-out split, applies
LMDS (local-maximum point detection with non-maximum suppression),
and reports F1 / precision / recall / MAE / RMSE per class.

## Inference on new imagery

`tools/infer.py` runs the original `HerdNet` model end-to-end:

```bash
python tools/infer.py <images_dir> <model.pth>
```

Outputs land in `<images_dir>/<date>_HerdNet_results/<date>_detections.csv`.
For OWL-C / OWL-D / OWL-T inference, use `tools/test.py` with the
corresponding test config.

## Tiling large images

Drone orthomosaics and high-resolution aerial frames generally exceed
model input size. `tools/patcher.py` tiles them into model-ready patches
with configurable overlap:

```bash
python tools/patcher.py <images_dir> <height> <width> <overlap>
```

The output directory mirrors the input layout, with each large image
replaced by its grid of patches plus a CSV mapping patches back to
source-image coordinates.

## OWL-D weights

`OWLD_S/B/L/H` load DINOv3 ViT backbones from `weights/dinov3_*.pth`.
The default filenames are hard-coded on each class as
`_DEFAULT_WEIGHTS_FILENAME`. See [INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md)
for the download steps.

To override the path or use a different DINOv3 release (e.g. the SAT-493M
satellite-imagery variant of ViT-L), set `model.weights` in the training
config.

## Custom models

The `MODELS` registry in `animaloc.models` allows new architectures to be
added via decorator:

```python
from animaloc.models import MODELS
import torch.nn as nn

@MODELS.register()
class MyModel(nn.Module):
    def __init__(self, ...): ...
    def forward(self, x): ...
```

Once registered, `MyModel` is selectable from any training config via
`model.name: MyModel`.

## Verifying the install (smoke tests)

The `tests/` directory has minimal end-to-end smoke tests that verify a
fresh install works without needing real data:

```bash
# 1. Forward-pass test for all 6 OWL models (~30 s on CPU)
python tests/smoke_forward.py

# 2. Build a synthetic mini-dataset (4 train + 2 val 512x512 images)
python tests/make_synthetic_dataset.py

# 3. Train OWL-C for one epoch on the synthetic data
WANDB_MODE=disabled python tools/train.py train=owlc_smoketest

# 4. Evaluate the resulting checkpoint
CKPT=$(ls -t outputs/*/*/best_model.pth | head -1 | xargs realpath)
WANDB_MODE=disabled python tools/test.py test=owlc_smoketest \
    "++test.model.pth_file=$CKPT"
```

OWL-D variants additionally need DINOv3 weights under `weights/`; if
present, `tests/smoke_forward.py` exercises all six OWL classes
(otherwise OWL-D is skipped). See [tests/README.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/tests/README.md).

The smoke configs `configs/train/owlc_smoketest.yaml`,
`configs/train/owld_s_smoketest.yaml`, and their `configs/test/`
counterparts are also good templates to copy when you want to point at
your own real dataset: edit `csv_file`, `root_dir`, `class_def`, and the
`training_settings.{batch_size,epochs}` fields.
