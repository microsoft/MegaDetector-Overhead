---
description: "End-to-end caribou demo for MegaDetector-Overhead: download the Zenodo OWL-C weights and test patches, run OWL-C inference on GPU or CPU, and visualize the predictions."
tags:
  - demo
  - quickstart
  - caribou
  - OWL-C
  - inference
  - visualization
  - PyTorch-Wildlife
---

# Caribou Demo (download → infer → visualize)

This walkthrough takes you from a fresh clone to **visualized OWL-C predictions**
on real caribou aerial patches. It uses the public
[Caribou Aerial Survey Dataset](datasets.md) on Zenodo (weights + test patches),
runs the same evaluation stack as `tools/test.py`, and renders the detections
onto the patches as PNGs.

The demo **auto-detects** your hardware: it runs on a CUDA GPU when one is
available and otherwise falls back to CPU. It makes **no assumption** that you
have a GPU.

!!! note "About the weights"
    The Zenodo release labels the checkpoint "HerdNet (DLA-34)". In this repo the
    same DLA-34 detection branch is registered as **OWL-C**, so the demo loads it
    under `model.name: OWLC`. They are the same network.

## Prerequisites

Install the environment with `uv` (see [Installation](installation.md)):

```bash
uv sync
uv run python -c "import animaloc.models, dinov3; print('OK')"
```

You also need `curl` and `unzip` on your `PATH` (both are standard on Linux/macOS).

## One command

```bash
./tools/demo_caribou.sh
```

This will:

1. Download `Caribou-OWL-C.pth` (216 MB) and `test.zip` (1.2 GB) from Zenodo
   into `demo_data/` (skipped if already present).
2. Verify the weights' SHA-256 against the published checksum.
3. Build a deterministic **50-patch subset** (40 annotated + 10 background).
4. Auto-detect the device (GPU if available, else CPU).
5. Run OWL-C inference (`tools/test.py`) with Weights & Biases disabled.
6. Render predictions onto every patch with `tools/visualize_detections.py`.

Outputs:

| Path | Contents |
|---|---|
| `demo_data/run/metrics_results.csv` | F1 / precision / recall / MAE / RMSE |
| `demo_data/run/detections.csv` | One row per detection (`images, x, y, dscores, …`) |
| `demo_data/viz/*.png` | Patches with **green = ground truth, red = predictions** |

### Options

```bash
./tools/demo_caribou.sh --device cpu        # force CPU
./tools/demo_caribou.sh --device cuda        # force GPU
./tools/demo_caribou.sh --full               # run the full 2,607-patch test set
./tools/demo_caribou.sh --subset-size 100    # larger subset
./tools/demo_caribou.sh --score-threshold 0.3
```

## Expected results

On the default 50-patch subset (229 ground-truth points) you should see numbers
close to:

```
recall ≈ 0.98   precision ≈ 0.89   f1 ≈ 0.93
```

These match the per-patch validation regime reported for the checkpoint
(val F1 = 0.937). The full test set reproduces the paper headline
(F1 = 0.965 at τ = 20 px); see [Datasets](datasets.md). GPU and CPU produce
**identical detections** — only the speed differs (on a Tesla V100 the subset
runs ~25× faster than CPU).

## Compare all OWL models

`tools/demo_owl_models.sh` runs **all released pretrained models** on the caribou
data, visualizes each one's predictions, and prints a side-by-side metrics table.
It downloads the checkpoints from the same [Zenodo record](https://zenodo.org/records/20802844):

```bash
./tools/demo_owl_models.sh                       # all models, auto device
./tools/demo_owl_models.sh --models "owl-c owl-t"  # a subset
./tools/demo_owl_models.sh --device cpu --full     # full test set on CPU
```

| Key | Checkpoint | Registry | Training data |
|---|---|---|---|
| `caribou-owl-c` | `Caribou-OWL-C.pth` | `OWLC` | Caribou (in-domain reference) |
| `owl-c` | `OWL-C.pth` | `OWLC` | General overhead benchmark |
| `owl-t` | `OWL-T.pth` | `OWLT` | General overhead benchmark |
| `owl-d` | `OWL-D.pth` | `OWLD_H` | General overhead benchmark |

Each model writes `demo_data/run_<model>/` (metrics + detections) and
`demo_data/viz_<model>/` (overlays), plus a combined
`demo_data/model_comparison.csv`. Example on the default 50-patch subset:

```
        model device  recall  precision  f1_score
caribou-owl-c    cpu  0.9782     0.8854    0.9295
        owl-c    cpu  0.8734     0.8130    0.8421
        owl-t    cpu  0.8472     0.8661    0.8565
```

!!! note "Zero-shot vs in-domain"
    `owl-c` / `owl-t` / `owl-d` are trained on **other** public overhead datasets,
    not caribou — so on the caribou test set they run **zero-shot** and score below
    the in-domain `caribou-owl-c` (which hits the F1 = 0.965 headline). That gap is
    expected and is exactly what this comparison illustrates.

!!! warning "OWL-D needs a GPU"
    `owl-d` uses a DINOv3 ViT-H+/16 backbone (3.5 GB checkpoint). It is included
    **only when a CUDA GPU is available** and is skipped automatically on CPU-only
    machines. It loads entirely from `OWL-D.pth` (no separate Meta DINOv3 download
    required for inference).

## Manual walkthrough

If you prefer to run the steps yourself:

```bash
# 1. Download the caribou test patches + the caribou OWL-C weights
mkdir -p demo_data/weights demo_data/test
curl -fL -o demo_data/weights/best_model.pth \
    "https://zenodo.org/api/records/20802844/files/Caribou-OWL-C.pth/content"
curl -fL -o demo_data/test.zip \
    "https://zenodo.org/api/records/20802844/files/test.zip/content"
unzip -q demo_data/test.zip -d demo_data/test

# 2. Run OWL-C eval (CPU shown; use ++test.device_name=cuda for GPU)
export OWL_DEMO_DATA="$(pwd)/demo_data"
WANDB_MODE=disabled uv run python tools/test.py test=owlc_caribou_demo \
    ++test.device_name=cpu \
    ++test.model.pth_file="$OWL_DEMO_DATA/weights/best_model.pth" \
    ++test.dataset.root_dir="$OWL_DEMO_DATA/test" \
    ++test.dataset.csv_file="$OWL_DEMO_DATA/test/gt.csv" \
    ++hydra.run.dir="$OWL_DEMO_DATA/run"

# 3. Visualize predictions onto the patches
#    (predictions are saved in the model's down-sampled space; OWL-C uses
#     down_ratio=2, so pass --pred-scale 2 to map them onto the patch)
uv run python tools/visualize_detections.py \
    --detections "$OWL_DEMO_DATA/run/detections.csv" \
    --images-dir "$OWL_DEMO_DATA/test" \
    --output-dir "$OWL_DEMO_DATA/viz" \
    --gt "$OWL_DEMO_DATA/test/gt.csv" \
    --score-threshold 0.2 --pred-scale 2 --all-images
```

The portable demo config lives at `configs/test/owlc_caribou_demo.yaml` — unlike
the author-specific eval configs, it hardcodes no machine paths (they come from
`OWL_DEMO_DATA` or `++` overrides) and defaults to CPU.

## Evaluation operating point

The demo config (`configs/test/owlc_caribou_demo.yaml`) evaluates with:

* **Match radius τ = 20 image px.** `evaluator.threshold: 10` is measured on the
  half-resolution heatmap (`down_ratio: 2`, stitcher `up: False`); ground truth is
  down-sampled by the same factor, so 10 heatmap px = 20 original px.
* **Confidence (peak selection) `adapt_ts: 0.3`** (LMDS), with `neg_ts: 0.1` and a
  `(3, 3)` peak kernel.

This mirrors the per-patch **validation** regime (val F1 ≈ 0.937). The paper's
headline F1 = 0.965 is reported at a slightly different operating point
(c\* = 0.20); see [Datasets](datasets.md).

!!! note "Detection coordinate space"
    With `up: False`, `tools/test.py` writes `detections.csv` in the model's
    **down-sampled** space (x, y in 0…255 for a 512-px patch at `down_ratio=2`).
    Ground truth in `gt.csv` is in original 512-px space. The visualizer's
    `--pred-scale 2` rescales predictions so the two overlay correctly.

## Visualizing detections on your own runs

`tools/visualize_detections.py` works with any `detections.csv` produced by
`tools/test.py`:

```bash
uv run python tools/visualize_detections.py \
    --detections path/to/detections.csv \
    --images-dir path/to/patches \
    --output-dir path/to/viz \
    --pred-scale 2 \
    [--gt path/to/gt.csv] [--score-threshold 0.2] [--all-images]
```

Predicted points are drawn in red; if `--gt` is given, ground-truth points are
drawn in green. Each patch is captioned with its predicted (and GT) point count.
Pass `--pred-scale` equal to the model's `down_ratio` (2 for OWL-C) so the
down-sampled predictions land on the full-resolution patch; ground truth is never
scaled.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `wandb: ERROR ...` or a login prompt | The demo sets `WANDB_MODE=disabled`. Running `tools/test.py` by hand requires `WANDB_MODE=disabled` (or `wandb login`). |
| `CUDA: False` even though `nvidia-smi` shows a GPU | A plain `uv sync` installs the **CPU** build. Install a GPU build with `uv pip install torch torchvision --torch-backend=auto` (see [Installation → GPU support](installation.md#gpu-support)). |
| `RuntimeError: ... unable to find an engine` on an older GPU | Some newer wheels omit kernels for older architectures (e.g. Volta / V100). Use `uv pip install torch torchvision --torch-backend=cu124`, which includes them. |
| Red prediction dots look shifted toward the top-left / "smaller" | Predictions are in the model's down-sampled space — pass `--pred-scale 2` (the OWL-C `down_ratio`) to the visualizer. |
| `ImportError: libGL.so.1` / `libgthread-2.0.so.0` | Image libs need system glib/GL. The project pins `opencv-python-headless`; re-run `uv sync` if it was replaced. |
| Checksum mismatch on weights | A corrupted/partial download. Delete `demo_data/weights/` and re-run. |

## See also

* [Datasets](datasets.md) — dataset details and the Zenodo record
* [Training, Evaluation, and Inference](training.md) — the full eval/inference stack
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T families
