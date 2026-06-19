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

1. Download `weights.zip` (216 MB) and `test.zip` (1.2 GB) from Zenodo into
   `demo_data/` (skipped if already present).
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

## Manual walkthrough

If you prefer to run the steps yourself:

```bash
# 1. Download + extract
mkdir -p demo_data/weights demo_data/test
curl -fL -o demo_data/weights.zip \
    "https://zenodo.org/api/records/20767534/files/weights.zip/content"
curl -fL -o demo_data/test.zip \
    "https://zenodo.org/api/records/20767534/files/test.zip/content"
unzip -q demo_data/weights.zip -d demo_data/weights
unzip -q demo_data/test.zip   -d demo_data/test

# 2. Run OWL-C eval (CPU shown; use ++test.device_name=cuda for GPU)
export OWL_DEMO_DATA="$(pwd)/demo_data"
WANDB_MODE=disabled uv run python tools/test.py test=owlc_caribou_demo \
    ++test.device_name=cpu \
    ++test.model.pth_file="$OWL_DEMO_DATA/weights/best_model.pth" \
    ++test.dataset.root_dir="$OWL_DEMO_DATA/test" \
    ++test.dataset.csv_file="$OWL_DEMO_DATA/test/gt.csv" \
    ++hydra.run.dir="$OWL_DEMO_DATA/run"

# 3. Visualize predictions onto the patches
uv run python tools/visualize_detections.py \
    --detections "$OWL_DEMO_DATA/run/detections.csv" \
    --images-dir "$OWL_DEMO_DATA/test" \
    --output-dir "$OWL_DEMO_DATA/viz" \
    --gt "$OWL_DEMO_DATA/test/gt.csv" \
    --score-threshold 0.2 --all-images
```

The portable demo config lives at `configs/test/owlc_caribou_demo.yaml` — unlike
the author-specific eval configs, it hardcodes no machine paths (they come from
`OWL_DEMO_DATA` or `++` overrides) and defaults to CPU.

## Visualizing detections on your own runs

`tools/visualize_detections.py` works with any `detections.csv` produced by
`tools/test.py`:

```bash
uv run python tools/visualize_detections.py \
    --detections path/to/detections.csv \
    --images-dir path/to/patches \
    --output-dir path/to/viz \
    [--gt path/to/gt.csv] [--score-threshold 0.2] [--all-images]
```

Predicted points are drawn in red; if `--gt` is given, ground-truth points are
drawn in green. Each patch is captioned with its predicted (and GT) point count.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `wandb: ERROR ...` or a login prompt | The demo sets `WANDB_MODE=disabled`. Running `tools/test.py` by hand requires `WANDB_MODE=disabled` (or `wandb login`). |
| `CUDA: False` even though `nvidia-smi` shows a GPU | The pinned PyTorch is built for **CUDA 13**. If your NVIDIA driver only supports CUDA ≤ 12.x, `torch.cuda.is_available()` is `False` and the demo correctly runs on CPU. Update your driver to a CUDA-13-capable version, or install a PyTorch build matching your local CUDA (see the [PyTorch install matrix](https://pytorch.org/get-started/locally/)). |
| `RuntimeError: ... unable to find an engine` on an older GPU | PyTorch wheels may omit kernels for older architectures (e.g. Volta / V100). Use a PyTorch build that includes your GPU's compute capability. |
| `ImportError: libGL.so.1` / `libgthread-2.0.so.0` | Image libs need system glib/GL. The project pins `opencv-python-headless`; re-run `uv sync` if it was replaced. |
| Checksum mismatch on weights | A corrupted/partial download. Delete `demo_data/weights/` and re-run. |

## See also

* [Datasets](datasets.md) — dataset details and the Zenodo record
* [Training, Evaluation, and Inference](training.md) — the full eval/inference stack
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T families
