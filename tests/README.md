# MegaDetector-Overhead smoke tests

## OWL notebook helper tests

These focused tests use temporary images and synthetic heatmaps, without network
access or released model weights:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_owl_demo.py' -v
```

They cover verified sample extraction/cache reuse, invalid archives and downloads,
explicit model/device choices, small/odd/multi-tile inputs, original-pixel
coordinates, threshold/count consistency, empty detections, saved overlays,
source mapping, exact release-asset staging, and isolated model-cache selection.

Real-asset checks for the notebook backend:

```bash
python tools/demo_owl_notebook.py prepare
python tools/demo_owl_notebook.py infer \
    --images-dir demo_data/owl_notebook/OWL_DATA \
    --selection patches --output-dir demo_data/owl_notebook/check_owld_patches
python tools/demo_owl_notebook.py infer \
    --images-dir demo_data/owl_notebook/OWL_DATA \
    --selection full --output-dir demo_data/owl_notebook/check_owld_full
python tools/demo_owl_notebook.py infer --model owl-c --device cpu \
    --images-dir demo_data/owl_notebook/OWL_DATA \
    --selection patches --output-dir demo_data/owl_notebook/check_owlc_cpu
```

Use a new output directory on each run. The default OWL-D route requires CUDA;
the third command explicitly selects OWL-C for CPU. These samples have no ground
truth, so predicted counts are not accuracy measurements.

For actual notebook validation, install the `notebook` extra while preserving
your selected CPU/GPU dependency group. This runner executes **all notebook
cells from a fresh kernel** and only changes configuration in the executed copy:

```bash
python tests/execute_owl_notebook.py --full \
    --output demo_data/owl_notebook/executed_owld.ipynb
python tests/execute_owl_notebook.py --model owl-c --device cpu \
    --output demo_data/owl_notebook/executed_owlc_cpu.ipynb
```

Choose a new output filename on reruns. The runner retains partial output if a
cell fails and returns the execution error, rather than treating it as success.
Executed artifacts belong in ignored `demo_data/`, not in the source notebook.
The full-resolution switch processes both large images without resizing.
Keep source notebook outputs cleared.
Use `--model-cache PATH` to select a separate model cache in the executed
notebook without changing its source or the existing default cache.
Local backend checks do not establish that hosted Colab or public asset
downloads work; test those paths separately before release.

## Existing training and forward-pass smoke tests

Minimal end-to-end smoke tests that verify a fresh install can:

1. **Construct and forward-pass all 6 OWL models** (OWL-C, OWL-T,
   OWL-D-S/B/L/H) on a random (1, 3, 512, 512) input.
2. **Build a tiny synthetic point-annotated dataset**.
3. **Run one training epoch** of OWL-C (CPU, batch_size=1) on that
   dataset using `tools/train.py`.
4. **Run evaluation** on the resulting checkpoint using `tools/test.py`.

These are NOT a substitute for running on real data — they verify the
plumbing only (configs parse, datasets load, models forward, gradients
flow, checkpoints save/load, metrics compute).

## Prerequisites

* `uv sync` has succeeded (see [INSTALL.md](../INSTALL.md)).
* For OWL-D smoke tests only: DINOv3 backbone weights are present
  under `weights/` at the repo root. See INSTALL.md for download
  instructions. Without weights, OWL-D tests are skipped.

## Run

Activate the venv first (`source .venv/bin/activate` after `uv sync`), then run
with plain `python`:

```bash
# 1. Forward-pass smoke test (all 6 OWL models)
python tests/smoke_forward.py

# 2. Synthetic dataset (writes /tmp/owl-smoketest/)
python tests/make_synthetic_dataset.py

# 3. OWL-C training smoke (one epoch, batch_size=1, CPU)
WANDB_MODE=disabled python tools/train.py train=owlc_smoketest

# 4. OWL-C evaluation smoke on the checkpoint produced above
CKPT=$(ls -t outputs/*/*/best_model.pth | head -1 | xargs realpath)
WANDB_MODE=disabled python tools/test.py test=owlc_smoketest \
    "++test.model.pth_file=$CKPT"

# 5. (Optional) OWL-D-S training+eval smoke (requires DINOv3 weights)
WANDB_MODE=disabled python tools/train.py train=owld_s_smoketest
CKPT=$(ls -t outputs/*/*/best_model.pth | head -1 | xargs realpath)
WANDB_MODE=disabled python tools/test.py test=owld_s_smoketest \
    "++test.model.pth_file=$CKPT"
```

Expected runtime on CPU: ~1 min for forward-pass + dataset, ~30 s for
OWL-C train, ~5 s for OWL-C eval, ~25 s for OWL-D-S train (frozen
backbone), ~5 s for OWL-D-S eval.

Metrics on synthetic data are meaningless (4 train + 2 val images, 1
epoch, batch_size=1, random init). What matters is that every step
completes without error.

## Expected output

The forward-pass test ends with:

```
=== Forward-pass smoke summary ===
  OWLC      PASS            out shape=(1, 1, 256, 256)
  OWLT      PASS            out shape=(1, 1, 256, 256)
  OWLD_S    PASS            out shapes=[(1, 1, 256, 256)]
  OWLD_B    PASS            out shapes=[(1, 1, 256, 256)]
  OWLD_L    PASS            out shapes=[(1, 1, 256, 256)]
  OWLD_H    PASS            out shapes=[(1, 1, 256, 256)]

6/6 models passed; exit=0
```

OWL-D-* will be skipped (CONSTRUCT_FAIL with `FileNotFoundError` on the
weight path) if DINOv3 weights are not present under `weights/`.

The training smoke run completes with output like:

```
[TRAINING] - Epoch: [1] [4/4] eta: ... loss: ...
[VALIDATION] - Epoch: [1] ...
Best model saved - Epoch 1 - Validation value: ...
Training complete | Best f1_score: ... at epoch 1
```

The evaluation smoke run writes `metrics_results.csv`,
`confusion_matrix.csv`, `detections.csv`, and `plots/precision_recall_curve.png`
under `outputs/<date>/<time>/`.
