# MegaDetector-Overhead smoke tests

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
