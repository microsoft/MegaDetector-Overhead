#!/usr/bin/env bash
#
# demo_caribou.sh — end-to-end caribou OWL-C demo.
#
# Downloads the Zenodo OWL-C weights + caribou test patches, runs OWL-C
# inference over a small subset (or the full test set), and renders the
# predictions onto the patches as PNGs.
#
# The device is auto-detected: the demo uses a CUDA GPU when
# torch.cuda.is_available() is True, otherwise it falls back to CPU. It makes
# NO assumption that a GPU is present. Override with --device.
#
# Usage:
#   tools/demo_caribou.sh [options]
#
# Options:
#   --device {auto,cuda,cpu}   Device to run on. Default: auto.
#   --full                     Use the full 2,607-patch test set (slow on CPU).
#                              Default: a ~50-patch subset.
#   --subset-size N            Number of patches in the subset. Default: 50.
#   --score-threshold F        Min confidence for a drawn detection. Default: 0.2.
#   --data-dir DIR             Where to store/download data. Default: ./demo_data.
#   -h, --help                 Show this help and exit.
#
# Re-running is cheap: downloads, extraction, and the subset are skipped when
# they already exist.
set -euo pipefail

ZENODO="https://zenodo.org/api/records/20767534/files"
WEIGHTS_SHA256="8206535afd52e1990fd5e4248e92968dfe05196dd7f44693a2f854a027512bc5"

DEVICE="auto"
FULL=0
SUBSET_SIZE=50
SCORE_THRESHOLD="0.2"
DATA_DIR="demo_data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --full) FULL=1; shift;;
    --subset-size) SUBSET_SIZE="$2"; shift 2;;
    --score-threshold) SCORE_THRESHOLD="$2"; shift 2;;
    --data-dir) DATA_DIR="$2"; shift 2;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

# Resolve repo root (this script lives in <repo>/tools/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Make uv available even when ~/.local/bin is not on PATH.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install it first (see INSTALL.md)." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"   # absolute
WEIGHTS_DIR="$DATA_DIR/weights"
TEST_DIR="$DATA_DIR/test"
SUBSET_DIR="$DATA_DIR/subset"
RUN_DIR="$DATA_DIR/run"
VIZ_DIR="$DATA_DIR/viz"

echo "==> Repo:      $REPO_ROOT"
echo "==> Data dir:  $DATA_DIR"

# ---------------------------------------------------------------------------
# 1. Download + verify weights
# ---------------------------------------------------------------------------
mkdir -p "$WEIGHTS_DIR"
if [[ ! -f "$WEIGHTS_DIR/best_model.pth" ]]; then
  echo "==> Downloading weights.zip (216 MB) ..."
  curl -fL --retry 3 -o "$DATA_DIR/weights.zip" "$ZENODO/weights.zip/content"
  unzip -o -q "$DATA_DIR/weights.zip" -d "$WEIGHTS_DIR"
  rm -f "$DATA_DIR/weights.zip"
fi
echo "==> Verifying weights SHA-256 ..."
GOT_SHA="$(sha256sum "$WEIGHTS_DIR/best_model.pth" | awk '{print $1}')"
if [[ "$GOT_SHA" != "$WEIGHTS_SHA256" ]]; then
  echo "ERROR: weights checksum mismatch!" >&2
  echo "  expected $WEIGHTS_SHA256" >&2
  echo "  got      $GOT_SHA" >&2
  exit 1
fi
echo "    OK ($GOT_SHA)"

# ---------------------------------------------------------------------------
# 2. Download + extract test patches
# ---------------------------------------------------------------------------
mkdir -p "$TEST_DIR"
if [[ ! -f "$TEST_DIR/gt.csv" ]]; then
  echo "==> Downloading test.zip (1.2 GB) ..."
  curl -fL --retry 3 -o "$DATA_DIR/test.zip" "$ZENODO/test.zip/content"
  echo "==> Extracting test patches ..."
  unzip -o -q "$DATA_DIR/test.zip" -d "$TEST_DIR"
  rm -f "$DATA_DIR/test.zip"
fi
echo "    test patches: $(find "$TEST_DIR" -name '*.png' | wc -l)"

# ---------------------------------------------------------------------------
# 3. Choose evaluation directory (subset or full)
# ---------------------------------------------------------------------------
if [[ "$FULL" -eq 1 ]]; then
  EVAL_DIR="$TEST_DIR"
  echo "==> Using FULL test set."
else
  echo "==> Building ${SUBSET_SIZE}-patch subset ..."
  SUBSET_SIZE="$SUBSET_SIZE" TEST_DIR="$TEST_DIR" SUBSET_DIR="$SUBSET_DIR" \
    uv run python - <<'PY'
import os, random, shutil
import pandas as pd

test_dir = os.environ["TEST_DIR"]
sub = os.environ["SUBSET_DIR"]
size = int(os.environ["SUBSET_SIZE"])
random.seed(42)

gt = pd.read_csv(os.path.join(test_dir, "gt.csv"))
all_png = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))
neg = [f for f in all_png if "_neg_" in f]
pos = sorted(gt["images"].unique().tolist())

n_neg = max(1, size // 5)            # ~20% background patches
n_pos = max(1, size - n_neg)
pos_pick = sorted(random.sample(pos, min(n_pos, len(pos))))
neg_pick = sorted(random.sample(neg, min(n_neg, len(neg))))
picks = pos_pick + neg_pick

if os.path.isdir(sub):
    shutil.rmtree(sub)
os.makedirs(sub)
for f in picks:
    shutil.copy(os.path.join(test_dir, f), os.path.join(sub, f))
gt[gt["images"].isin(picks)].to_csv(os.path.join(sub, "gt.csv"), index=False)
print(f"    subset: {len(picks)} patches "
      f"({len(pos_pick)} annotated + {len(neg_pick)} background)")
PY
  EVAL_DIR="$SUBSET_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Resolve device (auto-detect → cuda if available, else cpu)
# ---------------------------------------------------------------------------
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo cpu)"
fi
echo "==> Device: $DEVICE"
if [[ "$DEVICE" == "cuda" ]]; then
  uv run python -c 'import torch; print("    GPU:", torch.cuda.get_device_name(0))' 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 5. Run OWL-C evaluation
# ---------------------------------------------------------------------------
rm -rf "$RUN_DIR"
echo "==> Running OWL-C inference ..."
export OWL_DEMO_DATA="$DATA_DIR"
WANDB_MODE=disabled uv run python tools/test.py test=owlc_caribou_demo \
  ++test.device_name="$DEVICE" \
  ++test.model.pth_file="$WEIGHTS_DIR/best_model.pth" \
  ++test.dataset.root_dir="$EVAL_DIR" \
  ++test.dataset.csv_file="$EVAL_DIR/gt.csv" \
  ++hydra.run.dir="$RUN_DIR"

DET_CSV="$RUN_DIR/detections.csv"
echo "==> Metrics:     $RUN_DIR/metrics_results.csv"
echo "==> Detections:  $DET_CSV"

# ---------------------------------------------------------------------------
# 6. Visualize predictions on the patches
# ---------------------------------------------------------------------------
echo "==> Rendering prediction overlays ..."
uv run python tools/visualize_detections.py \
  --detections "$DET_CSV" \
  --images-dir "$EVAL_DIR" \
  --output-dir "$VIZ_DIR" \
  --gt "$EVAL_DIR/gt.csv" \
  --score-threshold "$SCORE_THRESHOLD" \
  --pred-scale 2 \
  --all-images

echo
echo "==> Done. Annotated patches: $VIZ_DIR"
echo "    (green = ground truth, red = OWL-C predictions)"
