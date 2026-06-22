#!/usr/bin/env bash
#
# demo_owl_models.sh — run and compare the released OWL models on caribou.
#
# Downloads the caribou test patches and the pretrained OWL checkpoints from
# Zenodo (record 20802844), runs each model over a subset (or the full test set),
# visualizes the predictions, and prints a side-by-side metrics comparison.
#
# Models:
#   caribou-owl-c  Caribou-OWL-C.pth  (OWLC)    in-domain caribou model (reference)
#   owl-c          OWL-C.pth          (OWLC)    general overhead benchmark
#   owl-t          OWL-T.pth          (OWLT)    general overhead benchmark
#   owl-d          OWL-D.pth          (OWLD_H)  general; DINOv3 ViT-H+/16 (GPU only)
#
# The general OWL-C/OWL-T/OWL-D models are trained on public overhead datasets,
# NOT caribou — running them here is a zero-shot, cross-domain check, so expect
# lower numbers than the in-domain caribou-owl-c.
#
# OWL-D is heavy (3.5 GB, ViT-H+/16). It runs ONLY when a CUDA GPU is available;
# on CPU-only machines it is skipped automatically.
#
# Usage:
#   tools/demo_owl_models.sh [options]
#
# Options:
#   --device {auto,cuda,cpu}  Device. Default: auto (cuda if available, else cpu).
#   --models "a b c"          Space-separated subset of:
#                             caribou-owl-c owl-c owl-t owl-d. Default: all.
#   --full                    Use the full 2,607-patch test set (slow). Default: subset.
#   --subset-size N           Subset size. Default: 50.
#   --score-threshold F       Min confidence for a drawn detection. Default: 0.2.
#   --data-dir DIR            Data location. Default: ./demo_data.
#   -h, --help                Show this help and exit.
#
# Environment:
#   UV_RUN   Command used to run Python (default "uv run"). Set to
#            "uv run --no-sync" on networks that block PyTorch's wheel host so the
#            existing .venv is used without a re-sync.
#
# Re-running is cheap: downloads, extraction, and the subset are skipped when
# they already exist.
set -euo pipefail

ZENODO="https://zenodo.org/api/records/20802844/files"
CARIBOU_SHA256="8206535afd52e1990fd5e4248e92968dfe05196dd7f44693a2f854a027512bc5"

DEVICE="auto"
MODELS="caribou-owl-c owl-c owl-t owl-d"
FULL=0
SUBSET_SIZE=50
SCORE_THRESHOLD="0.2"
DATA_DIR="demo_data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --full) FULL=1; shift;;
    --subset-size) SUBSET_SIZE="$2"; shift 2;;
    --score-threshold) SCORE_THRESHOLD="$2"; shift 2;;
    --data-dir) DATA_DIR="$2"; shift 2;;
    -h|--help) sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install it first (see INSTALL.md)." >&2
  exit 1
fi
UV_RUN="${UV_RUN:-uv run}"

mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
MODELS_DIR="$DATA_DIR/models"
TEST_DIR="$DATA_DIR/test"
SUBSET_DIR="$DATA_DIR/subset"
mkdir -p "$MODELS_DIR" "$TEST_DIR"

echo "==> Repo:      $REPO_ROOT"
echo "==> Data dir:  $DATA_DIR"
echo "==> Models:    $MODELS"

# Per-model metadata: file name, config, registry display, GPU-only flag.
model_file()   { case "$1" in
  caribou-owl-c) echo "Caribou-OWL-C.pth";; owl-c) echo "OWL-C.pth";;
  owl-t) echo "OWL-T.pth";; owl-d) echo "OWL-D.pth";;
  *) echo ""; esac; }
model_config() { case "$1" in
  caribou-owl-c|owl-c) echo "owlc_caribou_demo";; owl-t) echo "owlt_caribou_demo";;
  owl-d) echo "owld_caribou_demo";; *) echo ""; esac; }
model_gpu_only() { [[ "$1" == "owl-d" ]]; }

# ---------------------------------------------------------------------------
# 1. Resolve device (auto-detect -> cuda if available, else cpu)
# ---------------------------------------------------------------------------
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$($UV_RUN python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo cpu)"
fi
echo "==> Device:    $DEVICE"
if [[ "$DEVICE" == "cuda" ]]; then
  $UV_RUN python -c 'import torch; print("    GPU:", torch.cuda.get_device_name(0))' 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 2. Download the test patches
# ---------------------------------------------------------------------------
if [[ ! -f "$TEST_DIR/gt.csv" ]]; then
  echo "==> Downloading test.zip (1.2 GB) ..."
  curl -fL --retry 3 -o "$DATA_DIR/test.zip" "$ZENODO/test.zip/content"
  echo "==> Extracting test patches ..."
  unzip -o -q "$DATA_DIR/test.zip" -d "$TEST_DIR"
  rm -f "$DATA_DIR/test.zip"
fi
echo "    test patches: $(find "$TEST_DIR" -name '*.png' | wc -l)"

# ---------------------------------------------------------------------------
# 3. Build the eval directory (subset or full)
# ---------------------------------------------------------------------------
if [[ "$FULL" -eq 1 ]]; then
  EVAL_DIR="$TEST_DIR"
  echo "==> Using FULL test set."
else
  echo "==> Building ${SUBSET_SIZE}-patch subset ..."
  SUBSET_SIZE="$SUBSET_SIZE" TEST_DIR="$TEST_DIR" SUBSET_DIR="$SUBSET_DIR" \
    $UV_RUN python - <<'PY'
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

n_neg = max(1, size // 5)
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

export OWL_DEMO_DATA="$DATA_DIR"
SUMMARY_CSV="$DATA_DIR/model_comparison.csv"
echo "model,device,recall,precision,f1_score,mae,rmse" > "$SUMMARY_CSV"

# ---------------------------------------------------------------------------
# 4. Run each selected model: download -> eval -> visualize
# ---------------------------------------------------------------------------
for m in $MODELS; do
  file="$(model_file "$m")"; cfg="$(model_config "$m")"
  if [[ -z "$file" || -z "$cfg" ]]; then
    echo "==> [skip] unknown model '$m'"; continue
  fi

  mdev="$DEVICE"
  if model_gpu_only "$m" && [[ "$DEVICE" != "cuda" ]]; then
    echo "==> [skip] $m needs a CUDA GPU (device is '$DEVICE')."
    continue
  fi

  echo
  echo "================ $m ($file) ================"
  if [[ ! -f "$MODELS_DIR/$file" ]]; then
    echo "==> Downloading $file ..."
    curl -fL --retry 3 -o "$MODELS_DIR/$file" "$ZENODO/$file/content"
  fi
  if [[ "$m" == "caribou-owl-c" ]]; then
    got="$(sha256sum "$MODELS_DIR/$file" | awk '{print $1}')"
    if [[ "$got" != "$CARIBOU_SHA256" ]]; then
      echo "ERROR: $file checksum mismatch (expected $CARIBOU_SHA256, got $got)" >&2
      exit 1
    fi
    echo "    SHA-256 OK"
  fi

  run_dir="$DATA_DIR/run_$m"; viz_dir="$DATA_DIR/viz_$m"
  rm -rf "$run_dir"
  echo "==> Evaluating ($mdev) ..."
  WANDB_MODE=disabled $UV_RUN python tools/test.py "test=$cfg" \
    ++test.device_name="$mdev" \
    ++test.model.pth_file="$MODELS_DIR/$file" \
    ++test.dataset.root_dir="$EVAL_DIR" \
    ++test.dataset.csv_file="$EVAL_DIR/gt.csv" \
    ++hydra.run.dir="$run_dir"

  echo "==> Visualizing ..."
  $UV_RUN python tools/visualize_detections.py \
    --detections "$run_dir/detections.csv" \
    --images-dir "$EVAL_DIR" \
    --output-dir "$viz_dir" \
    --gt "$EVAL_DIR/gt.csv" \
    --score-threshold "$SCORE_THRESHOLD" \
    --pred-scale 2 \
    --all-images

  # Append the binary-row metrics to the comparison CSV.
  MODEL="$m" DEVICE="$mdev" RUN_DIR="$run_dir" SUMMARY_CSV="$SUMMARY_CSV" \
    $UV_RUN python - <<'PY'
import os, pandas as pd
run = os.environ["RUN_DIR"]
df = pd.read_csv(os.path.join(run, "metrics_results.csv"))
row = df[df["class"].astype(str) == "binary"]
row = row.iloc[0] if len(row) else df.iloc[-1]
with open(os.environ["SUMMARY_CSV"], "a") as fh:
    fh.write("{m},{d},{r:.4f},{p:.4f},{f:.4f},{mae:.4f},{rmse:.4f}\n".format(
        m=os.environ["MODEL"], d=os.environ["DEVICE"],
        r=row["recall"], p=row["precision"], f=row["f1_score"],
        mae=row["mae"], rmse=row["rmse"]))
PY
  echo "==> $m done. Metrics: $run_dir/metrics_results.csv  |  Overlays: $viz_dir"
done

# ---------------------------------------------------------------------------
# 5. Comparison summary
# ---------------------------------------------------------------------------
echo
echo "================ Model comparison (caribou test subset) ================"
SUMMARY_CSV="$SUMMARY_CSV" $UV_RUN python - <<'PY'
import os, pandas as pd
df = pd.read_csv(os.environ["SUMMARY_CSV"])
if len(df):
    print(df.to_string(index=False))
else:
    print("(no models were run)")
PY
echo
echo "Comparison CSV: $SUMMARY_CSV"
echo "Per-model overlays: $DATA_DIR/viz_<model>/   (green = GT, red = prediction)"
echo "Note: general OWL-C/OWL-T/OWL-D are zero-shot on caribou (trained on other"
echo "      overhead datasets); caribou-owl-c is the in-domain reference."
