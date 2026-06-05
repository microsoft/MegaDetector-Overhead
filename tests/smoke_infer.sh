#!/usr/bin/env bash
# Smoke test for tools/infer.py.
#
# 1. Ensures /tmp/owl-smoketest/ has the synthetic dataset (regenerates if not).
# 2. Ensures an OWL-C checkpoint is available (runs the OWL-C training smoke
#    if none is found under outputs/).
# 3. Runs tools/infer.py against the val/ split with --model OWLC.
# 4. Verifies the detections CSV exists, has > 0 rows, and contains the
#    expected schema columns.
#
# Designed to run on CPU, no GPU required, ~30 seconds end-to-end after
# the training smoke has already produced a checkpoint.
#
# Exit codes:
#   0 = smoke pass
#   non-zero = something is broken; the failing step prints details

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SMOKETEST_DATA="/tmp/owl-smoketest"
OUTPUT_DIR="/tmp/owl-smoketest-infer"

echo "==> Step 1/4: ensure synthetic dataset at ${SMOKETEST_DATA}"
if [[ ! -d "${SMOKETEST_DATA}/val" ]]; then
    uv run python tests/make_synthetic_dataset.py
fi

echo "==> Step 2/4: ensure an OWL-C checkpoint exists"
CKPT="$(ls -t outputs/*/*/best_model.pth 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
    echo "    no checkpoint found; running owlc_smoketest training (one epoch on CPU)"
    WANDB_MODE=disabled uv run python tools/train.py train=owlc_smoketest
    CKPT="$(ls -t outputs/*/*/best_model.pth | head -1)"
fi
CKPT="$(realpath "${CKPT}")"
echo "    using checkpoint: ${CKPT}"

echo "==> Step 3/4: run tools/infer.py --model OWLC"
mkdir -p "${OUTPUT_DIR}"
WANDB_MODE=disabled uv run python tools/infer.py \
    "${SMOKETEST_DATA}/val/" \
    "${CKPT}" \
    --model OWLC \
    -device cpu \
    --output-dir "${OUTPUT_DIR}"

echo "==> Step 4/4: verify output"
CSV="$(ls -t "${OUTPUT_DIR}"/*_detections.csv | head -1)"
if [[ -z "${CSV}" ]]; then
    echo "    FAIL: no detections CSV under ${OUTPUT_DIR}"
    exit 1
fi
ROW_COUNT="$(($(wc -l < "${CSV}") - 1))"
echo "    found: ${CSV} (${ROW_COUNT} detections)"

if [[ "${ROW_COUNT}" -le 0 ]]; then
    echo "    FAIL: empty detections CSV"
    exit 1
fi

# The exact column set depends on the family (HerdNet vs Detection_Branch);
# at minimum, every detection row must have images, x, y, labels.
HEADER="$(head -1 "${CSV}")"
for col in images x y labels; do
    if [[ "${HEADER}" != *"${col}"* ]]; then
        echo "    FAIL: column ${col!r} missing from CSV header: ${HEADER}"
        exit 1
    fi
done

echo
echo "OK: tools/infer.py smoke test passed (${ROW_COUNT} detections, schema OK)"
echo "    CSV: ${CSV}"
