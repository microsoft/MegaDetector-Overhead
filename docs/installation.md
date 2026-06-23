---
description: "How to install and set up MegaDetector-Overhead for wildlife detection in drone and aerial imagery."
tags:
  - MegaDetector-Overhead installation
  - aerial detection setup
  - PyTorch-Wildlife
  - drone wildlife survey
  - overhead imagery AI
  - uv install
---

# Installation

MegaDetector-Overhead uses [uv](https://github.com/astral-sh/uv) to manage
a single Python 3.11 environment containing the `animaloc` training/eval
package, the DINOv3 encoder, and all transitive dependencies.

For the full step-by-step (including the DINOv3 weights download and
troubleshooting), see [INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md)
at the repo root.

## Quickstart

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
# The installer only updates PATH for *new* shells, so make uv available now:
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone and sync
git clone https://github.com/microsoft/MegaDetector-Overhead
cd MegaDetector-Overhead
uv sync                  # CPU PyTorch; for a GPU see "GPU support" below

# 3. Smoke test
uv run python -c "import animaloc.models, dinov3; print('OK')"
```

`uv sync` will:

1. Auto-download CPython 3.11 if missing.
2. Build `.venv/` at the repo root.
3. Install the pinned dependency set from `uv.lock` (~140 packages).
4. Install the root `animaloc` package editably.
5. Install the vendored `dinov3/` package editably from its own setup.py.

## Requirements

* Python ≥ 3.11 (DINOv3 requires PEP 604 syntax).
* Linux x86_64 with glibc ≥ 2.28. Other platforms work but are untested.
* ~6 GB free for DINOv3 weights (downloaded separately — see
  [INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md)).
* CUDA-capable GPU recommended for training; CPU works for small
  inference jobs.

## GPU support

A plain `uv sync` installs the **CPU** build of PyTorch on every platform (works
everywhere, no GPU assumed). To use a GPU, sync the **dependency-group** matching
your NVIDIA driver — this installs a CUDA build that is pinned in `uv.lock`
(reproducible):

```bash
# Pick ONE group matching your driver's CUDA version (`nvidia-smi`):
uv sync --no-default-groups --group cu121   # CUDA 12.1+ (incl. Volta / V100)
uv sync --no-default-groups --group cu124   # CUDA 12.4+
uv sync --no-default-groups --group cu128   # CUDA 12.8+ (recent GPUs)

uv run --no-default-groups --group cu121 \
    python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Pick the highest `cuXXX` that is **≤** your driver's CUDA version. **Volta GPUs
(Tesla V100) require `cu121`** — `cu124`/`cu128` drop Volta kernels and raise
`RuntimeError: ... unable to find an engine`. Because `cpu` is the default group,
a bare `uv sync`/`uv run` returns to CPU, so pass the group on every GPU command
(or `uv run --no-sync` after syncing). Full details are in
[INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md).

## Next steps

* [Training, Evaluation, and Inference](training.md) — end-to-end workflow
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T family
