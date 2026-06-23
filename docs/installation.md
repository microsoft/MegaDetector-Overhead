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

Two reproducible install options (both pinned in `uv.lock`):

```bash
uv sync                                    # CPU (default) — or `make sync`
uv sync --no-default-groups --group gpu    # GPU (NVIDIA) — or `make sync-gpu`
```

Then **activate the venv** and run Python directly (this uses whichever build you
synced and never reverts it — no per-command flags):

```bash
source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

The `gpu` group installs **torch 2.5.1+cu121**, which covers NVIDIA **Volta
(sm_70, Tesla V100)** through **Hopper (sm_90)**. This is driven by the GPU
*architecture*, not the driver — newer `cu124`/`cu128` wheels drop Volta kernels
(a V100 on those raises `RuntimeError: ... unable to find an engine`), and a
CUDA-12.1 build runs fine on newer drivers. Avoid bare `uv run` on the GPU (it
re-syncs to the CPU default); use the activated venv. Full details, including how
to add a `cu128` group for Blackwell GPUs, are in
[INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md).

## Next steps

* [Training, Evaluation, and Inference](training.md) — end-to-end workflow
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T family
