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
everywhere, no GPU assumed). To use a GPU, install a matching build **after**
syncing — uv auto-detects your driver and picks the right wheel:

```bash
uv sync                                                  # CPU baseline
uv pip install torch torchvision --torch-backend=auto    # swap in the GPU build
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

`--torch-backend=auto` queries the installed driver and selects the most
compatible PyTorch build (falling back to CPU if no GPU is found). You can also
pin one explicitly, e.g. `--torch-backend=cu124`. Older GPUs (Volta / V100) are
covered by `cu121`/`cu124` but may be missing from `cu128`; if you see
`RuntimeError: ... unable to find an engine`, use `--torch-backend=cu124`. Full
details are in
[INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md).

## Next steps

* [Training, Evaluation, and Inference](training.md) — end-to-end workflow
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T family
