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

# 3. Activate the venv, then smoke test with plain `python`
source .venv/bin/activate
python -c "import animaloc.models, dinov3; print('OK')"
```

`uv sync` will:

1. Auto-download CPython 3.11 if missing.
2. Build `.venv/` at the repo root.
3. Install the pinned dependency set from `uv.lock` (~140 packages).
4. Install the root `animaloc` package editably.
5. Install the vendored `dinov3/` package editably from its own setup.py.

## Requirements

* Python ≥ 3.11 and < 3.13 for the project backend.
* Linux x86_64 with glibc ≥ 2.28. Other platforms work but are untested.
* ~6 GB free for separate DINOv3 training weights (see
  [INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md)).
  The OWL inference notebook instead uses the approximately 3.5 GB full
  `OWL-D.pth` / `OWLD_H` checkpoint, which already includes its frozen
  backbone and requires no separate Meta download.
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

## OWL inference notebook

Install the notebook extra while preserving your selected PyTorch group:

```bash
uv sync --locked --no-default-groups --group gpu --extra notebook
.venv/bin/python -m jupyterlab notebooks/owl_inference_demo.ipynb
```

For CPU instead, use `uv sync --locked --extra notebook` and explicitly select
`MODEL='owl-c'`, `DEVICE='cpu'` in the notebook. Default OWL-D requires CUDA and
never silently substitutes another model. In VS Code, select the checkout's
`.venv` kernel; optional named-kernel registration and Windows commands are in
[INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md#owl-inference-notebook).
Local setup reuses the existing backend without automatically re-syncing it.

The Colab setup path uses a configurable repository ref and a uv-managed
Python 3.11 backend, even if the hosted kernel has a different Python version.
It clones only into an absent dedicated directory and validates existing
checkouts without resetting them. **Actual Colab validation is pending** and
does not block the local Jupyter/VS Code release;
use `REPO_REF='owl-notebook-v1'` for the matching release.
The manifest downloads the exact sample ZIP from that GitHub release.
SheepCounter Public Domain, HerdNet CC BY-NC-SA 4.0,
and caribou CC BY-NC-SA 4.0 terms are documented with their evidence sources;
both public model downloads and their published checksums have
been verified in isolated caches, although network requests may fail
intermittently. Set `ARCHIVE` to use the exact `OWL_SAMPLE_DATA.zip` locally
instead. See [demo/access details](demo.md#owl-inference-notebook).

## Next steps

* [Training, Evaluation, and Inference](training.md) — end-to-end workflow
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T family
* [Demos](demo.md) — OWL inference notebook and existing caribou shell workflows
