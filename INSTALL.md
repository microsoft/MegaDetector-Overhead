# Installation

MegaDetector-Overhead is installed with [uv](https://github.com/astral-sh/uv).
A single `uv sync` builds a Python 3.11 environment with all dependencies,
the `animaloc` training/eval package, and the vendored `dinov3/` encoder
in editable mode.

## 1. Install uv (one-time, per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(See the [uv docs](https://github.com/astral-sh/uv) for Windows / macOS
installers.)

## 2. Clone and sync the environment

```bash
git clone https://github.com/microsoft/MegaDetector-Overhead
cd MegaDetector-Overhead
uv sync
```

This will:

1. Download CPython 3.11 if not already present on the host (uv reads
   the version from `.python-version`).
2. Create `.venv/` at the repo root.
3. Resolve and install the locked dependency set from `uv.lock`
   (140 packages including PyTorch, torchvision, Hydra, wandb,
   albumentations, the DINOv3 transitive deps, and the geospatial stack).
4. Install the `megadetector-overhead` root project editably, which
   exposes the `animaloc` package.
5. Install the vendored `dinov3/` subtree editably from its own
   `setup.py`.

Activate the venv (optional — `uv run <cmd>` works without activation):

```bash
source .venv/bin/activate
```

## 3. Smoke test

```bash
uv run python -c "import animaloc.models, dinov3; print('OK')"
```

Expected output: `OK`.

A more thorough check that lists the registered models:

```bash
uv run python -c "from animaloc.models import MODELS; print(sorted(MODELS.registry_names))"
```

Expected:

```
['FasterRCNNResNetFPN', 'HerdNet', 'OWLC', 'OWLD_B', 'OWLD_H', 'OWLD_L', 'OWLD_S', 'OWLT', 'SemSegDLA']
```

For end-to-end verification (forward pass on every OWL model, plus a
mini training + eval run on synthetic data), use the bundled smoke
tests:

```bash
uv run python tests/smoke_forward.py
uv run python tests/make_synthetic_dataset.py
WANDB_MODE=disabled uv run python tools/train.py train=owlc_smoketest
```

See [`tests/README.md`](https://github.com/microsoft/MegaDetector-Overhead/blob/main/tests/README.md).

## 4. Download DINOv3 weights (one-time, ~6 GB total)

The OWL-D family loads DINOv3 ViT backbones at training time. Weights
are NOT vendored (5.8 GB, governed by the
[DINOv3 License](dinov3/LICENSE.md)). Download them from Meta's official
release and place them under `weights/` at the repo root:

```bash
mkdir -p weights/
# Replace <URL> with the official Meta DINOv3 release link for each backbone
# (see https://github.com/facebookresearch/dinov3 for the download form).
wget <URL> -O weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
wget <URL> -O weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
wget <URL> -O weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
wget <URL> -O weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

The default filenames above match the paths hard-coded into
`animaloc/models/owl_d.py` (`_DEFAULT_WEIGHTS_FILENAME`). To use a
different filename or directory, override `weights` in the training
config.

`weights/` is `.gitignore`d.

## 5. (Optional) Install development dependencies

```bash
uv sync --extra dev    # adds pytest, ruff, mypy
uv sync --extra docs   # adds mkdocs + material theme for building this site
```

## Python version

DINOv3 requires Python ≥ 3.11 (uses PEP 604 union syntax internally),
so the project pins `>=3.11,<3.13`. uv reads `.python-version` and
will auto-download CPython 3.11 if your system Python is older or newer.

## GPU support

PyTorch wheels include CPU and CUDA builds. Verify CUDA after install:

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count(), 'devices')"
```

For other CUDA versions or ROCm, see the
[PyTorch install matrix](https://pytorch.org/get-started/locally/) and
add the appropriate index URL to `pyproject.toml`'s `[tool.uv.sources]`
section.

## Troubleshooting

* **`ImportError: libgthread-2.0.so.0`** — opencv-python's GUI bindings
  need glib. We pin `opencv-python-headless` instead. If the headless
  build was accidentally replaced by `opencv-python`, run
  `uv sync --reinstall-package opencv-python-headless`.
* **`uv sync` resolver failure** — delete `uv.lock`, re-run `uv lock`,
  and re-sync. If the failure persists, open an issue with the resolver
  log.
* **DINOv3 weights file not found at training time** — verify the file
  name under `weights/` matches the `_DEFAULT_WEIGHTS_FILENAME` constant
  on the corresponding `OWLD_*` class in `animaloc/models/owl_d.py`.
