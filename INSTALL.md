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

!!! note "If `uv` is not found after installing"
    The installer places `uv` in `~/.local/bin` and adds it to your `PATH` — but
    that change only applies to **new** shells. If `uv sync` reports
    `uv: command not found` in the same terminal, either open a new shell or run:

    ```bash
    source "$HOME/.local/bin/env"          # if the installer created it
    # ...or add the directory to PATH directly:
    export PATH="$HOME/.local/bin:$PATH"   # for the current shell
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # persist it
    ```

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
   (PyTorch, torchvision, Hydra, wandb, albumentations, the DINOv3
   transitive deps, and the geospatial stack).
4. Install the `megadetector-overhead` root project editably, which
   exposes the `animaloc` package.
5. Install the vendored `dinov3/` subtree editably from its own
   `setup.py`.

A plain `uv sync` installs the **CPU** build of PyTorch, so it works on any
machine and makes no assumption that you have a GPU. To run on a CUDA GPU,
sync one matching CUDA extra instead — see [GPU support](#gpu-support) below.

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

There are two install options. Both pin PyTorch in `uv.lock`, so they're fully
reproducible.

**CPU (default)** — works on any machine, no GPU assumed:

```bash
uv sync            # or: make sync
```

**GPU** — a CUDA build for NVIDIA GPUs:

```bash
uv sync --no-default-groups --group gpu     # or: make sync-gpu
```

Then **activate the virtualenv** and run Python directly — this uses whichever
build you synced and never reverts it:

```bash
source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count(), 'devices')"
python tools/test.py ...
```

(`deactivate` to leave the venv. On Windows: `.venv\Scripts\activate`.)

!!! tip "Use the activated venv, not `uv run`, on the GPU"
    `cpu` is the default group, so a bare `uv sync` **or `uv run`** re-syncs the
    environment back to the CPU build. Activating the venv (above) and calling
    `python` directly avoids that entirely — no per-command flags. If you prefer
    `uv run`, you must pass the group every time:
    `uv run --no-default-groups --group gpu python ...`. The bundled demo scripts
    sidestep this by calling the venv interpreter directly.

### Which GPU build / why `gpu` = CUDA 12.1

The `gpu` group installs **torch 2.5.1+cu121**, whose kernels cover NVIDIA
**Volta (sm_70, e.g. Tesla V100)** through **Hopper (sm_90)**. This is chosen by
the GPU's **architecture**, not your driver version: newer `cu124`/`cu128` wheels
**drop Volta kernels**, so a V100 on those builds raises
`RuntimeError: ... unable to find an engine`. A CUDA-12.1 build runs fine on newer
drivers (they are backward-compatible), so `--group gpu` works across the common
NVIDIA data-center/workstation GPUs.

!!! note "Blackwell / very new GPUs"
    GPUs that require CUDA 12.8+ (e.g. Blackwell, sm_100/120) are not covered by
    cu121. Add a `cu128` group in `pyproject.toml` following the same pattern as
    `gpu` (point its source at the `pytorch-cu128` index) and sync that instead.

## Troubleshooting

* **`torch.cuda.is_available()` is `False` even though `nvidia-smi` shows a GPU**
  — you have the CPU build, or a bare `uv sync`/`uv run` reverted it. Sync the GPU
  group (`uv sync --no-default-groups --group gpu`) and run via the **activated
  venv** (`source .venv/bin/activate`), not bare `uv run` (see GPU support above).
* **`RuntimeError: ... unable to find an engine`** — the wheel lacks kernels for
  your GPU's architecture. The `gpu` group (cu121) covers Volta–Hopper; very new
  GPUs need a cu128 group (see note above).
* **`ImportError: libgthread-2.0.so.0`** — opencv-python's GUI bindings
  need glib. We pin `opencv-python-headless` instead. If the headless
  build was accidentally replaced by `opencv-python`, run
  `uv sync --reinstall-package opencv-python-headless`.
* **`uv sync` resolver failure** — delete `uv.lock`, re-run `uv lock`,
  and re-sync. If the failure persists, open an issue with the resolver
  log.
* **`uv lock` or `uv sync` fails fetching `download-r2.pytorch.org` (TLS /
  connection errors)** — some corporate or cloud networks block PyTorch's CDN
  (`download-r2.pytorch.org`), which serves both wheel *metadata* (needed by
  `uv lock`) and the wheels themselves (needed by `uv sync`). On such networks:
    - `uv lock` can't run locally. The lockfile is regenerated automatically by
      the **Update uv.lock** GitHub Actions workflow
      (`.github/workflows/update-lock.yml`) whenever `pyproject.toml` changes;
      you can also trigger it by hand from the repo's Actions tab, then
      `git pull` to get the refreshed `uv.lock`.
    - `uv sync` may fail to download the PyTorch wheels. Ask your network admin
      to allowlist `download.pytorch.org` and `download-r2.pytorch.org`, run the
      install from an unrestricted network, or install PyTorch manually (e.g.
      `uv pip install torch torchvision --torch-backend=auto`, or download the
      wheels with `curl` from `download.pytorch.org` and
      `uv pip install ./<wheel>.whl`).

  This only affects networks that block that host — a normal `uv sync` downloads
  the CPU wheels without issue.
* **DINOv3 weights file not found at training time** — verify the file
  name under `weights/` matches the `_DEFAULT_WEIGHTS_FILENAME` constant
  on the corresponding `OWLD_*` class in `animaloc/models/owl_d.py`.
