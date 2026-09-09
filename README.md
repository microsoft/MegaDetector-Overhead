# MegaDetector-Overhead

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![PyTorch-Wildlife](https://img.shields.io/badge/PyTorch--Wildlife-ecosystem-green)](https://github.com/microsoft/Biodiversity)

**Open-source AI for detecting wildlife in overhead and aerial imagery.**

MegaDetector-Overhead extends the [MegaDetector](https://github.com/microsoft/MegaDetector) detection framework to drone and UAV survey imagery, handling the unique challenges of overhead perspectives: small targets, variable altitude, and nadir-angle distortion. It is powered by [PyTorch-Wildlife](https://github.com/microsoft/Biodiversity) and is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) ecosystem.

This repository ships the training, evaluation, and inference stack for the **OWL** model family:

| Model | Backbone | Notes |
|---|---|---|
| **OWL-C** | DLA-34 (HerdNet detection branch) | Baseline; fast inference |
| **OWL-T** | DLA-34 + Swin transformer multiscale residual | Sharper localization on cluttered backgrounds |
| **OWL-D** (S / B / L / H) | DINOv3 ViT + DPT decoder | Highest quality; foundation-model encoder |

The legacy `HerdNet` multi-class model is also available. See [Model Zoo](docs/model_zoo.md) for the full list.

**Pretrained weights:** all OWL benchmark checkpoints (the caribou-specific
`Caribou-OWL-C` plus the general `OWL-C` / `OWL-T` / `OWL-D` models) are released
on [Zenodo](https://zenodo.org/records/20802844). See [Datasets](docs/datasets.md).

---

## Documentation

* [Installation](INSTALL.md) — environment, notebook setup, and optional training backbone weights
* [OWL inference notebook](notebooks/owl_inference_demo.ipynb) — OWL-D-first, annotation-free prediction on four patches; optional tiled full images and explicit CPU OWL-C
* [Datasets](docs/datasets.md) — caribou data + pretrained model weights (Zenodo)
* [Model Zoo](docs/model_zoo.md) — the OWL-C / OWL-D / OWL-T family + pretrained checkpoints
* [Demos](docs/demo.md) — notebook setup/access notes and the existing caribou evaluation/model-comparison shell demos
* [Training, Evaluation, and Inference](docs/training.md) — end-to-end workflow

---

## Quick Start

The environment is managed with [uv](https://github.com/astral-sh/uv). One `uv sync` builds a Python 3.11 venv with all dependencies, the `animaloc` training package, and the vendored DINOv3 encoder.

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
# The installer only updates PATH for *new* shells, so make uv available now:
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone and sync
git clone https://github.com/microsoft/MegaDetector-Overhead
cd MegaDetector-Overhead
uv sync                  # CPU build of PyTorch (works everywhere)
# For a GPU, sync the GPU build instead:
#   uv sync --no-default-groups --group gpu     # see INSTALL.md → GPU support

# 3. Activate the venv, then run with plain `python`
source .venv/bin/activate
python -c "import animaloc.models, dinov3; print('OK')"
```

See [INSTALL.md](INSTALL.md) for troubleshooting and training-time DINOv3 weights.

### Try the OWL inference notebook

[`notebooks/owl_inference_demo.ipynb`](notebooks/owl_inference_demo.ipynb) uses
the released **OWL-D / `OWLD_H`** checkpoint by default. It displays a six-image
contact sheet, runs four 512×512 patches, and exports predicted counts,
original-pixel coordinates, heatmaps, point overlays, and run metadata.
Set `RUN_FULL_RESOLUTION=True` for the two 5472×3648 images with overlapping
tiles, or separately enable your own image directory. No annotation files or
accuracy metrics are involved.

```bash
# From a checkout containing the notebook and its backend:
uv sync --locked --no-default-groups --group gpu --extra notebook
.venv/bin/python -m jupyterlab notebooks/owl_inference_demo.ipynb
```

For CPU, use `uv sync --locked --extra notebook` and explicitly set
`MODEL='owl-c'`, `DEVICE='cpu'` in the notebook. OWL-D never silently falls back
to another model. Its approximately **3.5 GB** full checkpoint includes the
frozen DINOv3 backbone; **no separate Meta weights download** is needed for this
inference path. A compatible CUDA GPU and sufficient host/GPU memory are needed.
Use the `.venv` interpreter directly after syncing; bare `uv run` can restore
CPU PyTorch.

**Local notebook execution verified (2026-09-08):** the full 23-cell notebook
completed through nbclient with OWL-D on a V100 32 GB GPU in FP32, including
all four patches and both 5472×3648 images (160 tiles each). A separate
explicit OWL-C CPU notebook run also completed. Outputs include environment
information, progress, tables, and inline figures. Reproduce with
[`tests/execute_owl_notebook.py`](tests/execute_owl_notebook.py); see
[runner commands and local artifacts](docs/demo.md#local-notebook-execution)
and [backend counts/resource observations](docs/demo.md#local-backend-validation).
These are inference results, not accuracy metrics.

**Sample access:** the notebook downloads the exact `OWL_SAMPLE_DATA.zip`
from [release owl-notebook-v1](https://github.com/microsoft/MegaDetector-Overhead/releases/tag/owl-notebook-v1).
Set `ARCHIVE` to reuse a local copy instead. All sample sources and terms are documented:
the contributor confirms SheepCounter Public Domain and HerdNet CC BY-NC-SA 4.0;
the caribou release's CC BY-NC-SA 4.0 terms are verified from its primary source.
Redistribution permission was confirmed by the sample provider.
Do not substitute the caribou test ZIP. The notebook,
backend, and manifest are distributed together. The configurable Colab setup uses a uv-managed
Python backend rather than the hosted kernel, but **actual Colab validation is
pending** and does not block the local release. Both model checkpoints have
separately passed real clean-cache public downloads and published-checksum
verification. The model/data release declares CC BY-NC-SA 4.0; see
[dataset terms](docs/datasets.md#sample-provenance-and-publication-gates).
The notebook is in this repository; the sample ZIP is a release asset outside
Git history. The documented locked setup also installed and ran in a fresh
environment without author Conda/PYTHONPATH dependencies; see
[execution evidence](docs/demo.md#local-notebook-execution) for scope and
intermittent model-download limitations.

---

## Repository Layout

```
animaloc/    # Training/eval package vendored from HerdNet (MIT)
dinov3/      # DINOv3 encoder vendored from facebookresearch/dinov3 (DINOv3 License)
tools/       # train.py, test.py, infer.py, patcher.py
notebooks/   # OWL inference notebook and exact sample manifest
configs/     # Hydra configs for OWL-C / OWL-D / OWL-T training and eval
docs/        # MkDocs Material site (build with `make docs`)
```

See [NOTICE](NOTICE) for upstream attribution and third-party licenses.

---

## Ecosystem

| Repository | Description |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | Umbrella hub — PyTorch-Wildlife, MegaDetector, ecosystem overview |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal, human, and vehicle detection for camera-trap images |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | **This repo** — wildlife detection in aerial and drone imagery |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic AI for audio-based wildlife monitoring |
| [microsoft/MegaDetector-Sonar](https://github.com/microsoft/MegaDetector-Sonar) | Sonar-based wildlife detection for aquatic monitoring |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch |

---

## Citation

If you use MegaDetector-Overhead in your research, please cite:

```bibtex
@article{chacon2026overhead,
  title={Overhead Wildlife Locator (OWL): Benchmarking Weakly Supervised Learning for Aerial Wildlife Surveys},
  author={Chac{\'o}n, Isai Daniel and Miao, Zhongqi and Demuro, Bruno and Robinson, Caleb and Dodhia, Rahul and Otarashvili, Lasha and Holmberg, Jason and Larsen, Kirk and Frederick, Howard and Pamperin, Nathan J and others},
  journal={arXiv preprint arXiv:2606.13911},
  year={2026}
}
```
