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

* [Installation](INSTALL.md) — full install + DINOv3 weights download
* [Datasets](docs/datasets.md) — caribou data + pretrained model weights (Zenodo)
* [Model Zoo](docs/model_zoo.md) — the OWL-C / OWL-D / OWL-T family + pretrained checkpoints
* [Caribou Demo](docs/demo.md) — download → OWL inference (GPU/CPU) → visualize; run & compare all models (`tools/demo_caribou.sh`, `tools/demo_owl_models.sh`)
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
# For a GPU, install a matching build after syncing (uv auto-detects the driver):
#   uv sync --no-default-groups --group gpu     # GPU build; see INSTALL.md → GPU support

# 3. Smoke test
uv run python -c "import animaloc.models, dinov3; print('OK')"
```

See [INSTALL.md](INSTALL.md) for DINOv3 weights download and troubleshooting.

---

## Repository Layout

```
animaloc/    # Training/eval package vendored from HerdNet (MIT)
dinov3/      # DINOv3 encoder vendored from facebookresearch/dinov3 (DINOv3 License)
tools/       # train.py, test.py, infer.py, patcher.py
configs/     # Hydra configs for OWL-C / OWL-D / OWL-T training and eval
docs/        # MkDocs Material site (build with `uv run --extra docs mkdocs build`)
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
