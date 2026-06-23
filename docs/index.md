---
description: "MegaDetector-Overhead — Microsoft AI for Good Lab's open-source AI for detecting wildlife from overhead and aerial imagery captured by drones and UAVs."
tags:
  - MegaDetector-Overhead
  - overhead detection
  - drone wildlife survey
  - aerial imagery
  - UAV wildlife detection
  - PyTorch-Wildlife
  - conservation AI
---

# Microsoft MegaDetector-Overhead

**Open-source AI for detecting wildlife in overhead and aerial imagery.**

MegaDetector-Overhead is a toolkit from the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/) for detecting and localizing wildlife in imagery captured from drones, UAVs, and other aerial platforms. It extends the MegaDetector detection framework to handle the unique challenges of overhead perspectives: small targets, variable altitude, and nadir-angle distortion. It is powered by the [PyTorch-Wildlife](https://github.com/microsoft/Biodiversity) framework and is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) ecosystem.

---

## What It Does

MegaDetector-Overhead turns aerial survey imagery into structured wildlife detections:

1. **Preprocessing** — tile large orthomosaics and drone frames into model-ready patches with configurable overlap (`tools/patcher.py`)
2. **Detection** — locate animals in overhead imagery with the OWL-C / OWL-D / OWL-T model families (DLA + Swin + DINOv3 backbones)
3. **Post-processing** — merge tile-level peak detections back to full-image coordinates, suppress duplicates, and export results as CSV

---

## Get Started

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # make uv available in the current shell
git clone https://github.com/microsoft/MegaDetector-Overhead
cd MegaDetector-Overhead
uv sync                  # CPU PyTorch; for a GPU: `uv sync --no-default-groups --group gpu`
uv run python -c "import animaloc.models, dinov3; print('OK')"
```

Then see:

* [Installation](installation.md) — full install + DINOv3 weights download
* [Caribou Demo](demo.md) — download → OWL-C inference (GPU/CPU) → visualize
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T family
* [Training, Evaluation, and Inference](training.md) — end-to-end workflow

---

## Part of the Biodiversity Ecosystem

MegaDetector-Overhead is one model in a larger open-source ecosystem from the Microsoft AI for Good Lab.

| Repository | Description |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | Umbrella hub — PyTorch-Wildlife, MegaDetector, ecosystem overview |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal, human, and vehicle detection for camera-trap images |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | **This repo** — wildlife detection in aerial and drone imagery |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic AI for audio-based wildlife monitoring |
| [microsoft/MegaDetector-Sonar](https://github.com/microsoft/MegaDetector-Sonar) | Sonar-based wildlife detection for aquatic monitoring |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI-enabled edge device |
