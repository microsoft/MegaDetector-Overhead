# MegaDetector-Overhead

[![Docs](https://img.shields.io/badge/docs-microsoft.github.io-blue)](https://microsoft.github.io/MegaDetector-Overhead/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![PyTorch-Wildlife](https://img.shields.io/badge/PyTorch--Wildlife-ecosystem-green)](https://github.com/microsoft/Biodiversity)

**Open-source AI for detecting wildlife in overhead and aerial imagery.**

MegaDetector-Overhead extends the [MegaDetector](https://github.com/microsoft/MegaDetector) detection framework to drone and UAV survey imagery, handling the unique challenges of overhead perspectives: small targets, variable altitude, and nadir-angle distortion. It is powered by [PyTorch-Wildlife](https://github.com/microsoft/PytorchWildlife) and is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) ecosystem.

---

## Documentation

Full documentation at **[microsoft.github.io/MegaDetector-Overhead](https://microsoft.github.io/MegaDetector-Overhead/)**

---

## Quick Start

```bash
git clone https://github.com/microsoft/MegaDetector-Overhead
cd MegaDetector-Overhead
pip install -r requirements.txt
jupyter notebook demo/overhead_demo.ipynb
```

---

## Ecosystem

| Repository | Description |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | Umbrella hub — PyTorch-Wildlife, MegaDetector, ecosystem overview |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal, human, and vehicle detection for camera-trap images |
| [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife) | The collaborative deep learning framework for wildlife monitoring |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | **This repo** — wildlife detection in aerial and drone imagery |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic AI for audio-based wildlife monitoring |
| [microsoft/MegaDetector-Sonar](https://github.com/microsoft/MegaDetector-Sonar) | Sonar-based wildlife detection for aquatic monitoring |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch |
| [microsoft/SPARROW-Studio](https://github.com/microsoft/SPARROW-Studio) | Desktop application for all AI for Good Lab models |

---

## Citation

If you use MegaDetector-Overhead in your research, please cite:

```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```
