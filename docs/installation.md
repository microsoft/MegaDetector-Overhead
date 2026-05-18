---
description: "How to install and set up MegaDetector-Overhead for wildlife detection in drone and aerial imagery."
tags:
  - MegaDetector-Overhead installation
  - aerial detection setup
  - PyTorch-Wildlife
  - drone wildlife survey
  - overhead imagery AI
---

# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, but recommended for large aerial datasets)

## Install

```bash
git clone https://github.com/microsoft/MegaDetector-Overhead
cd MegaDetector-Overhead
pip install -r requirements.txt
```

## Verify

```python
from PytorchWildlife.models.overhead import OverheadDetector
print("MegaDetector-Overhead is ready.")
```

## GPU Setup

GPU acceleration is strongly recommended when processing large drone survey datasets. Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())  # should print True on a CUDA-enabled machine
```

## Next Steps

- Run the [demo notebook](https://github.com/microsoft/MegaDetector-Overhead/blob/main/demo/overhead_demo.ipynb) for an end-to-end walkthrough
- See the [Model Zoo](model_zoo.md) for available pre-trained models
- See the [README](https://github.com/microsoft/MegaDetector-Overhead#quick-start) for full CLI usage
