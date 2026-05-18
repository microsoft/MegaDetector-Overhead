---
description: "MegaDetector-Overhead model zoo — pre-trained models for wildlife detection in drone and aerial imagery."
tags:
  - MegaDetector-Overhead models
  - aerial wildlife detection models
  - drone survey AI
  - overhead model zoo
  - PyTorch-Wildlife
---

# Model Zoo

Pre-trained MegaDetector-Overhead models for detecting wildlife, humans, and vehicles in overhead and aerial imagery.

!!! note
    Model weights and download links will be published here as models are released. Check back or watch the [repository](https://github.com/microsoft/MegaDetector-Overhead) for updates.

---

## Planned Models

| Model | Input | Description | Status |
|---|---|---|---|
| `MDOverhead-v1` | Drone RGB | General-purpose wildlife detector for nadir drone imagery | Coming soon |
| `MDOverhead-UAV` | UAV RGB | Optimized for high-altitude fixed-wing UAV surveys | Coming soon |
| `MDOverhead-Thermal` | Thermal IR | Night-survey thermal imagery detector | Planned |

---

## Usage

Once a model is released, load it via PyTorch-Wildlife:

```python
from PytorchWildlife.models.overhead import OverheadDetector

model = OverheadDetector(weights="MDOverhead-v1")
results = model.predict("path/to/aerial_image.jpg")
print(results)
```

See the [demo notebook](https://github.com/microsoft/MegaDetector-Overhead/blob/main/demo/overhead_demo.ipynb) for a full end-to-end example.
