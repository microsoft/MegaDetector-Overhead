---
description: "MegaDetector-Overhead model zoo — the OWL-C, OWL-D, and OWL-T model families for wildlife detection in drone and aerial imagery."
tags:
  - MegaDetector-Overhead models
  - aerial wildlife detection models
  - OWL-C
  - OWL-D
  - OWL-T
  - DINOv3
  - HerdNet
  - drone survey AI
  - PyTorch-Wildlife
---

# Model Zoo

All three model families are vendored in `animaloc/models/` and registered
in the `MODELS` registry. They produce single-channel FIDT heatmaps and
share the same evaluation/stitching/peak-detection infrastructure.

| Model | Backbone | Source file | Registry name(s) | Notes |
|---|---|---|---|---|
| **OWL-C** | DLA-34 | `animaloc/models/owl_c.py` | `OWLC` | HerdNet detection branch — the baseline. |
| **OWL-T** | DLA-34 + Swin transformer multiscale residual | `animaloc/models/owl_t.py` | `OWLT` | Adds windowed self-attention at multiple scales for sharper localization on cluttered backgrounds. |
| **OWL-D-S** | DINOv3 ViT-S/16 + DPT decoder | `animaloc/models/owl_d.py` | `OWLD_S` | ~22M backbone params. Lightest; benefits most from full backbone fine-tuning. |
| **OWL-D-B** | DINOv3 ViT-B/16 + DPT decoder | `animaloc/models/owl_d.py` | `OWLD_B` | ~86M backbone params. Quality/speed balance; partial fine-tuning recommended. |
| **OWL-D-L** | DINOv3 ViT-L/16 + DPT decoder | `animaloc/models/owl_d.py` | `OWLD_L` | ~307M backbone params. Use with frozen backbone and feature caching. Two weight files available (LVD-1.6B default; SAT-493M for satellite imagery). |
| **OWL-D-H** | DINOv3 ViT-H+/16 + DPT decoder | `animaloc/models/owl_d.py` | `OWLD_H` | ~840M backbone params. Highest quality; strongly recommended to use frozen backbone with feature caching. |

The legacy `HerdNet` multi-class model is also registered (`HerdNet`).

!!! note
    Pretrained weights for the OWL family are not yet released. Watch the
    [repository](https://github.com/microsoft/MegaDetector-Overhead) for
    updates. DINOv3 backbone weights are downloaded separately from
    Meta — see [INSTALL.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/INSTALL.md).

## Loading a model

```python
from animaloc.models import MODELS

model = MODELS.get("OWLD_B")(
    down_ratio=2,
    freeze_backbone=True,
    unfreeze_last_n=0,
)
```

For training and evaluation via the bundled Hydra configs, see
[Training, Evaluation, and Inference](training.md).

## Architecture references

* OWL-C / OWL-T architecture (HerdNet detection branch + Swin residual):
  Delplanque et al., *ISPRS J. Photogramm. Remote Sens.* 197 (2023),
  https://doi.org/10.1016/j.isprsjprs.2023.01.025
* DINOv3 backbone: Meta AI, https://github.com/facebookresearch/dinov3
