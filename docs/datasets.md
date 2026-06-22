---
description: "Caribou aerial survey dataset for training and evaluating overhead wildlife detectors. Point-annotated 512×512 patches from the Porcupine and Central Arctic herds."
tags:
  - datasets
  - caribou
  - aerial wildlife detection
  - point annotations
  - OWL
  - benchmark
  - PyTorch-Wildlife
---

# Datasets

## Caribou Aerial Survey Dataset

Point-annotated 512×512 px aerial image patches for caribou detection and counting from overhead survey imagery. This dataset accompanies the OWL paper and enables reproducible evaluation of point-based object detectors on aerial wildlife imagery.

**[➜ Download the dataset and model weights from Zenodo](https://zenodo.org/records/20802844)**

!!! tip "Try it in one command"
    The [Caribou Demo](demo.md) downloads the patches + weights, runs OWL-C
    inference (GPU or CPU), and visualizes the predictions
    (`tools/demo_caribou.sh`). To run and **compare all four pretrained models**
    on caribou, use `tools/demo_owl_models.sh`.

---

### Overview

| Split | Source herd | Year | Patches | Annotated | Background | Point annotations |
|---|---|---|---|---|---|---|
| **Train** | Porcupine Caribou Herd (PCH), Alaska | 2017 | 23,517 | 18,322 | 5,195 | 273,268 |
| **Test** | Central Arctic Herd (CAH), Alaska | 2022 | 2,607 | 1,852 | 755 | 12,456 |

This is a **strict cross-herd and cross-temporal generalization benchmark**: models trained on PCH 2017 are evaluated on CAH 2022 without any per-deployment retraining.

---

### Contents

| File | Description |
|---|---|
| `train.zip` | 23,517 training patches (512×512 PNG) + `gt.csv` (273,268 annotations) |
| `test.zip` | 2,607 test patches (512×512 PNG) + `gt.csv` (12,456 annotations) |
| `Caribou-OWL-C.pth` | Caribou-specific OWL-C (DLA-34, epoch 14, val F1 = 0.937); reproduces the F1 = 0.965 headline below |
| `OWL-C.pth` | OWL-C general overhead-benchmark model (DLA-34 detection branch) |
| `OWL-T.pth` | OWL-T general overhead-benchmark model (DLA-34 + Swin multi-scale residual) |
| `OWL-D.pth` | OWL-D general overhead-benchmark model (DINOv3 ViT-H+/16 + DPT decoder) |
| `README.md` | Full dataset documentation, annotation format, and benchmark results |

The `OWL-C` / `OWL-T` / `OWL-D` checkpoints are trained on public overhead
datasets, **not** caribou; see the [Model Zoo](model_zoo.md) for details.

---

### Annotation format

Each split contains a `gt.csv` with point annotations in the following format:

| Column | Description |
|---|---|
| `images` | Patch filename (e.g., `patch_00001.png`) |
| `x` | Horizontal pixel coordinate of the animal center |
| `y` | Vertical pixel coordinate of the animal center |

This format is directly compatible with the `animaloc` training package used in this repository. See [Training, Evaluation, and Inference](training.md) for usage.

---

### Benchmark results

The pre-trained **`Caribou-OWL-C.pth`** weights reproduce the paper headline on the test split:

| Metric | Value |
|---|---|
| F1 score (τ = 20 px, c* = 0.20) | **0.965** |
| Precision | 0.975 |
| Recall | 0.955 |

!!! note
    **All OWL pretrained checkpoints are now released** — the caribou-specific
    `Caribou-OWL-C.pth` plus the three general overhead-benchmark models
    (`OWL-C.pth`, `OWL-T.pth`, `OWL-D.pth`). The general models are trained on
    public overhead datasets, not caribou, so evaluating them on the caribou test
    set is a zero-shot, cross-domain check (expect lower numbers than the
    in-domain `Caribou-OWL-C`). The [Caribou Demo](demo.md) runs and compares all
    four.

---

### Citation

If you use this dataset or code, please cite:

```bibtex
@article{chacon2026overhead,
  title={Overhead Wildlife Locator (OWL): Benchmarking Weakly Supervised Learning for Aerial Wildlife Surveys},
  author={Chac{\'o}n, Isai Daniel and Miao, Zhongqi and Demuro, Bruno and Robinson, Caleb and Dodhia, Rahul and Otarashvili, Lasha and Holmberg, Jason and Larsen, Kirk and Frederick, Howard and Pamperin, Nathan J and others},
  journal={arXiv preprint arXiv:2606.13911},
  year={2026}
}
```
