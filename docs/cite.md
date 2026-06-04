---
description: "How to cite MegaDetector-Overhead and its upstream sources — BibTeX and citation.cff entries for the PyTorch-Wildlife aerial wildlife detection system."
tags:
  - cite MegaDetector-Overhead
  - MegaDetector citation
  - HerdNet citation
  - DINOv3 citation
  - PyTorch-Wildlife citation
  - aerial wildlife research
  - BibTeX
---

# :fountain_pen: Cite Us

If you use MegaDetector-Overhead in your research, please cite **all** of
the following — the framework, the training stack we vendor, and the
foundation model that powers the OWL-D family.

## MegaDetector-Overhead / PyTorch-Wildlife

```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

You can also cite MegaDetector-Overhead as software directly. The [`citation.cff`](https://github.com/microsoft/MegaDetector-Overhead/blob/main/citation.cff) file in the repository is machine-readable and is used by GitHub's "Cite this repository" widget.

## HerdNet (animaloc package — basis for OWL-C and OWL-T)

```bibtex
@article{delplanque2023herdnet,
    title   = {From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning?},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume  = {197},
    pages   = {167-180},
    year    = {2023},
    issn    = {0924-2716},
    doi     = {10.1016/j.isprsjprs.2023.01.025},
    url     = {https://www.sciencedirect.com/science/article/pii/S092427162300031X},
    author  = {Alexandre Delplanque and Samuel Foucher and Jérôme Théau and Elsa Bussière and Cédric Vermeulen and Philippe Lejeune}
}
```

## DINOv3 (encoder backbone for OWL-D)

```bibtex
@misc{simeoni2025dinov3,
      title={DINOv3},
      author={Meta AI Research},
      year={2025},
      url={https://github.com/facebookresearch/dinov3},
}
```

See [dinov3/MODEL_CARD.md](https://github.com/microsoft/MegaDetector-Overhead/blob/main/dinov3/MODEL_CARD.md) for Meta's official DINOv3 model card and the most up-to-date citation.
