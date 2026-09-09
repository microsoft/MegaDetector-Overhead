---
description: "OWL notebook sample access and provenance gates, plus the annotated Caribou Aerial Survey Dataset and released model checkpoints."
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

## OWL notebook sample images

The [OWL inference notebook](demo.md#owl-inference-notebook) uses a separate,
exact sample archive named **`OWL_SAMPLE_DATA.zip`**, not the caribou training
or test ZIPs below. It is an annotation-free visual demonstration, not an
evaluation dataset.

| Property | Supplied archive |
|---|---|
| Archive size | 18,269,117 bytes (approximately 18.3 MB) |
| Uncompressed image bytes | 18,281,181 |
| Default inference inputs | Four RGB 512×512 patches |
| Optional full-image inputs | Two RGB 5472×3648 images |
| Archive image directory | `OWL_DATA/` |
| Default extraction directory | `demo_data/owl_notebook/OWL_DATA/` |
| Annotations | None; no ground-truth CSV, species labels, or evaluation metrics |
| Integrity/inventory record | `notebooks/sample_data.json` in the matching notebook checkout |
| Exact public archive URL | [OWL_SAMPLE_DATA.zip, release owl-notebook-v1](https://github.com/microsoft/MegaDetector-Overhead/releases/download/owl-notebook-v1/OWL_SAMPLE_DATA.zip) |
| Per-source attribution/license documentation | SheepCounter: Public Domain; HerdNet and caribou: CC BY-NC-SA 4.0; evidence provenance recorded per source |

Notebook preparation downloads the exact ZIP from the manifest URL.
`ARCHIVE` selects an existing local copy instead. `SAMPLE_URL` defaults to an
empty string so the manifest is the single source of the release URL.
Preparation verifies archive
integrity and the expected image inventory, safely extracts the files, and
generates `contact_sheet.png` and `sample_inventory.csv` in
`demo_data/owl_notebook/`. Valid local files are checked and reused on reruns.
Missing inputs or integrity failures produce explicit errors.

### Sample provenance and publication gates

The sample provider **has confirmed redistribution is permitted** and supplied
the following mapping. It is recorded per file in `notebooks/sample_data.json`,
not inferred solely from filenames.

| Images | Source | Terms and evidence |
|---|---|---|
| Two `DJI_...` patches | [SheepCounter](https://universe.roboflow.com/riisprivate/sheepcounter) | **Public Domain**, explicitly confirmed by the contributor on 2026-09-09. The source acknowledgment is retained; no specific CC0 instrument is inferred. |
| Two full-resolution hashed-name JPGs | [HerdNet-associated dataset, DOI 10.58119/ULG/MIRUU5](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5) | **CC BY-NC-SA 4.0**, explicitly confirmed by the contributor on 2026-09-09. Retain original dataset/creator attribution through the source DOI and the license notice. |
| Two `CAH_...` patches | [OWL models and caribou data, Zenodo 20802844](https://zenodo.org/records/20802844) | Primary record metadata and release README declare [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The README requests citation of Chacon et al. (2026), the OWL paper, arXiv:2606.13911. |

The Zenodo terms were checked against the
[record API](https://zenodo.org/api/records/20802844) and
[released README](https://zenodo.org/api/records/20802844/files/README.md/content).
Retain source credit, the license link, and applicable change notices.
CC BY-NC-SA includes noncommercial and share-alike conditions; do not treat
public access as unrestricted reuse. The software repository's MIT license
does **not** apply to the mixed-source sample imagery.

The SheepCounter and HerdNet license statements are contributor-confirmed,
not independently retrieved from their source pages in this environment.
This evidence distinction is retained in the manifest.

Keep the original ZIP bytes unchanged. Its attribution and source-specific
terms accompany it in the staged `SAMPLE_ATTRIBUTION.json` and release notes;
they are not inserted into the archive. See
[release preparation](demo.md#repository-publication-and-sample-release-preparation).

For each public notebook release, retain:

1. A stable, public URL for the exact archive, with checksum-verified clean
   download access and a corresponding manifest update.
2. Include the documented source-specific terms, creator/source credits,
   license links, and accompanying redistribution notice with the release asset.
3. A public repository ref containing the matching notebook, backend, and
   manifest, verified without private paths or credentials.
4. Fresh access to the pinned public model checkpoints, distinct from
   successful use of cached local weights.
5. Execution evidence for the routes being advertised. The local OWL-D
   patch/full-image and explicit OWL-C CPU notebooks have now completed
   through nbclient (see [local notebook execution](demo.md#local-notebook-execution)).
   The implemented **Colab setup path remains experimental**. It does not
   block local Jupyter/VS Code release, but must be tested in an actual hosted
   runtime before advertising hosted support.

No archive upload or public release is performed by notebook execution.
A fresh clone downloads the exact sample archive from the published manifest
URL. For your own authorized image directory, the notebook has a separate
`RUN_CUSTOM_IMAGES=True` / `CUSTOM_IMAGES_DIR` option with no annotation
requirement. The sum of retained localization peaks is not a population or
accuracy estimate, and FIDT heatmap sums are not animal counts.

### Notebook model checkpoint

The default `OWL-D.pth` in the existing
[Zenodo model release](https://zenodo.org/records/20802844) is the approximately
3.5 GB full **`OWLD_H`** checkpoint, including its frozen DINOv3 ViT-H+/16
backbone. The notebook loads it with `pretrained=False`, without a separate
Meta weights download. Check applicable model and
[DINOv3 license terms](https://github.com/microsoft/MegaDetector-Overhead/blob/main/dinov3/LICENSE.md);
checkpoint access does not resolve sample-image licensing.

Weights are cached under `demo_data/models/OWL-D.pth` or, for the explicit
CPU alternative, `demo_data/models/OWL-C.pth`. CPU users choose the **general**
OWL-C model, not the separate caribou-specific checkpoint. CUDA is required
for the notebook's default OWL-D route; it never silently swaps models.
Public model-download availability must be verified independently of cached
local model integrity.

As of the **2026-09-08 local validation checkpoint**, OWL-D CUDA patch and
full-resolution backend runs, plus explicit OWL-C CPU patch inference,
succeeded using cached weights (see
[backend evidence](demo.md#local-backend-validation)). Complete local nbclient
executions also succeeded for the OWL-D patch/full-image notebook and explicit
OWL-C CPU notebook. The recorded checkpoint
SHA-256 pins and file sizes have now been independently confirmed by the
[published release README](https://zenodo.org/api/records/20802844/files/README.md/content);
the cached files also match the record API's MD5 checksums. The release declares
**CC BY-NC-SA 4.0** for its models and caribou data; consult the release terms
and the additional DINOv3 terms for the included backbone. Clean-download
verification is separate from source metadata checks and from sample hosting.
Both OWL-D and OWL-C have now downloaded successfully from the public release
into initially empty isolated caches, matching the published sizes/SHA-256
hashes and leaving the working model cache unchanged.
The sample URL and source-specific terms are recorded in the manifest;
model and sample assets remain independently checksum-verified.

---

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
