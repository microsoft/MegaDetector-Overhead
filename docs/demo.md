---
description: "OWL inference notebook with explicit CPU and optional full-image routes, plus existing caribou evaluation and model-comparison shell demos."
tags:
  - demo
  - quickstart
  - caribou
  - OWL-C
  - OWL-D
  - notebook
  - inference
  - visualization
  - PyTorch-Wildlife
---

# OWL demos

## OWL inference notebook

Open
[`notebooks/owl_inference_demo.ipynb`](https://github.com/microsoft/MegaDetector-Overhead/blob/main/notebooks/owl_inference_demo.ipynb)
from a checkout that contains the notebook, `tools/demo_owl_notebook.py`, and
`notebooks/sample_data.json`. The matching release is
[owl-notebook-v1](https://github.com/microsoft/MegaDetector-Overhead/releases/tag/owl-notebook-v1).
The notebook is an annotation-free localization demo, separate from the
caribou evaluation workflows below.

This existing repository is the canonical notebook publication location.
GitHub previews the `.ipynb`; it does not execute it. Users clone the matching
revision and run it in Jupyter or VS Code. The sample ZIP belongs in a release
asset on the same repository, not in Git history; model weights stay on Zenodo.

The default model is **OWL-D / `OWLD_H`**: DINOv3 ViT-H+/16 with a DPT decoder
and frozen backbone. The approximately **3.5 GB** released `OWL-D.pth`
contains the full backbone and loads with `pretrained=False`; it needs **no
separate Meta weights download** for inference. A compatible CUDA GPU and
sufficient host/GPU memory are required for this notebook route. CPU users
must explicitly set `MODEL='owl-c'`, `DEVICE='cpu'`, selecting the released
general `OWL-C.pth`. Missing CUDA does not skip OWL-D or silently substitute a
model.

### Local Jupyter / VS Code

New users do **not** need the author's Conda environments. They need Git, an
installed [uv](https://docs.astral.sh/uv/getting-started/installation/), network
access, and sufficient disk/RAM. Default OWL-D additionally needs a compatible
NVIDIA GPU and driver; uv installs the locked PyTorch CUDA wheels, not the
host GPU driver. CPU users explicitly choose OWL-C instead.

Clone the notebook/backend together:

```bash
git clone https://github.com/microsoft/MegaDetector-Overhead.git
cd MegaDetector-Overhead
```

`uv sync --locked` reads `.python-version`, `pyproject.toml`, and `uv.lock`,
obtains the supported Python version if needed, and creates **this checkout's
own `.venv`**, including the notebook dependencies and vendored `animaloc`/
`dinov3` packages. It does not depend on a preexisting Conda installation.

```bash
# Run from a checkout containing the notebook and its backend.
uv sync --locked --no-default-groups --group gpu --extra notebook
# CPU alternative instead; also change MODEL/DEVICE in the notebook:
# uv sync --locked --extra notebook
.venv/bin/python -m ipykernel install --user \
    --name megadetector-overhead --display-name "MegaDetector Overhead"
.venv/bin/python -m jupyterlab notebooks/owl_inference_demo.ipynb
```

In VS Code, choose the **MegaDetector Overhead** or project `.venv` kernel.
Use `.venv\Scripts\python.exe` on Windows. Local setup checks and reuses the
existing environment rather than automatically syncing it. All backend stages
use the selected `.venv` interpreter directly; avoid bare `uv run`, which can
restore the default CPU PyTorch build. The optional `notebook` extra adds
JupyterLab, ipykernel, nbformat, and nbclient without changing default
CPU/GPU group selection.

The local notebook checks this environment and prints repair instructions
when it is missing; it deliberately does **not** run installers automatically.
Opening the `.ipynb` by itself is therefore not the complete first-time setup.
Once initialized, all model inference runs in `.venv`, even if VS Code uses
another compatible presentation kernel. GitHub is a preview, not a runtime.

If the existing environment is ready, launch Jupyter directly without syncing:

```bash
.venv/bin/jupyter lab notebooks/owl_inference_demo.ipynb
```

Choose **Restart Kernel and Run All Cells**. The default is OWL-D on CUDA with
four patches. Set `RUN_FULL_RESOLUTION=True` before restarting/running all to
include the two large images. Review the printed model/device, image inventory,
counts, inline figures, tables, and saved artifact paths.

The manifest downloads **`OWL_SAMPLE_DATA.zip` from release `owl-notebook-v1`**.
Leave `SAMPLE_URL=''` to use that URL. Set `ARCHIVE` to use an already downloaded
copy of the exact ZIP instead. Files are checksum-verified and cached under
`demo_data/`. Do not substitute the separate caribou test ZIP.

### What the notebook runs

1. Locate/validate the repository and managed backend, then inspect the
   interpreter, revision, PyTorch/CUDA, GPU resources, and chosen model/device
   **before downloading samples or weights**.
2. Verify and safely prepare all six images under
   `demo_data/owl_notebook/OWL_DATA`; show the inventory CSV and contact sheet.
3. Validate/reuse the selected checkpoint from `demo_data/models/`, or download
   its pinned public Zenodo file when missing. A local checkpoint override is
   available through `CHECKPOINT`.
4. Predict the **four 512×512 patches**, reporting settings, checkpoint loading,
   counts, per-image timing, and resources.
5. Display the summary, a detections CSV preview, and all four original /
   normalized-display FIDT heatmap / predicted-point comparison figures.
6. Optionally process **both 5472×3648 images** at their original dimensions
   when `RUN_FULL_RESOLUTION=True`, using overlapping tiles rather than
   silently resizing the survey imagery.
7. Independently enable `RUN_CUSTOM_IMAGES=True` and set `CUSTOM_IMAGES_DIR`
   for annotation-free inference on your image directory (`selection=all`).

Default inference uses 512-pixel tiles, 160-pixel overlap, mean stitching,
half-resolution heatmaps, and LMDS with a 3×3 peak kernel.
`ADAPT_THRESHOLD=0.3` is adaptive peak selection,
`NEGATIVE_THRESHOLD=0.1` suppresses weak/background responses, and
`SCORE_THRESHOLD=0.2` is a separate absolute retained-peak cutoff.
The same retained detections determine exported coordinates, overlays, and
counts. Every inference call creates a new timestamp/UUID directory under
`demo_data/owl_notebook/runs/`, so reruns cannot reuse stale outputs.

### Outputs and interpretation

| Artifact in each run directory | Contents |
|---|---|
| `detections.csv` | `images,x,y,dscores,labels`; generic animal points in **original-image pixels** |
| `summary.csv` | `images,width,height,count,seconds,tiles`; includes zero-count images |
| `metadata.json` | Run settings, checkpoint/runtime details, and actual figure filenames |
| `figures/` | Original / FIDT heatmap / point-overlay comparisons; optional detail figures |
| `overlays/` | Saved predicted-point overlays |

The frontend uses standard-library CSV/JSON and IPython display, not PyTorch
or `animaloc` imports in the host kernel. It consumes the `metadata.json`
`figures` list rather than deriving potentially long filenames. Local
`FileLink` exports work when Jupyter is launched from the repository root;
VS Code can open the printed paths. Colab users can use its Files panel or
`google.colab.files.download(str(PATCH_OUTPUT / 'detections.csv'))`.
Nothing is uploaded or published automatically.

**Predicted count is the number of retained peaks. FIDT heatmap sums are not
animal counts, and peak scores are not calibrated probabilities.** Display
normalization does not make heatmap colors comparable across images. The sum
of detections across demo images is not a population estimate, especially
where patches and full images overlap. No species identification or
precision/recall/F1 is provided for these unannotated samples. A true zero
prediction retains its summary row and detection CSV headers; errors must
remain errors.

!!! warning "Notebook coordinates are already in original-image pixels"
    Do **not** apply the legacy caribou evaluation visualizer's
    `--pred-scale 2` to notebook exports. The notebook backend converts
    heatmap coordinates exactly once.

### Backend stages from the command line

These are the same stages the notebook invokes through the project Python.
The environment check does not download assets; preparation uses the public
manifest URL unless an explicit local archive or URL override is given.

```bash
.venv/bin/python tools/demo_owl_notebook.py environment --model owl-d --device auto
.venv/bin/python tools/demo_owl_notebook.py prepare --data-dir demo_data/owl_notebook

# A NEW output directory is required for each inference invocation.
OUT="demo_data/owl_notebook/runs/$(.venv/bin/python -c 'import uuid; print(uuid.uuid4().hex)')"
.venv/bin/python tools/demo_owl_notebook.py infer \
    --model owl-d --device auto \
    --images-dir demo_data/owl_notebook/OWL_DATA --selection patches \
    --output-dir "$OUT" --score-threshold 0.2 \
    --adapt-threshold 0.3 --negative-threshold 0.1 \
    --tile-size 512 --overlap 160
```

For optional full images, select `full` and create another new output directory.
For your image directory, set `--images-dir` and `--selection all`.
For CPU, explicitly use `--model owl-c --device cpu` after syncing the CPU
environment. Optional inputs are `prepare --archive PATH`,
`prepare --sample-url URL` for an explicit URL override, and
`infer --checkpoint PATH` for the selected model. `infer --model-cache PATH`
selects a separate cache without changing `demo_data/models`; it cannot be
combined with `--checkpoint`. In the notebook, the equivalent option is
`MODEL_CACHE`.

Weights can be downloaded and verified without constructing a GPU model:

```bash
# Use new empty cache directories for a genuine fresh-download check.
.venv/bin/python tools/demo_owl_notebook.py fetch-model --model owl-d \
    --model-cache demo_data/owl_notebook/fresh_models/owl-d
.venv/bin/python tools/demo_owl_notebook.py fetch-model --model owl-c \
    --model-cache demo_data/owl_notebook/fresh_models/owl-c
```

A download error fails explicitly and does not fall back to the working cache.
Existing valid files in the selected cache are labeled as cached, not as newly
downloaded.

Downloads use the official Zenodo public file URL, with its official API
content endpoint as a network-failure fallback. Both must match the same
published size and SHA-256. Each endpoint has bounded retries, and an integrity
failure is never bypassed by trying a different URL. The endpoints can be
intermittent; an exhausted download fails explicitly rather than returning
empty predictions.

### Local backend validation

The following evidence was recorded on **2026-09-08 using the backend CLI**,
the supplied local archive, and cached checkpoints:

- **14 focused backend unit tests passed**, including mocked successful and
  corrupt downloads.
- **OWL-D / `OWLD_H`, CUDA:** real inference completed for all four 512×512
  patches, reporting retained predicted counts **58, 19, 14, 9** (**100 total**).
  This run used a Tesla V100 32 GB in **FP32**, with **3.543 GiB peak PyTorch GPU
  allocation**. That allocation is not total process/device memory or a
  minimum GPU requirement.
- **Explicit OWL-C CPU route:** real inference completed for the same four
  patches, reporting retained predicted counts **55, 17, 14, 9** (**95 total**).
- **OWL-D full-resolution route:** real tiled inference completed for both
  **5472×3648** images, using **160 tiles each** and reporting retained predicted
  counts **4 and 6**. Peak PyTorch GPU allocation was **3.5815 GiB** on the
  V100 32 GB. Both full images retained their original resolution.

These are localization counts in the backend's image order, **not ground-truth
counts or accuracy measurements**. They do not establish population size,
species identity, or superiority of either model. All these real backend routes
wrote their output artifacts. The observed approximately **3.6 GiB GPU tensor
allocation is not a universal minimum memory requirement**; allocator
reservations, CUDA/kernel overhead, and other memory use differ.
Actual end-to-end notebook execution has also succeeded separately through
nbclient, as documented below; this is not inferred merely from backend CLI
success.

Earlier Zenodo page/API requests returned timeouts/HTTP 504 from this host.
Subsequent retrieval of the primary record API and release README succeeded:
the published sizes and SHA-256 hashes match the configured OWL-D and OWL-C
pins, and the files also match the API's MD5 checksums. The release declares
CC BY-NC-SA 4.0; also review the included DINOv3 backbone terms.
Both **OWL-D and OWL-C were subsequently downloaded in full into separate,
initially empty caches**, and their sizes/SHA-256 hashes passed verification.
The working `demo_data/models/` cache was unchanged. Local download logs and
files are under `demo_data/owl_notebook/public_model_checks/`.
Use the isolated-cache commands above to repeat this check; choose new cache
directories to avoid merely reusing the successful downloads.
Fresh model requests on 2026-09-09 encountered intermittent gateway failures
and an interrupted transfer, so remote availability is not guaranteed.
The official-endpoint fallback and integrity checks do not hide final errors.

### Local notebook execution

**Verified on 2026-09-08:** nbclient successfully executed the complete
**23-cell notebook** with default OWL-D and the full-resolution switch enabled.
All four patches and both 5472×3648 images completed. A second complete
notebook execution explicitly configured `MODEL='owl-c'`, `DEVICE='cpu'`
and completed the four-patch CPU route. The recorded notebook outputs include
environment information, progress messages, tables, and inline figures.

Notebook counts matched the separately verified backend runs: OWL-D patch
counts **58, 19, 14, 9**, full-image counts **4 and 6**, and CPU OWL-C patch
counts **55, 17, 14, 9**. These are predicted localization counts, not
ground truth or accuracy measurements.

From the repository root, use
[`tests/execute_owl_notebook.py`](https://github.com/microsoft/MegaDetector-Overhead/blob/main/tests/execute_owl_notebook.py)
in the same checkout as the notebook/backend:

```bash
# Preserve the existing selected environment; do not use bare uv run.
.venv/bin/python tests/execute_owl_notebook.py --full \
    --output demo_data/owl_notebook/my_check_owld.ipynb
.venv/bin/python tests/execute_owl_notebook.py --model owl-c --device cpu \
    --output demo_data/owl_notebook/my_check_owlc_cpu.ipynb
```

Choose a **new output filename for each rerun**; the runner deliberately refuses
to overwrite previous executed notebooks. It returns exit code 0 and prints
`Executed notebook: ...` on success. On failure it retains the partial artifact
and reports the execution error. Use `--model-cache PATH` to run against an
explicit separate model cache.

The verified local artifacts are
`demo_data/owl_notebook/executed_owld.ipynb` and
`demo_data/owl_notebook/executed_owlc_cpu.ipynb`. Open them locally in
Jupyter/VS Code to inspect execution outputs. They and the CSV/PNG run
directories are **ignored local artifacts, not published repository files**.
The source `notebooks/owl_inference_demo.ipynb` keeps its outputs cleared.
The runner and notebook are distributed together at the matching repository ref.

**Fresh environment check (2026-09-09):** a separate working snapshot began
with an empty `.venv` that could not import PyTorch. The documented
`uv sync --locked --no-default-groups --group gpu --extra notebook` installed
the environment with author Conda/PYTHONPATH settings removed. All notebook
cells then completed for the four patches and both full images using the
previously verified checkpoint and supplied sample ZIP. Direct fresh model
downloads failed intermittently that day; this check proves environment
independence on the Linux/CUDA host, not remote-service uptime or every
operating system.

### Repository publication and sample release preparation

The publication layout is:

| Resource | Location |
|---|---|
| Notebook, helpers, manifest, dependency files, and documentation | Existing `microsoft/MegaDetector-Overhead` GitHub repository |
| `OWL_SAMPLE_DATA.zip` | GitHub release asset on that same repository |
| OWL model weights | Existing Zenodo record 20802844 |
| LinkedIn notebook link | Verified GitHub URL of the published `.ipynb`, not a local filesystem or sample ZIP URL |

To stage assets locally, use a new or empty ignored directory:

```bash
.venv/bin/python tools/demo_owl_notebook.py stage-release \
    --output-dir demo_data/owl_notebook/release_assets
```

This copies the exact 18,269,117-byte ZIP, verifies its SHA-256, and writes
`SHA256SUMS`, `SAMPLE_ATTRIBUTION.json`, and `RELEASE_NOTES.txt`.
It never uploads files, creates releases, or changes Git history. The notes
retain source-specific licenses and the origin of each confirmation:
**staged does not mean published**.
Choose another empty staging directory if regenerating after metadata changes;
existing artifacts are not overwritten.

After source terms are resolved and the repository owner explicitly approves
publication, the manual release procedure is:

1. Publish the notebook, backend, manifest, dependencies, and docs together in
   the existing repository, and identify the actual published revision.
2. Create the intended GitHub release on that repository and attach the exact
   ZIP plus completed attribution/checksum notes. Do not commit the ZIP.
3. Copy the real release-asset download URL into the manifest's `url` field
   and publish that manifest update. Do not assume a guessed tag/path exists.
4. In a separate checkout of the published revision with no local ZIP or
   cached weights, install the environment and execute the default notebook.
   Verify sample/model downloads, the four-image results, and exports. Test
   full-resolution and explicit CPU routes as applicable to the release.
5. Verify the actual GitHub notebook link and replace only `[NOTEBOOK_URL]`
   in the approved LinkedIn publication copy.

No step that publishes code or assets has been performed by release staging.

### Colab setup path and release gates

The notebook contains a **not-yet-verified Colab setup path**, not a promise of
hosted execution. It is experimental and **does not block the local
Jupyter/VS Code release**. Select a compatible GPU runtime for default OWL-D, and set
`REPO_REF='owl-notebook-v1'` for the matching notebook, backend, and manifest.
Setup clones only into an absent dedicated directory and validates any existing
checkout without resets, destructive updates, or overwrites. Configure
`REPO_PATH` to use an existing development checkout.

Hosted Python may not satisfy `>=3.11,<3.13`. If needed, setup installs `uv`
through pip in the disposable hosted environment and creates a managed project
Python 3.11 backend using the selected CPU/GPU group and notebook extra. Every
stage runs in a subprocess through that `.venv` Python; the hosted kernel
never imports PyTorch or `animaloc`. A ready backend is reused unless the
explicit `RESYNC_COLAB_ENVIRONMENT` repair switch is enabled.

When maintaining a release:

- Publish the matching notebook/backend/manifest ref and verify its access.
- Preserve the **exact sample ZIP URL** and verify a clean checksum-checked
  download. The local archive is not proof of public availability.
- Retain **source-specific terms and creator/source credits** with the asset:
  contributor-confirmed SheepCounter Public Domain and HerdNet CC BY-NC-SA 4.0,
  plus primary-source-verified caribou CC BY-NC-SA 4.0. Redistribution permission
  is confirmed; see
  [sample access notes](datasets.md#owl-notebook-sample-images).
- Verify fresh pinned Zenodo checkpoint downloads; a cached local model does
  not establish current public endpoint availability.
- Keep the verified local notebook evidence above distinct from hosted
  support. **Actual Colab execution must be tested separately** before
  advertising that support.

The source notebook has cleared execution outputs; the successful local
executed copies are retained only as ignored artifacts. Hosted execution is
still unverified.
For setup failures, follow [Installation](installation.md), fix CUDA or
explicitly choose CPU OWL-C, and keep full-resolution inference disabled if
resources are insufficient. Never bypass integrity checks or reinterpret a
failed download/load as an empty successful prediction.

---

## Caribou Demo (download → infer → visualize)

**The remaining sections describe the existing caribou evaluation shell
demos**, not the annotation-free notebook above. Their CPU auto-detection,
ground-truth metrics, and downsampled-coordinate conventions are unchanged.

This walkthrough takes you from a fresh clone to **visualized OWL-C predictions**
on real caribou aerial patches. It uses the public
[Caribou Aerial Survey Dataset](datasets.md) on Zenodo (weights + test patches),
runs the same evaluation stack as `tools/test.py`, and renders the detections
onto the patches as PNGs.

The demo **auto-detects** your hardware: it runs on a CUDA GPU when one is
available and otherwise falls back to CPU. It makes **no assumption** that you
have a GPU.

!!! note "About the weights"
    The Zenodo release labels the checkpoint "HerdNet (DLA-34)". In this repo the
    same DLA-34 detection branch is registered as **OWL-C**, so the demo loads it
    under `model.name: OWLC`. They are the same network.

## Prerequisites

Install the environment with `uv` (see [Installation](installation.md)):

```bash
uv sync
source .venv/bin/activate
python -c "import animaloc.models, dinov3; print('OK')"
```

You also need `curl` and `unzip` on your `PATH` (both are standard on Linux/macOS).

## One command

```bash
./tools/demo_caribou.sh
```

This will:

1. Download `Caribou-OWL-C.pth` (216 MB) and `test.zip` (1.2 GB) from Zenodo
   into `demo_data/` (skipped if already present).
2. Verify the weights' SHA-256 against the published checksum.
3. Build a deterministic **50-patch subset** (40 annotated + 10 background).
4. Auto-detect the device (GPU if available, else CPU).
5. Run OWL-C inference (`tools/test.py`) with Weights & Biases disabled.
6. Render predictions onto every patch with `tools/visualize_detections.py`.

Outputs:

| Path | Contents |
|---|---|
| `demo_data/run/metrics_results.csv` | F1 / precision / recall / MAE / RMSE |
| `demo_data/run/detections.csv` | One row per detection (`images, x, y, dscores, …`) |
| `demo_data/viz/*.png` | Patches with **green = ground truth, red = predictions** |

### Options

```bash
./tools/demo_caribou.sh --device cpu        # force CPU
./tools/demo_caribou.sh --device cuda        # force GPU
./tools/demo_caribou.sh --full               # run the full 2,607-patch test set
./tools/demo_caribou.sh --subset-size 100    # larger subset
./tools/demo_caribou.sh --score-threshold 0.3
```

## Expected results

On the default 50-patch subset (229 ground-truth points) you should see numbers
close to:

```
recall ≈ 0.98   precision ≈ 0.89   f1 ≈ 0.93
```

These match the per-patch validation regime reported for the checkpoint
(val F1 = 0.937). The full test set reproduces the paper headline
(F1 = 0.965 at τ = 20 px); see [Datasets](datasets.md). GPU and CPU produce
**identical detections** — only the speed differs (on a Tesla V100 the subset
runs ~25× faster than CPU).

## Compare all OWL models

`tools/demo_owl_models.sh` runs **all released pretrained models** on the caribou
data, visualizes each one's predictions, and prints a side-by-side metrics table.
It downloads the checkpoints from the same [Zenodo record](https://zenodo.org/records/20802844):

```bash
# CPU (default):
uv sync
./tools/demo_owl_models.sh --models "caribou-owl-c owl-c owl-t"

# GPU — sync the GPU build once; the demo scripts run through the venv directly,
# so they use it without reverting (see Installation → GPU support):
uv sync --no-default-groups --group gpu            # or: make sync-gpu
./tools/demo_owl_models.sh --device cuda           # includes owl-d on GPU

./tools/demo_owl_models.sh --device cpu --full     # full test set on CPU
```

| Key | Checkpoint | Registry | Training data |
|---|---|---|---|
| `caribou-owl-c` | `Caribou-OWL-C.pth` | `OWLC` | Caribou (in-domain reference) |
| `owl-c` | `OWL-C.pth` | `OWLC` | General overhead benchmark |
| `owl-t` | `OWL-T.pth` | `OWLT` | General overhead benchmark |
| `owl-d` | `OWL-D.pth` | `OWLD_H` | General overhead benchmark |

Each model writes `demo_data/run_<model>/` (metrics + detections) and
`demo_data/viz_<model>/` (overlays), plus a combined
`demo_data/model_comparison.csv`. Example on the default 50-patch subset (GPU):

```
        model device  recall  precision  f1_score
caribou-owl-c   cuda  0.9782     0.8854    0.9295
        owl-c   cuda  0.8734     0.8130    0.8421
        owl-t   cuda  0.8472     0.8661    0.8565
        owl-d   cuda  0.9563     0.9481    0.9522
```

Notably `owl-d` (a *general* model with a DINOv3 foundation backbone) nearly
matches the in-domain `caribou-owl-c` zero-shot — its backbone generalizes across
domains far better than the DLA/Swin encoders.

!!! note "Zero-shot vs in-domain"
    `owl-c` / `owl-t` / `owl-d` are trained on **other** public overhead datasets,
    not caribou — so on the caribou test set they run **zero-shot** and score below
    the in-domain `caribou-owl-c` (which hits the F1 = 0.965 headline). That gap is
    expected and is exactly what this comparison illustrates.

!!! warning "OWL-D needs a GPU"
    `owl-d` uses a DINOv3 ViT-H+/16 backbone (3.5 GB checkpoint). It is included
    **only when a CUDA GPU is available** and is skipped automatically on CPU-only
    machines. It loads entirely from `OWL-D.pth` (no separate Meta DINOv3 download
    required for inference).

## Manual walkthrough

If you prefer to run the steps yourself:

```bash
# 1. Download the caribou test patches + the caribou OWL-C weights
mkdir -p demo_data/weights demo_data/test
curl -fL -o demo_data/weights/best_model.pth \
    "https://zenodo.org/api/records/20802844/files/Caribou-OWL-C.pth/content"
curl -fL -o demo_data/test.zip \
    "https://zenodo.org/api/records/20802844/files/test.zip/content"
unzip -q demo_data/test.zip -d demo_data/test

# 2. Activate the venv, then run OWL-C eval
#    (CPU shown; for a GPU, `uv sync --no-default-groups --group gpu` first and
#     add ++test.device_name=cuda)
source .venv/bin/activate
export OWL_DEMO_DATA="$(pwd)/demo_data"
WANDB_MODE=disabled python tools/test.py test=owlc_caribou_demo \
    ++test.device_name=cpu \
    ++test.model.pth_file="$OWL_DEMO_DATA/weights/best_model.pth" \
    ++test.dataset.root_dir="$OWL_DEMO_DATA/test" \
    ++test.dataset.csv_file="$OWL_DEMO_DATA/test/gt.csv" \
    ++hydra.run.dir="$OWL_DEMO_DATA/run"

# 3. Visualize predictions onto the patches
#    (predictions are saved in the model's down-sampled space; OWL-C uses
#     down_ratio=2, so pass --pred-scale 2 to map them onto the patch)
python tools/visualize_detections.py \
    --detections "$OWL_DEMO_DATA/run/detections.csv" \
    --images-dir "$OWL_DEMO_DATA/test" \
    --output-dir "$OWL_DEMO_DATA/viz" \
    --gt "$OWL_DEMO_DATA/test/gt.csv" \
    --score-threshold 0.2 --pred-scale 2 --all-images
```

The portable demo config lives at `configs/test/owlc_caribou_demo.yaml` — unlike
the author-specific eval configs, it hardcodes no machine paths (they come from
`OWL_DEMO_DATA` or `++` overrides) and defaults to CPU.

## Evaluation operating point

The demo config (`configs/test/owlc_caribou_demo.yaml`) evaluates with:

* **Match radius τ = 20 image px.** `evaluator.threshold: 10` is measured on the
  half-resolution heatmap (`down_ratio: 2`, stitcher `up: False`); ground truth is
  down-sampled by the same factor, so 10 heatmap px = 20 original px.
* **Confidence (peak selection) `adapt_ts: 0.3`** (LMDS), with `neg_ts: 0.1` and a
  `(3, 3)` peak kernel.

This mirrors the per-patch **validation** regime (val F1 ≈ 0.937). The paper's
headline F1 = 0.965 is reported at a slightly different operating point
(c\* = 0.20); see [Datasets](datasets.md).

!!! note "Detection coordinate space"
    With `up: False`, `tools/test.py` writes `detections.csv` in the model's
    **down-sampled** space (x, y in 0…255 for a 512-px patch at `down_ratio=2`).
    Ground truth in `gt.csv` is in original 512-px space. The visualizer's
    `--pred-scale 2` rescales predictions so the two overlay correctly.

## Visualizing detections on your own runs

`tools/visualize_detections.py` works with any `detections.csv` produced by
`tools/test.py`:

```bash
python tools/visualize_detections.py \
    --detections path/to/detections.csv \
    --images-dir path/to/patches \
    --output-dir path/to/viz \
    --pred-scale 2 \
    [--gt path/to/gt.csv] [--score-threshold 0.2] [--all-images]
```

Predicted points are drawn in red; if `--gt` is given, ground-truth points are
drawn in green. Each patch is captioned with its predicted (and GT) point count.
Pass `--pred-scale` equal to the model's `down_ratio` (2 for OWL-C) so the
down-sampled predictions land on the full-resolution patch; ground truth is never
scaled.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `wandb: ERROR ...` or a login prompt | The demo sets `WANDB_MODE=disabled`. Running `tools/test.py` by hand requires `WANDB_MODE=disabled` (or `wandb login`). |
| `CUDA: False` even though `nvidia-smi` shows a GPU | A plain `uv sync` installs the **CPU** build. Sync the GPU build (`uv sync --no-default-groups --group gpu`) and run via the activated venv (`source .venv/bin/activate`), not bare `uv run` (see [Installation → GPU support](installation.md#gpu-support)). |
| `RuntimeError: ... unable to find an engine` on an older GPU | The wheel lacks kernels for your GPU's architecture. The `gpu` group (cu121) covers Volta (V100) – Hopper; very new GPUs need a cu128 group (see INSTALL.md). |
| Red prediction dots look shifted toward the top-left / "smaller" | Predictions are in the model's down-sampled space — pass `--pred-scale 2` (the OWL-C `down_ratio`) to the visualizer. |
| `ImportError: libGL.so.1` / `libgthread-2.0.so.0` | Image libs need system glib/GL. The project pins `opencv-python-headless`; re-run `uv sync` if it was replaced. |
| Checksum mismatch on weights | A corrupted/partial download. Delete `demo_data/weights/` and re-run. |

## See also

* [Datasets](datasets.md) — dataset details and the Zenodo record
* [Training, Evaluation, and Inference](training.md) — the full eval/inference stack
* [Model Zoo](model_zoo.md) — the OWL-C / OWL-D / OWL-T families
