"""Reproducible, annotation-free OWL notebook inference and asset preparation."""

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "notebooks/sample_data.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DETECTION_COLUMNS = ["images", "x", "y", "dscores", "labels"]
SUMMARY_COLUMNS = ["images", "width", "height", "count", "seconds", "tiles"]
MODEL_SPECS = {
    "owl-d": {
        "filename": "OWL-D.pth",
        "registry": "OWLD_H",
        "size": 3541435091,
        "sha256": "e28fc06d28fe010faefaffc5eb9bd0b85f98dad0805fca9f5256562cc3403d5b",
        "kwargs": {
            "pretrained": False, "head_conv": 64, "readout_type": "ignore",
            "freeze_backbone": True, "down_ratio": 2,
        },
    },
    "owl-c": {
        "filename": "OWL-C.pth",
        "registry": "OWLC",
        "size": 216392299,
        "sha256": "de97b6bbc50e20d914b179372a1ea42c569795591a3cf33b926ba1a6e659939a",
        "kwargs": {"pretrained": False, "head_conv": 64, "down_ratio": 2},
    },
}
CHECKSUM_PROVENANCE = (
    "SHA-256 and file size match the published Zenodo record 20802844 README: "
    "https://zenodo.org/api/records/20802844/files/README.md/content"
)


class DownloadError(RuntimeError):
    """A remote asset could not be fetched after bounded network retries."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    if path.stat().st_size != expected_size or sha256_file(path) != expected_sha256:
        raise ValueError(f"Integrity check failed: {path}. Move the corrupt file aside and retry.")


def download_file(url: str, destination: Path, size: int, sha256: str) -> Path:
    import requests

    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Asset download URL must be an absolute HTTPS URL")
    if destination.exists() or destination.is_symlink():
        verify_file(destination, size, sha256)
        print(f"Verified cached asset: {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as staging:
                partial = Path(staging) / "asset.part"
                print(f"Download attempt {attempt}/3: {url} ({size / 1e6:.1f} MB)", flush=True)
                with requests.get(url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    downloaded = 0
                    last_print = time.monotonic()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            downloaded += len(chunk)
                            if downloaded > size:
                                raise ValueError("Download exceeds the pinned asset size")
                            stream.write(chunk)
                            if time.monotonic() - last_print >= 5:
                                print(f"  {downloaded / 1e6:.1f}/{size / 1e6:.1f} MB", flush=True)
                                last_print = time.monotonic()
                verify_file(partial, size, sha256)
                if destination.exists():
                    verify_file(destination, size, sha256)
                else:
                    partial.replace(destination)
                return destination
        except requests.RequestException as exc:
            print(f"Download failed ({attempt}/3): {exc}", file=sys.stderr, flush=True)
            if attempt == 3:
                raise DownloadError(f"Could not download {url}; no completed asset was cached") from exc
            time.sleep(attempt)
    raise RuntimeError("Download retry loop ended unexpectedly")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def stage_release_assets(output_dir: Path, archive_path: Path | None = None,
                         manifest_path: Path = MANIFEST) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["attribution"]["sources"]
    source_ids = {source["id"] for source in sources}
    if len(source_ids) != len(sources) or any(entry.get("source_id") not in source_ids for entry in manifest["files"]):
        raise ValueError("Every sample must map to a unique documented source ID")
    archive_path = archive_path if archive_path is not None else ROOT / manifest["archive_name"]
    verify_file(archive_path, manifest["archive_size"], manifest["archive_sha256"])
    with ZipFile(archive_path) as archive:
        validate_archive_members(archive, manifest)
    if not output_dir.resolve().is_relative_to(ROOT / "demo_data"):
        raise ValueError("Release staging must stay under the repository's ignored demo_data directory")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Use a new or empty release staging directory; existing files will not be overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / manifest["archive_name"]
    shutil.copyfile(archive_path, target)
    verify_file(target, manifest["archive_size"], manifest["archive_sha256"])
    (output_dir / "SHA256SUMS").write_text(
        f"{manifest['archive_sha256']}  {manifest['archive_name']}\n", encoding="utf-8",
    )
    (output_dir / "SAMPLE_ATTRIBUTION.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    notes = [
        "OWL notebook sample release preparation",
        "",
        "STAGING ONLY: no upload or publication has been performed.",
        manifest["publication_status"],
        "",
        "Notebook: microsoft/MegaDetector-Overhead/notebooks/owl_inference_demo.ipynb",
        "Publish the notebook and matching backend in the existing GitHub repository.",
        "GitHub previews the notebook; users execute it in local Jupyter/VS Code.",
        "Attach the exact sample ZIP to a release on the same repository, outside Git history.",
        "Model weights remain on Zenodo record 20802844. Colab is experimental and nonblocking.",
        "",
        f"Asset: {manifest['archive_name']} ({manifest['archive_size']} bytes)",
        f"SHA-256: {manifest['archive_sha256']}",
        "The archive bytes are unchanged; attribution accompanies it in SAMPLE_ATTRIBUTION.json.",
        "",
        "Source mapping supplied by the contributor (not independently inferred):",
    ]
    for source in sources:
        notes.append(f"- {source['name']}: {source['url']}")
        notes.append(f"  License: {source.get('license_identifier', 'Not recorded')}")
        if source.get("license_url"):
            notes.append(f"  License URL: {source['license_url']}")
        if source.get("required_credit"):
            notes.append(f"  Credit: {source['required_credit']}")
        notes.append(f"  License/credit verification: {source['verification_status']}")
    notes.extend([
        "",
        "Before publication: retain the documented source-specific terms and credits;",
        "confirm public model access and obtain explicit approval to publish code/upload assets.",
        "After upload: set the exact public asset URL in notebooks/sample_data.json and verify",
        "a fresh checkout and clean downloads before replacing the approved post's notebook URL.",
        "Do not apply the software MIT license to the images or advertise unverified hosted support.",
    ])
    (output_dir / "RELEASE_NOTES.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Prepared exact archive and source notes in {output_dir}\n"
          f"STAGING ONLY. {manifest['publication_status']}", flush=True)
    return output_dir


def validate_archive_members(archive: ZipFile, manifest: dict) -> None:
    expected = {entry["path"]: entry for entry in manifest["files"]}
    seen = set()
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        if (
            path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or name in seen or stat.S_ISLNK(member.external_attr >> 16)
            or name.rstrip("/") != path.as_posix()
        ):
            raise ValueError(f"Unsafe or duplicate ZIP member: {name}")
        seen.add(name)
        if member.is_dir():
            if name != manifest["root"] + "/":
                raise ValueError(f"Unexpected ZIP directory: {name}")
        elif name not in expected or member.file_size != expected[name]["size"]:
            raise ValueError(f"Unexpected ZIP file or size: {name}")
    if {name for name in seen if not name.endswith("/")} != set(expected):
        raise ValueError("ZIP files do not match the sample manifest")


def verify_samples(base: Path, manifest: dict) -> list[dict]:
    from PIL import Image

    root = base / manifest["root"]
    if root.is_symlink():
        raise ValueError(f"Sample directory must not be a symlink: {root}")
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {path.relative_to(base).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(f"Extracted sample inventory differs from manifest: {root}")
    rows = []
    for entry in manifest["files"]:
        path = base / entry["path"]
        verify_file(path, entry["size"], entry["sha256"])
        with Image.open(path) as image:
            if image.size != (entry["width"], entry["height"]):
                raise ValueError(f"Unexpected image dimensions: {path}")
            image.verify()
        rows.append({"images": path.name, "width": entry["width"], "height": entry["height"]})
    return rows


def prepare_samples(data_dir: Path, archive_path: Path | None = None,
                    sample_url: str | None = None, manifest_path: Path = MANIFEST) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir.mkdir(parents=True, exist_ok=True)
    if archive_path is not None:
        source = archive_path
    elif sample_url or manifest["url"]:
        source = download_file(
            sample_url or manifest["url"], data_dir / manifest["archive_name"],
            manifest["archive_size"], manifest["archive_sha256"],
        )
    elif (ROOT / manifest["archive_name"]).is_file():
        source = ROOT / manifest["archive_name"]
    elif (data_dir / manifest["archive_name"]).is_file():
        source = data_dir / manifest["archive_name"]
    else:
        raise FileNotFoundError(
            "The public sample URL is not configured. Supply --archive /path/to/OWL_SAMPLE_DATA.zip "
            "or --sample-url HTTPS_URL for the exact released archive. Hosting is a publication gate."
        )
    verify_file(source, manifest["archive_size"], manifest["archive_sha256"])
    print(f"Sample archive: {source}\nSHA-256: {manifest['archive_sha256']}", flush=True)
    destination = data_dir / manifest["root"]
    with ZipFile(source) as archive:
        validate_archive_members(archive, manifest)
        if not destination.exists() and not destination.is_symlink():
            with tempfile.TemporaryDirectory(prefix=".extract-", dir=data_dir) as staging:
                staged = Path(staging)
                archive.extractall(staged)
                verify_samples(staged, manifest)
                (staged / manifest["root"]).rename(destination)
    rows = verify_samples(data_dir, manifest)
    write_csv(data_dir / "sample_inventory.csv", ["images", "width", "height"], rows)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for index, (axis, row) in enumerate(zip(axes.flat, rows), start=1):
        with Image.open(destination / row["images"]) as image:
            image.thumbnail((900, 600))
            axis.imshow(image)
        axis.set_title(f"Sample {index}: {row['width']} x {row['height']}")
        axis.axis("off")
        print(f"  {row['images']}: {row['width']} x {row['height']}")
    figure.tight_layout()
    figure.savefig(data_dir / "contact_sheet.png", dpi=110)
    plt.close(figure)
    print(f"Verified {len(rows)} images. {manifest['publication_status']}", flush=True)
    return destination


def resolve_device(model_name: str, requested: str, cuda_available: bool) -> str:
    if model_name not in MODEL_SPECS or requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unknown model/device choice")
    device = "cuda" if requested == "auto" and cuda_available else requested
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not cuda_available:
        raise ValueError("CUDA was requested but is unavailable. Check your runtime and GPU PyTorch installation.")
    if model_name == "owl-d" and device != "cuda":
        raise ValueError(
            "The OWL-D notebook route requires CUDA. Enable a compatible GPU or explicitly set "
            "MODEL='owl-c' / --model owl-c for CPU. The model has NOT been changed automatically."
        )
    return device


def environment_info(model_name: str, requested: str) -> dict:
    import torch

    device = resolve_device(model_name, requested, torch.cuda.is_available())
    info = {
        "python": sys.version.split()[0], "interpreter": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "model": model_name, "registry": MODEL_SPECS[model_name]["registry"],
        "device": device,
    }
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    if git.returncode:
        print(f"Repository revision unavailable: {git.stderr.strip()}", file=sys.stderr)
        info["revision"] = None
    else:
        info["revision"] = git.stdout.strip()
    if info["device"] == "cuda":
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(), free_vram_gib=free / 2**30, total_vram_gib=total / 2**30)
    print(json.dumps(info, indent=2), flush=True)
    return info


def model_download_urls(model_name: str) -> tuple[str, str]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    filename = MODEL_SPECS[model_name]["filename"]
    return (
        f"https://zenodo.org/records/20802844/files/{filename}?download=1",
        f"https://zenodo.org/api/records/20802844/files/{filename}/content",
    )


def fetch_model_checkpoint(model_name: str, cache_dir: Path | None = None) -> Path:
    urls = model_download_urls(model_name)
    spec = MODEL_SPECS[model_name]
    cache_dir = cache_dir if cache_dir is not None else ROOT / "demo_data/models"
    for index, url in enumerate(urls):
        try:
            return download_file(url, cache_dir / spec["filename"], spec["size"], spec["sha256"])
        except DownloadError:
            if index == len(urls) - 1:
                raise
            print("Primary Zenodo file download failed; trying the official alternate endpoint "
                  "with the same pinned size and SHA-256.", file=sys.stderr, flush=True)
    raise RuntimeError("No model download endpoint was attempted")


def load_demo_model(model_name: str, device: str, checkpoint_path: Path | None = None,
                    cache_dir: Path | None = None):
    import torch
    from animaloc.models import MODELS, LossWrapper

    if checkpoint_path is not None and cache_dir is not None:
        raise ValueError("Choose either --checkpoint or --model-cache, not both")
    spec = MODEL_SPECS[model_name]
    urls = model_download_urls(model_name)
    if checkpoint_path is None:
        path = fetch_model_checkpoint(model_name, cache_dir)
    else:
        path = checkpoint_path
        verify_file(path, spec["size"], spec["sha256"])
    print(f"Model: {spec['registry']} | {spec['size'] / 1e9:.3f} GB\nCheckpoint: {path}\n"
          f"SHA-256: {spec['sha256']}\n{CHECKSUM_PROVENANCE}", flush=True)
    start = time.perf_counter()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Released checkpoint must contain model_state_dict")
    model = LossWrapper(MODELS[spec["registry"]](**spec["kwargs"]), [])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval().requires_grad_(False).to(device)
    metadata = {
        "path": str(path), "url": urls[0], "download_urls": list(urls),
        "sha256": spec["sha256"], "size": spec["size"],
        "checksum_provenance": CHECKSUM_PROVENANCE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": time.perf_counter() - start,
    }
    print(f"Loaded {metadata['parameters']:,} parameters in {metadata['load_seconds']:.2f}s", flush=True)
    return model, metadata


def select_images(directory: Path, selection: str) -> list[Path]:
    from PIL import Image

    if selection not in {"patches", "full", "all"}:
        raise ValueError(f"Unknown image selection: {selection}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    selected = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            with Image.open(path) as image:
                width, height = image.size
                if min(width, height) < 2:
                    raise ValueError(f"Image must be at least 2x2 pixels: {path}")
                image.verify()
            small = width <= 512 and height <= 512
            if selection == "all" or (selection == "patches" and small) or (selection == "full" and not small):
                selected.append(path)
    if not selected:
        raise ValueError(f"No supported images matched selection={selection!r} in {directory}")
    return selected


def heatmap_detections(heatmap, image_name: str, width: int, height: int,
                       score_threshold: float, adapt_threshold: float, negative_threshold: float) -> list[dict]:
    import torch
    from animaloc.eval.lmds import HerdNet_Detection_Branch_LMDS

    if tuple(heatmap.shape) != (1, 1, height // 2, width // 2):
        raise ValueError(f"Unexpected heatmap shape for {image_name}: {tuple(heatmap.shape)}")
    if not torch.isfinite(heatmap).all():
        raise ValueError(f"Non-finite heatmap for {image_name}")
    lmds = HerdNet_Detection_Branch_LMDS(
        up=False, kernel_size=(3, 3), adapt_ts=adapt_threshold, neg_ts=negative_threshold,
    )
    _, locations, _, scores = lmds(heatmap)
    rows = []
    for (y, x), score in zip(locations[0], scores[0]):
        if score < score_threshold:
            continue
        x, y = float(x * 2), float(y * 2)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Prediction outside original image bounds: {image_name}, {x}, {y}")
        rows.append({"images": image_name, "x": x, "y": y, "dscores": float(score), "labels": 1})
    return rows


def render_prediction(image, heatmap, rows: list[dict], output: Path, stem: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from animaloc.vizual import draw_points

    overlay = draw_points(image.copy(), [(row["y"], row["x"]) for row in rows],
                          color="red", size=8)
    overlay.save(output / "overlays" / f"{stem}.png")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[1].imshow(image)
    axes[1].imshow(heatmap, extent=(0, image.width, image.height, 0),
                   cmap="magma", alpha=0.7, vmin=0, vmax=max(float(heatmap.max()), 1e-6))
    axes[1].set_title("FIDT heatmap (display-normalized)")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Predicted locations: {len(rows)} (red)")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    filename = f"figures/{stem}.png"
    figure.savefig(output / filename, dpi=120)
    plt.close(figure)
    figures = [filename]
    if max(image.size) > 512:
        center_x, center_y = (rows[0]["x"], rows[0]["y"]) if rows else (image.width / 2, image.height / 2)
        left = max(0, min(int(center_x) - 256, image.width - 512))
        top = max(0, min(int(center_y) - 256, image.height - 512))
        crop_file = f"figures/{stem}_detail.png"
        overlay.crop((left, top, left + 512, top + 512)).save(output / crop_file)
        figures.append(crop_file)
    return figures


def run_inference(images_dir: Path, output_dir: Path, model_name: str = "owl-d",
                  requested_device: str = "auto", selection: str = "patches",
                  checkpoint_path: Path | None = None, score_threshold: float = 0.2,
                  adapt_threshold: float = 0.3, negative_threshold: float = 0.1,
                  tile_size: int = 512, overlap: int = 160, cpu_threads: int = 4,
                  model_cache: Path | None = None) -> dict:
    import numpy as np
    from PIL import Image
    import torch
    from animaloc.eval.stitchers import HerdNet_Detection_Branch_Stitcher

    if tile_size < 32 or tile_size % 32 or overlap < 0 or overlap >= tile_size or overlap % 2:
        raise ValueError("Tile size must be a positive multiple of 32; overlap must be even and below tile size")
    if cpu_threads < 1:
        raise ValueError("CPU thread count must be positive")
    if not all(math.isfinite(value) and 0 <= value <= 1
               for value in (score_threshold, adapt_threshold, negative_threshold)):
        raise ValueError("Peak thresholds must be finite values in [0, 1]")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Use a new or empty output directory to avoid stale results: {output_dir}")
    images = select_images(images_dir, selection)
    info = environment_info(model_name, requested_device)
    torch.set_num_threads(cpu_threads)
    device = info["device"]
    model, checkpoint_info = load_demo_model(model_name, device, checkpoint_path, model_cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir()
    (output_dir / "overlays").mkdir()
    stitcher = HerdNet_Detection_Branch_Stitcher(
        model, size=(tile_size, tile_size), overlap=overlap, batch_size=1,
        down_ratio=2, up=False, reduction="mean", device_name=device,
    )
    settings = {
        "selection": selection, "tile_size": tile_size, "overlap": overlap, "tile_batch_size": 1,
        "down_ratio": 2, "score_threshold": score_threshold,
        "adapt_threshold": adapt_threshold, "negative_threshold": negative_threshold,
        "cpu_threads": cpu_threads, "coordinate_space": "original image pixels (x, y)",
    }
    print("Inference settings:", json.dumps(settings), flush=True)
    print("Counts are retained localization peaks, not heatmap sums or validated population counts.", flush=True)
    summary, detections, figures = [], [], []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, path in enumerate(images, start=1):
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        tiles = max(1, math.ceil((height - tile_size) / (tile_size - overlap)) + 1) * max(
            1, math.ceil((width - tile_size) / (tile_size - overlap)) + 1,
        )
        print(f"[{index}/{len(images)}] {path.name} | {width}x{height} | {tiles} tile(s)", flush=True)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32,
        )
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            heatmap = stitcher(tensor).cpu()
            rows = heatmap_detections(
                heatmap, path.name, width, height, score_threshold, adapt_threshold, negative_threshold,
            )
        seconds = time.perf_counter() - start
        detections.extend(rows)
        summary.append({
            "images": path.name, "width": width, "height": height,
            "count": len(rows), "seconds": round(seconds, 4), "tiles": tiles,
        })
        figures.extend(render_prediction(image, heatmap[0, 0].numpy(), rows, output_dir, f"{index:02d}_{path.stem}"))
        print(f"  Predicted count: {len(rows)} | inference + peak extraction: {seconds:.2f}s", flush=True)
        del heatmap, tensor, array, image
    write_csv(output_dir / "detections.csv", DETECTION_COLUMNS, detections)
    write_csv(output_dir / "summary.csv", SUMMARY_COLUMNS, summary)
    metadata = {
        "environment": info, "checkpoint": checkpoint_info, "settings": settings,
        "images_dir": str(images_dir.resolve()), "figures": figures,
        "images_processed": len(images), "total_predicted_detections": len(detections),
        "inference_seconds": sum(row["seconds"] for row in summary),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
        "interpretation": "Unannotated demonstration. Peak scores are not calibrated probabilities; no accuracy metrics.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Finished {len(images)} images; {len(detections)} predicted detections across demo images.\n"
          f"Peak GPU allocation: {metadata['peak_gpu_allocated_gib']} GiB\nResults: {output_dir}", flush=True)
    return metadata


DEMO_ERRORS = (OSError, ValueError, RuntimeError, BadZipFile)
