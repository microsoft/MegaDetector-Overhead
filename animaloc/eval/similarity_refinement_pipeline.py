"""
End-to-end similarity refinement pipeline for large overhead images.

Processes multiple full-size images by:
  1. Loading detector predictions (x, y, dscore) from a CSV
  2. For each image: tiling into 512x512 patches, computing per-patch DINO
     cosine-similarity maps, stitching them, running LMDS peak detection
  3. Writing refined detections to an output CSV in the same format as the input

Usage:
    python -m animaloc.eval.similarity_refinement_pipeline \\
        --csv detections.csv \\
        --image_dir /path/to/images \\
        --output_csv refined_detections.csv \\
        --model vits16 \\
        --threshold 0.1
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .similarity_stitcher import SimilarityMapStitcher

__all__ = ['load_detections_csv', 'run_refinement_pipeline', 'visualize_stitched_result']


def load_detections_csv(csv_path):
    """Load detections CSV grouped by image.

    Expected columns: images, labels, dscores, x, y

    Returns:
        dict: image_name -> list of dicts {label, dscore, x, y}
    """
    detections = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            detections[row["images"]].append({
                "label": float(row["labels"]),
                "dscore": float(row["dscores"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
            })
    return dict(detections)


def visualize_stitched_result(
    pil_image,
    stitched_map,
    input_detections,
    refined_detections,
    image_name,
    output_dir,
    threshold=0.1,
    aggregation='weighted',
    output_csv_name="refined_detections",
):
    """Generate a 3-panel diagnostic figure for one stitched image.

    Layout:
      (a) Original image + input detection points (colored by dscore)
      (b) Stitched similarity heatmap overlaid on image
      (c) Original image + LMDS refined predictions (colored by similarity score)

    Args:
        pil_image: PIL Image of the full-size image.
        stitched_map: [1, 1, H_tok, W_tok] normalized similarity map tensor.
        input_detections: list of dicts with 'x', 'y', 'dscore' (pre-threshold).
        refined_detections: list of dicts with 'x', 'y', 'dscores' (from LMDS).
        image_name: filename for the title and output path.
        output_dir: directory to save the visualization.
        threshold: confidence threshold used (for display).
        aggregation: aggregation mode used (for display).
        output_csv_name: base name of the output CSV (for display).
    """
    img_np = np.array(pil_image)
    img_h, img_w = img_np.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

    # ── (a) Original image + input detection points ──
    ax = axes[0]
    ax.imshow(img_np)
    if input_detections:
        in_x = [d['x'] for d in input_detections]
        in_y = [d['y'] for d in input_detections]
        in_s = [d['dscore'] for d in input_detections]
        sc = ax.scatter(
            in_x, in_y, c=in_s, cmap="hot", s=12, edgecolors="cyan",
            linewidths=0.5, vmin=0, vmax=max(max(in_s), 0.5),
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="dscore")
    ax.set_title(
        f"Input detections (n={len(input_detections)}, t\u2265{threshold:.2f})",
        fontsize=9,
    )
    ax.axis("off")

    # ── (b) Stitched similarity heatmap overlaid on image ──
    ax = axes[1]
    ax.imshow(img_np, alpha=0.3)
    sim_np = stitched_map.squeeze().numpy()
    im = ax.imshow(
        sim_np, cmap="jet", alpha=0.7, interpolation="bilinear",
        extent=[0, img_w, img_h, 0],
    )
    ax.set_title(
        f"Stitched similarity ({aggregation})\n"
        f"max={sim_np.max():.3f}, mean={sim_np.mean():.3f}",
        fontsize=9,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── (c) Original image + LMDS refined predictions ──
    ax = axes[2]
    ax.imshow(img_np)
    if refined_detections:
        ref_x = [d['x'] for d in refined_detections]
        ref_y = [d['y'] for d in refined_detections]
        ref_s = [d['dscores'] for d in refined_detections]
        sc2 = ax.scatter(
            ref_x, ref_y, c=ref_s, cmap="hot", s=30, edgecolors="lime",
            linewidths=0.8, vmin=0, vmax=1,
        )
        plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04, label="LMDS score")
    ax.set_title(
        f"Refined detections (n={len(refined_detections)})",
        fontsize=9,
    )
    ax.axis("off")

    plt.suptitle(
        f"{image_name}  \u2014  threshold {threshold:.2f}  \u2014  {img_w}\u00d7{img_h}px",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    viz_dir = os.path.join(output_dir, "viz_", output_csv_name)
    os.makedirs(viz_dir, exist_ok=True)
    stem = image_name.rsplit(".", 1)[0]
    out_path = os.path.join(viz_dir, f"{stem}_stitched_viz.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    return out_path


def run_refinement_pipeline(
    image_dir,
    detections_csv,
    output_csv,
    dino_model,
    n_layers=12,
    size=(512, 512),
    overlap=128,
    down_ratio=16,
    threshold=0.1,
    aggregation='weighted',
    use_hann=True,
    device='cuda',
    sim_upsample=1,
    lmds_kernel=5,
    lmds_adapt_ts=100.0 / 255.0,
    lmds_neg_ts=0.1,
    save_maps=False,
    maps_dir=None,
    n_viz=0,
    viz_seed=42,
):
    """Process all images and write refined detections to CSV.

    Args:
        image_dir: Directory containing full-size images.
        detections_csv: Path to input detections CSV.
        output_csv: Path for output refined detections CSV.
        dino_model: Loaded DINOv3 ViT model (eval mode).
        n_layers: Number of ViT layers.
        size: Patch size (height, width).
        overlap: Overlap in pixels (must be divisible by down_ratio).
        down_ratio: ViT patch_size.
        threshold: Confidence threshold for filtering input detections.
        aggregation: 'weighted' or 'unweighted'.
        use_hann: Apply Hann windowing.
        device: 'cuda' or 'cpu'.
        sim_upsample: Upsample factor for similarity map before LMDS.
        lmds_kernel: LMDS kernel size.
        lmds_adapt_ts: LMDS adaptive threshold.
        lmds_neg_ts: LMDS negative sample threshold.
        save_maps: If True, save stitched similarity maps as .npy files.
        maps_dir: Directory for saved maps (required if save_maps=True).
        n_viz: Number of images to visualize (0 = off, -1 = all).
        viz_seed: Random seed for selecting images to visualize.

    Returns:
        list of all refined detection dicts.
    """
    # Load input detections
    all_detections = load_detections_csv(detections_csv)
    image_names = sorted(all_detections.keys())
    total_dets = sum(len(v) for v in all_detections.values())
    print(f"Loaded {total_dets} detections across {len(image_names)} images")
    print(f"Threshold: {threshold}, Aggregation: {aggregation}")

    # Create stitcher
    stitcher = SimilarityMapStitcher(
        dino_model=dino_model,
        size=size,
        overlap=overlap,
        down_ratio=down_ratio,
        aggregation=aggregation,
        use_hann=use_hann,
        device=device,
        n_layers=n_layers,
        sim_upsample=sim_upsample,
        lmds_kernel=lmds_kernel,
        lmds_adapt_ts=lmds_adapt_ts,
        lmds_neg_ts=lmds_neg_ts,
    )

    if save_maps and maps_dir:
        os.makedirs(maps_dir, exist_ok=True)

    # Select images for visualization
    import random
    viz_set = set()
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    if n_viz != 0:
        # Find images that actually exist on disk
        existing = [n for n in image_names
                    if os.path.exists(os.path.join(image_dir, n))]
        if n_viz == -1:
            viz_set = set(existing)
        else:
            random.seed(viz_seed)
            viz_set = set(random.sample(existing, min(n_viz, len(existing))))
        print(f"Visualization enabled for {len(viz_set)} images "
              f"(seed={viz_seed})")

    all_results = []
    skipped = 0
    t0 = time.time()

    for img_idx, img_name in enumerate(image_names):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Filter detections by threshold
        img_dets = all_detections[img_name]
        filtered = [d for d in img_dets if d['dscore'] >= threshold]

        # Load image
        pil_image = Image.open(img_path).convert('RGB')

        # Run stitcher (stitch + LMDS + CSV formatting)
        result = stitcher(pil_image, filtered, image_name=img_name)

        all_results.extend(result['detections'])

        # Optionally save the stitched similarity map
        if save_maps and maps_dir:
            stem = img_name.rsplit('.', 1)[0]
            np.save(
                os.path.join(maps_dir, f"{stem}_stitched_sim.npy"),
                result['stitched_map'].squeeze().numpy(),
            )

        # Visualization
        if img_name in viz_set:
            viz_path = visualize_stitched_result(
                pil_image=pil_image,
                stitched_map=result['stitched_map'],
                input_detections=filtered,
                refined_detections=result['detections'],
                image_name=img_name,
                output_dir=output_dir,
                threshold=threshold,
                aggregation=aggregation,
                output_csv_name =output_csv.rsplit(".", 1)[0],
            )
            print(f"  Viz saved: {viz_path}")

        # Progress log
        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            elapsed = time.time() - t0
            n_refined = len(all_results)
            print(
                f"  [{img_idx + 1}/{len(image_names)}] "
                f"{img_name}: {len(result['detections'])} refined detections, "
                f"total so far: {n_refined}, elapsed: {elapsed:.1f}s"
            )

    # Write output CSV
    fieldnames = ["images", "labels", "dscores", "x", "y", "count_1"]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    elapsed = time.time() - t0
    print(f"\nDone. {len(all_results)} refined detections written to {output_csv}")
    print(f"Images processed: {len(image_names) - skipped} "
          f"(skipped {skipped} missing)")
    print(f"Total time: {elapsed:.1f}s")

    return all_results


# ── Model loading (self-contained, no dependency on patch_feature_visualization) ──

MODEL_REGISTRY = {
    "vits16":     ("dinov3_vits16",     12, 384),
    "vits16plus": ("dinov3_vits16plus", 12, 384),
    "vitb16":     ("dinov3_vitb16",     12, 768),
    "vitl16":     ("dinov3_vitl16",     24, 1024),
    "vith16plus": ("dinov3_vith16plus", 32, 1280),
    "vit7b16":    ("dinov3_vit7b16",    40, 4096),
}


def load_dino_model(model_name, device, weights_path, dinov3_root=None):
    """Load a DINOv3 ViT model.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g. 'vits16', 'vith16plus').
        device: torch.device.
        weights_path: Path to the .pth weights file.
        dinov3_root: Path to the dinov3 repo root (containing dinov3/hub/).
            If None, tries common locations.

    Returns:
        (model, n_layers): loaded model on device and its depth.
    """
    # Ensure dinov3 package is importable
    if dinov3_root is not None:
        if dinov3_root not in sys.path:
            sys.path.insert(0, dinov3_root)
    else:
        # Try common locations
        for candidate in [
            "/home/v-ichaconsil/azurefiles/dinov3",
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
        ]:
            if os.path.isdir(os.path.join(candidate, 'dinov3')):
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                break

    import dinov3.hub.backbones as backbones

    loader_func_name, depth, embed_dim = MODEL_REGISTRY[model_name]
    loader_fn = getattr(backbones, loader_func_name)
    model = loader_fn(pretrained=True, weights=weights_path)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"Model loaded: {model_name} ({loader_func_name}), "
        f"embed_dim={model.embed_dim}, depth={depth}, {n_params:.1f}M params"
    )
    print(f"  Weights: {os.path.basename(weights_path)}")
    return model, depth


def _parse_args():
    p = argparse.ArgumentParser(
        description="DINO similarity refinement pipeline for full-size images"
    )
    p.add_argument("--csv", type=str, required=True,
                   help="Input detections CSV")
    p.add_argument("--image_dir", type=str, required=True,
                   help="Directory with full-size images")
    p.add_argument("--output_csv", type=str, required=True,
                   help="Output refined detections CSV")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Confidence threshold (default: 0.1)")
    p.add_argument("--aggregation", type=str,
                   choices=["weighted", "unweighted"], default="weighted",
                   help="Similarity map aggregation mode")
    p.add_argument("--overlap", type=int, default=128,
                   help="Patch overlap in pixels (must be divisible by 16)")
    p.add_argument("--sim_upsample", type=int, default=1,
                   help="Upsample factor for similarity map before LMDS")
    p.add_argument("--lmds_kernel", type=int, default=5,
                   help="LMDS kernel size (must be odd)")
    p.add_argument("--lmds_adapt_ts", type=float, default=100.0 / 255.0,
                   help="LMDS adaptive threshold")
    p.add_argument("--lmds_neg_ts", type=float, default=0.1,
                   help="LMDS negative sample threshold")
    p.add_argument("--model", type=str, default="vits16",
                   choices=list(MODEL_REGISTRY.keys()),
                   help="DINOv3 ViT model variant")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to DINOv3 weights .pth file")
    p.add_argument("--dinov3_root", type=str, default=None,
                   help="Path to dinov3 repo root (containing dinov3/hub/). "
                        "Auto-detected if not provided.")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: auto)")
    p.add_argument("--use_hann", action="store_true", default=True,
                   help="Use Hann windowing (default: True)")
    p.add_argument("--no_hann", action="store_true",
                   help="Disable Hann windowing")
    p.add_argument("--save_maps", action="store_true",
                   help="Save stitched similarity maps as .npy")
    p.add_argument("--maps_dir", type=str, default=None,
                   help="Directory for saved maps")
    p.add_argument("--n_viz", type=int, default=0,
                   help="Number of images to visualize (0=off, -1=all)")
    p.add_argument("--viz_seed", type=int, default=42,
                   help="Random seed for selecting images to visualize")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model, n_layers = load_dino_model(
        args.model, device, args.weights, dinov3_root=args.dinov3_root,
    )

    use_hann = args.use_hann and not args.no_hann

    run_refinement_pipeline(
        image_dir=args.image_dir,
        detections_csv=args.csv,
        output_csv=args.output_csv,
        dino_model=model,
        n_layers=n_layers,
        overlap=args.overlap,
        threshold=args.threshold,
        aggregation=args.aggregation,
        use_hann=use_hann,
        device=str(device),
        sim_upsample=args.sim_upsample,
        lmds_kernel=args.lmds_kernel,
        lmds_adapt_ts=args.lmds_adapt_ts,
        lmds_neg_ts=args.lmds_neg_ts,
        save_maps=args.save_maps,
        maps_dir=args.maps_dir,
        n_viz=args.n_viz,
        viz_seed=args.viz_seed,
    )
