"""Overlay point detections (and optional ground truth) onto image patches.

This standalone tool fills a gap in the repo: ``tools/test.py`` only logs
visualizations to Weights & Biases, and the overlay code in ``tools/infer.py``
is commented out and hardcoded to the 7-class HerdNet. This script reads the
``detections.csv`` produced by ``tools/test.py`` (columns ``images, x, y`` and
a score column ``dscores`` or ``scores``) and writes annotated PNGs to disk.

Example
-------
    uv run python tools/visualize_detections.py \
        --detections outputs/<date>/<time>/detections.csv \
        --images-dir demo_data/subset \
        --output-dir demo_data/viz \
        --gt demo_data/subset/gt.csv \
        --score-threshold 0.2

Predicted points are drawn in red, ground-truth points (if ``--gt`` is given)
in green. A small legend/count caption is added to each patch.
"""

import argparse
import os
import sys

import pandas
import PIL
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animaloc.vizual import draw_points, draw_text  # noqa: E402

PIL.Image.MAX_IMAGE_PIXELS = None

_IMG_EXTS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".TIF", ".tiff", ".TIFF")


def _score_column(df: pandas.DataFrame) -> str | None:
    """Return the name of the confidence column, if present."""
    for col in ("dscores", "scores", "score"):
        if col in df.columns:
            return col
    return None


def _points_for(df: pandas.DataFrame, image: str) -> list:
    """Return detection points for one image as a list of (y, x) tuples.

    ``animaloc.vizual.draw_points`` expects points in (y, x) order.
    """
    rows = df[df["images"] == image]
    return [(float(r.y), float(r.x)) for r in rows.itertuples(index=False)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visualize_detections",
        description="Overlay point detections (and optional GT) on image patches.",
    )
    parser.add_argument("--detections", required=True, type=str,
                        help="Path to detections.csv produced by tools/test.py.")
    parser.add_argument("--images-dir", required=True, type=str,
                        help="Directory containing the patch images.")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Directory where annotated images are written.")
    parser.add_argument("--gt", type=str, default=None,
                        help="Optional ground-truth CSV (columns images,x,y) to also overlay.")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Keep only detections with score >= this value. Defaults to 0.0.")
    parser.add_argument("--size", type=int, default=8,
                        help="Diameter (px) of plotted points. Defaults to 8.")
    parser.add_argument("--pred-color", type=str, default="red",
                        help="Color for predicted points. Defaults to 'red'.")
    parser.add_argument("--gt-color", type=str, default="lime",
                        help="Color for ground-truth points. Defaults to 'lime'.")
    parser.add_argument("--all-images", action="store_true",
                        help="Also render images that have no detections "
                             "(e.g. background patches). By default only images "
                             "present in the detections CSV are rendered.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    det = pandas.read_csv(args.detections)
    if "images" not in det.columns:
        raise SystemExit(f"'images' column not found in {args.detections}")
    det["images"] = det["images"].apply(os.path.basename)

    score_col = _score_column(det)
    if score_col is not None and args.score_threshold > 0.0:
        det = det[det[score_col] >= args.score_threshold]

    gt = None
    if args.gt is not None:
        gt = pandas.read_csv(args.gt)
        gt["images"] = gt["images"].apply(os.path.basename)

    # Decide which images to render.
    if args.all_images:
        images = sorted(f for f in os.listdir(args.images_dir) if f.endswith(_IMG_EXTS))
    else:
        images = sorted(det["images"].unique().tolist())

    os.makedirs(args.output_dir, exist_ok=True)

    n_done = 0
    total_pred = 0
    for image in images:
        img_path = os.path.join(args.images_dir, image)
        if not os.path.isfile(img_path):
            print(f"  skip (missing): {image}")
            continue

        canvas = Image.open(img_path).convert("RGB")

        n_gt = 0
        if gt is not None:
            gt_pts = _points_for(gt, image)
            n_gt = len(gt_pts)
            canvas = draw_points(canvas, gt_pts, color=args.gt_color, size=args.size)

        pred_pts = _points_for(det, image)
        n_pred = len(pred_pts)
        total_pred += n_pred
        canvas = draw_points(canvas, pred_pts, color=args.pred_color, size=args.size)

        caption = f"pred={n_pred}"
        if gt is not None:
            caption += f"  gt={n_gt}"
        canvas = draw_text(canvas, caption, position=(6, 6), font_size=16)

        canvas.save(os.path.join(args.output_dir, image))
        n_done += 1

    print(f"Wrote {n_done} annotated image(s) to {args.output_dir} "
          f"({total_pred} predicted points total).")


if __name__ == "__main__":
    main()
