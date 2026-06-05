"""Create a synthetic point-annotated mini-dataset under /tmp/owl-smoketest/.

Used by the OWL training and evaluation smoke tests. The dataset has 4
train images + 2 val images, each 512x512 with 1-3 high-contrast
circular "animal" blobs whose centers are written to gt.csv in the
animaloc CSVDataset point format (columns: images, x, y, labels).

Run:
  uv run python tests/make_synthetic_dataset.py

Output:
  /tmp/owl-smoketest/
    train/  img_001.jpg ... img_004.jpg  gt.csv
    val/    img_005.jpg, img_006.jpg     gt.csv
"""

import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/tmp/owl-smoketest")
IMG_SIZE = 512
N_TRAIN = 4
N_VAL = 2
SEED = 42


def make_image(rng: random.Random) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    """Returns (image, [(x, y, label)...]) for one synthetic frame."""
    bg = (rng.randint(60, 140), rng.randint(60, 140), rng.randint(60, 140))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    # Background texture: a few darker rectangles for "ground".
    for _ in range(rng.randint(3, 6)):
        x0, y0 = rng.randint(0, IMG_SIZE - 64), rng.randint(0, IMG_SIZE - 64)
        x1 = x0 + rng.randint(32, 100)
        y1 = y0 + rng.randint(32, 100)
        shade = tuple(max(0, c - rng.randint(20, 50)) for c in bg)
        draw.rectangle([x0, y0, x1, y1], fill=shade)

    # 1-3 "animal" blobs per image, 40 px from the image border.
    points = []
    for _ in range(rng.randint(1, 3)):
        cx = rng.randint(40, IMG_SIZE - 40)
        cy = rng.randint(40, IMG_SIZE - 40)
        r = rng.randint(8, 14)
        blob_color = (rng.randint(200, 255), rng.randint(200, 255), rng.randint(50, 150))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=blob_color, outline=(0, 0, 0))
        points.append((cx, cy, 1))

    return img, points


def write_split(split: str, n: int, start_idx: int, rng: random.Random) -> None:
    split_dir = ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = [("images", "x", "y", "labels")]
    for i in range(n):
        idx = start_idx + i
        name = f"img_{idx:03d}.jpg"
        img, pts = make_image(rng)
        img.save(split_dir / name, quality=85)
        for x, y, lbl in pts:
            rows.append((name, x, y, lbl))

    with (split_dir / "gt.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    rng = random.Random(SEED)
    write_split("train", N_TRAIN, 1, rng)
    write_split("val", N_VAL, N_TRAIN + 1, rng)
    print(f"Wrote synthetic dataset to {ROOT}")
    print(f"  train: {N_TRAIN} images, gt.csv")
    print(f"  val:   {N_VAL} images, gt.csv")

    for split in ("train", "val"):
        csv_path = ROOT / split / "gt.csv"
        with csv_path.open() as f:
            lines = f.readlines()
        print(f"\n=== {csv_path} ({len(lines) - 1} annotations) ===")
        for line in lines[:6]:
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
