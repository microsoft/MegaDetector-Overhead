"""Run inference with any registered animaloc model on a folder of images.

This is the generic inference CLI. It works for the legacy multi-class
HerdNet model AND for the single-class OWL family (OWLC, OWLT,
OWLD_S/B/L/H). Add new models to `animaloc/registry/families.py` and
they become usable here without code changes.

## Quickstart

Run the legacy HerdNet model (backwards-compatible default):

    python tools/infer.py /path/to/images /path/to/herdnet.pth

Run an OWL-C model:

    python tools/infer.py /path/to/images /path/to/owlc.pth --model OWLC

Run an OWL-D-L model and write results elsewhere:

    python tools/infer.py /path/to/images /path/to/owld_l.pth \\
        --model OWLD_L --output-dir /tmp/owld_l_results --device cpu

Override a single model constructor kwarg:

    python tools/infer.py imgs/ ckpt.pth --model OWLT \\
        --model-kwarg down_ratio=4

See `animaloc/registry/families.py` for the supported model families
and their default kwargs/stitcher/evaluator/normalization.

## Outputs

A timestamped folder under `--output-dir` (or under `<images>/` by
default) containing:

    <date>_<model>_results/
        <date>_detections.csv   columns: images, x, y, labels, scores, [species]

The `species` column is included only when the checkpoint stores a
`classes` mapping (saved automatically by `tools/train.py`).

## Vendored from HerdNet

This script started as a copy of HerdNet's `tools/infer.py` (MIT,
Universite de Liege) and was rewritten in this repo to be model-
agnostic. See git log for the rewrite commit.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Optional

import albumentations as A
import pandas
import PIL
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import animaloc.eval.evaluators as evaluators_mod
import animaloc.eval.stitchers as stitchers_mod
import animaloc.models as models_mod
from animaloc.data.transforms import DownSample, Rotate90
from animaloc.datasets import CSVDataset
from animaloc.eval.metrics import PointsMetrics
from animaloc.models.utils import LossWrapper
from animaloc.registry.families import FAMILIES, resolve_family
from animaloc.utils.useful_funcs import current_date, mkdir
from torch.utils.data import DataLoader, SequentialSampler

warnings.filterwarnings("ignore")
PIL.Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

# Recognized string forms for boolean CLI values. The Python `bool()`
# builtin returns True for any non-empty string ("False" included), so we
# parse explicitly.
_TRUE_STRS = {"true", "1", "yes", "on", "t"}
_FALSE_STRS = {"false", "0", "no", "off", "f"}


def _parse_kv_value(raw: str) -> Any:
    """Coerce a CLI key=value string into int/float/bool/str.

    Order: int -> float -> bool (explicit string set) -> str.
    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    low = raw.lower()
    if low in _TRUE_STRS:
        return True
    if low in _FALSE_STRS:
        return False
    return raw


def _parse_kv_pair(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--model-kwarg expects key=value, got {s!r}"
        )
    k, v = s.split("=", 1)
    return k.strip(), _parse_kv_value(v.strip())


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def _parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="infer",
        description=(
            "Run a pretrained animaloc model (HerdNet or any OWL variant) "
            "on a folder of images and write the resulting detections to a "
            "CSV. Defaults to HerdNet for backwards compatibility."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Positional (unchanged).
    p.add_argument("root", type=str, help="path to the folder of input images")
    p.add_argument("pth", type=str, help="path to the .pth checkpoint")

    # Model selection.
    p.add_argument(
        "--model",
        type=str,
        default="HerdNet",
        choices=sorted(FAMILIES.keys()),
        help="registered model name (default: HerdNet, backwards-compat)",
    )
    p.add_argument(
        "--model-kwarg",
        action="append",
        type=_parse_kv_pair,
        default=[],
        metavar="KEY=VAL",
        help=(
            "override a single model constructor kwarg (repeatable). "
            "Coerces to int/float/bool/str. Use --model-kwarg key=value."
        ),
    )
    p.add_argument(
        "--stitcher",
        type=str,
        default=None,
        help="override the family-default stitcher class name",
    )
    p.add_argument(
        "--evaluator",
        type=str,
        default=None,
        help="override the family-default evaluator class name",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "explicitly set num_classes (only needed for legacy HerdNet "
            "checkpoints whose 'classes' metadata is missing AND whose "
            "head shape probe fails)"
        ),
    )

    # Normalization + geometry overrides.
    p.add_argument(
        "--mean", type=_parse_csv_floats, default=None,
        metavar="R,G,B", help="image normalization mean (override checkpoint/family)",
    )
    p.add_argument(
        "--std", type=_parse_csv_floats, default=None,
        metavar="R,G,B", help="image normalization std (override checkpoint/family)",
    )
    p.add_argument(
        "--down-ratio", type=int, default=None,
        help="downsample ratio (override family default)",
    )

    # LMDS post-processing knobs (model-agnostic).
    p.add_argument(
        "--lmds-kernel-size", type=_parse_csv_ints, default=(3, 3),
        metavar="H,W", help="LMDS kernel size (default 3,3)",
    )
    p.add_argument(
        "--lmds-adapt-ts", type=float, default=0.2,
        help="LMDS adaptive threshold (default 0.2)",
    )
    p.add_argument(
        "--lmds-neg-ts", type=float, default=0.1,
        help="LMDS negative threshold (HerdNet family only)",
    )

    # Output.
    p.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "where to write results (default: <root>/<date>_<model>_results). "
            "Useful when <root> is read-only or shared."
        ),
    )

    # Stitcher geometry + runtime knobs (kept from the original CLI).
    p.add_argument("-size", type=int, default=512, help="patch size for stitching")
    p.add_argument("-over", type=int, default=160, help="overlap for stitching")
    p.add_argument(
        "-device", type=str, default="cuda",
        help="'cpu' or 'cuda' (default cuda)",
    )
    p.add_argument("-pf", type=int, default=10, help="print frequency")
    p.add_argument(
        "-rot", type=int, default=0,
        help="number of 90-degree CCW rotations to apply",
    )
    p.add_argument(
        "--skip-model-inference",
        action="store_true",
        help="skip the inference step (debug-only; preserved from upstream)",
    )
    return p


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _load_checkpoint(pth_path: str, map_location: torch.device) -> dict:
    return torch.load(pth_path, map_location=map_location)


def _probe_herdnet_num_classes(state_dict: dict) -> Optional[int]:
    """Last-resort fallback: read the HerdNet classification head shape.

    Returns None if the layer is not present (i.e. not a HerdNet checkpoint).
    """
    key = "model.cls_head.2.weight"
    if key not in state_dict:
        return None
    return int(state_dict[key].shape[0])


def _resolve_num_classes(
    family_name: str,
    args: argparse.Namespace,
    checkpoint_meta: dict,
    state_dict: dict,
) -> Optional[int]:
    """Decide what num_classes to pass to the model constructor.

    For single-class OWL models: returns None (the constructor does
    not accept num_classes).

    For HerdNet (multi_class=True): tries in order:
        1. --num-classes CLI override
        2. len(checkpoint_meta['classes']) + 1 (binary + per-species)
        3. layer-shape probe on `model.cls_head.2.weight`
        4. raises with an actionable message
    """
    if not FAMILIES[family_name].multi_class:
        return None

    if args.num_classes is not None:
        return args.num_classes

    if "classes" in checkpoint_meta and checkpoint_meta["classes"]:
        # +1 for background class 0
        return len(checkpoint_meta["classes"]) + 1

    probed = _probe_herdnet_num_classes(state_dict)
    if probed is not None:
        return probed

    raise RuntimeError(
        f"Cannot determine num_classes for {family_name!r} model: the checkpoint "
        "has no 'classes' metadata and the head-shape probe failed. "
        "Pass --num-classes N explicitly."
    )


def _build_model(
    family_name: str,
    resolved: dict,
    num_classes: Optional[int],
) -> torch.nn.Module:
    """Instantiate the model class with resolved kwargs."""
    if family_name not in models_mod.__dict__:
        known = sorted(models_mod.MODELS.registry_names)
        raise KeyError(
            f"Model class {family_name!r} not found in animaloc.models. "
            f"Known registered models: {known}"
        )
    cls = models_mod.__dict__[family_name]
    kwargs = dict(resolved["model_kwargs"])
    if num_classes is not None:
        kwargs.setdefault("num_classes", num_classes)
    return cls(**kwargs)


def _build_stitcher(
    name: str,
    model: torch.nn.Module,
    size: int,
    overlap: int,
    down_ratio: int,
    device: torch.device,
):
    if name not in stitchers_mod.__dict__:
        raise KeyError(
            f"Stitcher class {name!r} not found in animaloc.eval.stitchers. "
            "Check FAMILIES or pass --stitcher explicitly."
        )
    cls = stitchers_mod.__dict__[name]
    return cls(
        model=model,
        size=(size, size),
        overlap=overlap,
        down_ratio=down_ratio,
        up=True,
        reduction="mean",
        device_name=device,
    )


def _build_evaluator(
    name: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    metrics: PointsMetrics,
    stitcher,
    device: torch.device,
    print_freq: int,
    work_dir: str,
    lmds_kwargs: dict,
):
    if name not in evaluators_mod.__dict__:
        raise KeyError(
            f"Evaluator class {name!r} not found in animaloc.eval.evaluators. "
            "Check FAMILIES or pass --evaluator explicitly."
        )
    cls = evaluators_mod.__dict__[name]
    return cls(
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        lmds_kwargs=lmds_kwargs,
        device_name=device,
        print_freq=print_freq,
        stitcher=stitcher,
        work_dir=work_dir,
        header="[INFERENCE]",
    )


def _validate_model_stitcher_shape(
    model: torch.nn.Module, family_name: str, size: int, device: torch.device
) -> None:
    """One-shot dummy forward to catch model/stitcher mismatches early.

    Cheap (one forward on a 3xSxS tensor on the target device) and
    catches the common 'used --model OWLC with --stitcher HerdNetStitcher'
    error class with a clear message instead of a deep-stack tuple-unpacking
    error later.

    The model is always wrapped in LossWrapper before this is called.
    LossWrapper.forward returns `(real_output, output_dict)` in eval
    mode, so we unwrap one level before inspecting shape.
    """
    family = FAMILIES[family_name]
    try:
        with torch.no_grad():
            model.eval()
            wrapped_out = model(torch.zeros(1, 3, size, size, device=device))
    except Exception as e:
        raise RuntimeError(
            f"Dummy forward failed for {family_name!r} with size={size}: "
            f"{type(e).__name__}: {e}"
        ) from e

    # Unwrap LossWrapper's (output, loss_dict) tuple to get the real
    # model output. Without this, every model looks like a 2-tuple.
    if isinstance(wrapped_out, tuple) and len(wrapped_out) == 2 and isinstance(wrapped_out[1], dict):
        real_out = wrapped_out[0]
    else:
        real_out = wrapped_out

    # Some OWL-family models (e.g. OWLD_*) return `(heatmap, None)` to
    # match an optional secondary head signature. Filter Nones so the
    # shape check sees only real tensors.
    if isinstance(real_out, (tuple, list)):
        real_tensors = [x for x in real_out if x is not None]
        n_outputs = len(real_tensors)
    else:
        n_outputs = 1

    if family.multi_class and n_outputs < 2:
        raise RuntimeError(
            f"{family_name!r} family expects a multi-output model (heatmap+classmap), "
            f"but the model returned {n_outputs} tensor(s). "
            f"Check the family definition or override --stitcher / --evaluator."
        )
    if not family.multi_class and n_outputs > 1:
        print(
            f"[WARN] {family_name!r} family expects a single-output model but "
            f"forward returned {n_outputs} outputs. Continuing — verify the result.",
            file=sys.stderr,
        )


def _make_inference_dataset(
    image_dir: str, mean: list[float], std: list[float], down_ratio: int, rot: int
) -> CSVDataset:
    """Build a CSVDataset of dummy (x=0, y=0, label=1) entries -- one per
    image in `image_dir`. The Evaluator path needs a dataloader of
    (image, target) pairs; ground truth values are discarded.
    """
    possible_extensions = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    img_names = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(possible_extensions)
    )
    if not img_names:
        raise FileNotFoundError(
            f"No images with extensions {possible_extensions} found in {image_dir!r}"
        )

    n = len(img_names)
    df = pandas.DataFrame(
        data={"images": img_names, "x": [0] * n, "y": [0] * n, "labels": [1] * n}
    )

    end_transforms = []
    if rot != 0:
        end_transforms.append(Rotate90(k=rot))
    end_transforms.append(DownSample(down_ratio=down_ratio, anno_type="point"))

    albu_transforms = [A.Normalize(mean=mean, std=std)]

    return CSVDataset(
        csv_file=df,
        root_dir=image_dir,
        albu_transforms=albu_transforms,
        end_transforms=end_transforms,
    )


def _attach_species_column(detections: pandas.DataFrame, classes: dict) -> pandas.DataFrame:
    """Add a 'species' column mapped from labels; emit raw label on miss.

    NOTE: The previous version of this code did
        df.dropna(inplace=True)
    after the .map(), which silently deleted every detection whose label
    was not in the mapping. That hid both real bugs (wrong classes dict)
    and minor mismatches. We now keep every row and warn about coverage
    gaps instead.
    """
    if not classes:
        return detections
    species = detections["labels"].map(classes)
    unmapped_labels = detections.loc[species.isna(), "labels"].unique().tolist()
    if unmapped_labels:
        print(
            f"[WARN] {len(unmapped_labels)} detection label(s) had no entry in "
            f"the classes mapping: {sorted(unmapped_labels)}. "
            "Keeping raw label; please update the checkpoint's `classes` metadata.",
            file=sys.stderr,
        )
        species = species.fillna(detections["labels"].astype(str))
    detections = detections.copy()
    detections["species"] = species
    return detections


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Map_location for torch.load: fall back to CPU if --device cuda but
    # no CUDA available, so checkpoints load without crashing.
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda requested but no CUDA available; using cpu.", file=sys.stderr)
        args.device = "cpu"
    device = torch.device(args.device)
    map_location = device

    if args.skip_model_inference:
        print("Skipping model inference (debug mode)")
        return 0

    # ---- Resolve family + checkpoint metadata ----
    print(f"Loading checkpoint: {args.pth}")
    checkpoint = _load_checkpoint(args.pth, map_location)

    overrides = dict(
        mean=args.mean,
        std=args.std,
        down_ratio=args.down_ratio,
        stitcher=args.stitcher,
        evaluator=args.evaluator,
        model_kwargs=dict(args.model_kwarg) if args.model_kwarg else None,
    )
    resolved = resolve_family(args.model, checkpoint_meta=checkpoint, overrides=overrides)
    print(f"Resolved family for --model {args.model}:")
    print(f"  stitcher    = {resolved['stitcher']}")
    print(f"  evaluator   = {resolved['evaluator']}")
    print(f"  down_ratio  = {resolved['down_ratio']}")
    print(f"  multi_class = {resolved['multi_class']}")

    # ---- num_classes resolution (HerdNet only) ----
    state_dict = checkpoint["model_state_dict"]
    num_classes = _resolve_num_classes(args.model, args, checkpoint, state_dict)
    if num_classes is not None:
        print(f"  num_classes = {num_classes}")

    # ---- Build + load model ----
    print(f"Building model {args.model}")
    model = _build_model(args.model, resolved, num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters  = {n_params:,}")

    model = LossWrapper(model, [])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] {len(missing)} missing key(s) in state_dict (first 5): {missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected key(s) in state_dict (first 5): {unexpected[:5]}", file=sys.stderr)
    model = model.to(device)

    # ---- Sanity check ----
    _validate_model_stitcher_shape(model, args.model, args.size, device)

    # ---- Output dir ----
    curr_date = current_date()
    output_dir = args.output_dir or os.path.join(
        args.root, f"{curr_date}_{args.model}_results"
    )
    mkdir(output_dir)
    print(f"Output dir: {output_dir}")

    # ---- Dataset / dataloader ----
    print(f"Listing images under {args.root}")
    dataset = _make_inference_dataset(
        image_dir=args.root,
        mean=resolved["mean"],
        std=resolved["std"],
        down_ratio=resolved["down_ratio"],
        rot=args.rot,
    )
    print(f"  found {len(dataset)} image(s)")

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        sampler=SequentialSampler(dataset),
    )

    # ---- Stitcher + Evaluator ----
    stitcher = _build_stitcher(
        resolved["stitcher"],
        model=model,
        size=args.size,
        overlap=args.over,
        down_ratio=resolved["down_ratio"],
        device=device,
    )

    # PointsMetrics needs a num_classes. For single-class OWL we use 2
    # (background + animal); for HerdNet, the real num_classes.
    metrics_num_classes = num_classes if num_classes is not None else 2
    metrics = PointsMetrics(20, num_classes=metrics_num_classes)

    # LMDS kwargs: HerdNet family uses neg_ts; Detection_Branch family does not.
    lmds_kwargs: dict[str, Any] = dict(
        kernel_size=tuple(args.lmds_kernel_size),
        adapt_ts=args.lmds_adapt_ts,
    )
    if FAMILIES[args.model].multi_class:
        lmds_kwargs["neg_ts"] = args.lmds_neg_ts

    evaluator = _build_evaluator(
        resolved["evaluator"],
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        stitcher=stitcher,
        device=device,
        print_freq=args.pf,
        work_dir=output_dir,
        lmds_kwargs=lmds_kwargs,
    )

    # ---- Run inference ----
    # We use the Evaluator pipeline (it already implements stitching +
    # LMDS post-processing) but pass dummy ground truth so the computed
    # metrics are meaningless and discarded. A future PR can factor out
    # a pure inference path that does not go through Evaluator at all.
    print(f"Running inference on {len(dataset)} image(s) ...")
    evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)

    # ---- Save detections ----
    print("Saving detections ...")
    detections = evaluator.detections
    classes_meta = resolved.get("classes") or {}
    detections = _attach_species_column(detections, classes_meta)

    out_csv = os.path.join(output_dir, f"{curr_date}_detections.csv")
    detections.to_csv(out_csv, index=False)
    print(f"Wrote {len(detections)} detection(s) to {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
