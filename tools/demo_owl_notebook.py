"""Backend stages for notebooks/owl_inference_demo.ipynb."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animaloc.utils.owl_demo import (  # noqa: E402
    DEMO_ERRORS, ROOT, environment_info, fetch_model_checkpoint, prepare_samples,
    run_inference, stage_release_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    env = stages.add_parser("environment", help="Check the chosen model and runtime")
    prepare = stages.add_parser("prepare", help="Verify and safely extract the six-image sample")
    stage = stages.add_parser("stage-release", help="Stage exact ZIP and attribution notes; never upload")
    fetch = stages.add_parser("fetch-model", help="Download/verify weights without constructing a model")
    infer = stages.add_parser("infer", help="Run annotation-free OWL prediction")
    for command in (env, infer, fetch):
        command.add_argument("--model", choices=("owl-d", "owl-c"), default="owl-d")
    for command in (env, infer):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for command in (infer, fetch):
        command.add_argument("--model-cache", type=Path, help="Isolated model cache; default demo_data/models")
    stage.add_argument("--archive", type=Path)
    stage.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--data-dir", type=Path, default=ROOT / "demo_data/owl_notebook")
    prepare.add_argument("--archive", type=Path)
    prepare.add_argument("--sample-url")
    infer.add_argument("--images-dir", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--selection", choices=("patches", "full", "all"), default="patches")
    infer.add_argument("--checkpoint", type=Path)
    infer.add_argument("--tile-size", type=int, default=512)
    infer.add_argument("--overlap", type=int, default=160)
    infer.add_argument("--score-threshold", type=float, default=0.2)
    infer.add_argument("--adapt-threshold", type=float, default=0.3)
    infer.add_argument("--negative-threshold", type=float, default=0.1)
    infer.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.stage == "environment":
            environment_info(args.model, args.device)
        elif args.stage == "prepare":
            prepare_samples(args.data_dir, args.archive, args.sample_url)
        elif args.stage == "stage-release":
            stage_release_assets(args.output_dir, args.archive)
        elif args.stage == "fetch-model":
            path = fetch_model_checkpoint(args.model, args.model_cache)
            print(f"Checkpoint verified: {path}")
        else:
            run_inference(
                args.images_dir, args.output_dir, args.model, args.device, args.selection,
                args.checkpoint, args.score_threshold, args.adapt_threshold, args.negative_threshold,
                args.tile_size, args.overlap, args.cpu_threads, args.model_cache,
            )
    except DEMO_ERRORS as exc:
        parser.exit(1, f"OWL demo failed: {exc}\n")


if __name__ == "__main__":
    main()
