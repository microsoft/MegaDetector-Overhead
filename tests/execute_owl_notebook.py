"""Execute the real OWL notebook and save its output under ignored demo_data/."""

import argparse
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def validate_demo_outputs(notebook, full: bool) -> None:
    cells = {cell.id: cell for cell in notebook.cells}
    for cell_id in ("environment-check", "prepare-samples", "infer-patches"):
        streams = "".join(output.get("text", "") for output in cells[cell_id].get("outputs", []))
        marker = {
            "environment-check": '"interpreter":',
            "prepare-samples": "Verified 6 images.",
            "infer-patches": "Finished 4 images;",
        }[cell_id]
        if marker not in streams:
            raise AssertionError(f"Expected demo print {marker!r} missing from {cell_id}")
    stages = ["show-patches"]
    if full:
        streams = "".join(output.get("text", "") for output in cells["infer-full"].get("outputs", []))
        if "Finished 2 images;" not in streams:
            raise AssertionError("Optional full-resolution stage did not finish both images")
        stages.append("show-full")
    for cell_id in stages:
        if not any("image/png" in output.get("data", {}) for output in cells[cell_id].get("outputs", [])):
            raise AssertionError(f"No inline image output in {cell_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("owl-d", "owl-c"), default="owl-d")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--model-cache", type=Path, help="Optional isolated cache for real download checks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "demo_data") or output.suffix != ".ipynb":
        parser.error("--output must be an .ipynb file under the repository's ignored demo_data/")
    if output.exists():
        parser.error("--output already exists; choose a new artifact name")

    notebook = nbformat.read(ROOT / "notebooks/owl_inference_demo.ipynb", as_version=4)
    nbformat.validate(notebook)
    configs = [cell for cell in notebook.cells if cell.id == "configuration"]
    if len(configs) != 1:
        raise ValueError("Notebook must have exactly one configuration cell")
    config = configs[0]
    for name, value in (
        ("MODEL", repr(args.model)), ("DEVICE", repr(args.device)),
        ("RUN_FULL_RESOLUTION", repr(args.full)),
        ("MODEL_CACHE", repr(str(args.model_cache.resolve())) if args.model_cache is not None else "None"),
    ):
        config.source, count = re.subn(rf"(?m)^{name}\s*=.*$", f"{name} = {value}", config.source)
        if count != 1:
            raise ValueError(f"Expected one {name} assignment in the configuration cell")

    def cell_start(cell, cell_index, **kwargs):
        print(f"Cell {cell_index}: {cell.id}", flush=True)

    client = NotebookClient(
        notebook, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=cell_start,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
        validate_demo_outputs(notebook, args.full)
    finally:
        nbformat.write(notebook, output)
    print(f"Executed notebook: {output}", flush=True)


if __name__ == "__main__":
    main()
