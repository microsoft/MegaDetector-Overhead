# MegaDetector-Overhead — common commands.
#
# Install (pick one):
#   make sync        CPU build of PyTorch (default; works anywhere)
#   make sync-gpu    GPU build (CUDA 12.1 / torch 2.5.1+cu121; supports
#                    NVIDIA Volta sm_70 e.g. Tesla V100 through Hopper sm_90)
#
# After installing, run things through the venv (no per-command flags):
#   source .venv/bin/activate
#   python tools/test.py ...
# ...or use `make smoke`, `make demo`, etc. below.

UV ?= uv

.PHONY: help sync sync-gpu smoke demo demo-gpu docs clean-demo

help:
	@echo "Targets:"
	@echo "  make sync       Install the CPU PyTorch build (uv sync)"
	@echo "  make sync-gpu   Install the GPU build (uv sync --no-default-groups --group gpu)"
	@echo "  make smoke      Import check (animaloc + dinov3 + torch device)"
	@echo "  make demo       Caribou OWL-C demo (tools/demo_caribou.sh)"
	@echo "  make demo-gpu   Multi-model comparison on GPU (tools/demo_owl_models.sh)"
	@echo "  make docs       Build the MkDocs site (uv run --group docs mkdocs build)"

# ── Environment ──────────────────────────────────────────────────────────────
sync:
	$(UV) sync

sync-gpu:
	$(UV) sync --no-default-groups --group gpu

# ── Convenience ──────────────────────────────────────────────────────────────
# These run through the project venv interpreter directly, so they use whichever
# build was synced (CPU or GPU) without reverting it.
PYBIN = .venv/bin/python

smoke:
	$(PYBIN) -c "import animaloc.models, dinov3, torch; \
		print('OK |', torch.__version__, '| CUDA:', torch.cuda.is_available())"

demo:
	./tools/demo_caribou.sh

demo-gpu:
	./tools/demo_owl_models.sh --device cuda

docs:
	$(UV) run --extra docs mkdocs build --strict

clean-demo:
	rm -rf demo_data/run* demo_data/viz* demo_data/model_comparison.csv
