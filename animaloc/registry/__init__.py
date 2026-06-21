"""Deployment-defaults registries for animaloc.

Reusable lookup tables that tell client code (CLI tools, notebooks)
which Stitcher, Evaluator, model_kwargs, normalization stats, etc. to
use for each registered model. The model classes themselves do NOT read
this — it's strictly a deployment / tooling concern. Keeping it out of
animaloc.models prevents accidental coupling between model code and
eval components.

Consumers:
    from animaloc.registry.families import FAMILIES, resolve_family
"""

from .families import FAMILIES, ModelFamily, resolve_family

__all__ = ["FAMILIES", "ModelFamily", "resolve_family"]
