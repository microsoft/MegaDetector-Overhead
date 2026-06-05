__copyright__ = """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

import sys
import os

# Allow `python tools/train.py` to be run from the repo root by adding the
# repo root to sys.path so `animaloc` resolves even without `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# 2026-04-25: /dev/shm on this host is 64 MiB (container default). The default
# DataLoader worker IPC ('file_descriptor' via /dev/shm) blows up with
# "RuntimeError: DataLoader worker (pid ...) exited unexpectedly" / bus error
# once num_workers>0 and tensors flow between workers and main. Switching to
# 'file_system' sharing routes IPC through file descriptors on the regular
# filesystem instead. Slower per-handoff but unbounded by /dev/shm size.
torch.multiprocessing.set_sharing_strategy("file_system")
import hydra
import animaloc
import wandb
import pandas
import os
import torchvision

import albumentations as A

from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from typing import Callable, Optional

from animaloc.models.utils import LossWrapper, load_model
from animaloc.eval import (
    Evaluator,
    PointsMetrics,
    Stitcher,
    BoxesMetrics,
    ImageLevelMetrics,
)

from animaloc.utils.seed import set_seed
from animaloc.utils.useful_funcs import current_date


def _set_species_labels(cls_dict: dict, df: pandas.DataFrame) -> None:
    assert "species" in df.columns
    cls_dict = dict(map(reversed, cls_dict.items()))
    df["labels"] = df["species"].map(cls_dict)


def _load_albu_transforms(tr_cfg: dict) -> list:
    # 2026-04-25: OmegaConf wraps YAML lists as ListConfig and YAML dicts as
    # DictConfig. Some albumentations transforms (e.g., GaussNoise.var_limit)
    # do strict isinstance checks against (int, float, tuple, list) and reject
    # ListConfig with a TypeError. Coerce to native Python so all transforms
    # accept their kwargs.
    from omegaconf import OmegaConf as _OC

    transforms = []
    for name, kwargs in tr_cfg.items():
        if _OC.is_config(kwargs):
            kwargs = _OC.to_container(kwargs, resolve=True)
        transforms.append(A.__dict__[name](**kwargs))

    return transforms


def _load_end_transforms(tr_cfg: DictConfig) -> Optional[list]:
    if tr_cfg is not None:
        transforms = []
        for name, kwargs in tr_cfg.items():
            if name == "MultiTransformsWrapper":
                tr_list = []
                for n, k in kwargs.items():
                    tr_list.append(animaloc.data.transforms.__dict__[n](**k))

                transforms.append(animaloc.data.transforms.__dict__[name](tr_list))

            else:
                transforms.append(animaloc.data.transforms.__dict__[name](**kwargs))

        return transforms

    else:
        return None


def _build_sampler(sampler_cfg: DictConfig, dl_kwargs: dict, dataset: Dataset) -> dict:
    dl_kwargs = dl_kwargs.copy()

    sampler = animaloc.data.samplers.__dict__[sampler_cfg.name]
    if sampler_cfg.data_source == "dataset":
        sampler = sampler(dataset, **dict(sampler_cfg.kwargs))
    else:
        raise NotImplementedError

    if sampler_cfg.batch:
        dl_kwargs.update(dict(batch_size=1, shuffle=False, batch_sampler=sampler))
    else:
        dl_kwargs.update(dict(shuffle=False, sampler=sampler))

    return dl_kwargs


def _get_collate_fn(cfg: DictConfig) -> Callable:
    fn = cfg.datasets.collate_fn
    if fn is not None:
        fn = animaloc.data.batch_utils.__dict__[fn]
    return fn


def _build_model(cfg: DictConfig) -> torch.nn.Module:

    name = cfg.model.name
    from_torchvision = cfg.model.from_torchvision

    if from_torchvision:
        assert name in torchvision.models.__dict__.keys(), (
            f"'{name}' unfound in torchvision's models"
        )

        model = torchvision.models.__dict__[name]

    else:
        assert name in animaloc.models.__dict__.keys(), (
            f"'{name}' class unfound, make sure you have included the class in the models list"
        )

        model = animaloc.models.__dict__[name]

    kwargs = dict(cfg.model.kwargs)
    for k in ["num_classes"]:
        kwargs.pop(k, None)

    num_classes = getattr(cfg.datasets, "num_classes", None)

    # Check if the model constructor accepts num_classes
    import inspect

    sig = inspect.signature(model)
    accepts_num_classes = "num_classes" in sig.parameters

    if accepts_num_classes:
        assert num_classes is not None, (
            f"'num_classes' should be set for {name} model, got {num_classes}"
        )
        model = model(**kwargs, num_classes=num_classes)
    else:
        model = model(**kwargs)

    return model


def _load_losses(cfg: DictConfig) -> tuple:
    criterions = []
    if cfg.losses is not None:
        for loss, args in cfg.losses.items():
            kwargs = {}
            if "kwargs" in args.keys():
                kwargs = dict(args.kwargs)

                if "weights" in kwargs.keys():
                    kwargs["weights"] = torch.Tensor(kwargs["weights"])
                elif "weight" in kwargs.keys():
                    kwargs["weight"] = torch.Tensor(kwargs["weight"]).to(
                        torch.device(cfg.device_name)
                    )

            crit_dict = {}
            if args.from_torch:
                crit_dict.update({"loss": torch.nn.__dict__[loss](**kwargs)})
            else:
                crit_dict.update(
                    {"loss": animaloc.train.losses.__dict__[loss](**kwargs)}
                )

            crit_dict.update(
                {
                    "idx": args.output_idx,
                    "idy": args.target_idx,
                    "lambda": args.lambda_const,
                    "name": args.print_name,
                }
            )

            criterions.append(crit_dict)

    return criterions


def _define_stitcher(model: torch.nn.Module, cfg: DictConfig) -> Stitcher:

    kwargs = dict(cfg.training_settings.stitcher.kwargs)
    for k in ["model", "size", "device_name"]:
        kwargs.pop(k, None)

    stitcher = animaloc.eval.stitchers.__dict__[cfg.training_settings.stitcher.name](
        model=model, size=cfg.datasets.img_size, **kwargs, device_name=cfg.device_name
    )

    return stitcher


def _define_evaluator(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, cfg: DictConfig
) -> Evaluator:

    name = cfg.training_settings.evaluator.name
    anno_type = cfg.datasets.anno_type

    assert name in animaloc.eval.evaluators.__dict__.keys(), (
        f"'{name}' class unfound, make sure you have included the class in the evaluators list"
    )

    num_classes = getattr(cfg.datasets, "num_classes", 2)  # default to 2 for evaluator

    if anno_type == "point":
        metrics = PointsMetrics(
            radius=cfg.training_settings.evaluator.threshold, num_classes=num_classes
        )
    elif anno_type == "bbox":
        metrics = BoxesMetrics(
            iou=cfg.training_settings.evaluator.threshold, num_classes=num_classes
        )
    elif anno_type == "image":
        metrics = ImageLevelMetrics(num_classes=num_classes)
    else:
        raise NotImplementedError

    stitcher = None
    if cfg.training_settings.stitcher is not None:
        stitcher = _define_stitcher(model, cfg)

    kwargs = dict(cfg.training_settings.evaluator.kwargs)
    for k in [
        "model",
        "dataloader",
        "metrics",
        "device_name",
        "stitcher",
        "header",
        "vizual_fn",
    ]:
        kwargs.pop(k, None)

    vizual_fn = None
    if cfg.training_settings.vizual_fn is not None:
        vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]

    evaluator = animaloc.eval.evaluators.__dict__[name](
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        device_name=cfg.device_name,
        stitcher=stitcher,
        header="[TEST]",
        vizual_fn=vizual_fn,
        **kwargs,
    )

    return evaluator


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = cfg.train

    # Set the seed
    print(f"Setting the seed to {cfg.seed}")
    set_seed(cfg.seed)

    # Prepare datasets and dataloaders
    print("Building datasets ...")
    device = torch.device(cfg.device_name)

    train_args = cfg.datasets.train
    val_args = cfg.datasets.validate

    train_df = pandas.read_csv(train_args.csv_file)
    # _set_species_labels(dict(cfg.datasets.class_def), train_df)
    if {"x_min", "y_min", "x_max", "y_max"}.issubset(
        train_df.columns
    ) and "x" not in train_df.columns:
        train_df["x"] = ((train_df["x_min"] + train_df["x_max"]) / 2).astype(int)
        train_df["y"] = ((train_df["y_min"] + train_df["y_max"]) / 2).astype(int)

    train_dataset = animaloc.datasets.__dict__[train_args.name](
        csv_file=train_df,
        root_dir=train_args.root_dir,
        albu_transforms=_load_albu_transforms(train_args.albu_transforms),
        end_transforms=_load_end_transforms(train_args.end_transforms),
    )

    # 2026-04-25: num_workers and pin_memory added. The previous DataLoader was
    # constructed without these, defaulting to num_workers=0 (single-threaded
    # I/O) and pin_memory=False, which left the GPU idle ~90% of the time
    # waiting on PNG decode + albumentations on a network-mounted dataset.
    # Configurable via cfg.training_settings.{num_workers,pin_memory}; defaults
    # tuned for the V100 + multi-core training host (4 workers is conservative;
    # bump to 8 for very large datasets).
    _num_workers = cfg.training_settings.get("num_workers", 4)
    _pin_memory = cfg.training_settings.get("pin_memory", True)

    train_dl_kwargs = dict(
        batch_size=cfg.training_settings.batch_size,
        shuffle=True,
        collate_fn=_get_collate_fn(cfg),
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
    )

    if train_args.sampler is not None:
        train_dl_kwargs = _build_sampler(
            train_args.sampler, dl_kwargs=train_dl_kwargs, dataset=train_dataset
        )

    train_dataloader = DataLoader(train_dataset, **train_dl_kwargs)

    val_dataloader = None
    if val_args is not None:
        val_df = pandas.read_csv(val_args.csv_file)
        # _set_species_labels(dict(cfg.datasets.class_def), val_df)
        if {"x_min", "y_min", "x_max", "y_max"}.issubset(
            val_df.columns
        ) and "x" not in val_df.columns:
            val_df["x"] = ((val_df["x_min"] + val_df["x_max"]) / 2).astype(int)
            val_df["y"] = ((val_df["y_min"] + val_df["y_max"]) / 2).astype(int)

        val_dataset = animaloc.datasets.__dict__[val_args.name](
            csv_file=val_df,
            root_dir=val_args.root_dir,
            albu_transforms=_load_albu_transforms(val_args.albu_transforms),
            end_transforms=_load_end_transforms(val_args.end_transforms),
        )

        # 2026-04-25: same num_workers/pin_memory as train loader to keep GPU
        # fed during per-epoch validation passes (especially after FolderDataset
        # was added to validation, which made val_dataset ~5k patches).
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=_get_collate_fn(cfg),
            num_workers=_num_workers,
            pin_memory=_pin_memory,
            persistent_workers=_num_workers > 0,
        )

    work_dir = None

    # Set up wandb
    print("Connecting to Weights & Biases ...")
    settings = cfg.training_settings
    losses = cfg.losses
    if losses is not None:
        losses = list(cfg.losses.keys())

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=dict(
            batch_size=settings.batch_size,
            optimizer=settings.optimizer,
            lr=settings.lr,
            weight_decay=settings.weight_decay,
            warmup_iters=settings.warmup_iters,
            epochs=settings.epochs,
            losses=losses,
            seed=cfg.seed,
            data_augmentation=list(cfg.datasets.train.albu_transforms.keys()),
            input_size=cfg.datasets.img_size,
            **cfg.model.kwargs,
        ),
    )

    date = current_date()
    wandb.run.name = f"{date}_" + cfg.wandb_run + f"_RUN_{wandb.run.id}"

    # Build the model
    print("Building the model ...")
    model = _build_model(cfg)

    # Prepare for training
    print("Preparing for training ...")
    criterions = _load_losses(cfg)
    model = LossWrapper(model, criterions).to(device)

    if cfg.model.load_from is not None:
        partial_load = getattr(cfg.model, "partial_load", False)
        encoder_only = getattr(cfg.model, "encoder_only", False)
        model = load_model(model, cfg.model.load_from, partial_load, encoder_only)

        # Optional layer freezing. All kept models (HerdNet, OWLC, OWLT,
        # OWLD_S/B/L/H) implement a .freeze(layers) method on the underlying
        # nn.Module. The LossWrapper exposes it via .model. We check
        # `hasattr` rather than matching cfg.model.name so the logic stays
        # robust to future model renames.
        if hasattr(getattr(model, "model", model), "freeze"):
            if cfg.model.freeze is not None:
                model.model.freeze(layers=list(cfg.model.freeze))
                print(f"Layers {list(cfg.model.freeze)} freezed")

    if cfg.training_settings.optimizer == "adam":
        backbone_lr_factor = getattr(cfg.training_settings, "backbone_lr_factor", None)
        if backbone_lr_factor is not None:
            backbone_params = [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and "backbone" in n
            ]
            other_params = [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and "backbone" not in n
            ]
            param_groups = [
                {
                    "params": backbone_params,
                    "lr": cfg.training_settings.lr * backbone_lr_factor,
                },
                {"params": other_params},
            ]
            optimizer = torch.optim.Adam(
                param_groups,
                lr=cfg.training_settings.lr,
                weight_decay=cfg.training_settings.weight_decay,
            )
            print(
                f"Discriminative LR: backbone {cfg.training_settings.lr * backbone_lr_factor:.1e}, "
                f"decoder {cfg.training_settings.lr:.1e}"
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.training_settings.lr,
                weight_decay=cfg.training_settings.weight_decay,
            )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training_settings.lr,
            weight_decay=cfg.training_settings.weight_decay,
        )

    # Watch the model's gradients during training
    wandb.watch(model)

    # Evaluator ?
    evaluator = None
    validate_on = "f1_score"
    select = "max"

    if cfg.training_settings.evaluator is not None:
        assert val_dataloader is not None, (
            "A validation dataset must be defined to build an evaluator"
        )

        evaluator = _define_evaluator(model, val_dataloader, cfg)
        select = cfg.training_settings.evaluator.select_mode
        validate_on = cfg.training_settings.evaluator.validate_on

    # Start training & validation
    auto_lr = cfg.training_settings.auto_lr
    if auto_lr:
        auto_lr = dict(cfg.training_settings.auto_lr)

    vizual_fn = None
    if cfg.training_settings.vizual_fn is not None:
        vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]

    trainer = animaloc.train.trainers.__dict__[cfg.training_settings.trainer](
        model,
        train_dataloader,
        optimizer=optimizer,
        num_epochs=cfg.training_settings.epochs,
        auto_lr=auto_lr,
        # adaloss = cfg.training_settings.adaloss,
        val_dataloader=val_dataloader,
        evaluator=evaluator,
        device_name=cfg.device_name,
        vizual_fn=vizual_fn,
        work_dir=work_dir,
        print_freq=cfg.training_settings.print_freq,
        valid_freq=cfg.training_settings.valid_freq,
    )

    if cfg.model.resume_from is not None:
        print(f"Resuming training from '{cfg.model.resume_from}' ...")
        trainer.resume(
            pth_path=cfg.model.resume_from,
            select=select,
            validate_on=validate_on,
            load_optim=True,
            wandb_flag=True,
        )
    else:
        print("Starting training ...")
        trainer.start(
            cfg.training_settings.warmup_iters,
            select=select,
            validate_on=validate_on,
            wandb_flag=True,
        )

    # Add information in .pth files
    for pth_name in ["best_model.pth", "latest_model.pth"]:
        path = os.path.join(os.curdir, pth_name)
        pth_file = torch.load(path)
        norm_trans = _load_albu_transforms(train_args.albu_transforms)[-1]
        pth_file["classes"] = dict(cfg.datasets.class_def)
        pth_file["mean"] = list(norm_trans.mean)
        pth_file["std"] = list(norm_trans.std)
        torch.save(pth_file, path)


if __name__ == "__main__":
    main()
