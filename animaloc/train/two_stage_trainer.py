# two_stage_trainer.py
import os
import sys
import math
import torch
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Callable, Any, Dict

from .trainers import TRAINERS

from ..utils.torchvision_utils import SmoothedValue, reduce_dict
from ..utils.logger import CustomLogger
from ..eval.evaluators import Evaluator
from ..data.transforms import UnNormalize
from ..utils.registry import Registry
from .adaloss import Adaloss

@TRAINERS.register()
class TwoStageFineTuneTrainer:
    """
    Two-stage finetuning Trainer for HerdNet + Swin refinement.

    Stage 1 (epoch 1 default):
        - Freeze: backbone (base_0), decoder (dla_up), bottleneck_conv
        - Train: multiscale_swin + loc_head
        - LRs: Swin 1e-4, Head 2e-5
        - No scheduler (single epoch), optional warmup steps

    Stage 2 (epochs 2..N):
        - Unfreeze everything
        - Param groups + LRs:
            Swin 5e-5, loc_head 2e-5, dla_up 2e-5, backbone 1e-5
        - Scheduler: ReduceLROnPlateau (mode=max, patience=2, factor=0.5, cooldown=1, min_lr=1e-6)
        - Warmup not needed here (already stabilized)

    Notes:
        - Uses Adam, weight_decay=5e-4 for all groups
        - Gradient clipping max_norm=1.0
        - Keeps the same evaluator & logging style as your base Trainer
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer],  # ignored; we build per stage
        num_epochs: int = 10,
        lr_milestones: Optional[List[int]] = None,   # not used here
        auto_lr: Union[bool, dict] = False,          # dict for ReduceLROnPlateau overrides
        adaloss: Optional[str] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        evaluator: Optional[Evaluator] = None,
        vizual_fn: Optional[Callable] = None,
        work_dir: Optional[str] = None,
        device_name: str = 'cuda',
        print_freq: int = 50,
        valid_freq: int = 1,
        csv_logger: bool = False,

        # --- Two-stage hyperparams (Option A for 10 epochs) ---
        stage1_epochs: int = 1,
        stage1_warmup_iters: Optional[int] = 400,   # ~5% steps or fixed; None to disable
        stage1_lrs: Dict[str, float] = None,        # {'swin':1e-4, 'head':2e-5}
        stage2_lrs: Dict[str, float] = None,        # {'swin':5e-5, 'head':2e-5, 'dlaup':2e-5, 'backbone':1e-5}
        weight_decay: float = 5e-4,
        grad_clip_max_norm: float = 1.0
    ) -> None:

        # validations
        assert isinstance(model, torch.nn.Module), "model must be nn.Module"
        assert isinstance(train_dataloader, torch.utils.data.DataLoader), "train_dataloader must be DataLoader"
        assert isinstance(val_dataloader, (torch.utils.data.DataLoader, type(None))), "val_dataloader must be DataLoader or None"
        assert isinstance(auto_lr, (bool, dict)), "auto_lr must be bool or dict"
        assert valid_freq <= num_epochs, "valid_freq must be <= num_epochs"

        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = num_epochs

        self.print_freq = print_freq
        self.valid_freq = valid_freq
        self.evaluator = evaluator
        self.vizual_fn = vizual_fn
        self.lr_milestones = lr_milestones  # unused
        self.csv_logger = csv_logger

        # Working directory
        self.work_dir = work_dir or os.getcwd()

        # Loggers
        self.train_logger = CustomLogger(delimiter=' ', filename='training', work_dir=self.work_dir, csv=self.csv_logger)
        self.val_logger   = CustomLogger(delimiter=' ', filename='validation', work_dir=self.work_dir, csv=self.csv_logger)

        # Auto LR / scheduler config
        self.auto_lr = auto_lr
        self.auto_lr_flag = bool(auto_lr)

        # Adaloss (optional)
        self.adaloss = None
        if isinstance(adaloss, str):
            assert hasattr(self.train_dataloader.dataset, 'end_params'), 'Missing end_params in training dataset'
            assert hasattr(self.train_dataloader.dataset, 'update_end_transforms'), 'Missing update_end_transforms()'
            assert adaloss in self.train_dataloader.dataset.end_params, 'adaloss parameter not found in dataset end_params'
            self.adaparam = adaloss
            self.adaloss = Adaloss(self.train_dataloader.dataset.end_params[adaloss], w=3, delta_max=1)

        # Two-stage defaults
        self.stage1_epochs = int(stage1_epochs)
        self.stage1_warmup_iters = stage1_warmup_iters
        self.grad_clip_max_norm = grad_clip_max_norm

        self.stage1_lrs = stage1_lrs or {'swin': 1e-4, 'head': 2e-5}
        self.stage2_lrs = stage2_lrs or {'swin': 5e-5, 'head': 2e-5, 'dlaup': 2e-5, 'backbone': 1e-5}
        self.weight_decay = weight_decay

        # Internal state
        self.optimizer = None
        self.scheduler = None
        self.losses = torch.tensor(0.0)
        self.best_val = float('-inf')  # for mode='max' by default in Option A use-cases

    # -------------------------
    # Public API
    # -------------------------
    def start(
        self,
        warmup_iters: Optional[int] = None,  # ignored; use stage1_warmup_iters
        checkpoints: str = 'best',
        select: str = 'max',             # typically we maximize AP / F1
        validate_on: str = 'mAP',        # or 'all' for sum of losses
        wandb_flag: bool = False
    ) -> torch.nn.Module:

        if warmup_iters is not None:
            self.stage1_warmup_iters = warmup_iters

        assert checkpoints in ['best', 'all']
        assert select in ['min', 'max']

        # Track best
        self.best_val = float('inf') if select == 'min' else float('-inf')

        # --- Stage 1: freeze backbone/decoder, train Swin + Head ---
        self._freeze_modules(['base_0', 'dla_up', 'bottleneck_conv'])
        self._build_optimizer_stage1()   # sets self.optimizer
        self.scheduler = None            # no scheduler for 1 epoch (by default)

        # Train stage 1 epochs
        for epoch in range(1, self.stage1_epochs + 1):
            train_loss = self._train_epoch(epoch, warmup_iters=self.stage1_warmup_iters, wandb_flag=wandb_flag)
            if wandb_flag:
                wandb.log({'train_loss': train_loss, 'epoch': epoch})
                wandb.log({'lr': self.optimizer.param_groups[0]['lr']})

            # Validate (first/last or by valid_freq)
            val_output = self._maybe_validate_and_checkpoint(
                epoch=epoch,
                checkpoints=checkpoints,
                select=select,
                validate_on=validate_on,
                wandb_flag=wandb_flag
            )

            # Adaloss updates
            self._maybe_update_adaloss()

            # Save latest
            self._save_checkpoint(epoch, 'latest')

        # --- Stage 2: unfreeze all, param groups + scheduler ---
        self._unfreeze_modules(['base_0', 'dla_up', 'bottleneck_conv'])
        self._build_optimizer_stage2()
        self._build_scheduler_stage2()   # sets self.scheduler

        # Epochs 2..N (resume counting from stage1_epochs)
        for epoch in range(self.stage1_epochs + 1, self.epochs + 1):
            train_loss = self._train_epoch(epoch, warmup_iters=None, wandb_flag=wandb_flag)
            if wandb_flag:
                wandb.log({'train_loss': train_loss, 'epoch': epoch})
                wandb.log({'lr': self.optimizer.param_groups[0]['lr']})

            # Validate & checkpoint
            val_output = self._maybe_validate_and_checkpoint(
                epoch=epoch,
                checkpoints=checkpoints,
                select=select,
                validate_on=validate_on,
                wandb_flag=wandb_flag
            )

            # Step scheduler on validation metric (if any)
            if self.scheduler is not None:
                metric_for_scheduler = val_output if val_output is not None else (self.best_val if select == 'max' else -self.best_val)
                self.scheduler.step(metric_for_scheduler)

            # Adaloss updates
            self._maybe_update_adaloss()

            # Save latest
            self._save_checkpoint(epoch, 'latest')

        if wandb_flag:
            wandb.run.summary['best_validation'] = self.best_val
            wandb.run.finish()

        return self.model

    # -------------------------
    # Stage builders / groups
    # -------------------------
    def _build_optimizer_stage1(self):
        swin_params, head_params, _, _ = self._group_params()

        self.optimizer = torch.optim.Adam([
            {"params": swin_params, "lr": self.stage1_lrs['swin'], "weight_decay": self.weight_decay},
            {"params": head_params, "lr": self.stage1_lrs['head'], "weight_decay": self.weight_decay},
        ], betas=(0.9, 0.999), eps=1e-8)

    def _build_optimizer_stage2(self):
        swin_params, head_params, dlaup_params, backbone_params = self._group_params()

        self.optimizer = torch.optim.Adam([
            {"params": swin_params,     "lr": self.stage2_lrs['swin'],     "weight_decay": self.weight_decay},
            {"params": head_params,     "lr": self.stage2_lrs['head'],     "weight_decay": self.weight_decay},
            {"params": dlaup_params,    "lr": self.stage2_lrs['dlaup'],    "weight_decay": self.weight_decay},
            {"params": backbone_params, "lr": self.stage2_lrs['backbone'], "weight_decay": self.weight_decay},
        ], betas=(0.9, 0.999), eps=1e-8)

    def _build_scheduler_stage2(self):
        if self.auto_lr is True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=2, factor=0.5, cooldown=1, min_lr=1e-6, verbose=True
            )
        elif isinstance(self.auto_lr, dict):
            # allow overrides (e.g., mode, patience)
            kwargs = dict(mode='max', patience=2, factor=0.5, cooldown=1, min_lr=1e-6, verbose=True)
            kwargs.update(self.auto_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs)
        else:
            self.scheduler = None

    def _group_params(self):
        swin_params     = [p for n, p in self.model.named_parameters() if "multiscale_swin" in n and p.requires_grad]
        head_params     = [p for n, p in self.model.named_parameters() if "loc_head" in n and p.requires_grad]
        dlaup_params    = [p for n, p in self.model.named_parameters() if "dla_up" in n and p.requires_grad]
        backbone_params = [p for n, p in self.model.named_parameters() if (("base_0" in n) or ("bottleneck_conv" in n)) and p.requires_grad]
        return swin_params, head_params, dlaup_params, backbone_params

    # -------------------------
    # Train / Eval loops
    # -------------------------
    def _train_epoch(self, epoch: int, warmup_iters: Optional[int] = None, wandb_flag: bool = False) -> float:
        self.model.train()

        # logging meters
        self.train_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'[TRAINING] - Epoch: [{epoch}]'

        # warmup LR (manual)
        start_lr_scheduler = None
        if warmup_iters is not None and warmup_iters > 0:
            start_lr_scheduler = self._warmup_lr_scheduler(min(warmup_iters, len(self.train_dataloader)-1), 1. / max(1, warmup_iters))

        batches_losses = []
        for i, (images, targets) in enumerate(self.train_logger.log_every(self.train_dataloader, self.print_freq, header)):
            images, targets = self._prepare_data(images, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self.model(images, targets)       # training—model returns dict of losses

            if wandb_flag:
                wandb.log(loss_dict)

            losses = sum(loss for loss in loss_dict.values())
            batches_losses.append(losses.detach())

            # guard against NaNs
            if not math.isfinite(losses.item()):
                print("Loss is {}, stopping training".format(losses.item()))
                print({k: float(v.detach()) for k, v in loss_dict.items()})
                sys.exit(1)

            losses.backward()
            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            self.optimizer.step()

            # adaloss feed
            if self.adaloss is not None:
                self.adaloss.feed(losses)

            # warmup step
            if start_lr_scheduler is not None:
                start_lr_scheduler.step()

            # reduced (for logging)
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            self.train_logger.update(loss=losses_reduced, **loss_dict_reduced)
            self.train_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        mean_loss = torch.stack(batches_losses).mean().item()
        print(f'{header} mean loss: {mean_loss:.4f}')
        return mean_loss

    @torch.no_grad()
    def _evaluate(self, epoch: int, reduction: str = 'mean', wandb_flag: bool = False, returns: str = 'mAP') -> float:
        assert self.val_dataloader is not None or self.evaluator is not None, "No validation dataloader/evaluator provided"
        self.model.eval()

        header = f'[VALIDATION] - Epoch: [{epoch}]'
        if self.evaluator is not None:
            self._prepare_evaluator('validation', epoch)
            viz = bool(wandb_flag)
            val_output = self.evaluator.evaluate(returns=returns, viz=viz)
            print(f'{self.evaluator.header} {returns}: {val_output:.4f}')
            return float(val_output)

        # Fallback: compute losses from val loader
        batches_losses = []
        for i, (images, targets) in enumerate(self.val_logger.log_every(self.val_dataloader, self.print_freq, header)):
            images, targets = self._prepare_data(images, targets)
            output, loss_dict = self.model(images, targets)  # eval returns (output, loss_dict)

            losses = sum(loss for loss in loss_dict.values())
            if returns != 'all':  # use specific key if desired
                losses = loss_dict[returns]

            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            self.val_logger.update(loss=losses_reduced, **loss_dict_reduced)
            batches_losses.append(losses)

            if wandb_flag and self.vizual_fn is not None and (i % self.print_freq == 0 or i == len(self.val_dataloader) - 1):
                fig = self._vizual(image=images, target=targets, output=output)
                wandb.log({'validation_vizuals': fig})

        batches_losses = torch.stack(batches_losses)
        if reduction == 'mean':
            out = batches_losses.mean().item()
            print(f'{header} mean loss: {out:.4f}')
            return out
        else:  # 'sum'
            out = batches_losses.sum().item()
            print(f'{header} sum loss: {out:.4f}')
            return out

    # -------------------------
    # Utils
    # -------------------------
    def _prepare_data(self, images, targets):
        images = images.to(self.device)
        if isinstance(targets, (list, tuple)):
            targets = [t.to(self.device) for t in targets]
        else:
            targets = targets.to(self.device)
        return images, targets

    def _warmup_lr_scheduler(self, warmup_iters: int, warmup_factor: float):
        def warmup_func(x):
            if x >= warmup_iters:
                return 1.0
            alpha = float(x) / float(max(1, warmup_iters))
            return warmup_factor * (1 - alpha) + alpha
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_func)

    def _freeze_modules(self, module_names: List[str]):
        for name in module_names:
            if hasattr(self.model, name):
                getattr(self.model, name).requires_grad_(False)

    def _unfreeze_modules(self, module_names: List[str]):
        for name in module_names:
            if hasattr(self.model, name):
                getattr(self.model, name).requires_grad_(True)

    def _prepare_evaluator(self, filename: str, epoch: int) -> None:
        if self.evaluator is not None:
            self.evaluator.model = self.model
            self.evaluator.logs_filename = filename
            self.evaluator.header = f'[{filename.upper()}] - Epoch: [{epoch}]'

    def _is_best(self, val_output: float, mode: str = 'max') -> bool:
        if mode == 'min':
            if val_output < self.best_val:
                self.best_val = val_output
                return True
            return False
        else:  # 'max'
            if val_output > self.best_val:
                self.best_val = val_output
                return True
            return False

    def _save_checkpoint(self, epoch: int, mode: str) -> None:
        check_dir = self.work_dir
        if mode == 'all':
            outpath = os.path.join(check_dir, f'epoch_{epoch}.pth')
        elif mode == 'best':
            outpath = os.path.join(check_dir, 'best_model.pth')
        elif mode == 'latest':
            outpath = os.path.join(check_dir, 'latest_model.pth')
        else:
            outpath = os.path.join(check_dir, f'epoch_{epoch}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': (self.optimizer.state_dict() if self.optimizer is not None else {}),
            'loss': self.losses,
            'best_val': self.best_val
        }, outpath)

    def _maybe_validate_and_checkpoint(self, epoch: int, checkpoints: str, select: str, validate_on: str, wandb_flag: bool):
        val_output = None
        # validate at frequency or on first/last epoch
        if (self.val_dataloader is not None or self.evaluator is not None) and \
           (epoch % self.valid_freq == 0 or epoch in [1, self.epochs]):

            val_output = self._evaluate(epoch, reduction='mean', wandb_flag=wandb_flag, returns=validate_on)
            if wandb_flag:
                wandb.log({validate_on: val_output, 'epoch': epoch})

            # save
            if checkpoints == 'best':
                if self._is_best(val_output, mode=select):
                    print(f'Best model saved - Epoch {epoch} - Validation ({validate_on}): {val_output:.6f}')
                    self._save_checkpoint(epoch, 'best')
            elif checkpoints == 'all':
                self._save_checkpoint(epoch, 'all')
        return val_output

    def _maybe_update_adaloss(self):
        if self.adaloss is not None:
            self.adaloss.step()
            self.train_dataloader.dataset.load_end_param(self.adaparam, self.adaloss.param)
            self.train_dataloader.dataset.update_end_transforms()
            if self.val_dataloader is not None:
                self.val_dataloader.dataset.end_params = self.train_dataloader.dataset.end_params
                self.val_dataloader.dataset.update_end_transforms()

    def _vizual(self, image: Any, target: Any, output: Any):
        if self.vizual_fn is None:
            return None
        fig = self.vizual_fn(image=image, target=target, output=output)
        return fig

    def resume(
        self,
        pth_path: str,
        checkpoints: str = 'best',
        select: str = 'max',          # keep 'max' if you resume with metrics like f1/mAP
        validate_on: str = 'mAP',
        load_optim: bool = False,     # True = try to load optimizer state (only safe if same stage/param-groups)
        wandb_flag: bool = False
    ) -> torch.nn.Module:
        """
        Resume training from a checkpoint. Safe with best_model.pth or latest_model.pth.
        Handles stage detection (1 or 2), param groups, and scheduler.

        Args:
            pth_path: path to .pth checkpoint (e.g., best_model.pth)
            checkpoints: 'best' or 'all' for saving policy going forward
            select: 'max' (typical for f1/mAP) or 'min'
            validate_on: metric/loss key to drive validation & scheduler
            load_optim: if True, attempts to load optimizer state (works only if param-groups match)
            wandb_flag: log to W&B
        """
        assert checkpoints in ['best', 'all']
        assert select in ['min', 'max']

        # --- Load checkpoint ---
        ckpt = torch.load(pth_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state_dict'])
        resume_epoch = int(ckpt.get('epoch', 0))
        self.losses = ckpt.get('loss', torch.tensor(0.0))

        # Best value bookkeeping
        if 'best_val' in ckpt:
            self.best_val = float(ckpt['best_val'])
        else:
            self.best_val = float('-inf') if select == 'max' else float('inf')

        # --- Decide which stage we are resuming into ---
        # If the checkpoint was saved at epoch e, we will start at e+1.
        # Stage 1 is epochs [1..stage1_epochs], Stage 2 is [stage1_epochs+1..]
        next_epoch = resume_epoch + 1
        in_stage1 = (next_epoch <= self.stage1_epochs)

        # Freeze/unfreeze + build optimizer/scheduler according to the stage
        if in_stage1:
            # Stage 1: freeze backbone/decoder; no scheduler
            self._freeze_modules(['base_0', 'dla_up', 'bottleneck_conv'])
            self._build_optimizer_stage1()
            self.scheduler = None
        else:
            # Stage 2: unfreeze everything; scheduler enabled if auto_lr set
            self._unfreeze_modules(['base_0', 'dla_up', 'bottleneck_conv'])
            self._build_optimizer_stage2()
            self._build_scheduler_stage2()

        # Try to load optimizer state only if requested AND param groups match
        if load_optim and 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict']:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print("[resume] Warning: could not load optimizer state (likely stage/param-group mismatch). "
                      "Continuing with freshly built optimizer.\n", repr(e))

        # W&B lr log
        if wandb_flag and len(self.optimizer.param_groups) > 0:
            wandb.log({'lr': self.optimizer.param_groups[0]['lr']})

        # --- Training loop from next_epoch .. self.epochs ---
        for epoch in range(next_epoch, self.epochs + 1):
            # Handle stage transition exactly at stage1_epochs -> stage2
            if in_stage1 and epoch == self.stage1_epochs + 1:
                # switch to Stage 2
                self._unfreeze_modules(['base_0', 'dla_up', 'bottleneck_conv'])
                self._build_optimizer_stage2()
                self._build_scheduler_stage2()
                in_stage1 = False

            # (No warmup on resume; it only applied to epoch 1 in Stage 1)
            train_loss = self._train_epoch(epoch, warmup_iters=None, wandb_flag=wandb_flag)
            if wandb_flag:
                wandb.log({'train_loss': train_loss, 'epoch': epoch})
                wandb.log({'lr': self.optimizer.param_groups[0]['lr']})

            # Validate & checkpoint (first/last or by valid_freq)
            val_output = self._maybe_validate_and_checkpoint(
                epoch=epoch,
                checkpoints=checkpoints,
                select=select,
                validate_on=validate_on,
                wandb_flag=wandb_flag
            )

            # Step Stage-2 scheduler (if enabled)
            if (self.scheduler is not None) and (not in_stage1):
                metric_for_scheduler = val_output if val_output is not None else (
                    self.best_val if select == 'max' else -self.best_val
                )
                self.scheduler.step(metric_for_scheduler)

            # Adaloss updates
            self._maybe_update_adaloss()

            # Always save latest
            self._save_checkpoint(epoch, 'latest')

        if wandb_flag:
            wandb.run.summary['best_validation'] = self.best_val
            wandb.run.finish()

        return self.model
