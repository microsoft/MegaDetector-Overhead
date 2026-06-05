import torch
import torch.nn as nn

from .register import LOSSES


@LOSSES.register()
class WeightedMSELoss(nn.Module):
    """MSE loss with optional upweighting of positive (non-zero) regions.

    For FIDT / Gaussian density targets most pixels are 0; a plain MSE
    optimises towards predicting 0 everywhere.  This loss multiplies the
    per-pixel squared error by a weight map:

        weight(x,y) = 1  +  pos_weight * (target(x,y) > threshold)

    so pixels near annotations receive ``1 + pos_weight`` times the loss.

    Args:
        pos_weight (float): Extra weight added on positive pixels.
            Defaults to 10.
        threshold (float): Minimum target value to count as positive.
            Defaults to 0.01.
        reduction (str): 'mean' or 'sum'. Defaults to 'mean'.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        threshold: float = 0.01,
        reduction: str = 'mean',
    ):
        super().__init__()
        assert reduction in ('mean', 'sum')
        self.pos_weight = pos_weight
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.pos_weight * (target > self.threshold).float()
        loss = weight * (output - target) ** 2
        return loss.mean() if self.reduction == 'mean' else loss.sum()
