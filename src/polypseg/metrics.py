"""Loss functions and per-image segmentation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_DIMS = (1, 2, 3)


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(_DIMS)
    denominator = probs.sum(_DIMS) + targets.sum(_DIMS)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of binary cross-entropy and soft Dice loss."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * soft_dice_loss(logits, targets)


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Per-image Dice, IoU, precision and recall (each a tensor of shape ``(batch,)``)."""
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum(_DIMS)
    fp = (preds * (1 - targets)).sum(_DIMS)
    fn = ((1 - preds) * targets).sum(_DIMS)
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }
