"""Training curves and qualitative result figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from polypseg.data import IMAGENET_MEAN, IMAGENET_STD
from polypseg.inference import overlay_mask
from polypseg.metrics import segmentation_metrics


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_curves(history: list[dict], path: str | Path) -> None:
    plt = _pyplot()
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="BCE + Dice loss", title="Loss")
    ax_loss.legend()

    ax_metric.plot(epochs, [h["val_dice"] for h in history], label="Dice")
    ax_metric.plot(epochs, [h["val_iou"] for h in history], label="IoU")
    ax_metric.set(xlabel="epoch", ylabel="score", title="Validation metrics", ylim=(0, 1))
    ax_metric.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def per_sample_dice(model: torch.nn.Module, dataset, device: torch.device) -> list[float]:
    model.eval()
    scores = []
    for i in range(len(dataset)):
        image, mask, _ = dataset[i]
        logits = model(image.unsqueeze(0).to(device))
        dice = segmentation_metrics(logits, mask.unsqueeze(0).to(device))["dice"]
        scores.append(dice.item())
    return scores


def _to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (tensor.cpu() * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def save_prediction_grid(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    indices: list[int],
    path: str | Path,
    title: str = "",
) -> None:
    """One row per sample: image | ground truth (green) | prediction (red) with its Dice."""
    plt = _pyplot()
    model.eval()
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)), squeeze=False)

    for row, idx in enumerate(indices):
        image_t, mask_t, name = dataset[idx]
        logits = model(image_t.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
        dice = segmentation_metrics(logits, mask_t.unsqueeze(0).to(device))["dice"].item()

        image = Image.fromarray(_to_uint8_image(image_t))
        panels = [
            (image, "image"),
            (overlay_mask(image, mask_t[0].numpy() > 0.5, color=(40, 220, 40)), "ground truth"),
            (overlay_mask(image, pred), f"prediction (Dice {dice:.3f})"),
        ]
        for col, (panel, label) in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(label if row else f"{label}", fontsize=9)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(name)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
