"""Inference helpers shared by the CLI and the API (no training-only dependencies)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from polypseg.data import build_transforms
from polypseg.model import ResNetUNet


class Segmenter:
    """Loads a checkpoint written by ``polypseg.train`` and segments PIL images."""

    def __init__(self, model: ResNetUNet, img_size: int, device: torch.device, meta: dict):
        self.model = model.to(device).eval()
        self.img_size = img_size
        self.device = device
        self.meta = meta
        # Same deterministic pipeline as validation: resize -> scale -> ImageNet normalisation.
        self.transform = build_transforms(img_size, train=False)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> Segmenter:
        device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=device_, weights_only=True)
        model = ResNetUNet(pretrained=False)
        model.load_state_dict(ckpt["model"])
        meta = {"epoch": ckpt.get("epoch"), "val_dice": ckpt.get("val_dice")}
        return cls(model, int(ckpt["img_size"]), device_, meta)

    @torch.inference_mode()
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """Foreground probability map at the original image resolution, shape (H, W)."""
        image = image.convert("RGB")
        width, height = image.size
        x = self.transform(image).as_subclass(torch.Tensor).unsqueeze(0).to(self.device)
        probs = torch.sigmoid(self.model(x))
        probs = F.interpolate(probs, size=(height, width), mode="bilinear", align_corners=False)
        return probs[0, 0].cpu().numpy()

    def predict(self, image: Image.Image, threshold: float = 0.5) -> np.ndarray:
        """Boolean mask at the original image resolution, shape (H, W)."""
        return self.predict_proba(image) > threshold


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a coloured mask over the image and draw its one-pixel outline."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    inside = mask.astype(bool)
    rgb = np.array(color, dtype=np.float32)

    out = base.copy()
    out[inside] = (1 - alpha) * base[inside] + alpha * rgb

    padded = np.pad(inside, 1)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    out[inside & ~eroded] = rgb
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
