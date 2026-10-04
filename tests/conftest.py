from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from polypseg.model import ResNetUNet


def _make_sample(
    rng: np.random.Generator, size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    width, height = size
    background = rng.integers(60, 160, size=(height, width, 3), dtype=np.uint8)
    image = Image.fromarray(background)
    mask = Image.new("L", size, 0)
    cx, cy = (
        int(rng.integers(width // 4, 3 * width // 4)),
        int(rng.integers(height // 4, 3 * height // 4)),
    )
    r = int(rng.integers(min(size) // 8, min(size) // 4))
    box = [cx - r, cy - r, cx + r, cy + r]
    ImageDraw.Draw(mask).ellipse(box, fill=255)
    ImageDraw.Draw(image).ellipse(box, fill=(200, 90, 90))
    return image, mask


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """A tiny Kvasir-SEG look-alike: images/ and masks/ with matching file names."""
    rng = np.random.default_rng(0)
    root = tmp_path / "Kvasir-SEG"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for i in range(20):
        size = (96 + 8 * (i % 3), 80 + 8 * (i % 2))
        image, mask = _make_sample(rng, size)
        image.save(root / "images" / f"sample_{i:03d}.jpg")
        mask.save(root / "masks" / f"sample_{i:03d}.jpg")
    return tmp_path  # parent folder, like the real download


@pytest.fixture()
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Randomly initialised model saved in the same format train.py writes."""
    torch.manual_seed(0)
    model = ResNetUNet(pretrained=False)
    path = tmp_path / "best.pt"
    torch.save(
        {"model": model.state_dict(), "img_size": 64, "epoch": 1, "val_dice": 0.0, "config": {}},
        path,
    )
    return path


@pytest.fixture()
def sample_image() -> Image.Image:
    image, _ = _make_sample(np.random.default_rng(1), (120, 90))
    return image
