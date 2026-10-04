"""Kvasir-SEG dataset, deterministic splits and transforms."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_data_dir(data_dir: str | Path, max_depth: int = 4) -> Path:
    """Return the folder that directly contains ``images/`` and ``masks/``.

    ``data_dir`` may be that folder itself or any parent of it. Downloads nest it differently
    (e.g. ``archive/Kvasir-SEG/Kvasir-SEG/images``), so subfolders are searched breadth-first and
    the shallowest match wins.
    """
    root = Path(data_dir)
    queue = [(root, 0)] if root.is_dir() else []
    while queue:
        folder, depth = queue.pop(0)
        if (folder / "images").is_dir() and (folder / "masks").is_dir():
            return folder
        if depth < max_depth:
            queue.extend((child, depth + 1) for child in sorted(folder.iterdir()) if child.is_dir())
    raise FileNotFoundError(
        f"Could not find a folder containing 'images/' and 'masks/' within {max_depth} levels of "
        f"{root}. Run `python -m polypseg.download` or pass the correct --data-dir."
    )


def list_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    """List (image, mask) path pairs, sorted by file name."""
    root = resolve_data_dir(data_dir)
    masks = {p.stem: p for p in (root / "masks").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs: list[tuple[Path, Path]] = []
    for image in sorted((root / "images").iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask = masks.get(image.stem)
        if mask is None:
            raise FileNotFoundError(f"No mask found for image {image.name}")
        pairs.append((image, mask))
    if not pairs:
        raise FileNotFoundError(f"No images found in {root / 'images'}")
    return pairs


def make_splits(
    pairs: list[tuple[Path, Path]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    """Deterministic random train/val/test split at the image level."""
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be between 0 and 1")
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(pairs) * test_frac))
    n_val = max(1, round(len(pairs) * val_frac))
    test_idx = order[:n_test]
    val_idx = order[n_test : n_test + n_val]
    train_idx = order[n_test + n_val :]
    return {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx],
    }


def build_transforms(img_size: int, train: bool) -> v2.Compose:
    """Joint image/mask transforms. The eval pipeline is also used at inference time."""
    steps: list = [
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
    ]
    if train:
        steps += [
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.RandomApply(
                [v2.RandomAffine(degrees=25, translate=(0.08, 0.08), scale=(0.85, 1.15))],
                p=0.7,
            ),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)],
                p=0.8,
            ),
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(steps)


class KvasirSegDataset(Dataset):
    """Returns ``(image, mask, file_name)`` with image (3,H,W) float and mask (1,H,W) in {0,1}."""

    def __init__(self, pairs: list[tuple[Path, Path]], img_size: int = 352, train: bool = False):
        self.pairs = list(pairs)
        self.transform = build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask_arr = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
        mask = tv_tensors.Mask(torch.from_numpy(mask_arr).unsqueeze(0))
        image_t, mask_t = self.transform(image, mask)
        return (
            image_t.as_subclass(torch.Tensor),
            mask_t.as_subclass(torch.Tensor).float(),
            image_path.name,
        )
