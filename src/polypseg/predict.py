"""Command-line inference.

Example:
    python -m polypseg.predict --ckpt models/best.pt --image some_polyp.jpg --out-dir predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from polypseg.data import IMAGE_EXTENSIONS
from polypseg.inference import Segmenter, mask_to_image, overlay_mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint written by polypseg.train")
    parser.add_argument("--image", required=True, help="Image file or folder of images")
    parser.add_argument("--out-dir", default="predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.image)
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        images = [source]
    if not images:
        raise SystemExit(f"No images found in {source}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segmenter = Segmenter.from_checkpoint(args.ckpt)

    for path in images:
        image = Image.open(path).convert("RGB")
        mask = segmenter.predict(image, args.threshold)
        mask_to_image(mask).save(out_dir / f"{path.stem}_mask.png")
        overlay_mask(image, mask).save(out_dir / f"{path.stem}_overlay.png")
        print(f"{path.name}: {mask.mean():.1%} of pixels predicted as polyp")


if __name__ == "__main__":
    main()
