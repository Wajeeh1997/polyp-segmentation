"""Download and extract Kvasir-SEG.

The dataset is provided by Simula for research and educational use; see
https://datasets.simula.no/kvasir-seg/ for the terms. It is not redistributed in this repository.

Example:
    python -m polypseg.download --out-dir data
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--url", default=KVASIR_SEG_URL)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "Kvasir-SEG"
    if (target / "images").is_dir() and (target / "masks").is_dir():
        print(f"Dataset already present at {target}")
        return

    archive = out_dir / "kvasir-seg.zip"
    print(f"Downloading {args.url} ...")
    urllib.request.urlretrieve(args.url, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
    archive.unlink()

    images = len(list((target / "images").glob("*")))
    masks = len(list((target / "masks").glob("*")))
    print(f"Done: {images} images and {masks} masks in {target}")


if __name__ == "__main__":
    main()
