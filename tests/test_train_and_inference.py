import json

import numpy as np

from polypseg import predict, train
from polypseg.inference import Segmenter, mask_to_image, overlay_mask


def test_train_smoke_run_writes_all_artifacts(synthetic_dataset, tmp_path):
    out_dir = tmp_path / "run"
    train.main(
        [
            "--data-dir", str(synthetic_dataset),
            "--out-dir", str(out_dir),
            "--override",
            "train.epochs=2", "train.batch_size=4", "train.num_workers=0", "train.amp=false",
            "data.img_size=64", "model.pretrained=false",
        ]
    )  # fmt: skip
    for name in ("best.pt", "metrics.json", "history.csv", "curves.png", "samples.png",
                 "failure_cases.png", "splits.json"):  # fmt: skip
        assert (out_dir / name).is_file(), name
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert 0.0 <= metrics["test"]["dice"] <= 1.0
    assert metrics["n_train"] + metrics["n_val"] + metrics["n_test"] == 20

    # The checkpoint it wrote must be loadable for inference.
    segmenter = Segmenter.from_checkpoint(out_dir / "best.pt", device="cpu")
    assert segmenter.img_size == 64


def test_segmenter_returns_mask_at_original_resolution(tiny_checkpoint, sample_image):
    segmenter = Segmenter.from_checkpoint(tiny_checkpoint, device="cpu")
    proba = segmenter.predict_proba(sample_image)
    mask = segmenter.predict(sample_image, threshold=0.5)
    assert proba.shape == mask.shape == (sample_image.height, sample_image.width)
    assert mask.dtype == bool
    assert ((proba >= 0) & (proba <= 1)).all()


def test_overlay_and_mask_images(sample_image):
    mask = np.zeros((sample_image.height, sample_image.width), dtype=bool)
    mask[20:50, 30:70] = True
    overlay = overlay_mask(sample_image, mask)
    assert overlay.size == sample_image.size
    assert np.asarray(overlay)[35, 50].tolist() != np.asarray(sample_image)[35, 50].tolist()
    assert mask_to_image(mask).getextrema() == (0, 255)


def test_predict_cli_writes_outputs(tiny_checkpoint, sample_image, tmp_path):
    image_path = tmp_path / "polyp.jpg"
    sample_image.save(image_path)
    out_dir = tmp_path / "out"
    predict.main(["--ckpt", str(tiny_checkpoint), "--image", str(image_path),
                  "--out-dir", str(out_dir)])  # fmt: skip
    assert (out_dir / "polyp_mask.png").is_file()
    assert (out_dir / "polyp_overlay.png").is_file()
