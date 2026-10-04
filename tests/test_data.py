import shutil

import pytest
import torch

from polypseg.data import KvasirSegDataset, list_pairs, make_splits


def test_list_pairs_accepts_parent_folder(synthetic_dataset):
    pairs = list_pairs(synthetic_dataset)
    assert len(pairs) == 20
    assert all(img.stem == mask.stem for img, mask in pairs)


def test_list_pairs_finds_nested_kaggle_style_layout(synthetic_dataset, tmp_path):
    # archive/Kvasir-SEG/Kvasir-SEG/{images,masks}, as in the Kaggle download
    inner = tmp_path / "archive" / "Kvasir-SEG" / "Kvasir-SEG"
    inner.parent.mkdir(parents=True)
    shutil.move(str(synthetic_dataset / "Kvasir-SEG"), str(inner))
    assert len(list_pairs(tmp_path / "archive")) == 20
    assert len(list_pairs(inner)) == 20


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_pairs(tmp_path / "does_not_exist")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        list_pairs(tmp_path / "empty")


def test_splits_are_deterministic_disjoint_and_complete(synthetic_dataset):
    pairs = list_pairs(synthetic_dataset)
    a = make_splits(pairs, 0.1, 0.2, seed=7)
    b = make_splits(pairs, 0.1, 0.2, seed=7)
    assert a == b
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (14, 2, 4)
    names = [img.name for split in a.values() for img, _ in split]
    assert len(names) == len(set(names)) == 20


def test_different_seed_changes_split(synthetic_dataset):
    pairs = list_pairs(synthetic_dataset)
    assert make_splits(pairs, seed=1) != make_splits(pairs, seed=2)


def test_dataset_item_shapes_and_binary_masks(synthetic_dataset):
    pairs = list_pairs(synthetic_dataset)
    for train in (False, True):
        ds = KvasirSegDataset(pairs, img_size=64, train=train)
        image, mask, name = ds[0]
        assert image.shape == (3, 64, 64) and image.dtype == torch.float32
        assert mask.shape == (1, 64, 64)
        assert set(mask.unique().tolist()) <= {0.0, 1.0}
        assert name.endswith(".jpg")
