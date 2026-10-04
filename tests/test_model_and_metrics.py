import pytest
import torch

from polypseg.metrics import BCEDiceLoss, segmentation_metrics, soft_dice_loss
from polypseg.model import ResNetUNet


@pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 100, 120)])
def test_model_output_matches_input_size(shape):
    model = ResNetUNet(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(*shape))
    assert out.shape == (shape[0], 1, shape[2], shape[3])


def test_perfect_prediction_scores_one():
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1
    logits = (target * 2 - 1) * 20  # confident and correct
    m = segmentation_metrics(logits, target)
    for name in ("dice", "iou", "precision", "recall"):
        assert torch.allclose(m[name], torch.ones(2), atol=1e-4)


def test_disjoint_prediction_scores_zero():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1
    logits = (1 - target * 2) * 20  # predicts exactly the opposite half
    m = segmentation_metrics(logits, target)
    assert m["dice"].item() < 1e-3
    assert m["iou"].item() < 1e-3


def test_loss_is_lower_for_better_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1
    good = (target * 2 - 1) * 5
    bad = -good
    criterion = BCEDiceLoss(0.5)
    assert criterion(good, target) < criterion(bad, target)
    assert soft_dice_loss(good, target) < soft_dice_loss(bad, target)
