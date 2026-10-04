"""U-Net with an ImageNet-pretrained ResNet34 encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample, concatenate the encoder skip connection, then two 3x3 convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class ResNetUNet(nn.Module):
    """Binary segmentation network. Returns logits with the same spatial size as the input."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 1/2, 64 ch
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1  # 1/4, 64 ch
        self.layer2 = backbone.layer2  # 1/8, 128 ch
        self.layer3 = backbone.layer3  # 1/16, 256 ch
        self.layer4 = backbone.layer4  # 1/32, 512 ch

        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 64, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        d = self.dec3(s4, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        d = self.dec0(d, s0)
        d = F.interpolate(d, size=size, mode="bilinear", align_corners=False)
        return self.head(d)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
