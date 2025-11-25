import torch
import torch.nn as nn
from .layers import ComplexConv2d


class JammingSuppressionNet(nn.Module):
    """
    Fixed U-Net:
    1. LeakyReLU instead of ReLU.
    2. Skip connections.
    3. Input/Output dimensions match [B, 2*MN, F, T].
    """

    def __init__(self, in_channels=16):  # 16 = 4x4 elements
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            ComplexConv2d(in_channels, 32),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ComplexConv2d(32, 64),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ComplexConv2d(64, 128),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            ComplexConv2d(128 + 64, 64),  # Skip connection input size
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            ComplexConv2d(64 + 32, 32),  # Skip connection input size
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output
        self.out_conv = ComplexConv2d(32, in_channels)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with Skips & Size Fix
        d1 = self.up1(b)
        if d1.shape[2:] != e2.shape[2:]:
            d1 = nn.functional.interpolate(d1, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e2], dim=1)  # Concatenate
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)  # Concatenate
        d2 = self.dec2(d2)

        # Output
        out = self.out_conv(d2)

        # Residual Connection (Global Skip)
        out = out + x * 0.1

        return out