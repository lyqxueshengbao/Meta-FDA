import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ComplexConv2d


class JammingSuppressionNet(nn.Module):
    """
    修复版 U-Net:
    1. 可学习的残差权重
    2. 更鲁棒的跳跃连接
    3. 改进的激活函数
    """

    def __init__(self, in_channels=16):
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
            ComplexConv2d(128 + 64, 64),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            ComplexConv2d(64 + 32, 32),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output
        self.out_conv = ComplexConv2d(32, in_channels)

        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

    def _align_size(self, x, target):
        """鲁棒的尺寸对齐"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:],
                              mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with Skip Connections
        d1 = self.up1(b)
        d1 = self._align_size(d1, e2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self._align_size(d2, e1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        # Output
        out = self.out_conv(d2)

        # 可学习的残差连接
        weight = torch.sigmoid(self.residual_weight)
        out = out + x * weight

        return out
