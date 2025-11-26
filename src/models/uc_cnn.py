import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ComplexConv2d

class ComplexBatchNorm2d(nn.Module):
    """
    Complex Batch Normalization
    Applies BatchNorm2d independently to real and imaginary parts.
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x):
        ch = x.shape[1] // 2
        xr, xi = x[:, :ch, :, :], x[:, ch:, :, :]
        xr = self.bn_r(xr)
        xi = self.bn_i(xi)
        return torch.cat([xr, xi], dim=1)

class ComplexReLU(nn.Module):
    """
    Complex ReLU (CReLU)
    Applies ReLU independently to real and imaginary parts.
    """
    def forward(self, x):
        ch = x.shape[1] // 2
        return torch.cat([F.relu(x[:, :ch, :, :]), F.relu(x[:, ch:, :, :])], dim=1)

class DoubleConv(nn.Module):
    """(ComplexConv => ComplexBN => ComplexReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(in_channels, out_channels),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(),
            ComplexConv2d(out_channels, out_channels),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UCCNN(nn.Module):
    """
    UC-CNN: U-Shaped Complex CNN for Mainlobe Deceptive Jamming Suppression
    Based on standard U-Net architecture but with Complex-valued operations.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up1 = DoubleConv(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up2 = DoubleConv(128 + 64, 64)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up3 = DoubleConv(64 + 32, 32)
        
        # Output
        self.outc = ComplexConv2d(32, out_channels)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder
        d1 = self.up1(x4)
        # Handle padding issues if dimensions are not perfect multiples of 2
        if d1.shape[2:] != x3.shape[2:]:
            d1 = F.interpolate(d1, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.conv_up1(d1)
        
        d2 = self.up2(d1)
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)
        
        d3 = self.up3(d2)
        if d3.shape[2:] != x1.shape[2:]:
            d3 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv_up3(d3)
        
        logits = self.outc(d3)
        return logits
