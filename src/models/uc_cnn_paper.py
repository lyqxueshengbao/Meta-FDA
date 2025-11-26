"""
UC-CNN: U-Shaped Complex CNN for Mainlobe Deceptive Jamming Suppression
基于论文核心思想的简化实现

核心组件:
1. 复数卷积 (Complex Convolution)
2. 多尺度特征融合 (3×3 + 5×5)
3. 混合注意力机制 (Channel + Spatial)
4. U-Net 编解码器结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """复数卷积 - 论文公式 (4)"""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv_r = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)
        self.conv_i = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)

    def forward(self, x):
        c = x.shape[1] // 2
        xr, xi = x[:, :c], x[:, c:]
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexBatchNorm2d(nn.Module):
    """复数批归一化"""
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x):
        c = x.shape[1] // 2
        return torch.cat([self.bn_r(x[:, :c]), self.bn_i(x[:, c:])], dim=1)


class ComplexReLU(nn.Module):
    """复数ReLU"""
    def forward(self, x):
        c = x.shape[1] // 2
        return torch.cat([F.relu(x[:, :c]), F.relu(x[:, c:])], dim=1)


class MultiScaleBlock(nn.Module):
    """多尺度特征融合块 (3×3 + 5×5)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        mid_c = out_c // 2
        
        self.branch3 = nn.Sequential(
            ComplexConv2d(in_c, mid_c, 3, 1),
            ComplexBatchNorm2d(mid_c),
            ComplexReLU()
        )
        self.branch5 = nn.Sequential(
            ComplexConv2d(in_c, mid_c, 5, 2),
            ComplexBatchNorm2d(mid_c),
            ComplexReLU()
        )
        self.fusion = nn.Sequential(
            ComplexConv2d(out_c, out_c, 3, 1),
            ComplexBatchNorm2d(out_c),
            ComplexReLU()
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        return self.fusion(torch.cat([b3, b5], dim=1))


class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        attn = self.fc(x).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_v, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, max_v], dim=1)))
        return x * attn


class HybridAttention(nn.Module):
    """混合注意力 (通道 + 空间)"""
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class UCCNN_Paper(nn.Module):
    """
    UC-CNN 论文实现
    
    输入: [B, 2*C, H, W] - C个复数通道的STFT数据
    输出: [B, 2*C, H, W] - 恢复的纯净信号
    """
    def __init__(self, in_channels=16):
        super().__init__()
        
        # 编码器
        self.enc1 = MultiScaleBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = MultiScaleBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = MultiScaleBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # 瓶颈层 + 注意力
        self.bottleneck = nn.Sequential(
            MultiScaleBlock(128, 256),
            HybridAttention(256 * 2)  # 2*256 通道 (实部+虚部)
        )
        
        # 解码器
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = MultiScaleBlock(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = MultiScaleBlock(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = MultiScaleBlock(64 + 32, 32)
        
        # 输出
        self.out = ComplexConv2d(32, in_channels, 1, 0)

    def _align(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # 瓶颈
        b = self.bottleneck(self.pool3(e3))
        
        # 解码 + 跳跃连接
        d3 = self.dec3(torch.cat([self._align(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._align(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._align(self.up1(d2), e1), e1], dim=1))
        
        return self.out(d1)


if __name__ == "__main__":
    model = UCCNN_Paper(in_channels=16)
    x = torch.randn(2, 32, 64, 17)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
