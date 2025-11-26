import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力 - 修复版"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # ✅ 关键修复：因为 forward 中只使用了一半通道(实部)作为输入
        # 所以 Linear 层的输入维度应该是 channels // 2
        half_channels = channels // 2

        # 防止 reduction 过大导致维度为 0
        hidden_dim = max(1, half_channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(half_channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, half_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 处理复数张量：[B, 2C, H, W]
        b, c, h, w = x.shape
        ch = c // 2  # 实部通道数

        # 只对实部做注意力（简化处理）
        # x[:, :ch, :, :] 维度是 [B, ch, H, W]
        avg_out = self.avg_pool(x[:, :ch, :, :]).view(b, ch)
        max_out = self.max_pool(x[:, :ch, :, :]).view(b, ch)

        # 此时 avg_out 是 [B, ch]，fc 接受的也是 ch 维输入
        avg_out = self.fc(avg_out).view(b, ch, 1, 1)
        max_out = self.fc(max_out).view(b, ch, 1, 1)

        scale = self.sigmoid(avg_out + max_out)

        # 同时应用到实部和虚部
        # scale 维度是 [B, ch, 1, 1]，广播乘法
        return torch.cat([
            x[:, :ch, :, :] * scale,
            x[:, ch:, :, :] * scale
        ], dim=1)


class SpatialAttention(nn.Module):
    """空间注意力 - 保持不变"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度求平均和最大值，压缩通道信息
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))

        return x * scale


class HybridAttention(nn.Module):
    """混合注意力 - 对外接口"""

    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x