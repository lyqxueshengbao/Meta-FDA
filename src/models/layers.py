# -*- coding:utf-8 -*-
"""
作者：李钰钦
日期：2025年11月25日
"""
import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    """
    复数 2D 卷积层
    处理输入: [Batch, 2*Channels, Height, Width]
    """

    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv_r = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)
        self.conv_i = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)

    def forward(self, x):
        # 拆分实部和虚部 (假设前一半通道是实部，后一半是虚部)
        ch = x.shape[1] // 2
        xr, xi = x[:, :ch, :, :], x[:, ch:, :, :]

        # 复数乘法: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)

        return torch.cat([yr, yi], dim=1)


def complex_relu(x):
    """复数 ReLU: 分别对实部和虚部做 ReLU"""
    ch = x.shape[1] // 2
    return torch.cat([torch.relu(x[:, :ch, :, :]), torch.relu(x[:, ch:, :, :])], dim=1)