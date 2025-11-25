# -*- coding:utf-8 -*-
"""
作者：李钰钦
日期：2025年11月25日
"""
import torch
import numpy as np
from src.simulation.radar_env import FDARadarSimulator
from src.models.backbone import JammingSuppressionNet

print("=" * 60)
print("🔍 维度验证测试 (v2.0 修正版)")
print("=" * 60)

# 1. 测试仿真器
sim = FDARadarSimulator(M=8, N=8)
X, Y = sim.generate_batch(batch_size=2, jamming_type='DFTJ')

print(f"\n📡 仿真器输出:")
print(f"   X.shape = {X.shape}  # 期望: [2, 128, F, T]")
print(f"   Y.shape = {Y.shape}  # 期望: [2, 2, F, T]")

# 2. 测试网络
model = JammingSuppressionNet(in_channels=64)
try:
    output = model(X)
    print(f"\n🧠 网络输出:")
    print(f"   Output: {output.shape}  # 期望: [2, 2, F, T]")

    assert output.shape == Y.shape, "❌ 输出形状不匹配!"
    print(f"\n✅ 所有测试通过!")

except Exception as e:
    print(f"\n❌ 错误: {e}")