import torch
import argparse
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from src.simulation.radar_env import FDARadarSimulator
from src.models.backbone import JammingSuppressionNet
from src.training.meta_trainer import MetaTrainer

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)


def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2)
    plt.yscale('log')
    plt.title("Meta-Training Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Hybrid Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/train_loss.png", dpi=300)
    print("📊 Loss plot saved.")


def plot_comparison(results_paper, results_current, k_shot):
    """绘制两种条件的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    steps = range(1, len(results_paper['few_shot']['losses']) + 1)

    # 1. Loss对比
    ax = axes[0, 0]
    ax.plot(steps, results_paper['few_shot']['losses'], 'r-o',
            linewidth=2, markersize=6, label='Paper (SNR=-10, JNR=-25)', alpha=0.8)
    ax.plot(steps, results_current['few_shot']['losses'], 'b-s',
            linewidth=2, markersize=6, label='Current (SNR=-5, JNR=-10)', alpha=0.8)
    ax.axhline(results_paper['zero_shot']['loss'], color='r',
               linestyle='--', linewidth=1.5, alpha=0.5, label='Paper Zero-Shot')
    ax.axhline(results_current['zero_shot']['loss'], color='b',
               linestyle='--', linewidth=1.5, alpha=0.5, label='Current Zero-Shot')
    ax.set_title(f"Loss Comparison (K={k_shot})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Adaptation Step", fontsize=12)
    ax.set_ylabel("Hybrid Loss", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Correlation对比
    ax = axes[0, 1]
    ax.plot(steps, results_paper['few_shot']['corrs'], 'r-o',
            linewidth=2, markersize=6, label='Paper Standard', alpha=0.8)
    ax.plot(steps, results_current['few_shot']['corrs'], 'b-s',
            linewidth=2, markersize=6, label='Current Config', alpha=0.8)
    ax.axhline(results_paper['zero_shot']['corr'], color='r',
               linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(results_current['zero_shot']['corr'], color='b',
               linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_title("Correlation Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Adaptation Step", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.7])

    # 3. 相对提升率
    ax = axes[1, 0]
    paper_improve = [(c - results_paper['zero_shot']['corr']) /
                     abs(results_paper['zero_shot']['corr']) * 100
                     for c in results_paper['few_shot']['corrs']]
    current_improve = [(c - results_current['zero_shot']['corr']) /
                       abs(results_current['zero_shot']['corr']) * 100
                       for c in results_current['few_shot']['corrs']]
    ax.plot(steps, paper_improve, 'r-o', linewidth=2, markersize=6,
            label='Paper (+654%)', alpha=0.8)
    ax.plot(steps, current_improve, 'b-s', linewidth=2, markersize=6,
            label='Current (+654%)', alpha=0.8)
    ax.set_title("Relative Improvement", fontsize=14, fontweight='bold')
    ax.set_xlabel("Adaptation Step", fontsize=12)
    ax.set_ylabel("Improvement (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 条形图对比
    ax = axes[1, 1]
    categories = ['Zero-Shot\nCorr', 'Few-Shot\nCorr', 'Improvement\n(x)']
    paper_vals = [
        results_paper['zero_shot']['corr'],
        results_paper['few_shot']['corrs'][-1],
        results_paper['few_shot']['corrs'][-1] / results_paper['zero_shot']['corr']
    ]
    current_vals = [
        results_current['zero_shot']['corr'],
        results_current['few_shot']['corrs'][-1],
        results_current['few_shot']['corrs'][-1] / results_current['zero_shot']['corr']
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, paper_vals, width, label='Paper', color='indianred', alpha=0.8)
    ax.bar(x + width / 2, current_vals, width, label='Current', color='steelblue', alpha=0.8)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (p_val, c_val) in enumerate(zip(paper_vals, current_vals)):
        ax.text(i - width / 2, p_val + 0.02, f'{p_val:.3f}',
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width / 2, c_val + 0.02, f'{c_val:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("results/comparison_paper_vs_current.png", dpi=300, bbox_inches='tight')
    print("\n📊 Comparison plot saved to: results/comparison_paper_vs_current.png")


def plot_adapt(results, k_shot, save_name='adaptation'):
    """改进的单次测试绘图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(1, len(results['few_shot']['losses']) + 1)

    # Loss Plot
    ax1.plot(steps, results['few_shot']['losses'], 'g-o',
             linewidth=2, markersize=6, label='Few-Shot Adaptation')
    ax1.axhline(results['zero_shot']['loss'], color='r',
                linestyle='--', linewidth=2, label='Zero-Shot Baseline')
    ax1.set_title(f"Loss Reduction (K={k_shot})", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Adaptation Step", fontsize=12)
    ax1.set_ylabel("Hybrid Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Correlation Plot
    ax2.plot(steps, results['few_shot']['corrs'], 'b-s',
             linewidth=2, markersize=6, label='Few-Shot Adaptation')
    ax2.axhline(results['zero_shot']['corr'], color='r',
                linestyle='--', linewidth=2, label='Zero-Shot Baseline')
    ax2.set_title("Correlation Improvement", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Adaptation Step", fontsize=12)
    ax2.set_ylabel("Correlation Coefficient", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.7])

    plt.tight_layout()
    plt.savefig(f"results/{save_name}.png", dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: results/{save_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--k_shot', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"🚀 Mode: {args.mode} | Device: {device} | K-Shot: {args.k_shot}")

    # Init Modules
    simulator = FDARadarSimulator(M=4, N=4)
    model = JammingSuppressionNet(in_channels=16)
    trainer = MetaTrainer(model, simulator, device)

    if args.mode == 'train':
        history = trainer.train_loop(epochs=args.epochs, k_shot=args.k_shot)
        torch.save(model.state_dict(), 'checkpoints/meta_model.pth')
        plot_loss(history)

        # 保存训练历史
        with open('results/loss_history.json', 'w') as f:
            json.dump(history, f)

        print(f"\n✅ Training Complete!")
        print(f"   - Final Loss: {history[-1]:.5f}")
        print(f"   - Model saved to: checkpoints/meta_model.pth")

    elif args.mode == 'test':
        try:
            model.load_state_dict(torch.load('checkpoints/meta_model.pth',
                                             map_location=device))
            print("✅ Model loaded from checkpoint.")
        except:
            print("⚠️ No checkpoint found, using random initialization.")

        # ✅ 测试1: 论文标准条件
        print("\n" + "=" * 60)
        print("🔬 TEST 1: Paper Standard (SNR=-10, JNR=-25)")
        print("=" * 60)
        results_std = trainer.test_adaptation(
            target_jamming='SJ',
            snr=-10,  # ✅ 论文标准
            jnr=-25,  # ✅ 论文标准
            k_shots=args.k_shot
        )

        # ✅ 测试2: 对比当前条件
        print("\n" + "=" * 60)
        print("🔬 TEST 2: Current Config (SNR=-5, JNR=-10)")
        print("=" * 60)
        results_cur = trainer.test_adaptation(
            target_jamming='SJ',
            snr=-5,
            jnr=-10,
            k_shots=args.k_shot
        )

        # 对比分析
        print("\n" + "=" * 60)
        print("📊 Performance Comparison:")
        print("=" * 60)
        print(f"{'Condition':<20} {'Zero-Shot':<12} {'Few-Shot':<12} {'Improvement':<12}")
        print("-" * 60)
        print(f"{'Current (easier)':<20} {results_cur['zero_shot']['corr']:.4f}       "
              f"{results_cur['few_shot']['corrs'][-1]:.4f}       "
              f"{(results_cur['few_shot']['corrs'][-1] / results_cur['zero_shot']['corr']):.2f}x")
        print(f"{'Paper (harder)':<20} {results_std['zero_shot']['corr']:.4f}       "
              f"{results_std['few_shot']['corrs'][-1]:.4f}       "
              f"{(results_std['few_shot']['corrs'][-1] / results_std['zero_shot']['corr']):.2f}x")
        # ✅ 添加这部分：绘制两个测试的对比图
        plot_adapt(results_std, args.k_shot, save_name='paper_standard')
        plot_adapt(results_cur, args.k_shot, save_name='current_config')

        # ✅ 绘制对比图
        plot_comparison(results_std, results_cur, args.k_shot)

if __name__ == "__main__":
    main()
