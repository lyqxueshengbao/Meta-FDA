import torch
import argparse
import matplotlib.pyplot as plt
import os
import json
from src.simulation.radar_env import FDARadarSimulator
from src.models.uc_cnn import UCCNN
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


def plot_adapt(results, k_shot):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MSE Plot
    steps = range(1, len(results['few_shot']['losses']) + 1)
    ax1.plot(steps, results['few_shot']['losses'], 'g-o',
             linewidth=2, markersize=6, label='Few-Shot Adaptation')
    ax1.axhline(results['zero_shot']['loss'], color='r',
                linestyle='--', linewidth=2, label='Zero-Shot Baseline')
    ax1.set_title(f"Loss Reduction (K={k_shot})", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Adaptation Step", fontsize=12)
    ax1.set_ylabel("Hybrid Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(useOffset=False, style='plain')

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
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("results/adaptation.png", dpi=300)
    print("📊 Adaptation plot saved.")


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
    simulator = FDARadarSimulator(M=4, N=4, n_range=64, n_angle=64)
    # UC-CNN: 2 channels (Real + Imag) for Range-Angle Map
    model = UCCNN(in_channels=1, out_channels=1)  # 1 complex channel = 2 real channels
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

        # Test on DFTJ (主瓣欺骗干扰) - 论文主要测试场景
        print("\n" + "="*60)
        print("🔬 TEST: DFTJ (Mainlobe Deceptive Jamming)")
        print("   Paper Standard Condition: SNR=-10dB, SIR=-25dB")
        print("="*60)
        
        # 测试不同 SIR 条件
        for sir in [0, -5, -10, -15, -20, -25]:
            results = trainer.test_adaptation(
                target_jamming='DFTJ',  # 主瓣欺骗干扰
                snr=-10,
                sir=sir,
                k_shots=args.k_shot
            )
            zero_corr = results['zero_shot']['corr']
            final_corr = results['few_shot']['corrs'][-1]
            ratio = 10**(-sir/10)
            improvement = (final_corr - zero_corr) / abs(zero_corr) * 100 if zero_corr != 0 else 0
            print(f"SIR={sir:3d}dB (干扰{ratio:6.1f}x信号): Zero={zero_corr:.4f} → Few={final_corr:.4f} (+{improvement:.1f}%)")
        
        # 选择 SIR=-25dB 作图
        results = trainer.test_adaptation('DFTJ', snr=-10, sir=-25, k_shots=args.k_shot)
        plot_adapt(results, args.k_shot)

        # 打印详细结果
        print(f"\n{'=' * 50}")
        print(f"📊 Final Results Summary (SIR=-25dB):")
        print(f"{'=' * 50}")
        print(f"Zero-Shot Performance:")
        print(f"  - Loss: {results['zero_shot']['loss']:.5f}")
        print(f"  - Correlation: {results['zero_shot']['corr']:.4f}")
        print(f"\nFew-Shot Performance (After {len(results['few_shot']['corrs'])} steps):")
        print(f"  - Loss: {results['few_shot']['losses'][-1]:.5f}")
        print(f"  - Correlation: {results['few_shot']['corrs'][-1]:.4f}")


if __name__ == "__main__":
    main()
