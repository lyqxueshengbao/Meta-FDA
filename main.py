import torch
import argparse
import matplotlib.pyplot as plt
import os
import json
from src.simulation.radar_env import FDARadarSimulator
from src.models.backbone import JammingSuppressionNet
from src.training.meta_trainer import MetaTrainer

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)


def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.yscale('log')
    plt.title("Meta-Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig("results/train_loss.png")
    print("📊 Loss plot saved.")


def plot_adapt(results, k_shot):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MSE
    ax1.plot(results['few_shot']['losses'], 'g-o', label='Few-Shot')
    ax1.axhline(results['zero_shot']['loss'], color='r', linestyle='--', label='Zero-Shot')
    ax1.set_title(f"MSE Adaptation (K={k_shot})")
    ax1.set_xlabel("Step")
    ax1.legend()

    # Correlation
    ax2.plot(results['few_shot']['corrs'], 'b-s', label='Few-Shot')
    ax2.axhline(results['zero_shot']['corr'], color='r', linestyle='--', label='Zero-Shot')
    ax2.set_title("Correlation Adaptation")
    ax2.set_xlabel("Step")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/adaptation.png")
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

    # Init Modules (M=4, N=4 -> 16 channels)
    simulator = FDARadarSimulator(M=4, N=4)
    model = JammingSuppressionNet(in_channels=16)
    trainer = MetaTrainer(model, simulator, device)

    if args.mode == 'train':
        history = trainer.train_loop(epochs=args.epochs, k_shot=args.k_shot)
        torch.save(model.state_dict(), 'checkpoints/meta_model.pth')
        plot_loss(history)
        with open('results/loss_history.json', 'w') as f:
            json.dump(history, f)

    elif args.mode == 'test':
        try:
            model.load_state_dict(torch.load('checkpoints/meta_model.pth', map_location=device))
            print("✅ Model loaded.")
        except:
            print("⚠️ Using random initialization.")

        # Test on unseen task (SJ)
        results = trainer.test_adaptation(target_jamming='SJ', snr=-10, jnr=-10, k_shots=args.k_shot)
        plot_adapt(results, args.k_shot)

        improvement = (results['few_shot']['corrs'][-1] - results['zero_shot']['corr'])
        print(f"\n📈 Final Correlation Improvement: +{improvement:.4f}")


if __name__ == "__main__":
    main()