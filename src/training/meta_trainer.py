import torch
import learn2learn as l2l
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class MetaTrainer:
    def __init__(self, model, simulator, device='cuda'):
        self.device = device
        self.simulator = simulator
        self.model = model.to(device)
        # Use first_order=True for speed/stability with LeakyReLU
        self.maml = l2l.algorithms.MAML(self.model, lr=0.005, first_order=True)
        self.optimizer = optim.Adam(self.maml.parameters(), lr=0.0005)
        self.loss_fn = torch.nn.MSELoss()

        self.train_tasks = [
            {'type': 'DFTJ', 'snr': 0, 'jnr': 0},
            {'type': 'ISRJ', 'snr': 0, 'jnr': 0},
        ]

        print(f"✅ Meta Trainer Initialized")
        print(f"   - Inner LR: 0.005, Outer LR: 0.0005")

    def _sample_task(self):
        idx = np.random.randint(len(self.train_tasks))
        return self.train_tasks[idx]

    def train_loop(self, epochs=500, tasks_per_batch=2, k_shot=5):
        loss_history = []
        pbar = tqdm(range(epochs), desc="Meta-Training")

        for epoch in pbar:
            self.optimizer.zero_grad()
            meta_loss = 0.0

            for task_idx in range(tasks_per_batch):
                task = self._sample_task()
                learner = self.maml.clone()

                sx, sy = self.simulator.generate_batch(k_shot, task['type'], task['snr'], task['jnr'])
                sx, sy = sx.to(self.device), sy.to(self.device)

                # Debug Epoch 0
                if epoch == 0 and task_idx == 0:
                    print(f"\n[Debug Epoch 0] X shape: {sx.shape}, Y shape: {sy.shape}")
                    print(f"   X mean: {sx.mean():.4f}, Y mean: {sy.mean():.4f}")

                # Inner Loop
                for _ in range(5):
                    pred = learner(sx)
                    loss = self.loss_fn(pred, sy)
                    learner.adapt(loss)

                qx, qy = self.simulator.generate_batch(k_shot, task['type'], task['snr'], task['jnr'])
                qx, qy = qx.to(self.device), qy.to(self.device)

                q_pred = learner(qx)
                q_loss = self.loss_fn(q_pred, qy)
                meta_loss += q_loss

            meta_loss /= tasks_per_batch
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.maml.parameters(), 1.0)
            self.optimizer.step()

            loss_history.append(meta_loss.item())

            if (epoch + 1) % 50 == 0:
                pbar.set_postfix({'Loss': f'{meta_loss.item():.5f}'})

        return loss_history

    def _compute_correlation(self, pred, target):
        p = pred.reshape(pred.size(0), -1)
        t = target.reshape(target.size(0), -1)
        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)
        corr = torch.sum(p * t, dim=1) / (torch.norm(p, dim=1) * torch.norm(t, dim=1) + 1e-8)
        return corr.mean().item()

    def test_adaptation(self, target_jamming='SJ', snr=-10, jnr=-10, k_shots=10):
        print(f"\n{'=' * 40}")
        print(f"🧪 Testing Adaptation: {target_jamming} (K={k_shots})")
        print(f"{'=' * 40}")

        sx, sy = self.simulator.generate_batch(k_shots, target_jamming, snr, jnr)
        tx, ty = self.simulator.generate_batch(50, target_jamming, snr, jnr)
        sx, sy = sx.to(self.device), sy.to(self.device)
        tx, ty = tx.to(self.device), ty.to(self.device)

        learner = self.maml.clone()

        # Zero-Shot Stats
        with torch.no_grad():
            z_pred = learner(tx)
            z_loss = self.loss_fn(z_pred, ty).item()
            z_corr = self._compute_correlation(z_pred, ty)

        print(f"Zero-Shot: MSE={z_loss:.5f}, Corr={z_corr:.4f}")

        losses, corrs = [], []
        for step in range(10):
            pred = learner(sx)
            loss = self.loss_fn(pred, sy)
            learner.adapt(loss)

            with torch.no_grad():
                t_pred = learner(tx)
                losses.append(self.loss_fn(t_pred, ty).item())
                corrs.append(self._compute_correlation(t_pred, ty))

            if (step + 1) % 5 == 0:
                print(f"Step {step + 1}: MSE={losses[-1]:.5f}, Corr={corrs[-1]:.4f}")

        return {
            'zero_shot': {'loss': z_loss, 'corr': z_corr},
            'few_shot': {'losses': losses, 'corrs': corrs}
        }