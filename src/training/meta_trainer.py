# meta_trainer.py (完整版)
import torch
import learn2learn as l2l
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class ComplexHybridLoss(torch.nn.Module):
    def __init__(self, lambda1=0.3, lambda2=0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        ch = pred.shape[1] // 2
        pred_r, pred_i = pred[:, :ch], pred[:, ch:]
        tgt_r, tgt_i = target[:, :ch], target[:, ch:]

        pred_amp = torch.sqrt(pred_r ** 2 + pred_i ** 2 + 1e-8)
        tgt_amp = torch.sqrt(tgt_r ** 2 + tgt_i ** 2 + 1e-8)
        L_amp = self.mse(pred_amp, tgt_amp)

        dot = pred_r * tgt_r + pred_i * tgt_i
        L_phase = 1 - (dot / (pred_amp * tgt_amp + 1e-8)).mean()

        L_mse = self.mse(pred, target)

        return L_amp + self.lambda1 * L_phase + self.lambda2 * L_mse


class MetaTrainer:
    def __init__(self, model, simulator, device='cuda'):
        self.device = device
        self.simulator = simulator
        self.model = model.to(device)

        self.maml = l2l.algorithms.MAML(self.model, lr=0.01, first_order=True)
        self.optimizer = optim.Adam(self.maml.parameters(), lr=0.0005)
        self.loss_fn = ComplexHybridLoss(lambda1=0.3, lambda2=0.5)

        # ✅ 对标论文的课程学习
        self.curriculum = {
            'easy': [
                {'type': 'SJ', 'snr': -5, 'jnr': -10},
                {'type': 'DFTJ', 'snr': -5, 'jnr': -10},
                {'type': 'ISRJ', 'snr': -5, 'jnr': -10},
            ],
            'medium': [
                {'type': 'SJ', 'snr': -10, 'jnr': -15},
                {'type': 'DFTJ', 'snr': -10, 'jnr': -15},
                {'type': 'ISRJ', 'snr': -10, 'jnr': -15},
                {'type': 'SRJ', 'snr': -10, 'jnr': -15},
            ],
            'hard': [
                {'type': 'SJ', 'snr': -10, 'jnr': -25},  # ✅ 论文条件
                {'type': 'DFTJ', 'snr': -10, 'jnr': -25},  # ✅ 论文条件
                {'type': 'ISRJ', 'snr': -10, 'jnr': -22},
                {'type': 'SRJ', 'snr': -10, 'jnr': -20},
            ],
            'extreme': [  # ✅ 新增超难任务
                {'type': 'SJ', 'snr': -10, 'jnr': -30},
                {'type': 'DFTJ', 'snr': -10, 'jnr': -28},
            ]
        }

        print(f"✅ Meta Trainer Initialized")
        print(f"   - Target: SNR=-10dB, SIR=-25dB (Paper Standard)")
        print(f"   - Curriculum: Easy → Medium → Hard → Extreme")

    def _sample_task(self, epoch, total_epochs):
        progress = epoch / total_epochs

        if progress < 0.25:
            stage = 'easy'
        elif progress < 0.5:
            stage = 'medium'
        elif progress < 0.8:
            stage = 'hard'
        else:
            stage = 'extreme'  # ✅ 后20%训练极端条件

        tasks = self.curriculum[stage]
        idx = np.random.randint(len(tasks))
        return tasks[idx], stage

    def train_loop(self, epochs=800, tasks_per_batch=2, k_shot=5):
        loss_history = []
        pbar = tqdm(range(epochs), desc="Meta-Training")

        for epoch in pbar:
            self.optimizer.zero_grad()
            meta_loss = 0.0

            for task_idx in range(tasks_per_batch):
                task, stage = self._sample_task(epoch, epochs)
                learner = self.maml.clone()

                sx, sy = self.simulator.generate_batch(
                    k_shot, task['type'], task['snr'], task['jnr']
                )
                sx, sy = sx.to(self.device), sy.to(self.device)

                # Inner Loop
                for _ in range(10):
                    pred = learner(sx)
                    loss = self.loss_fn(pred, sy)
                    learner.adapt(loss)

                qx, qy = self.simulator.generate_batch(
                    k_shot, task['type'], task['snr'], task['jnr']
                )
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
                pbar.set_postfix({
                    'Loss': f'{meta_loss.item():.5f}',
                    'Stage': stage
                })

        return loss_history

    def _compute_correlation(self, pred, target):
        p = pred.reshape(pred.size(0), -1)
        t = target.reshape(target.size(0), -1)

        p = (p - p.mean(dim=1, keepdim=True)) / (p.std(dim=1, keepdim=True) + 1e-8)
        t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-8)

        corr = (p * t).sum(dim=1) / p.shape[1]
        return corr.mean().item()

    def test_adaptation(self, target_jamming='SJ', snr=-10, jnr=-25, k_shots=10):
        print(f"\n{'=' * 50}")
        print(f"🧪 Testing: {target_jamming} | SNR={snr}dB, SIR={jnr}dB, K={k_shots}")
        print(f"{'=' * 50}")

        sx, sy = self.simulator.generate_batch(k_shots, target_jamming, snr, jnr)
        tx, ty = self.simulator.generate_batch(50, target_jamming, snr, jnr)
        sx, sy = sx.to(self.device), sy.to(self.device)
        tx, ty = tx.to(self.device), ty.to(self.device)

        learner = self.maml.clone()

        with torch.no_grad():
            z_pred = learner(tx)
            z_loss = self.loss_fn(z_pred, ty).item()
            z_corr = self._compute_correlation(z_pred, ty)

        print(f"Zero-Shot: Loss={z_loss:.5f}, Corr={z_corr:.4f}")

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
                print(f"Step {step + 1}: Loss={losses[-1]:.5f}, Corr={corrs[-1]:.4f}")

        improvement = (corrs[-1] - z_corr) / abs(z_corr + 1e-8) * 100
        print(f"\n📈 Improvement: {improvement:+.1f}% | {corrs[-1] / z_corr:.1f}x")

        return {
            'zero_shot': {'loss': z_loss, 'corr': z_corr},
            'few_shot': {'losses': losses, 'corrs': corrs}
        }
