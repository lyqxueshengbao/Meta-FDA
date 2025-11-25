import numpy as np
import torch
from scipy.signal import stft


class FDARadarSimulator:
    def __init__(self, M=4, N=4, fc=10e9, delta_f=30e3, fs=2e6, duration=256e-6):
        """
        Updated Config:
        - M=4, N=4 (16 channels to reduce compute load)
        - fs=2e6 (Higher sampling rate)
        - duration=256e-6 (512 samples for better STFT resolution)
        """
        self.M, self.N = M, N
        self.fc, self.delta_f = fc, delta_f
        self.c = 3e8
        self.d = self.c / (2 * self.fc)
        self.fs = fs
        self.duration = duration
        self.samples = int(fs * duration)  # 512 samples
        print(f"✅ Radar Config: {M}x{N} Elements, {self.samples} Samples")

    def _get_steering_vector(self, theta, r):
        theta_rad = np.deg2rad(theta)
        m_idx = np.arange(self.M)
        n_idx = np.arange(self.N)

        phase_t = -1j * 2 * np.pi * (self.fc + m_idx * self.delta_f) * \
                  (2 * r / self.c - m_idx * self.d * np.sin(theta_rad) / self.c)
        a_t = np.exp(phase_t)

        phase_r = -1j * 2 * np.pi * self.fc * (n_idx * self.d * np.sin(theta_rad) / self.c)
        a_r = np.exp(phase_r)

        return np.kron(a_t, a_r)

    def _apply_stft(self, signal_matrix):
        """
        Input: [MN, T] -> Output: [MN, F, T_stft] (Complex)
        """
        # nperseg=64 -> F=33, T_stft=17 (Approx) - Better resolution
        f, t, Zxx = stft(signal_matrix, fs=self.fs, nperseg=64, noverlap=32, axis=-1)
        return Zxx

    def _complex_to_tensor(self, data):
        """Convert complex numpy array to [2*C, H, W] tensor"""
        real = torch.FloatTensor(data.real)
        imag = torch.FloatTensor(data.imag)
        return torch.cat([real, imag], dim=0)

    def generate_batch(self, batch_size, jamming_type='DFTJ', snr=10, jnr=0):
        """
        Returns:
            X: [B, 2*MN, F, T] (Mixed Signal)
            Y: [B, 2*MN, F, T] (Clean Target Signal - Full Channel)
        """
        X_list, Y_list = [], []
        mn = self.M * self.N

        for _ in range(batch_size):
            # 1. Target
            theta = np.random.uniform(-60, 60)
            r = np.random.uniform(5e3, 15e3)
            a_tgt = self._get_steering_vector(theta, r)

            sig_pwr = 10 ** (snr / 10)
            s_tgt = (np.random.randn(self.samples) + 1j * np.random.randn(self.samples)) * np.sqrt(sig_pwr / 2)
            tgt_signal = np.outer(a_tgt, s_tgt)  # [MN, T]

            # 2. Jamming
            jam_pwr = 10 ** (jnr / 10)

            if jamming_type == 'DFTJ':
                n_false = np.random.randint(3, 8)
                jam_signal = np.zeros_like(tgt_signal)
                for _ in range(n_false):
                    j_theta = theta + np.random.uniform(-2, 2)
                    j_r = r + np.random.uniform(-2000, 2000)
                    a_jam = self._get_steering_vector(j_theta, j_r)
                    s_jam = (np.random.randn(self.samples) + 1j * np.random.randn(self.samples)) * np.sqrt(
                        jam_pwr / n_false)
                    jam_signal += np.outer(a_jam, s_jam)

            elif jamming_type == 'ISRJ':
                mask = (np.random.rand(self.samples) > 0.5).astype(float)
                jam_signal = tgt_signal * mask * np.sqrt(jam_pwr)

            elif jamming_type == 'SRJ':
                jam_signal = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(
                    jam_pwr * 0.5)

            else:  # SJ
                jam_signal = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(
                    jam_pwr)

            # 3. Noise
            noise = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(
                0.1)  # Lower noise floor

            # 4. Mix
            X_time = tgt_signal + jam_signal + noise
            Y_time = tgt_signal  # [MN, T] - Full channel reference

            # 5. STFT & Tensor Conversion
            X_stft = self._apply_stft(X_time)
            Y_stft = self._apply_stft(Y_time)

            X_list.append(self._complex_to_tensor(X_stft))
            Y_list.append(self._complex_to_tensor(Y_stft))

        return torch.stack(X_list), torch.stack(Y_list)