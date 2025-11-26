import numpy as np
import torch
from scipy.signal import stft


class FDARadarSimulator:
    def __init__(self, M=4, N=4, fc=10e9, delta_f=30e3, fs=2e6, duration=256e-6):
        """
        FDA-MIMO Radar Simulator - 按照论文设置
        
        论文配置:
        - M=4 发射阵元, N=4 接收阵元 -> 16 虚拟通道
        - 使用 STFT 转换到时频域
        
        参数:
            M: 发射阵元数
            N: 接收阵元数
            fc: 载频 (Hz)
            delta_f: FDA频率步进 (Hz)
            fs: 采样率 (Hz)
            duration: 脉冲持续时间 (s)
        """
        self.M, self.N = M, N
        self.fc, self.delta_f = fc, delta_f
        self.c = 3e8
        self.d = self.c / (2 * self.fc)  # 半波长间距
        self.fs = fs
        self.duration = duration
        self.samples = int(fs * duration)
        
        print(f"✅ FDA-MIMO Config: {M}Tx × {N}Rx = {M*N} Virtual Channels")
        print(f"   Samples: {self.samples}, STFT Mode")

    def _get_fda_steering_vector(self, theta, r):
        """
        FDA-MIMO 联合导向矢量
        
        FDA 特性: 相位同时与角度 θ 和距离 r 相关
        这是区分主瓣欺骗干扰的物理基础
        """
        theta_rad = np.deg2rad(theta)
        m_idx = np.arange(self.M)
        n_idx = np.arange(self.N)
        
        # FDA 发射导向矢量 (距离-角度耦合)
        tau_r = 2 * r / self.c
        tau_theta = m_idx * self.d * np.sin(theta_rad) / self.c
        phase_t = -1j * 2 * np.pi * (self.fc + m_idx * self.delta_f) * (tau_r - tau_theta)
        a_t = np.exp(phase_t)
        
        # 接收导向矢量
        phase_r = -1j * 2 * np.pi * self.fc * n_idx * self.d * np.sin(theta_rad) / self.c
        a_r = np.exp(phase_r)
        
        return np.kron(a_t, a_r)  # [MN] 虚拟阵列导向矢量

    def _apply_stft(self, signal_matrix):
        """
        STFT 变换 - 论文公式 (3)
        将时域信号转换到时频域
        
        Input: [MN, T] -> Output: [MN, F, T_stft] (Complex)
        """
        # 使用较小的窗口以获得更好的时间分辨率
        nperseg = min(64, self.samples // 4)
        noverlap = nperseg // 2
        
        f, t, Zxx = stft(signal_matrix, fs=self.fs, nperseg=nperseg, 
                        noverlap=noverlap, axis=-1)
        return Zxx

    def _complex_to_tensor(self, data):
        """
        转换复数数据到张量格式
        论文格式: [2*C, H, W] - 前半通道实部，后半通道虚部
        """
        real = torch.FloatTensor(data.real)
        imag = torch.FloatTensor(data.imag)
        return torch.cat([real, imag], dim=0)

    def generate_batch(self, batch_size, jamming_type='DFTJ', snr=10, sir=0):
        """
        生成训练批次 - STFT 时频域数据
        
        参数:
            batch_size: 批次大小
            jamming_type: 干扰类型
                - 'DFTJ': 密集假目标干扰 (Dense False Target Jamming)
                - 'SRJ':  切片重构干扰 (Slice Reconstruction Jamming)
                - 'ISRJ': 间歇采样转发干扰 (Intermittent Sampling Retransmission)
                - 'SJ':   压制性干扰 (Suppressive Jamming)
            snr: 信噪比 (dB)
            sir: 信干比 (dB), 负值表示干扰更强
                 
        Returns:
            X: [B, 2*MN, F, T] (受干扰信号的STFT)
            Y: [B, 2*MN, F, T] (纯净目标信号的STFT)
        """
        X_list, Y_list = [], []
        mn = self.M * self.N

        for _ in range(batch_size):
            # 1. 目标参数 (随机生成)
            theta_tgt = np.random.uniform(-50, 50)  # 目标角度
            r_tgt = np.random.uniform(5e3, 15e3)    # 目标距离
            
            # 目标信号功率
            sig_pwr = 10 ** (snr / 10)
            
            # 2. 生成目标时域信号
            a_tgt = self._get_fda_steering_vector(theta_tgt, r_tgt)
            # 目标波形 (复高斯)
            s_tgt = (np.random.randn(self.samples) + 1j * np.random.randn(self.samples)) * np.sqrt(sig_pwr / 2)
            tgt_signal = np.outer(a_tgt, s_tgt)  # [MN, T]

            # 3. 生成干扰信号
            jam_pwr = sig_pwr / (10 ** (sir / 10))  # SIR 定义
            
            if jamming_type == 'DFTJ':
                # 密集假目标干扰 - 多个主瓣方向的假目标
                n_false = np.random.randint(3, 8)
                jam_signal = np.zeros_like(tgt_signal)
                for k in range(n_false):
                    # 干扰角度接近目标 (主瓣欺骗)
                    j_theta = theta_tgt + np.random.uniform(-5, 5)
                    # 干扰距离与目标不同 (RGPO效果)
                    j_r = r_tgt + np.random.uniform(-5000, 5000)
                    
                    a_jam = self._get_fda_steering_vector(j_theta, j_r)
                    # 每个假目标的功率
                    s_jam = (np.random.randn(self.samples) + 1j * np.random.randn(self.samples)) * np.sqrt(jam_pwr / n_false / 2)
                    jam_signal += np.outer(a_jam, s_jam)

            elif jamming_type == 'ISRJ':
                # 间歇采样转发干扰 - 周期性采样并转发
                duty_cycle = np.random.uniform(0.3, 0.7)
                period = np.random.randint(20, 50)
                mask = np.zeros(self.samples)
                for i in range(0, self.samples, period):
                    end = min(i + int(period * duty_cycle), self.samples)
                    mask[i:end] = 1
                
                # 转发时添加延时和频移
                delay = np.random.randint(5, 20)
                shifted_tgt = np.roll(tgt_signal, delay, axis=-1)
                jam_signal = shifted_tgt * mask * np.sqrt(jam_pwr / sig_pwr)

            elif jamming_type == 'SRJ':
                # 切片重构干扰
                n_slices = np.random.randint(3, 6)
                slice_len = self.samples // n_slices
                jam_signal = np.zeros_like(tgt_signal)
                
                for i in range(n_slices):
                    start = i * slice_len
                    end = min((i + 1) * slice_len, self.samples)
                    # 每个切片添加随机相位
                    phase_shift = np.exp(1j * np.random.uniform(0, 2*np.pi))
                    jam_signal[:, start:end] = tgt_signal[:, start:end] * phase_shift * np.sqrt(jam_pwr / sig_pwr)

            else:  # SJ - 压制性干扰 (宽带噪声)
                jam_signal = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(jam_pwr / 2)

            # 4. 噪声
            noise_pwr = 1.0  # 归一化噪声功率
            noise = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(noise_pwr / 2)

            # 5. 混合信号
            mixed_signal = tgt_signal + jam_signal + noise  # [MN, T]

            # 6. STFT 变换到时频域
            X_stft = self._apply_stft(mixed_signal)   # [MN, F, T_stft]
            Y_stft = self._apply_stft(tgt_signal)     # [MN, F, T_stft]

            # 7. 归一化 (按混合信号最大幅度)
            scale = np.abs(X_stft).max() + 1e-8
            X_stft = X_stft / scale
            Y_stft = Y_stft / scale

            X_list.append(self._complex_to_tensor(X_stft))
            Y_list.append(self._complex_to_tensor(Y_stft))

        return torch.stack(X_list), torch.stack(Y_list)