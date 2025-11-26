import numpy as np
import torch
from scipy.signal import stft


class FDARadarSimulator:
    def __init__(self, M=4, N=4, fc=10e9, delta_f=30e3, fs=2e6, duration=256e-6,
                 n_range=64, n_angle=64):
        """
        FDA-MIMO Radar Simulator with Range-Angle Domain Processing
        
        参数:
            M: 发射阵元数
            N: 接收阵元数
            fc: 载频 (Hz)
            delta_f: FDA频率步进 (Hz)
            fs: 采样率 (Hz)
            duration: 脉冲持续时间 (s)
            n_range: 距离维FFT点数
            n_angle: 角度维FFT点数 (波束形成分辨率)
        """
        self.M, self.N = M, N
        self.fc, self.delta_f = fc, delta_f
        self.c = 3e8
        self.d = self.c / (2 * self.fc)  # 半波长间距
        self.fs = fs
        self.duration = duration
        self.samples = int(fs * duration)
        
        # Range-Angle Map 分辨率
        self.n_range = n_range
        self.n_angle = n_angle
        
        # 角度扫描范围 (-60° to 60°)
        self.angle_grid = np.linspace(-60, 60, n_angle)
        
        print(f"✅ FDA-MIMO Config: {M}Tx × {N}Rx, Range-Angle Map: {n_range}×{n_angle}")

    def _get_fda_steering_vector(self, theta, r):
        """
        FDA-MIMO 联合发射-接收导向矢量
        
        FDA 特性: 相位同时与角度 θ 和距离 r 相关
        a_t(θ,r) = exp(-j2π(fc + m·Δf)(2r/c - m·d·sinθ/c))
        """
        theta_rad = np.deg2rad(theta)
        m_idx = np.arange(self.M)
        n_idx = np.arange(self.N)
        
        # FDA 发射导向矢量 (距离-角度耦合)
        tau_r = 2 * r / self.c  # 距离时延
        tau_theta = m_idx * self.d * np.sin(theta_rad) / self.c  # 角度时延
        phase_t = -1j * 2 * np.pi * (self.fc + m_idx * self.delta_f) * (tau_r - tau_theta)
        a_t = np.exp(phase_t)
        
        # 接收导向矢量 (仅角度相关)
        phase_r = -1j * 2 * np.pi * self.fc * n_idx * self.d * np.sin(theta_rad) / self.c
        a_r = np.exp(phase_r)
        
        return a_t, a_r

    def _generate_range_angle_map(self, signal_time, theta_true, r_true):
        """
        将时域信号转换到距离-角度域 (Range-Angle Map)
        
        步骤:
        1. 匹配滤波 (距离压缩) -> 得到距离维
        2. 波束形成 (角度扫描) -> 得到角度维
        
        输入: signal_time [MN, T] - 虚拟阵元时域信号
        输出: range_angle_map [n_range, n_angle] - 复数距离-角度图
        """
        mn = self.M * self.N
        
        # 1. 距离压缩 (Range FFT)
        # 对每个虚拟阵元做FFT得到距离profile
        range_profiles = np.fft.fft(signal_time, n=self.n_range, axis=-1)  # [MN, n_range]
        
        # 2. 波束形成 (Beamforming)
        # 对每个角度，计算波束形成输出
        range_angle_map = np.zeros((self.n_range, self.n_angle), dtype=complex)
        
        for i, theta in enumerate(self.angle_grid):
            # 获取该角度的导向矢量
            a_t, a_r = self._get_fda_steering_vector(theta, r_true)
            # 虚拟阵列导向矢量
            a_v = np.kron(a_t, a_r)  # [MN]
            
            # 波束形成: w^H * X (对每个距离单元)
            # range_profiles: [MN, n_range]
            # a_v: [MN]
            bf_output = np.conj(a_v) @ range_profiles  # [n_range]
            range_angle_map[:, i] = bf_output
        
        return range_angle_map

    def _complex_to_tensor(self, data):
        """Convert complex numpy array to [2, H, W] tensor (Real, Imag channels)"""
        real = torch.FloatTensor(data.real)
        imag = torch.FloatTensor(data.imag)
        return torch.stack([real, imag], dim=0)  # [2, H, W]

    def generate_batch(self, batch_size, jamming_type='DFTJ', snr=10, sir=0):
        """
        生成训练批次 - 距离-角度域数据
        
        参数:
            batch_size: 批次大小
            jamming_type: 干扰类型 ('DFTJ', 'ISRJ', 'SRJ', 'SJ')
            snr: Signal-to-Noise Ratio (dB)
            sir: Signal-to-Interference Ratio (dB)
                 SIR = -25dB 表示干扰功率是信号的 316 倍
                 
        Returns:
            X: [B, 2, n_range, n_angle] (干扰污染的距离-角度图)
            Y: [B, 2, n_range, n_angle] (干净的目标距离-角度图)
        """
        X_list, Y_list = [], []
        mn = self.M * self.N

        for _ in range(batch_size):
            # 1. 目标参数
            theta_tgt = np.random.uniform(-50, 50)  # 目标角度
            r_tgt = np.random.uniform(5e3, 15e3)    # 目标距离
            
            # 目标信号功率
            sig_pwr = 10 ** (snr / 10)
            
            # 2. 生成目标时域信号
            a_t_tgt, a_r_tgt = self._get_fda_steering_vector(theta_tgt, r_tgt)
            a_v_tgt = np.kron(a_t_tgt, a_r_tgt)  # 虚拟阵列导向矢量
            
            s_tgt = (np.random.randn(self.samples) + 1j * np.random.randn(self.samples)) * np.sqrt(sig_pwr / 2)
            tgt_signal = np.outer(a_v_tgt, s_tgt)  # [MN, T]

            # 3. 生成干扰信号
            jam_pwr = sig_pwr / (10 ** (sir / 10))  # SIR定义
            
            if jamming_type == 'DFTJ':
                # 分布式假目标干扰 - 主瓣方向相近，但距离不同
                n_false = np.random.randint(3, 8)
                jam_signal = np.zeros_like(tgt_signal)
                for _ in range(n_false):
                    # 干扰角度接近目标（主瓣欺骗）
                    j_theta = theta_tgt + np.random.uniform(-3, 3)
                    # 干扰距离与目标不同（RGPO效果）
                    j_r = r_tgt + np.random.uniform(-3000, 3000)
                    
                    a_t_jam, a_r_jam = self._get_fda_steering_vector(j_theta, j_r)
                    a_v_jam = np.kron(a_t_jam, a_r_jam)
                    
                    s_jam = (np.random.randn(self.samples) + 1j * np.random.randn(self.samples)) * np.sqrt(jam_pwr / n_false / 2)
                    jam_signal += np.outer(a_v_jam, s_jam)

            elif jamming_type == 'ISRJ':
                # 间歇采样转发干扰
                mask = (np.random.rand(self.samples) > 0.5).astype(float)
                jam_signal = tgt_signal * mask * np.sqrt(jam_pwr / sig_pwr)

            elif jamming_type == 'SRJ':
                # 噪声转发干扰
                jam_signal = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(jam_pwr / 2)

            else:  # SJ - 压制式干扰
                jam_signal = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(jam_pwr / 2)

            # 4. 噪声
            noise_pwr = 1.0  # 归一化噪声功率
            noise = (np.random.randn(mn, self.samples) + 1j * np.random.randn(mn, self.samples)) * np.sqrt(noise_pwr / 2)

            # 5. 混合信号
            mixed_signal = tgt_signal + jam_signal + noise  # [MN, T]

            # 6. 转换到距离-角度域
            X_ra = self._generate_range_angle_map(mixed_signal, theta_tgt, r_tgt)  # 干扰图
            Y_ra = self._generate_range_angle_map(tgt_signal, theta_tgt, r_tgt)    # 干净目标图

            # 7. 归一化 (按混合信号最大幅度)
            scale = np.abs(X_ra).max() + 1e-8
            X_ra = X_ra / scale
            Y_ra = Y_ra / scale

            X_list.append(self._complex_to_tensor(X_ra))
            Y_list.append(self._complex_to_tensor(Y_ra))

        return torch.stack(X_list), torch.stack(Y_list)