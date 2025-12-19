import numpy as np

def add_gaussian_noise(signal, snr_db, random_seed):
    np.random.seed(random_seed)
    signal_power = np.var(signal)
    noise_power = signal_power / (10 ** (snr_db / 10))  #计算噪声功率
    noise = np.random.normal(0, np.sqrt(noise_power), size=(signal.shape[0], signal.shape[1], signal.shape[2]))  #生成高斯分布噪声点
    return signal + noise

def add_pulse_noise(signal, random_seed, probability=0.3, amplitude=0.05):
    np.random.seed(random_seed)
    mask = np.random.random((signal.shape[0], signal.shape[1], signal.shape[2])) < probability    #随机添加冲击噪声
    noise = amplitude * np.random.randn(signal.shape[0], signal.shape[1], signal.shape[2]) * mask
    return signal + noise

def add_mixed_noise(signal, snr_db=20, impulse_prob=0.3, impulse_amp=0.05, random_seed=42):
    signal = add_gaussian_noise(signal, snr_db, random_seed)
    signal = add_pulse_noise(signal, random_seed, impulse_prob, impulse_amp)
    return signal