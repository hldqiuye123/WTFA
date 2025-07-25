import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 生成信号
# ----------------
# 设置采样率（Hz）和时间（秒）
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# 设置信号频率（Hz）
f = 5

# 生成正弦波信号（这将是我们的实部）
x_real = np.sin(2 * np.pi * f * t)

# 执行Hilbert变换
# -----------------
# 使用Scipy的hilbert函数来计算Hilbert变换
x_hilbert = hilbert(x_real)

# 提取Hilbert变换的虚部
x_imag = x_hilbert.imag

# 计算相位差
# ----------------
# 计算实部和虚部的相位
phase_real = np.angle(x_real + 1j * np.zeros_like(x_real))
phase_imag = np.angle(np.zeros_like(x_imag) + 1j * x_imag)

# 计算相位差，并进行解包以避免相位跳跃
phase_diff = np.unwrap(phase_imag - phase_real)

# 绘图
# ------------
# 绘制实部、虚部和相位差
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, x_real)
plt.title("Real Part")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(t, x_imag)
plt.title("Imaginary Part (Hilbert Transform)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(t, phase_diff)
plt.title("Phase Difference")
plt.xlabel("Time [s]")
plt.ylabel("Phase [radians]")

plt.tight_layout()
plt.show()

# 输出平均相位差
# ---------------------
mean_phase_diff = np.mean(phase_diff)
print(f"Mean Phase Difference (in radians): {mean_phase_diff}")
print(f"Mean Phase Difference (in degrees): {np.degrees(mean_phase_diff)}")
