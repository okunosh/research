import numpy as np
import matplotlib.pyplot as plt

# スケーリング係数
scale_factor = 6

# スケーリングされた関数
def scaled_func(t):
    t_scaled = t / scale_factor - 2
    sec = 16 / (np.exp(t_scaled) + np.exp(-t_scaled))**2
    y = 3 + 3 * np.exp(t_scaled * 1/6) * sec
    return y

# 周期関数にするために正弦項を追加
def periodic_func(t):
    period = 24  # 周期を24時間に設定
    sin_term = np.sin(2 * np.pi * (t-20) / period)
    return scaled_func(t) - sin_term

# プロット
t_values = np.linspace(0, 48, 1000)  # 時間tを0から48時間までプロット
y_values = scaled_func(t_values)

#y = np.where(t_values>20, periodic_func(t_values), scaled_func(t_values))

plt.plot(t_values, y_values, label='Periodic Function with Period 24 hours')
plt.axvline(x=24, color='r', linestyle='--', label='t=24 hours')
plt.grid()
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Function Value')
plt.title('Periodic Function with Period of 24 Hours')
plt.show(block=False)
