import numpy as np
import matplotlib.pyplot as plt

# ایجاد سیگنال نمونه: ترکیب سینوسی و نویز
fs = 500                # فرکانس نمونه‌برداری
t = np.arange(0, 2, 1/fs)  # 2 ثانیه
f_signal = 5            # فرکانس سیگنال اصلی
x = np.sin(2 * np.pi * f_signal * t) + 0.5 * np.random.randn(len(t))  # سیگنال با نویز

# فیلتر میانگین متحرک
window_size = 20        # اندازه پنجره فیلتر
def moving_average_filter(signal, window):
    filtered = np.convolve(signal, np.ones(window)/window, mode='same')
    return filtered

y = moving_average_filter(x, window_size)

# رسم سیگنال‌ها
plt.figure(figsize=(12,8))
plt.plot(t, x, label='Input Signal (Noisy)', alpha=0.6)
plt.plot(t, y, label='Filtered Signal (Moving Average)', linewidth=2)
plt.title('Moving Average Low-pass Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
