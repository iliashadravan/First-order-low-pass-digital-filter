import numpy as np
import matplotlib.pyplot as plt

# پارامترهای نمونه‌برداری و سیگنال
fs = 1000  # فرکانس نمونه‌برداری
t = np.arange(0, 1, 1 / fs)  # بردار زمان 1 ثانیه

# سیگنال ترکیبی: 5 هرتز و 150 هرتز و نویز سفید
f_low = 5
f_high = 150
signal = np.sin(2 * np.pi * f_low * t) + 0.5 * np.sin(2 * np.pi * f_high * t)
signal += 0.2 * np.random.randn(len(t))

# تبدیل فوریه سریع
signal_fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(t), 1 / fs)

# طراحی فیلتر پایین‌گذر ایده‌آل: گذر فرکانس تا 20 هرتز
cutoff = 20
ideal_filter = np.abs(freqs) < cutoff

# اعمال فیلتر در حوزه فرکانس
filtered_fft = signal_fft * ideal_filter

# تبدیل معکوس فوریه
filtered_signal = np.fft.ifft(filtered_fft).real

# رسم سیگنال‌ها و پاسخ فرکانسی
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(t, signal, label='Original Signal (Noisy)')
plt.title('Original Signal in Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(freqs[:len(freqs) // 2], np.abs(signal_fft)[:len(freqs) // 2], label='Original Spectrum')
plt.plot(freqs[:len(freqs) // 2], np.abs(filtered_fft)[:len(freqs) // 2], label='Filtered Spectrum', linestyle='--')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, label='Filtered Signal (Ideal LPF)')
plt.title('Filtered Signal in Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
