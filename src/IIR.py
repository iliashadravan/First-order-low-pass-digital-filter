import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

# پارامترها
fs = 1000        # فرکانس نمونه‌برداری (Hz)
cutoff = 50      # فرکانس قطع (Hz)
order = 2        # مرتبه فیلتر باترورث

# طراحی فیلتر باترورث پایین‌گذر
b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)

# تولید سیگنال نمونه: ترکیب دو سینوسی + نویز
t = np.arange(0, 1.0, 1/fs)
x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.random.randn(len(t))

# اعمال فیلتر روی سیگنال
y = lfilter(b, a, x)

# رسم پاسخ فرکانسی فیلتر
w, h = freqz(b, a, worN=8000)
freq = w * fs / (2 * np.pi)

# رسم سیگنال‌ها و پاسخ فرکانسی
plt.figure(figsize=(14,10))

plt.subplot(3,1,1)
plt.plot(t, x, label='Input Signal (Noisy)')
plt.title('Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, y, label='Filtered Signal (Butterworth IIR)')
plt.title('Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(freq, 20 * np.log10(abs(h)), 'b')
plt.title('Frequency Response of Butterworth Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.xlim(0, fs/2)

plt.tight_layout()
plt.show()
