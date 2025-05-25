import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz

# پارامترها
fs = 1000          # فرکانس نمونه برداری (Hz)
cutoff = 50        # فرکانس قطع (Hz)
numtaps = 51       # تعداد ضرایب فیلتر (طول فیلتر)

# طراحی فیلتر FIR پایین گذر با پنجره هان
fir_coeff = firwin(numtaps, cutoff, fs=fs, window='hann')

# ایجاد سیگنال نمونه (ترکیب فرکانس‌های مختلف)
t = np.arange(0, 1.0, 1/fs)
x = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*200*t) + 0.2*np.random.randn(len(t))

# اعمال فیلتر
y = lfilter(fir_coeff, 1.0, x)

# نمایش پاسخ فرکانسی فیلتر
w, h = freqz(fir_coeff, worN=8000)
freq = w * fs / (2 * np.pi)

# رسم نتایج
plt.figure(figsize=(14,10))

plt.subplot(3,1,1)
plt.plot(t, x, label='Input Signal (Noisy)')
plt.title('Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, y, label='Filtered Signal (FIR Low-pass)')
plt.title('Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(freq, 20 * np.log10(abs(h)), 'b')
plt.title('Frequency Response of FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.xlim(0, fs/2)

plt.tight_layout()
plt.show()
