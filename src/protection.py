import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

fs = 1000
fc = 10
T = 1 / fs
RC = 1 / (2 * np.pi * fc)

alpha = T / (RC + T)

t = np.arange(0, 1, T)
f1 = 5
f2 = 100

x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
x += 0.3 * np.random.randn(len(t))

y = np.zeros_like(x)
y[0] = x[0]

for n in range(1, len(x)):
    y[n] = alpha * x[n] + (1 - alpha) * y[n - 1]

b = [alpha]
a = [1, -(1 - alpha)]

w, h = freqz(b, a, worN=8000)
freq = w * fs / (2 * np.pi)

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Input signal (noisy)', alpha=0.6)
plt.plot(t, y, label='Low-pass filtered output', linewidth=2)
plt.title('Input and Output of First Order Digital Low-pass Filter')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(freq, 20 * np.log10(abs(h)), 'b')
plt.title('Frequency Response (Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.xlim(0, fs / 2)

plt.subplot(3, 1, 3)
plt.plot(freq, np.angle(h), 'r')
plt.title('Frequency Response (Phase)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.xlim(0, fs / 2)

plt.tight_layout()
plt.show()

roots = np.roots(a)
print("Denominator roots (stability check):", roots)
if np.all(np.abs(roots) < 1):
    print("Filter is stable.")
else:
    print("Filter is unstable.")
