import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Sampling parameters
fs = 1000  # Sampling frequency in Hz
fc = 10    # Cutoff frequency in Hz

# Time constant and alpha coefficient for the filter
T = 1 / fs
RC = 1 / (2 * np.pi * fc)
alpha = T / (RC + T)

# Time vector for 1 second
t = np.arange(0, 1, T)

# Create a noisy input signal: low freq + high freq + white noise
f1 = 5     # Low frequency component
f2 = 100   # High frequency component
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
x += 0.3 * np.random.randn(len(t))  # Adding white noise

# Initialize output signal
y = np.zeros_like(x)
y[0] = x[0]

# Apply first-order low-pass filter using difference equation
for n in range(1, len(x)):
    y[n] = alpha * x[n] + (1 - alpha) * y[n - 1]

# Filter coefficients for frequency response
b = [alpha]
a = [1, -(1 - alpha)]

# Frequency response
w, h = freqz(b, a, worN=8000)
freq = w * fs / (2 * np.pi)

# Plot input and output signals
plt.figure(figsize=(14, 12))

plt.subplot(4, 1, 1)
plt.plot(t, x, label='Input signal (noisy)', alpha=0.6)
plt.plot(t, y, label='Filtered output', linewidth=2)
plt.title('Time-Domain Signals')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Plot magnitude response
plt.subplot(4, 1, 2)
plt.plot(freq, 20 * np.log10(abs(h)), 'b')
plt.title('Frequency Response (Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.xlim(0, fs / 2)

# Plot phase response
plt.subplot(4, 1, 3)
plt.plot(freq, np.angle(h), 'r')
plt.title('Frequency Response (Phase)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.xlim(0, fs / 2)

# Plot impulse response of the filter
impulse = np.zeros(100)
impulse[0] = 1
response = np.zeros_like(impulse)
response[0] = impulse[0]
for n in range(1, len(impulse)):
    response[n] = alpha * impulse[n] + (1 - alpha) * response[n - 1]

plt.subplot(4, 1, 4)
plt.stem(response, use_line_collection=True)
plt.title('Impulse Response of the Filter')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

# Stability check
roots = np.roots(a)
print("Denominator roots (stability check):", roots)
if np.all(np.abs(roots) < 1):
    print("Filter is stable.")
else:
    print("Filter is unstable.")
