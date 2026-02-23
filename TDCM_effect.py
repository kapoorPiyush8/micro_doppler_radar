import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
fs = 10000  # Sampling frequency (Hz)
T = 1       # Duration (seconds)
t = np.linspace(0, T, fs * T, endpoint=False)

fc = 1000   # Incident Carrier Frequency (Hz)
fm = 100    # Switching Frequency (Hz) 


carrier_signal = np.exp(1j * 2 * np.pi * fc * t)

# TDCM modulation sequence switches between +1 (0 deg phase) and -1 (180 deg phase) at rate fm 

modulation_sequence = np.sign(np.sin(2 * np.pi * fm * t))

#reflected signal (carrier * modulation sequence)

reflected_signal = carrier_signal * modulation_sequence

# --- (FFT) ---

N = len(t)
xf = np.fft.fftfreq(N, 1 / fs)
yf_carrier = np.fft.fft(carrier_signal) / N
yf_reflected = np.fft.fft(reflected_signal) / N


plt.figure(figsize=(10, 6))

# positive frequencies centered around carrier
mask = (xf > fc - 3*fm) & (xf < fc + 3*fm)

plt.plot(xf[mask], np.abs(yf_carrier[mask]), 'b--', alpha=0.6, label='Original Incident Carrier (fc)')
plt.plot(xf[mask], np.abs(yf_reflected[mask]), 'r-', linewidth=2, label='TDCM Reflected Signal')

plt.title("TDCM Effect: Generation of Harmonic Sidebands")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Normalized)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.annotate(f'Carrier ({fc} Hz)\nSuppressed', xy=(fc, 0.05), xytext=(fc, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), ha='center')
plt.annotate(f'Upper Harmonic\n(fc + fm: {fc+fm} Hz)', xy=(fc+fm, 0.6), xytext=(fc+fm+50, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
plt.annotate(f'Lower Harmonic\n(fc - fm: {fc-fm} Hz)', xy=(fc-fm, 0.6), xytext=(fc-fm-150, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
# plt.savefig("tdcm_spectrum_output.png") #  save  image
plt.show()
