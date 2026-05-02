import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks
from scipy.io import wavfile

def analyze_music(filename: str):
    fs, x = wavfile.read(filename)
    if x.ndim > 1:
        x = x[:, 0]
    nperseg = 2048
    noverlap = 1536
    f, t, Zxx = stft(x, fs, nperseg=nperseg, noverlap=noverlap, window="hann")


    freq_mask = (f >= 250) & (f <= 1000)
    f = f[freq_mask]
    Zxx = Zxx[freq_mask, :]

    magnitude = np.abs(Zxx)

    plt.figure("Spectrogram", figsize=(12,6))
    plt.pcolormesh(t, f, 20*np.log10(magnitude+1e-6), shading='gouraud', cmap='magma')
    plt.colorbar(label='Amplitude [dB]')
    plt.ylim(250, 1000)
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram (Single Notes)")
    plt.yticks(np.arange(250, 1001, 50))  
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_music("6413-15.wav")