import scipy.io.wavfile as wavfile
from statsmodels.tsa.stattools import acf
import numpy as np
import pyreaper
from scipy.signal.windows import triang
import matplotlib.pyplot as plt


def read_file(path):
    fs, x =wavfile.read(path)
    x = x.astype(np.float32)
    x /= np.max(np.abs(x))
    return fs, x


def mono_wav(fs, x, new_path):
    x_mono = x[:, 0]
    wavfile.write(new_path, fs, x_mono)
    return x_mono


def my_acf(x,m):
    n = len(x)
    R=0
    mean= np.mean(x)
    for i in range(0, n-m):
        R+=(x[i]-mean)*(x[i+m]-mean)
    return R/((n-m)*np.var(x))


def check_my_acf(x, n):
    x=x[:n]
    acf_res = acf(x, adjusted=True,  nlags=5000)
    my_acf_res = [my_acf(x, m) for m in range(0, 5001)]

    print("Библиотечная АКФ:", acf_res[:15])
    print("Моя АКФ:", my_acf_res[:15])
    acf_res = acf(x, adjusted=True,  nlags=5000)
    lags = np.arange(0, len(acf_res))
    plt.figure(figsize=(10, 4))
    plt.plot(lags, acf_res)
    plt.xlabel("Кол-во отсчетов сдвига")
    plt.ylabel("АКФ")
    plt.title("Автокорреляционная функция сигнала")
    plt.grid(True)
    plt.show()
    return acf_res, my_acf_res


def f_main(x, fs, f_min=50, f_max=500):
    m_min = int(fs / f_max)
    m_max = int(fs / f_min)

    acf_full = acf(x, adjusted=True, nlags=m_max)
    m_s = np.arange(m_min, m_max + 1)
    acf_vals = acf_full[m_s]

    peak_idx = 10 + np.argmax(acf_vals[10:])
    peak_m = m_s[peak_idx]
    f0 = fs / peak_m
    return f0


def plot_spectrum(frequencies, spectrum):
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, spectrum)
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.title("Амплитудный спектр сигнала")
    plt.grid(True)
    plt.show()


def my_exp(fs, f, len):
    omega=2*np.pi*f/fs
    return np.exp(-1j*omega*np.arange(len))


def my_dtft(x, fs, f):
    if hasattr(f, '__iter__'):
        result = np.empty(len(f))
        for i, value in enumerate(f):
            result[i] = np.abs(np.dot(x, my_exp(fs, value, x.shape[0])))
        return result
    else:
        result = np.abs(np.dot(x, my_exp(fs, f, x.shape[0])))
        return result
    

def analyze_reaper(x, fs):
    n = len(x)         
    t = np.linspace(0, (n-1)/fs, n)
    int16_info = np.iinfo(np.int16)
    x = x * min(int16_info.min, int16_info.max)
    x = x.astype(np.int16)
    pm_times, pm, f_times, f, _ = pyreaper.reaper(x, fs)
    plt.figure('[Reaper] Pitch Marks')
    plt.plot(t, x)
    plt.scatter(pm_times[pm == 1], x[(pm_times * fs).astype(int)][pm == 1], marker='x', color='red')
    plt.figure('[Reaper] Fundamental Frequency')
    plt.plot(f_times, f)
    print('Average fundamental frequency:', np.mean(f[f != -1]))
    plt.show()

def psola(x, fs, k):
    int16_info = np.iinfo(np.int16)
    x = x * min(int16_info.min, int16_info.max)
    x = x.astype(np.int16)

    pm_times, pm, f_times, f, _ = pyreaper.reaper(x, fs)
    centers = pm_times[pm == 1]
    centers_idx = (centers * fs).astype(int)
    centers_idx = centers_idx[(centers_idx >= 0) & (centers_idx < len(x))]
    
    T = int(np.mean(np.diff(centers_idx)))
    window = triang(2*T)

    segments = []
    for c in centers_idx:
        start = max(0, c - T)
        end = min(len(x), c + T)
        seg = x[start:end].copy()
        
        if len(seg) > T//2:  
            if len(seg) >= len(window):
                seg = seg[:len(window)] * window
            else:
                seg = seg * window[:len(seg)]
            segments.append(seg)
    new_T =  int(T / abs(k))
 
    out_len = len(centers_idx) * new_T
    y = np.zeros(out_len)
    w = np.zeros(out_len)
    
    pos = 0
    for seg in segments:
        end = min(pos + len(seg), out_len)
        y[pos:end] += seg[:end-pos]
        w[pos:end] += 1
        pos += new_T
        if pos >= out_len:
            break
    mask = w > 0
    y[mask] /= w[mask]
    max_abs = np.max(np.abs(y))
    y = y / max_abs
    return y


if __name__ == "__main__":

#первое задание 
    fs, x=read_file("speech.wav")

    #x=mono_wav(fs, x_two, "speech_2.wav")
    acf_res, my_acf_res=check_my_acf(x, 5000)
    f0=f_main(x, fs)
    print(f"Основная частота: {f0}")

# #второе задание
    freqs = np.arange(40, 501, 1)
    spectrum = my_dtft(x, fs, freqs)
    plot_spectrum(freqs, spectrum)
    idx_max = np.argmax(spectrum)
    f0 = freqs[idx_max]
    print(f"Частота основного тона: {f0} Гц")

# # #третье задание
    analyze_reaper(x, fs)
    
#четвертое задание
    fs, x =read_file("genana_2.wav")
    total_len = len(x)
    cut = int(5 * fs)
    third_part = total_len - int(3 * fs) 
    x1 = x[:cut]    
    x2 = x[cut:third_part] 
    x3=x[third_part:]
    y1 = psola(x1, fs, k=0.9) 
    y2 = psola(x2, fs, k=1.2) 
    y3 = psola(x3, fs, k=1.8) 
    y = np.concatenate([y1, y2, y3])
    y /= np.max(np.abs(y))
    y_int16 = (y * 32767).astype(np.int16)
    wavfile.write("dialog.wav", fs, y_int16) 