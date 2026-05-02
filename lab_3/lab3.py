import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io.wavfile import read, write

def shift(x, fs, dt, at, f):
    n=len(x)
    t=(n-1)/fs 
    time=np.linspace(0, t, n)
    t_delay=np.zeros(len(time))

    for i in range(len(t_delay)):
        t_delay[i]=time[i]+dt+at*np.sin(2*np.pi*f*time[i])
    
    if not np.all(np.diff(t_delay) > 0):
        print("Проверка на строгое возрастание отсчётов не прошла")

    interp_func=interp1d(t_delay, x, bounds_error=False)
    result=np.zeros(len(t_delay))
    min_t=min(t_delay)

    for i in range(len(t_delay)):
        if time[i]>=min_t:
            result[i]=interp_func(time[i])
    return result


def chorus(path:str, new_path: str):
    fs, x = read(path)

    x=x.astype(np.float32)
    max_abs = np.max(np.abs(x))
    x = x / max_abs

    result = x.copy()
    chorus_params = [
        (0.020, 0.010, 3),
        (0.025, 0.012, 2.5),
        (0.030, 0.015, 2),
        (0.035, 0.010, 1.5),
        (0.040, 0.008, 1)
    ]

    for dt, at, f in chorus_params:
        result += shift(x, fs, dt, at, f)

    result /= np.max(np.abs(result))

    t = np.arange(len(x)) / fs
    dt, at, f = chorus_params[0]
    shifted = shift(x, fs, dt, at, f)
    plt.figure(figsize=(10, 6))
    start = int(0.05 * fs)
    end = start + 2000  
    plt.plot(t[start:end], x[start:end], label="оригинал")
    plt.plot(t[start:end], shifted[start:end], label="задержанный")
    plt.plot(t[start:end], result[start:end], label="хорус")
    plt.legend()
    plt.show()

    write(new_path, fs, result.astype(np.float32))


if __name__ == "__main__":
    chorus("speech.wav", "result.wav")
    #chorus("lab_2/genana_2.wav", "lab_3/result_2.wav")
