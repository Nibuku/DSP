import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
from scipy.io.wavfile import write


def tone(f: float, t: float, waveform="harmonic", fs=44100) -> np.ndarray:
    n = int(fs * t)
    time = np.linspace(0, t, n)
    match waveform:
        case "harmonic":
            s = np.sin(2 * np.pi * f * time)
        case "square":
            s = square(2 * np.pi * f * time)
        case "triangle":
            s = sawtooth(2 * np.pi * f * time, width=0.5)
        case "sawtooth":
            s = sawtooth(2 * np.pi * f * time)
        case _:
            raise ValueError("Uncorrect parametr")
    return s.astype(np.float32)


def musical_tone(
    f: float, t: float, waveform="harmonic", fs: int = 44100, db: float = -40.0
) -> np.ndarray:
    if db > 0:
        raise ValueError("Uncorrect db-value")
    n = int(fs * t)
    s = np.zeros(n)
    a = [1, 0.3, 0.1, 0.05, 0.03] 
    for k, A_k in enumerate(a, start=1):
        s += A_k * tone(k * f, t, waveform, fs)
    s=s/np.max(np.abs(s))
    if db == 0:
        e = np.ones(n)
    else:
        A = 10 ** (db / 20)
        e = np.exp(np.linspace(0, np.log(A), n))
    x = s * e
    return x.astype(np.float32)


def check_tone(signal: np.ndarray, fs: int, path: str, n=600):
    plt.figure(figsize=(15,5))
    
    plt.stem(np.arange(n), signal[:n])
    plt.xlabel("Отсчёт")
    plt.ylabel("Амплитуда")
    plt.show()
    
    write(path, fs, signal)


def play_melody(melody, waveform="harmonic", fs=44100, db=-40):
    pieces = []
    with open("freq.json", "r") as f:
        NOTE_FREQ = json.load(f)
    for note, t in melody:
        f = NOTE_FREQ[note]
        tone = musical_tone(f=f, t=t, waveform=waveform, fs=fs, db=db)
        pieces.append(tone)

    return np.concatenate(pieces)


if __name__ == "__main__":
    # harmonic_signal=tone(720, 10)
    # check_tone(harmonic_signal, 44100, "harmonic_720.wav")

    # square_tone=tone(440, 10, "square")
    # check_tone(square_tone, 44100, "square_440.wav")

    # triangle_tone=tone(528, 7, "triangle", 48000)
    # check_tone(triangle_tone, 48000, "triangle_528.wav")

    # sawtooth_tone=tone(432, 15, "sawtooth", 52000)
    # check_tone( sawtooth_tone, 52000, "sawtooth_432.wav")

    # music=musical_tone(720, 5, db=-80)
    # check_tone(music, 44100, "music.wav", n=5000)

    melody = [
        ("C5", 0.22),
        ("C5", 0.22),
        ("C5", 0.22),
        ("D5", 0.45),
        ("E5", 0.45),
        ("B4", 0.22),
        ("B4", 0.22),
        ("B4", 0.22),
        ("C5", 0.45),
        ("B4", 0.45),
        ("C5", 0.22),
        ("C5", 0.22),
        ("B4", 0.3),
        ("A4", 0.22),
        ("A4", 0.22),
        ("C5", 0.3),
        ("C5", 0.22),
        ("B4", 0.45),
        ("A4", 0.22),
        ("A4", 0.22),
        ("C5", 0.3),
        ("C5", 0.22),
        ("C5", 0.22),
        ("A4", 0.45),
        ("C5", 0.45),
        ("C5", 0.22),
        ("D5", 0.45),
        ("C5", 0.22),
        ("E5", 0.45),
        ("B4", 0.22),
        ("G5", 0.22),
        ("B4", 0.22),
        ("C5", 0.45),
        ("E5", 0.45),
        ("C5", 0.22),
        ("C5", 0.22),
        ("B4", 0.3),
        ("A4", 0.22),
        ("A4", 0.22),
        ("C5", 0.3),
        ("C5", 0.22),
        ("B4", 0.45),
        ("A4", 0.22),
        ("A4", 0.22),
        ("C5", 0.3),
        ("C5", 0.22),
        ("C5", 0.22),
        ("A4", 0.45),
        ("C5", 0.45),
    ]

    # song = play_melody(melody)
    # write("waka_waka.wav", 44100, song.astype(np.float32))
 

    # s s s d |f|     # C5 C5 C5 D5 |E5|
    # a a a s |d|     # B4 B4 B4 C5 |B4|
    # s  s a  p       # C5 C5 B4 A4
    # p s  s a       # A4 C5 C5 B4
    # p  p s  s s     # A4 A4 C5 C5
    # p |s|           # C5 A4 |C5|
    # s  d|s  f       # C5 D5 C5 | E5
    # a  a a s|d      # B4 B4 B4 C5 | D5
    # s  s a  p       # C5 C5 B4 A4
    # p s s a         # A4 C5 C5 B4
    # p p s s         # A4 A4 C5 C5
    # s p |s          # C5 A4 | C5
    # melody = [
    # ("C5", 0.22), ("C5", 0.22), ("C5", 0.22), ("D5", 0.45), ("E5", 0.45),
    # ("B4", 0.22), ("B4", 0.22), ("B4", 0.22), ("C5", 0.45), ("B4", 0.45),
    # ("C5", 0.22), ("C5", 0.22), ("B4", 0.3), ("A4", 0.22),
    # ("A4", 0.22), ("C5", 0.3), ("C5", 0.22), ("B4", 0.45),
    # ("A4", 0.22), ("A4", 0.22), ("C5", 0.3),
    # ("C5", 0.22), ("C5", 0.22), ("A4", 0.45), ("C5", 0.45),

    # ("C5", 0.22), ("D5", 0.45), ("C5", 0.22), ("E5", 0.45),
    # ("B4", 0.22), ("G5", 0.22), ("B4", 0.22), ("C5", 0.45), ("E5", 0.45),

    # ("C5", 0.22), ("C5", 0.22), ("B4", 0.3), ("A4", 0.22),
    # ("A4", 0.22), ("C5", 0.3), ("C5", 0.22), ("B4", 0.45),

    # ("A4", 0.22), ("A4", 0.22), ("C5", 0.3), ("C5", 0.22),
    # ("C5", 0.22), ("A4", 0.45), ("C5", 0.45)]
