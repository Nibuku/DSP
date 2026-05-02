import unittest

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import stft
from scipy.io import wavfile


def dft(x: np.ndarray) -> np.ndarray:
    N=len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N): 
        for n in range(N): 
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def real_stft(x: np.ndarray, segment: int, overlap: int) -> np.ndarray:
    n = x.shape[0]
    assert len(x.shape) == 1
    assert segment < n
    assert overlap < segment

    step = segment - overlap
    stft = []

    for i in range(0, n - segment + 1, step):
        segment_x = x[i:(i + segment)]
        spectrum = dft(segment_x)
        stft.append(spectrum[:segment//2+1])
        

    x = np.array(stft).T
    return x


class Test(unittest.TestCase):
    class Params:
        def __init__(self, n: int, segment: int, overlap: int) -> None:
            self.n = n
            self.segment = segment
            self.overlap = overlap

        def __str__(self) -> str:
            return f"n={self.n} segment={self.segment} overlap={self.overlap}"

    def test_dft(self) -> None:
        for n in (10, 11, 12, 13, 14, 15, 16):
            with self.subTest(n=n):
                np.random.seed(0)
                x = np.random.rand(n) + 1j * np.random.rand(n)
                actual = dft(x)
                expected = fft(x)
                self.assertTrue(np.allclose(actual, expected))

    #@unittest.skip
    def test_stft(self) -> None:
        params_list = (
            Test.Params(50, 10, 5),
            Test.Params(50, 10, 6),
            Test.Params(50, 10, 7),
            Test.Params(50, 10, 8),
            Test.Params(50, 10, 9),
            Test.Params(101, 15, 7),
            Test.Params(101, 15, 8),
        )

        for params in params_list:
            with self.subTest(params=str(params)):
                np.random.seed(0)
                x = np.random.rand(params.n)
                actual = real_stft(x, params.segment, params.overlap)
                _, _, expected = stft(
                    x,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    padded=False,
                    window="boxcar",
                )
                assert isinstance(expected, np.ndarray)
                self.assertTrue(np.allclose(actual, params.segment * expected))


def main() -> None:
    unittest.main(exit=False)
    # fs, x=wavfile.read("6413-05.wav")
    # if x.ndim > 1:
    #     x = x[:, 0]
    # f, t, Zxx = stft(x, fs)

    # plt.figure("Spectrogram")
    # plt.pcolormesh(t, f, np.abs(Zxx) ** 2)
    # plt.ylim(0, 2000)
    # plt.xlabel("Time [sec]")
    # plt.ylabel("Frequency [Hz]")
    # plt.show()

if __name__ == "__main__":
    main()
