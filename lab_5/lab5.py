import unittest

import numpy as np
from scipy.signal import stft, istft
from scipy.fft import ifft
import scipy.io.wavfile as wavfile


def idft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    new_x = np.zeros(N, dtype=complex)
    for i in range(N):
        sum = 0
        for j in range(N):
            sum += x[j] * np.exp(2j * np.pi * j * i / N)
        new_x[i] = sum / N
    return new_x


def real_istft(spectrum: np.ndarray, segment: int, overlap: int) -> np.ndarray:
    assert len(spectrum.shape) == 2
    assert spectrum.shape[0] == segment // 2 + 1

    step = segment - overlap
    frames_number = spectrum.shape[1]
    new_len = step * (frames_number - 1) + segment

    output = np.zeros(new_len, dtype=float)
    weight = np.zeros(new_len, dtype=float)

    for i in range(frames_number):
        new_spectrum = np.zeros(segment, dtype=complex)
        new_spectrum[: segment // 2 + 1] = spectrum[:, i]

        for k in range(1, segment // 2 + 1):
            new_spectrum[segment - k] = np.conj(spectrum[k, i])

        frame = idft(new_spectrum).real

        N = i * step
        output[N : N + segment] += frame
        weight[N : N + segment] += 1

    return output / weight


class Test(unittest.TestCase):
    class Params:
        def __init__(self, n: int, segment: int, overlap: int) -> None:
            self.n = n
            self.segment = segment
            self.overlap = overlap

        def __str__(self) -> str:
            return f"n={self.n} segment={self.segment} overlap={self.overlap}"

    def test_idft(self) -> None:
        for n in (10, 11, 12, 13, 14, 15, 16):
            with self.subTest(n=n):
                np.random.seed(0)
                x = np.random.rand(n) + 1j * np.random.rand(n)
                actual = idft(x)
                expected = ifft(x)
                self.assertTrue(np.allclose(actual, expected))

    def test_istft_unmodified(self) -> None:
        self._test_istft(False)

    def test_istft_modified(self) -> None:
        self._test_istft(True)

    def _test_istft(self, modify: bool) -> None:
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

                _, _, s = stft(
                    x,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    padded=False,
                    window="boxcar",
                )

                assert isinstance(s, np.ndarray)

                if modify:
                    low_pass_filter = np.concatenate(
                        (
                            np.ones(s.shape[0] // 2),
                            np.zeros(s.shape[0] - s.shape[0] // 2),
                        )
                    )
                    for column in np.arange(s.shape[1]):
                        s[:, column] = s[:, column] * low_pass_filter

                _, expected = istft(
                    s,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    window="boxcar",
                )

                assert isinstance(expected, np.ndarray)

                actual = real_istft(s * params.segment, params.segment, params.overlap)

                self.assertTrue(np.allclose(actual, expected))


def main() -> None:
    # unittest.main()

    fs, x = wavfile.read("speech.wav")
    segment_ms = 20
    segment = int(fs * segment_ms / 1000)
    overlap = segment // 2
    window = "triangle"

    _, _, stft_spech = stft(x, fs=fs, nperseg=segment, noverlap=overlap, window=window)

    stft_spech = abs(stft_spech)

    _, robotic_speech = istft(
        stft_spech, fs=fs, nperseg=segment, noverlap=overlap, window=window
    )

    robotic_speech = robotic_speech / max(
        abs(max(robotic_speech)), abs(min(robotic_speech))
    )

    wavfile.write("speech_2.wav", rate=fs, data=robotic_speech)


if __name__ == "__main__":
    main()
