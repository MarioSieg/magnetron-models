# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import math
from collections.abc import Iterator

from magnetron import Tensor, dtype

SAMPLE_RATE: int = 16000
N_FFT: int = 400
HOP_LENGTH: int = 160
CHUNK_LENGTH: int = 30
N_SAMPLES: int = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES: int = N_SAMPLES // HOP_LENGTH
N_FREQS: int = N_FFT // 2 + 1


def load_audio(path: str) -> Tensor:
    audio, sr = Tensor.load_audio(path)
    audio = audio.mean(dim=0)
    if sr != SAMPLE_RATE:
        n: int = audio.shape[0]
        target: int = round(n * SAMPLE_RATE / sr)
        audio = audio.reshape(1, 1, 1, n).interpolate(size=(1, target), mode='bilinear', antialias=True).reshape(target)
    return audio


def _hz_to_mel(f: float) -> float:
    f_sp: float = 200.0 / 3.0
    min_log_hz: float = 1000.0
    if f < min_log_hz:
        return f / f_sp
    return min_log_hz / f_sp + math.log(f / min_log_hz) / (math.log(6.4) / 27.0)


def _mel_to_hz(m: float) -> float:
    f_sp: float = 200.0 / 3.0
    min_log_hz: float = 1000.0
    min_log_mel: float = min_log_hz / f_sp
    if m < min_log_mel:
        return m * f_sp
    return min_log_hz * math.exp((math.log(6.4) / 27.0) * (m - min_log_mel))


def mel_filters(n_mels: int) -> Tensor:
    lo: float = _hz_to_mel(0.0)
    hi: float = _hz_to_mel(SAMPLE_RATE / 2.0)
    hz: list[float] = [_mel_to_hz(lo + (hi - lo) * i / (n_mels + 1)) for i in range(n_mels + 2)]
    fdiff = Tensor([hz[i + 1] - hz[i] for i in range(n_mels + 1)], dtype=dtype.float32).reshape(-1, 1)
    fft_freqs = Tensor.arange(0, N_FREQS, dtype=dtype.float32).reshape(1, -1) * (SAMPLE_RATE / N_FFT)
    ramps = Tensor(hz, dtype=dtype.float32).reshape(-1, 1) - fft_freqs
    lower = -ramps[:-2] / fdiff[:-1]
    upper = ramps[2:] / fdiff[1:]
    enorm = Tensor([2.0 / (hz[i + 2] - hz[i]) for i in range(n_mels)], dtype=dtype.float32).reshape(-1, 1)
    return lower.min(upper).clamp_min(0.0) * enorm


def _dft_basis() -> tuple[Tensor, Tensor]:
    n = Tensor.arange(0, N_FFT, dtype=dtype.int64).reshape(-1, 1)
    k = Tensor.arange(0, N_FREQS, dtype=dtype.int64).reshape(1, -1)
    phase = ((n * k) % N_FFT).cast(dtype.float32) * (2.0 * math.pi / N_FFT)
    window = 0.5 - 0.5 * (Tensor.arange(0, N_FFT, dtype=dtype.float32) * (2.0 * math.pi / N_FFT)).cos()
    window = window.reshape(-1, 1)
    return (phase.cos() * window).contiguous(), (phase.sin() * window).contiguous()


class LogMelSpectrogram:
    def __init__(self, n_mels: int) -> None:
        self.n_mels = n_mels
        self.filters_t = mel_filters(n_mels).T.contiguous()
        self.cos_basis, self.sin_basis = _dft_basis()

    def __call__(self, chunk: Tensor) -> Tensor:
        n: int = chunk.shape[0]
        if n < N_SAMPLES:
            chunk = chunk.pad((0, N_SAMPLES - n))
        padded = chunk.pad((N_FFT // 2, N_FFT // 2), 'reflect').contiguous()
        frames = padded.strided_view((N_FRAMES, N_FFT), (HOP_LENGTH, 1))
        re = frames @ self.cos_basis
        im = frames @ self.sin_basis
        power = re.sqr() + im.sqr()
        log_spec = (power @ self.filters_t).clamp_min(1e-10).log10()
        log_spec = log_spec.max(log_spec.max() - 8.0)
        return ((log_spec + 4.0) / 4.0).T.contiguous()


def chunks(audio: Tensor) -> Iterator[tuple[float, Tensor]]:
    n: int = audio.shape[0]
    for start in range(0, max(n, 1), N_SAMPLES):
        yield start / SAMPLE_RATE, audio[start : min(start + N_SAMPLES, n)].contiguous()
