"""Compute-only D4 STFT magnitudes and energy-normalized mel amplitudes."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window

from audiospylt._optional import import_optional
from audiospylt.analysis.spectrum import (
    Spectrum, _finite_scalar, _mono_audio, _positive_int, _scaled_magnitude,
)
from audiospylt.io.audio import _sample_rate

__all__ = [
    "Spectrogram",
    "spectrogram",
    "MelSpectrogram",
    "mel_spectrogram",
]


@dataclass(frozen=True)
class Spectrogram:
    """Float64 frequencies (bins,), times (frames,), and D4 amp (bins, frames)."""

    freqs: np.ndarray
    times: np.ndarray
    amp: np.ndarray
    sr: int
    frame_length: int
    hop: int
    n_fft: int
    window: str | tuple | float
    center: bool

    def frame(self, i: int) -> Spectrum:
        """Copy column i as a Spectrum, with Python negative-index semantics.

        Parameters
        ----------
        i : int
            Frame index; out-of-bounds indices raise IndexError.

        Returns
        -------
        Spectrum
            Independent copies of frequencies and amplitudes, with this
            spectrogram's sample rate, frame/FFT lengths, and window.
        """
        return Spectrum(
            self.freqs.copy(), self.amp[:, index(i)].copy(), self.sr,
            self.frame_length, self.n_fft, self.window,
        )


def spectrogram(
    audio: np.ndarray,
    sr: int,
    *,
    frame_length: int = 2048,
    hop: int = 512,
    window: str | tuple | float = "hann",
    zero_pad: int = 1,
    center: bool = True,
) -> Spectrogram:
    """Compute a vectorized STFT in D4 linear peak amplitude re full scale.

    Parameters
    ----------
    audio : numpy.ndarray
        Finite 1-D real samples. Use to_mono(x) or x[0] for multichannel audio.
    sr : int
        Positive sample rate in Hz.
    frame_length : int, default 2048
        Samples per periodic window, at least 3.
    hop : int, default 512
        Positive spacing of frame starts in samples.
    window : str, tuple, or float, default 'hann'
        Passed unchanged to scipy.signal.get_window with fftbins=True.
    zero_pad : int, default 1
        Positive FFT-length multiplier; default is display-sized. Peak picking
        on frames needs zero_pad=4 for the accuracy documented by find_peaks.
    center : bool, default True
        Pad frame_length//2 zeros on each side, with times k*hop/sr (librosa
        convention). Edge frames include zeros, so a stationary sine reads
        lower in the first/last frames. False requires at least frame_length
        samples and uses times (k*hop + frame_length//2)/sr.

    Returns
    -------
    Spectrogram
        Float64 amp (bins, frames), freqs, and times. There are
        1 + (padded_length - frame_length)//hop frames; trailing partial
        frames are discarded. D4 scaling is exactly shared with spectrum:
        2/sum(w), with DC and even-length Nyquist halved (odd FFTs have no
        Nyquist bin). For center=False each column equals spectrum of its
        audio slice bit for bit. Inputs are never modified.

    Notes
    -----
    Convert to dB with amplitude_to_db(S.amp): 20*log10 re 1.0. The 0.6
    10*log10 of magnitude halved the dynamic range. At zero_pad=1 Hann QIFFT
    errors reach 7.8 cents / 0.32 dB at 4 cycles per window; use zero_pad=4
    for partial estimates (1.02 cents / 0.007 dB at 4 cycles).

    Raises
    ------
    ValueError
        Invalid shape, nonfinite audio, rate, frame length, hop, zero_pad, or
        insufficient samples with center=False.
    """
    array = _mono_audio(audio)
    sr = _sample_rate(sr)
    frame_length = _positive_int(frame_length, "frame_length", 3)
    hop = _positive_int(hop, "hop")
    n_fft = frame_length * _positive_int(zero_pad, "zero_pad")
    if center:
        array = np.pad(array, (frame_length // 2, frame_length // 2))
    elif len(array) < frame_length:
        raise ValueError(
            f"audio has {len(array)} samples; center=False needs at least "
            f"frame_length={frame_length}"
        )
    frames = (sliding_window_view(array, frame_length)[::hop]
              if len(array) >= frame_length else np.empty((0, frame_length)))
    amp = _scaled_magnitude(frames, window, n_fft).T
    times = np.arange(len(frames), dtype=np.float64) * hop
    if not center:
        times += frame_length // 2
    times /= sr
    return Spectrogram(
        np.fft.rfftfreq(n_fft, 1 / sr), times, amp,
        sr, frame_length, hop, n_fft, window, center,
    )


@dataclass(frozen=True)
class MelSpectrogram:
    """Float64 mel band centres (Hz), times, and D4 amp (n_mels, frames)."""

    freqs: np.ndarray
    times: np.ndarray
    amp: np.ndarray
    sr: int
    n_mels: int
    fmin: float
    fmax: float


def mel_spectrogram(
    spec: Spectrogram, *, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None,
) -> MelSpectrogram:
    """Combine STFT-bin energies into D4 amplitudes on Slaney mel bands.

    Parameters
    ----------
    spec : Spectrogram
        D4 magnitude spectrogram with its periodic window metadata.
    n_mels : int, default 128
        Positive band count (booleans rejected).
    fmin, fmax : float, optional
        Band edges in Hz, with 0 <= fmin < fmax; fmax defaults to sr/2.

    Returns
    -------
    MelSpectrogram
        Peak-1 triangular filters B (librosa norm=None, default Slaney scale),
        with amp = sqrt(B @ spec.amp**2 / (P * ENBW)), where P is
        n_fft/frame_length and ENBW = frame_length*sum(w**2)/sum(w)**2.
        Band centres exclude the outer edges of librosa.mel_frequencies.
        Needs ``audiospylt[librosa]``; librosa's warnings are not suppressed.

    Notes
    -----
    Parseval gives sum(scaled_sine_bins**2) approximately a**2 * P * ENBW,
    so band values are D4 peak amplitudes, independent of zero-padding;
    dB remains 20*log10 re 1.0. Narrow bands under-read when narrower than
    the window's main lobe. With frame 2048, n_mels=64, sr=44100, a sine at
    a band centre measures -0.15 dB at 4.8 kHz and -0.04 dB at 17 kHz, but
    -0.9 dB at 677 Hz. Zero_pad 1 versus 4 differs by at most 0.13 dB.
    Inputs are never modified.

    Raises
    ------
    ValueError
        Invalid band count or nonfinite/invalid frequency bounds.
    ImportError
        librosa is not installed (``audiospylt[librosa]``).
    """
    n_mels = _positive_int(n_mels, "n_mels")
    fmin = _finite_scalar(fmin, "fmin")
    fmax = spec.sr / 2 if fmax is None else _finite_scalar(fmax, "fmax")
    if fmin < 0 or fmax <= fmin:
        raise ValueError("frequency bounds must satisfy 0 <= fmin < fmax")

    librosa = import_optional("librosa", extra="librosa", feature="mel spectrogram support")
    bands = librosa.filters.mel(
        sr=spec.sr, n_fft=spec.n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm=None,
    )
    weights = get_window(spec.window, spec.frame_length, fftbins=True)
    enbw = spec.frame_length * np.sum(weights**2) / weights.sum()**2
    padding = spec.n_fft / spec.frame_length
    amp = np.sqrt(bands @ spec.amp**2 / (padding * enbw))
    freqs = librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
    return MelSpectrogram(freqs, spec.times.copy(), amp, spec.sr, n_mels, fmin, fmax)
