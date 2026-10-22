"""D4 peak-amplitude spectra and amplitude/dB conversions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.signal import get_window

from audiospylt.io.audio import _sample_rate

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "Spectrum",
    "spectrum",
    "amplitude_to_db",
    "db_to_amplitude",
]


@dataclass(frozen=True)
class Spectrum:
    """A one-sided D4 spectrum, with float64 frequencies and peak amplitudes.

    DC and even-length Nyquist bins have no one-sided doubling. Other bins
    include that factor and periodic-window coherent-gain correction. Arrays
    are independent of the audio input; freezing prevents field reassignment.
    """

    freqs: np.ndarray
    amp: np.ndarray
    sr: int
    frame_length: int
    n_fft: int
    window: str | tuple | float

    @property
    def bin_hz(self) -> float:
        """Return the spacing of zero-padded FFT bins in Hz."""
        return self.sr / self.n_fft


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _finite_scalar(value: float, name: str) -> float:
    try:
        if np.ndim(value) != 0:
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite; got {value!r}") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim != 1:
        raise ValueError(
            f"audio shape {array.shape} must be 1-D (samples,); "
            "2-D is read as (channels, samples). Use to_mono(x) from "
            "audiospylt.io.audio or x[0] to select one channel."
        )
    if np.iscomplexobj(array):
        raise ValueError("audio must contain real, finite samples")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("audio must contain only finite samples")
    return array


def _scaled_magnitude(
    frames: np.ndarray, window: str | tuple | float, n_fft: int,
) -> np.ndarray:
    """Scale one or many frames on the last axis identically in D4 units."""
    weights = get_window(window, frames.shape[-1], fftbins=True)
    gain = weights.sum()
    if not np.isfinite(gain) or gain == 0:
        raise ValueError("window must have a finite, nonzero sum for amplitude scaling")
    amp = np.abs(np.fft.rfft(frames * weights, n=n_fft, axis=-1)) * 2 / gain
    amp[..., 0] /= 2
    if n_fft % 2 == 0:
        amp[..., -1] /= 2
    return amp


def spectrum(
    audio: np.ndarray, sr: int, *, window: str | tuple | float = "hann", zero_pad: int = 4,
) -> Spectrum:
    """Compute a one-sided spectrum in D4 linear peak amplitude re full scale.

    Parameters
    ----------
    audio : numpy.ndarray
        Finite 1-D real samples, at least three long. Multichannel audio must
        first use ``to_mono(x)`` or select a channel with ``x[0]``.
    sr : int
        Positive sample rate in Hz (booleans rejected).
    window : str, tuple, or float, default 'hann'
        Passed to scipy.signal.get_window with ``fftbins=True`` (periodic),
        and stored as given.
    zero_pad : int, default 4
        Positive FFT-length multiplier (booleans rejected).

    Returns
    -------
    Spectrum
        Float64 bins scaled as ``abs(rfft(audio * w)) * 2 / sum(w)``.
        DC is halved; the last bin is halved only for even ``n_fft`` (Nyquist).
        An isolated on-bin sine of amplitude a reads a; constants and
        alternating signs read their magnitudes at DC and Nyquist.

    Notes
    -----
    D4 amplitudes feed ``amp * sin(phase)`` synthesis unchanged; dB uses
    ``20*log10(amp)`` re 1.0, so a full-scale sine is 0 dB. Between-bin
    values under-read (scalloping: Hann up to 1.42 dB at zero_pad=1).
    Use ``audiospylt.analysis.peaks.find_peaks`` for partial amplitudes.
    Measured isolated-sine QIFFT-dB worst errors for Hann, zero_pad=4 are
    1.02 cents / 0.007 dB at 4 cycles per window and 0.016 cents / 0.001 dB
    at 22 cycles. At zero_pad=1, errors reach 7.8 cents / 0.32 dB at 4 cycles.
    Negative-frequency-image interference sets the roughly 1-cent floor at
    4 cycles; more zero-padding does not remove it. Boxcar and Hamming are
    biased under QIFFT; see find_peaks for measurements across windows.

    Raises
    ------
    ValueError
        Invalid audio shape, nonfinite samples, rate, length, or zero_pad.
    """
    array = _mono_audio(audio)
    sr = _sample_rate(sr)
    frame_length = _positive_int(len(array), "frame_length", 3)
    n_fft = frame_length * _positive_int(zero_pad, "zero_pad")
    return Spectrum(
        np.fft.rfftfreq(n_fft, 1 / sr), _scaled_magnitude(array, window, n_fft),
        sr, frame_length, n_fft, window,
    )


@overload
def amplitude_to_db(amp: float | np.floating | np.integer, *, floor_db: float = ...) -> float: ...
@overload
def amplitude_to_db(amp: ArrayLike, *, floor_db: float = ...) -> np.ndarray: ...
def amplitude_to_db(amp: ArrayLike, *, floor_db: float = -120.0) -> float | np.ndarray:
    """Convert D4 peak amplitude to dB re 1.0, flooring zero and small values.

    Parameters
    ----------
    amp : float or array-like
        Nonnegative amplitudes. Negative values and NaNs raise ValueError.
    floor_db : float, default -120.0
        Finite lower limit in dB.

    Returns
    -------
    float or numpy.ndarray
        ``20*log10(maximum(amp, 10**(floor_db/20)))``: a full-scale sine is
        0 dB. Scalar input returns a Python float; arrays retain their shape
        and use float64. The input is never modified.
    """
    floor_db = _finite_scalar(floor_db, "floor_db")
    array = np.asarray(amp, dtype=np.float64)
    if np.any(np.isnan(array) | (array < 0)):
        raise ValueError("amp must be nonnegative and must not contain NaN")
    # Log-domain flooring is equivalent to flooring the amplitude, and also
    # handles finite floors whose linear amplitudes underflow or overflow.
    with np.errstate(divide="ignore"):
        result = np.maximum(20 * np.log10(array), floor_db)
    return float(result) if array.ndim == 0 else result


@overload
def db_to_amplitude(db: float | np.floating | np.integer) -> float: ...
@overload
def db_to_amplitude(db: ArrayLike) -> np.ndarray: ...
def db_to_amplitude(db: ArrayLike) -> float | np.ndarray:
    """Convert dB re full scale to D4 peak amplitude with ``10**(db/20)``.

    Parameters
    ----------
    db : float or array-like
        Decibels re 1.0; negative infinity maps to zero.

    Returns
    -------
    float or numpy.ndarray
        Python float for scalar input, otherwise a same-shape float64 array.
        The input is never modified.
    """
    array = np.asarray(db, dtype=np.float64)
    result = np.power(10.0, array / 20)
    return float(result) if array.ndim == 0 else result
