"""Temporal f0 (librosa pYIN, refined on QIFFT harmonics) and snapshot harmonic f0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from audiospylt._optional import import_optional
from audiospylt.analysis.peaks import find_peaks
from audiospylt.analysis.spectrum import (
    Spectrum, _finite_scalar, _mono_audio, _positive_int, _scaled_magnitude,
)
from audiospylt.io.audio import _sample_rate
from audiospylt.model.schema import validate

__all__ = [
    "F0Contour",
    "pyin",
    "HarmonicF0",
    "harmonic_f0",
]

# Refit settings (D20): harmonics below 5 kHz (or 2*fmax), matched within +-50
# cents, first on harmonics 1..8, then on the whole series around that estimate.
_REFIT_MAX_HZ = 5000.0
_REFIT_RATIO = 2 ** (50 / 1200)
_REFIT_FIRST = 8
_REFIT_REL_DB = -60.0
_REFIT_ZERO_PAD = 4
_BLOCK = 256  # frames per FFT block, bounding memory on long files
_WEIGHTS = ("amp", "equal")
_SEED_CELLS = 2**22


@dataclass(frozen=True)
class F0Contour:
    """Per-frame f0 in Hz on the spectrogram() frame grid (frames,).

    f0 and f0_pyin are NaN where unvoiced. amp is the D4 amplitude of the
    fundamental's spectral peak: NaN where unvoiced, 0 where voiced but no
    peak lies within 50 cents of f0 (a missing fundamental).
    """

    times: np.ndarray
    f0: np.ndarray
    voiced: np.ndarray
    voiced_prob: np.ndarray
    f0_pyin: np.ndarray
    refined: np.ndarray
    amp: np.ndarray
    sr: int
    frame_length: int
    hop: int
    fmin: float
    fmax: float


def _match(freq: np.ndarray, amp: np.ndarray, f0: float, count: int, narrow: bool):
    """Return harmonic numbers, frequencies and amplitudes of matched peaks.

    Harmonic k takes the strongest peak within +-50 cents of k*f0; with narrow,
    also within +-f0/4 Hz, so windows of neighbouring harmonics never overlap.
    """
    ks = np.arange(1, count + 1, dtype=np.float64)
    centre = ks * f0
    lo, hi = centre / _REFIT_RATIO, centre * _REFIT_RATIO
    if narrow:
        lo, hi = np.maximum(lo, centre - f0 / 4), np.minimum(hi, centre + f0 / 4)
    start, stop = np.searchsorted(freq, lo), np.searchsorted(freq, hi, side="right")
    matched = [(k, start[i] + int(np.argmax(amp[start[i]:stop[i]])))
               for i, k in enumerate(ks) if stop[i] > start[i]]
    harmonics = np.array([k for k, _ in matched], dtype=np.float64)
    rows = np.array([j for _, j in matched], dtype=np.int64)
    return harmonics, freq[rows], amp[rows]


def _log_fit(freq: np.ndarray, harmonics: np.ndarray, weights: np.ndarray) -> float:
    """Minimise the weighted squared log2 residuals of freq against harmonics*f0."""
    return float(2 ** (np.sum(weights * (np.log2(freq) - np.log2(harmonics))) / np.sum(weights)))


def _refit_frame(peaks: pd.DataFrame, f0: float, max_hz: float, refine: bool):
    """Return (f0 or NaN when not refined, fundamental amplitude) for one frame."""
    freq, amp = peaks["freq"].to_numpy(), peaks["amp"].to_numpy()
    if not refine:
        _, _, amps = _match(freq, amp, f0, 1, narrow=False)
        return np.nan, (float(amps[0]) if amps.size else 0.0)
    estimate = np.nan
    for count, narrow in ((min(_REFIT_FIRST, int(max_hz // f0)), False), (None, True)):
        reference = f0 if np.isnan(estimate) else estimate
        count = int(max_hz // reference) if count is None else count
        harmonics, freqs, amps = _match(freq, amp, reference, max(count, 1), narrow)
        if not harmonics.size:
            break
        estimate = _log_fit(freqs, harmonics, amps)
    if np.isnan(estimate):
        return np.nan, 0.0
    _, _, amps = _match(freq, amp, estimate, 1, narrow=False)
    return estimate, (float(amps[0]) if amps.size else 0.0)


def pyin(
    audio: np.ndarray,
    sr: int,
    *,
    fmin: float,
    fmax: float,
    frame_length: int = 2048,
    hop: int = 512,
    refine: bool = True,
) -> F0Contour:
    """Track f0 over time with librosa's pYIN, refined on spectral harmonics.

    Parameters
    ----------
    audio : numpy.ndarray
        Finite 1-D real samples. Use to_mono(x) or x[0] for multichannel audio.
    sr : int
        Positive sample rate in Hz.
    fmin, fmax : float
        Search range in Hz, with 0 < fmin < fmax <= sr/2. Required: a range
        that is wrong for the instrument fails silently (octave errors or
        unvoiced frames), so there is no default.
    frame_length : int, default 2048
        Samples per frame, at least 3. librosa warns when fewer than two
        periods of fmin fit into a frame; those frames tend to read unvoiced.
    hop : int, default 512
        Positive spacing of frame starts in samples.
    refine : bool, default True
        Replace pYIN's value in each voiced frame by a fit to that frame's
        harmonics (see Notes). False keeps pYIN's values.

    Returns
    -------
    F0Contour
        One value per frame, centred like spectrogram(audio, sr,
        frame_length=frame_length, hop=hop): times k*hop/sr, the same frame
        count. f0_pyin keeps pYIN's values and refined marks the frames whose
        f0 came from the fit. Inputs are never modified.

    Notes
    -----
    pYIN (librosa.pyin with its default 0.1-semitone pitch bins) returns f0 on
    a 10-cent grid, biased upwards at low f0: a 73.8 Hz sine reads +21 cents
    with frame 2048 at 44.1 kHz, +11 cents with frame 4096. The refit takes
    each voiced frame's Hann spectrum with 4x zero-padding (the same frame as
    spectrogram), finds QIFFT peaks within 60 dB of the frame's strongest peak
    below max(5 kHz, 2*fmax) (at most sr/2), and matches harmonic k to the strongest peak within 50 cents
    of k*f0 (and within f0/4 Hz once a first estimate exists). The f0 that
    minimises the amplitude-weighted squared log2 residuals of the matched
    peaks is first fitted on harmonics 1-8 around pYIN's value, then on all
    harmonics in that range around that estimate. A frame keeps pYIN's value
    when no peak matches. Measured errors after the refit: 73.8 Hz sine at
    most 1.2 cents, 223.3 Hz sine 0.02 cents, a sawtooth with +-50 cent
    vibrato at 5.5 Hz 1.6 cents (95th percentile). The refit follows pYIN's
    octave: it cannot correct an octave error. The first call in a process
    compiles librosa's decoder (several seconds). Needs
    ``audiospylt[librosa]``; librosa's warnings are not suppressed.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite audio, rate, frequency range, frame length or
        hop.
    ImportError
        librosa is not installed (``audiospylt[librosa]``).
    """
    array = _mono_audio(audio)
    sr = _sample_rate(sr)
    fmin = _finite_scalar(fmin, "fmin")
    fmax = _finite_scalar(fmax, "fmax")
    if not 0 < fmin < fmax <= sr / 2:
        raise ValueError(f"need 0 < fmin < fmax <= sr/2 = {sr / 2:g}; got {fmin!r}, {fmax!r}")
    frame_length = _positive_int(frame_length, "frame_length", 3)
    hop = _positive_int(hop, "hop")
    if not isinstance(refine, (bool, np.bool_)):
        raise ValueError(f"refine must be True or False; got {refine!r}")

    librosa = import_optional("librosa", extra="librosa", feature="pyin f0 tracking")
    f0_pyin, voiced, voiced_prob = librosa.pyin(
        array, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length,
        hop_length=hop, center=True, pad_mode="constant",
    )
    f0_pyin = np.asarray(f0_pyin, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0_pyin)
    count = len(f0_pyin)

    padded = np.pad(array, (frame_length // 2, frame_length // 2))
    frames = sliding_window_view(padded, frame_length)[::hop]
    if len(frames) != count:  # pragma: no cover - guards librosa's framing convention
        raise RuntimeError(f"pyin returned {count} frames, the spectrogram grid has {len(frames)}")
    n_fft = frame_length * _REFIT_ZERO_PAD
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    max_hz = min(max(_REFIT_MAX_HZ, 2 * fmax), sr / 2)
    fitted = np.full(count, np.nan)
    amp = np.full(count, np.nan)
    rows = np.flatnonzero(voiced)
    for begin in range(0, len(rows), _BLOCK):
        block = rows[begin:begin + _BLOCK]
        magnitudes = _scaled_magnitude(frames[block], "hann", n_fft)
        for i, magnitude in zip(block, magnitudes):
            spec = Spectrum(freqs, magnitude, sr, frame_length, n_fft, "hann")
            peaks = find_peaks(spec, min_db=None, rel_db=_REFIT_REL_DB, fmax=max_hz)
            fitted[i], amp[i] = _refit_frame(peaks, float(f0_pyin[i]), max_hz, bool(refine))
    refined = np.isfinite(fitted)
    f0 = np.where(refined, fitted, np.where(voiced, f0_pyin, np.nan))
    return F0Contour(
        np.arange(count, dtype=np.float64) * hop / sr, f0, voiced,
        np.asarray(voiced_prob, dtype=np.float64), np.where(voiced, f0_pyin, np.nan),
        refined, amp, sr, frame_length, hop, fmin, fmax,
    )


@dataclass(frozen=True)
class HarmonicF0:
    """The f0 that best explains a snapshot's partials as harmonics.

    harmonics (int64) and cents (float64) align with the snapshot's rows;
    rows with freq 0 get harmonic 0 and NaN cents and are not fitted.
    candidate_f0 / candidate_rms hold every local optimum in [fmin, fmax],
    ascending in frequency.
    """

    f0: float
    rms_cents: float
    harmonics: np.ndarray
    cents: np.ndarray
    candidate_f0: np.ndarray
    candidate_rms: np.ndarray
    fmin: float
    fmax: float
    weight: str
    tol_cents: float


def _rms_cents(freq: np.ndarray, weights: np.ndarray, f0: float):
    harmonics = np.maximum(np.rint(freq / f0), 1.0)
    cents = 1200 * np.log2(freq / (harmonics * f0))
    return float(np.sqrt(np.sum(weights * cents**2) / np.sum(weights))), harmonics, cents


def harmonic_f0(
    snapshot: pd.DataFrame,
    *,
    fmin: float,
    fmax: float,
    weight: str = "amp",
    tol_cents: float = 3.0,
) -> HarmonicF0:
    """Estimate one f0 that explains a snapshot's partials as integer harmonics.

    Parameters
    ----------
    snapshot : pandas.DataFrame
        A v1 ``snapshot`` table, e.g. from ``find_peaks``, with at least one
        row whose freq is > 0. Filter it first (``rel_db``, ``prominence_db``,
        ``fmin``): every row takes part.
    fmin, fmax : float
        Search range in Hz, with 0 < fmin < fmax.
    weight : {'amp', 'equal'}, default 'amp'
        Weight each row by its D4 amplitude, or weight rows equally.
    tol_cents : float, default 3.0
        Non-negative. The result is the highest candidate whose RMS error is
        at most tol_cents above the lowest; 0 returns the lowest.

    Returns
    -------
    HarmonicF0
        The chosen f0, its weighted RMS error in cents, each row's harmonic
        number round(freq/f0) (at least 1) and residual in cents, and all
        candidates. Inputs are never modified.

    Notes
    -----
    The objective is the 0.6 ``search_f0`` one: the weighted RMS of
    1200*log2(freq / (n*f0)) with n = max(round(freq/f0), 1). For fixed n it
    has the closed-form minimum log2 f0 = sum(w*(log2 freq - log2 n)) /
    sum(w); starting from a log-spaced grid fine enough that the highest
    partial changes harmonic number at most once per step, rounding and
    fitting alternate until n is stable, which finds every local optimum.
    A subharmonic f0/k explains every partial at least as well as f0, so the
    lowest error tends to sit an octave or more too low. On a cello D2
    (108 peaks, amplitude weights) the true 74.40 Hz scores 17.79 cents and
    37.20 Hz 17.23 cents; the next octave up scores 190-270 cents.
    tol_cents prefers the higher candidate in such near ties. Cost grows with
    the number of rows and the range in octaves times max(freq)/fmin: 0.01 s
    for the cello's 108 rows over 35-150 Hz, about 15 s for 3000 random rows
    up to 22 kHz over 20-2000 Hz.

    Raises
    ------
    SchemaError
        Invalid snapshot.
    ValueError
        No row with freq > 0, invalid range, weight, or tol_cents, or zero
        total weight.
    """
    table = validate(snapshot, "snapshot")
    fmin = _finite_scalar(fmin, "fmin")
    fmax = _finite_scalar(fmax, "fmax")
    if not 0 < fmin < fmax:
        raise ValueError(f"need 0 < fmin < fmax; got {fmin!r}, {fmax!r}")
    if weight not in _WEIGHTS:
        raise ValueError(f"weight must be one of {_WEIGHTS}; got {weight!r}")
    tol_cents = _finite_scalar(tol_cents, "tol_cents")
    if tol_cents < 0:
        raise ValueError(f"tol_cents must be >= 0; got {tol_cents!r}")
    all_freq = table["freq"].to_numpy()
    used = all_freq > 0
    if not used.any():
        raise ValueError("snapshot needs at least one row with freq > 0")
    freq = all_freq[used]
    weights = table["amp"].to_numpy()[used] if weight == "amp" else np.ones(len(freq))
    if not np.sum(weights) > 0:
        raise ValueError("weights must have a positive sum (all amplitudes are 0)")

    step = np.log2(1 + fmin / (2 * freq.max()))
    grid = 2 ** np.arange(np.log2(fmin), np.log2(fmax), step)
    seeds: dict[bytes, np.ndarray] = {}
    chunk = max(1, _SEED_CELLS // len(freq))  # bounds the (grid, rows) matrix
    for begin in range(0, len(grid), chunk):
        numbers = np.maximum(np.rint(freq / grid[begin:begin + chunk, None]), 1)
        for harmonics in np.unique(numbers, axis=0):
            seeds.setdefault(harmonics.tobytes(), harmonics)
    candidates = {fmin, fmax}
    for harmonics in seeds.values():
        for _ in range(50):
            f0 = _log_fit(freq, harmonics, weights)
            updated = np.maximum(np.rint(freq / f0), 1.0)
            if np.array_equal(updated, harmonics):
                break
            harmonics = updated
        if fmin <= f0 <= fmax:
            candidates.add(f0)
    candidate_f0 = np.array(sorted(candidates))
    candidate_rms = np.array([_rms_cents(freq, weights, f0)[0] for f0 in candidate_f0])
    within = candidate_rms <= candidate_rms.min() + tol_cents
    chosen = float(candidate_f0[np.flatnonzero(within)[-1]])
    rms, harmonics, cents = _rms_cents(freq, weights, chosen)
    all_harmonics = np.zeros(len(all_freq), dtype=np.int64)
    all_harmonics[used] = harmonics.astype(np.int64)
    all_cents = np.full(len(all_freq), np.nan)
    all_cents[used] = cents
    return HarmonicF0(
        chosen, rms, all_harmonics, all_cents, candidate_f0, candidate_rms,
        fmin, fmax, weight, tol_cents,
    )
