"""Framewise spectral peaks and weighted MQ-style partial tracking."""

from __future__ import annotations

from dataclasses import asdict
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_number
from scipy.optimize import linear_sum_assignment

from audiospylt.analysis.peaks import _frame_peaks, _peak_settings
from audiospylt.analysis.spectrogram import Spectrogram, spectrogram
from audiospylt.analysis.spectrum import _finite_scalar
from audiospylt.model.schema import validate

__all__ = ["spectrogram_peaks", "link_peaks", "track_partials"]


def spectrogram_peaks(
    spec: Spectrogram,
    *,
    min_db: float | None = -60.0,
    rel_db: float | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
    min_distance_hz: float | None = None,
    prominence_db: float | None = None,
    interpolate: bool = True,
    floor_db: float = -120.0,
) -> pd.DataFrame:
    """Pick spectral peaks in every frame using the find_peaks array core.

    Parameters
    ----------
    spec : Spectrogram
        D4 amplitude spectrogram. Frames without peaks contribute no rows.
    min_db : float or None, default -60.0
        Absolute amplitude threshold in dB re full scale; None disables it.
    rel_db : float or None
        Nonpositive offset from each frame's own loudest frequency/absolute
        survivor, rather than the loudest peak across the whole spectrogram.
    fmin, fmax : float, optional
        Inclusive interpolated-frequency bounds in Hz; fmax defaults to sr/2.
        Require 0 <= fmin < fmax.
    min_distance_hz : float or None
        Positive minimum spacing, greedily keeping the loudest survivors.
    prominence_db : float or None
        Nonnegative minimum whole-spectrum prominence in dB.
    interpolate : bool, default True
        Use QIFFT-dB interpolation; False keeps raw bins.
    floor_db : float, default -120.0
        Finite floor shared by detection, interpolation, and prominence.

    Returns
    -------
    pandas.DataFrame
        Validated peaks table sorted by frame then freq, with a RangeIndex.
        Columns are frame, time, freq, amp, bin (nullable Int64), prominence.
        Empty results retain all columns and dtypes. attrs['meta'] records
        analysis dimensions, window, interpolation, and peak settings.

    Notes
    -----
    Each frame equals find_peaks(spec.frame(i), **same_options) bit for bit.
    Peak accuracy and filter order: see find_peaks Notes and Returns.

    Raises
    ------
    TypeError
        Input is not a Spectrogram. Use find_peaks for a single Spectrum.
    ValueError
        Invalid peak settings or amplitudes, as in find_peaks.
    """
    if not isinstance(spec, Spectrogram):
        raise TypeError(f"spec must be a Spectrogram; got {type(spec).__name__}; "
                        "use find_peaks for a single Spectrum")
    settings = _peak_settings(
        spec.sr, min_db=min_db, rel_db=rel_db, fmin=fmin, fmax=fmax,
        min_distance_hz=min_distance_hz, prominence_db=prominence_db,
        interpolate=interpolate, floor_db=floor_db,
    )
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    columns: tuple[list[np.ndarray], ...] = ([], [], [], [])
    for i, time in enumerate(spec.times):
        values = _frame_peaks(spec.amp[:, i], spec.sr / spec.n_fft, settings)
        frames.append(np.full(len(values[0]), i, dtype=np.int64))
        times.append(np.full(len(values[0]), time, dtype=np.float64))
        for parts, array in zip(columns, values):
            parts.append(array)

    def joined(parts: list[np.ndarray], dtype: str = "float64") -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

    result = validate(pd.DataFrame({
        "frame": joined(frames, "int64"), "time": joined(times),
        "freq": joined(columns[0]), "amp": joined(columns[1]),
        "bin": pd.array(joined(columns[2], "int64"), dtype="Int64"),
        "prominence": joined(columns[3]),
    }), "peaks")
    options = asdict(settings)
    options.pop("interpolate")
    result.attrs["meta"] = {
        "sr": spec.sr, "frame_length": spec.frame_length, "n_fft": spec.n_fft,
        "window": str(spec.window), "interpolation": "qifft-db" if interpolate else "none",
        "hop": spec.hop, "center": spec.center, "n_frames": len(spec.times), **options,
    }
    return result


def _predict_frequency(last_frequency: float) -> float:
    """Predict the next point by holding the last measured frequency."""
    return last_frequency


def _link_scalar(value: float, name: str) -> float:
    """Check a finite real linking parameter, excluding strings and booleans."""
    if not is_number(value) or isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}")
    return _finite_scalar(value, name)


def _assign_frame(
    predicted: np.ndarray, frequencies: np.ndarray, max_cents: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate/peak indices maximizing gain over all partial matchings."""
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        distance = np.abs(1200 * np.log2(frequencies[np.newaxis, :] / predicted[:, np.newaxis]))
    gain = np.maximum(max_cents - distance, 0.0)
    # Each candidate can take a zero-gain dummy column. Peak columns can stay
    # unused too, even when an entirely in-gate full matching exists.
    cost = np.zeros((len(predicted), len(frequencies) + len(predicted)))
    cost[:, :len(frequencies)] = -gain
    candidates, peaks = linear_sum_assignment(cost)
    real = peaks < len(frequencies)
    candidates, peaks = candidates[real], peaks[real]
    linked = gain[candidates, peaks] > 0
    candidates, peaks = candidates[linked], peaks[linked]
    ordered_candidates = candidates[np.argsort(predicted[candidates], kind="stable")]
    ordered_peaks = np.sort(peaks)
    ordered_peaks = ordered_peaks[np.argsort(frequencies[ordered_peaks], kind="stable")]
    if np.all(distance[ordered_candidates, ordered_peaks] < max_cents):
        return ordered_candidates, ordered_peaks
    # Rounding can put an ordered pair at the gate; keep the solver's valid pairs.
    return candidates, peaks


def link_peaks(
    peaks: pd.DataFrame,
    *,
    max_cents: float = 50.0,
    max_gap: int = 0,
    min_duration: float = 0.05,
) -> pd.DataFrame:
    """Link measured peaks with one weighted assignment per frame.

    Parameters
    ----------
    peaks : pandas.DataFrame
        Valid peaks table with strictly positive frequencies, in any row order.
    max_cents : float, default 50.0
        Finite positive gate. Maximize total gain max_cents - absolute cents
        deviation; pairs at or beyond the gate never link. Equivalently,
        minimize total cents plus max_cents / 2 for every unlinked peak and
        every unlinked candidate. Rearranging links wins only when it lowers
        the total deviation. Unlinked peaks and candidates remain possible.
    max_gap : int, default 0
        Maximum missed frames before closing a track. Absent frame numbers
        count as missed frames. Prediction holds the last measured frequency.
    min_duration : float, default 0.05
        Finite nonnegative minimum last-time minus first-time span in seconds.
        Equality is kept. A one-point track has duration zero.

    Returns
    -------
    pandas.DataFrame
        Validated tracks with exactly track_id, time, freq, amp, sorted by
        track_id then time. Kept tracks are numbered from zero in first-time,
        first-frequency order. Every point retains its measured values.
        attrs['meta'] copies input metadata and records method='link_peaks',
        objective='weighted', max_cents, max_gap, and min_duration.

    Notes
    -----
    Ties are common. When the linked peaks all lie on one side of the linked
    candidates, every pairing has the same total. Links then keep the order
    of the predicted frequencies. With the current last-frequency prediction,
    linked tracks do not swap order from one frame to the next. Equal
    predicted frequencies keep candidate creation order. Equal peak
    frequencies keep peak row order after sorting.

    A track that resumes after missed frames has no rows in the gap, so
    tracks_to_events and synthesize glide linearly across it; max_gap=0 splits
    instead. Tracks end at their last measured point, and synthesize fades
    voice ends by default. One-point tracks give no events (tracks_to_events
    needs two points), hence min_duration > 0 by default.

    Raises
    ------
    SchemaError
        Input violates the peaks contract.
    ValueError
        Nonpositive frequencies (with zero-based row positions), or invalid
        max_cents, max_gap, or min_duration.
    """
    table = validate(peaks, "peaks")
    bad = np.flatnonzero(table.freq.to_numpy() <= 0)
    if bad.size:
        raise ValueError(f"freq must be > 0; row positions {bad.tolist()}")
    max_cents = _link_scalar(max_cents, "max_cents")
    if max_cents <= 0:
        raise ValueError("max_cents must be > 0")
    if isinstance(max_gap, (bool, np.bool_)) or not isinstance(max_gap, Integral) or max_gap < 0:
        raise ValueError("max_gap must be an integer >= 0")
    max_gap = int(max_gap)
    min_duration = _link_scalar(min_duration, "min_duration")
    if min_duration < 0:
        raise ValueError("min_duration must be >= 0")
    table = table.sort_values(["frame", "freq", "amp"], kind="stable").reset_index(drop=True)
    frame = table.frame.to_numpy()
    time, freq, amp = (table[name].to_numpy() for name in ("time", "freq", "amp"))
    tracks: list[list[int]] = []
    active: list[int] = []
    starts = np.r_[0, np.flatnonzero(frame[1:] != frame[:-1]) + 1, len(frame)]
    for start, stop in zip(starts[:-1], starts[1:]):
        if start == stop:
            continue
        k = int(frame[start])
        active = [i for i in active if int(frame[tracks[i][-1]]) >= k - 1 - max_gap]
        predicted = np.array([_predict_frequency(float(freq[tracks[i][-1]])) for i in active])
        candidates, matched = _assign_frame(predicted, freq[start:stop], max_cents)
        for candidate, peak in zip(candidates, matched):
            tracks[active[candidate]].append(int(start + peak))
        used = set(matched.tolist())
        for peak in range(stop - start):
            if peak not in used:
                active.append(len(tracks))
                tracks.append([int(start + peak)])
    kept = [rows for rows in tracks if time[rows[-1]] - time[rows[0]] >= min_duration]
    kept.sort(key=lambda rows: (time[rows[0]], freq[rows[0]]))
    rows = np.array([row for points in kept for row in points], dtype=np.int64)
    ids = np.repeat(np.arange(len(kept), dtype=np.int64), [len(points) for points in kept])
    result = validate(pd.DataFrame({
        "track_id": ids, "time": time[rows], "freq": freq[rows], "amp": amp[rows],
    }), "tracks")
    result.attrs["meta"] = {
        **peaks.attrs.get("meta", {}), "method": "link_peaks", "objective": "weighted",
        "max_cents": max_cents, "max_gap": max_gap, "min_duration": min_duration,
    }
    return result


def track_partials(
    audio: np.ndarray,
    sr: int,
    *,
    frame_length: int = 2048,
    hop: int = 256,
    zero_pad: int = 4,
    window: str | tuple | float = "hann",
    center: bool = True,
    min_db: float | None = -80.0,
    rel_db: float | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
    min_distance_hz: float | None = None,
    prominence_db: float | None = 20.0,
    max_cents: float = 50.0,
    max_gap: int = 0,
    min_duration: float = 0.05,
) -> pd.DataFrame:
    """Analyze mono audio into partial tracks through spectral peaks.

    Parameters
    ----------
    audio : numpy.ndarray
        Finite 1-D real samples, as for spectrogram.
    sr : int
        Positive sample rate in Hz.
    frame_length, hop, zero_pad : int
        STFT window length, frame spacing, and FFT multiplier in samples.
        Defaults are 2048, 256, and 4.
    window : str, tuple, or float, default 'hann'
        Periodic scipy.signal.get_window window.
    center : bool, default True
        Pad audio and place frame centers at k*hop/sr, as for spectrogram.
    min_db : float or None, default -80.0
        Absolute D4 peak threshold in dB re full scale.
    rel_db : float or None
        Nonpositive offset from each frame's own loudest survivor.
    fmin, fmax : float, optional
        Inclusive peak bounds in Hz; defaults are 0 and sr/2.
    min_distance_hz : float or None
        Positive minimum peak spacing, as for spectrogram_peaks.
    prominence_db : float or None, default 20.0
        Nonnegative whole-spectrum prominence floor in dB.
    max_cents : float, default 50.0
        Positive linking gate, as for link_peaks.
    max_gap : int, default 0
        Maximum number of missed frames, as for link_peaks.
    min_duration : float, default 0.05
        Minimum measured track span in seconds, as for link_peaks.

    Returns
    -------
    pandas.DataFrame
        Exactly link_peaks(spectrogram_peaks(spectrogram(audio, sr, ...),
        ...), ...), with the same errors. attrs['meta'] also records zero_pad.

    Notes
    -----
    With Hann at frame 2048 / hop 256 / zero_pad 4, linking plus a minimum
    length alone does not remove window side lobes. A 20 dB prominence floor
    removes most of them. window='blackmanharris' removes them and suits
    sparse spectra, at the cost of a main lobe twice as wide (8 vs 4 bins),
    which merges close partials sooner. Peak accuracy: see find_peaks Notes.

    Raises
    ------
    SchemaError
        Invalid intermediate table, as for spectrogram_peaks or link_peaks.
    ValueError
        Invalid audio or analysis, peak-picking, or linking parameters.
    """
    spec = spectrogram(audio, sr, frame_length=frame_length, hop=hop,
                       window=window, zero_pad=zero_pad, center=center)
    peaks = spectrogram_peaks(spec, min_db=min_db, rel_db=rel_db, fmin=fmin, fmax=fmax,
                              min_distance_hz=min_distance_hz, prominence_db=prominence_db)
    result = link_peaks(peaks, max_cents=max_cents, max_gap=max_gap, min_duration=min_duration)
    result.attrs["meta"]["zero_pad"] = zero_pad
    return result
