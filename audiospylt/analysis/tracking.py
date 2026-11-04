"""Framewise spectral peaks and weighted MQ-style partial tracking."""

from __future__ import annotations

import warnings
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

_LP_ORDER = 6  # Burg order of prediction='lp'
_LP_HISTORY = 64  # frames of log frequency it reads
_LOG2_FLOAT_RANGE = (-1074.0, 1024.0)  # open bounds of positive finite doubles


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


def _reflect_left(values: np.ndarray, length: int) -> np.ndarray:
    """Extend rows of at least two points to length with x(-i) = 2 x(0) - x(i)."""
    while values.shape[1] < length:
        take = min(values.shape[1] - 1, length - values.shape[1])
        values = np.concatenate([2 * values[:, :1] - values[:, take:0:-1], values], axis=1)
    return values


def _burg(samples: np.ndarray, valid: np.ndarray, order: int) -> np.ndarray:
    """Return Burg coefficients [1, a1, ...] for each row of samples.

    Rows are padded on the left: valid is False, and samples zero, before each
    row's first point. Every row needs more than order points.
    """
    forward = backward = samples
    coefficients = np.zeros((len(samples), order + 1))
    coefficients[:, 0] = 1.0
    for stage in range(order):
        # The padding stays as wide while the rows shrink from the left, so one
        # mask serves every stage. It drops each row's oldest forward error.
        ahead = forward[:, 1:] * valid[:, :forward.shape[1] - 1]
        behind = backward[:, :-1]
        energy = np.einsum("ij,ij->i", ahead, ahead) + np.einsum("ij,ij->i", behind, behind)
        cross = np.einsum("ij,ij->i", ahead, behind)
        reflection = np.divide(-2 * cross, energy, out=np.zeros_like(energy), where=energy > 0)
        reflection = reflection[:, np.newaxis]
        forward, backward = ahead + reflection * behind, behind + reflection * ahead
        coefficients[:, :stage + 2] += reflection * coefficients[:, stage + 1::-1]
    return coefficients


class _TrackHistory:
    """Frequency in Hz of every open track, one value per frame, newest last.

    Rows follow link_peaks' active list. All rows advance together, so a frame
    a track missed holds that track's own prediction. Columns left of a row's
    first value hold 1 Hz, whose log2 is the zero padding _burg needs.
    """

    def __init__(self) -> None:
        self.values = np.ones((0, _LP_HISTORY))
        self.filled = np.zeros(0, dtype=np.int64)

    def keep(self, rows: np.ndarray) -> None:
        self.values, self.filled = self.values[rows], self.filled[rows]

    def push(self, values: np.ndarray) -> None:
        """Advance every row by one frame."""
        self.values[:, :-1] = self.values[:, 1:]
        self.values[:, -1] = values
        self.filled = np.minimum(self.filled + 1, _LP_HISTORY)

    def add(self, values: np.ndarray) -> None:
        """Open one row per value."""
        rows = np.ones((len(values), _LP_HISTORY))
        rows[:, -1] = values
        self.values = np.concatenate([self.values, rows])
        self.filled = np.r_[self.filled, np.ones(len(values), dtype=np.int64)]

    def predict(self) -> np.ndarray:
        """Return each row's next frequency: a line plus a Burg residual step in log2.

        One-point rows and predictions outside the float range keep the stored
        value exactly, so they behave as prediction='hold' does.
        """
        result = self.values[:, -1].copy()
        moving = np.flatnonzero(self.filled > 1)
        if moving.size == 0:
            return result
        samples, filled = np.log2(self.values[moving]), self.filled[moving]
        least = _LP_ORDER + 1
        for short in range(2, least):
            rows = filled == short
            if rows.any():
                samples[rows, -least:] = _reflect_left(samples[rows, -short:], least)
        filled = np.maximum(filled, least)[:, np.newaxis]
        time = np.arange(1.0 - _LP_HISTORY, 1.0)  # newest frame at zero
        valid = time > -filled
        mean_time = (1.0 - filled) / 2
        mean = samples.sum(axis=1, keepdims=True) / filled
        centred = (samples - mean) * valid
        time_offset = (time - mean_time) * valid
        slope = (np.einsum("ij,ij->i", time_offset, centred)[:, np.newaxis]
                 / (filled * (filled ** 2 - 1) / 12))
        residual = centred - slope * time_offset
        coefficients = _burg(residual, valid, _LP_ORDER)
        step = -np.einsum("ij,ij->i", coefficients[:, 1:], residual[:, :-least:-1])
        predicted = (mean + slope * (1.0 - mean_time))[:, 0] + step
        # Falling back to the last value keeps fed-back predictions bounded.
        safe = (predicted > _LOG2_FLOAT_RANGE[0]) & (predicted < _LOG2_FLOAT_RANGE[1])
        result[moving[safe]] = np.exp2(predicted[safe])
        return result


def _link_scalar(value: float, name: str) -> float:
    """Check a finite real linking parameter, excluding strings and booleans."""
    if not is_number(value) or isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}")
    return _finite_scalar(value, name)


def _assign_frame(
    predicted: np.ndarray, frequencies: np.ndarray, max_cents: float, min_gate_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate/peak indices maximizing gain over all partial matchings."""
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        distance = np.abs(1200 * np.log2(frequencies[np.newaxis, :] / predicted[:, np.newaxis]))
        if min_gate_hz > 0:
            # Divide first: a subnormal floor would give 0 * inf on equal frequencies.
            offset = np.abs(frequencies[np.newaxis, :] - predicted[:, np.newaxis])
            distance = np.minimum(distance, offset / min_gate_hz * max_cents)
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
    # Cents are concave in Hz, so with a floor too this pairing loses no gain.
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
    prediction: str = "hold",
    min_gate_hz: float = 0.0,
) -> pd.DataFrame:
    """Link measured peaks with one weighted assignment per frame.

    Parameters
    ----------
    peaks : pandas.DataFrame
        Valid peaks table with strictly positive frequencies, in any row order.
    max_cents : float, default 50.0
        Finite positive gate. A pair's distance is the absolute cents deviation
        of the peak from the predicted frequency; see min_gate_hz for the one
        exception. Maximize total gain max_cents - distance; pairs at or beyond
        the gate never link. Equivalently, minimize total distance plus
        max_cents / 2 for every unlinked peak and every unlinked candidate.
        Rearranging links wins only when it lowers the total distance.
        Unlinked peaks and candidates remain possible.
    max_gap : int, default 0
        Maximum missed frames before closing a track. Absent frame numbers
        count as missed frames.
    min_duration : float, default 0.05
        Finite nonnegative minimum last-time minus first-time span in seconds.
        Equality is kept. A one-point track has duration zero.
    prediction : {'hold', 'lp'}, default 'hold'
        Where an open track is expected in the next frame. 'hold' uses its last
        measured frequency. 'lp' continues its motion over the last 64 frames
        of log frequency: a fitted line plus an order 6 Burg prediction of what
        the line leaves. See Notes.
    min_gate_hz : float, default 0.0
        Finite nonnegative floor under the gate: a pair also links when it lies
        within this many Hz, however many cents that is. A pair's distance is
        then the smaller of its cents and its Hz deviation scaled so that
        min_gate_hz counts as max_cents. 0 disables the floor.

    Returns
    -------
    pandas.DataFrame
        Validated tracks with exactly track_id, time, freq, amp, sorted by
        track_id then time. Kept tracks are numbered from zero in first-time,
        first-frequency order. Every point retains its measured values.
        attrs['meta'] copies input metadata and records method='link_peaks',
        objective='weighted', max_cents, max_gap, min_duration, prediction,
        and min_gate_hz.

    Notes
    -----
    Ties are common. When the linked peaks all lie on one side of the linked
    candidates, every pairing has the same total cents. Links then keep the order
    of the predicted frequencies. min_gate_hz can break such a tie, and the
    ordered pairing is then the best one. With prediction='hold', linked tracks
    do not swap order from one frame to the next; 'lp' predictions can cross,
    and their tracks with them. Equal predicted frequencies keep candidate
    creation order. Equal peak frequencies keep peak row order after sorting.

    prediction='lp' helps where a track must survive a hole while it moves,
    which needs max_gap > 0, or where a narrow gate loses a fast partial. A
    missed frame stores the track's own prediction, one value per frame, as
    the zombie peaks of Lagrange, Marchand, Raspaud and Rault (DAFx 2003) do.
    A track of two to six points is extended by their left reflection; a
    one-point track holds. This is not their form, which runs Burg on the raw
    frequencies: that form bridged fewer holes in moving partials than 'hold'
    did, so the line carries the motion here. Measured points alone reach the
    result, and a prediction outside the float range falls back to the last
    value.

    Prediction is an assumption, not a measurement. While two partials merge
    into one peak, the tracker has one frequency for both and cannot tell a
    crossing from two partials that turn back. 'lp' assumes the motion goes
    on, so it follows more crossings but swaps partials that turn back.
    'hold' does the reverse.

    min_gate_hz matters at low frequencies, where a cents gate is narrower than
    the analysis resolves: at 44.1 kHz with frame 2048 and zero_pad 4, one
    5.4 Hz spectrum step is about 220 cents at 40 Hz. Resolved low partials
    track without it. It adds tracks on low noise-like material, and those
    tracks follow noise paths, not partials.

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
        max_cents, max_gap, min_duration, prediction, or min_gate_hz.
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
    if not isinstance(prediction, str) or prediction not in ("hold", "lp"):
        raise ValueError(f"prediction must be 'hold' or 'lp'; got {prediction!r}")
    min_gate_hz = _link_scalar(min_gate_hz, "min_gate_hz")
    if min_gate_hz < 0:
        raise ValueError("min_gate_hz must be >= 0")
    table = table.sort_values(["frame", "freq", "amp"], kind="stable").reset_index(drop=True)
    frame = table.frame.to_numpy()
    time, freq, amp = (table[name].to_numpy() for name in ("time", "freq", "amp"))
    tracks: list[list[int]] = []
    active: list[int] = []
    history = _TrackHistory() if prediction == "lp" else None
    previous = -1
    starts = np.r_[0, np.flatnonzero(frame[1:] != frame[:-1]) + 1, len(frame)]
    for start, stop in zip(starts[:-1], starts[1:]):
        if start == stop:
            continue
        k = int(frame[start])
        open_tracks = [int(frame[tracks[i][-1]]) >= k - 1 - max_gap for i in active]
        active = [i for i, still_open in zip(active, open_tracks) if still_open]
        if history is not None:
            history.keep(np.array(open_tracks, dtype=bool))
            if active:  # any frame numbers skipped since `previous` are within max_gap
                for _ in range(k - 1 - previous):
                    history.push(history.predict())
            predicted = history.predict()
        else:
            predicted = np.array([_predict_frequency(float(freq[tracks[i][-1]])) for i in active])
        candidates, matched = _assign_frame(predicted, freq[start:stop], max_cents, min_gate_hz)
        for candidate, peak in zip(candidates, matched):
            tracks[active[candidate]].append(int(start + peak))
        used = set(matched.tolist())
        new_peaks = [peak for peak in range(stop - start) if peak not in used]
        if history is not None:
            predicted[candidates] = freq[start + matched]
            history.push(predicted)
            history.add(freq[start + np.array(new_peaks, dtype=np.int64)])
        for peak in new_peaks:
            active.append(len(tracks))
            tracks.append([int(start + peak)])
        previous = k
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
        "prediction": prediction, "min_gate_hz": min_gate_hz,
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
    min_duration: float = 0.1,
    prediction: str = "hold",
    min_gate_hz: float = 0.0,
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
    min_duration : float, default 0.1
        Minimum measured track span in seconds, as for link_peaks.
    prediction : {'hold', 'lp'}, default 'hold'
        Where an open track is expected next, as for link_peaks.
    min_gate_hz : float, default 0.0
        Hz floor under the linking gate, as for link_peaks.

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

    Warns
    -----
    UserWarning
        The result is empty, so synthesizing it gives no samples. The message
        names the settings to try. link_peaks alone does not warn.

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
    result = link_peaks(peaks, max_cents=max_cents, max_gap=max_gap, min_duration=min_duration,
                        prediction=prediction, min_gate_hz=min_gate_hz)
    result.attrs["meta"]["zero_pad"] = zero_pad
    if result.empty:
        warnings.warn(
            f"track_partials found no tracks in {len(peaks)} peaks. Try a lower min_db, "
            "prominence_db=None, a larger max_cents, min_gate_hz for low material, "
            "or a shorter min_duration.",
            UserWarning,
            stacklevel=2,
        )
    return result
