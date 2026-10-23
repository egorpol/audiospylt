"""Conversions between v1 table kinds."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from audiospylt.model.schema import validate

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from audiospylt.analysis.f0 import F0Contour

__all__ = ["snapshot_to_events", "f0_to_tracks", "tracks_to_events", "tracks_to_chords", "tracks_at"]


def snapshot_to_events(snapshot: pd.DataFrame, time_start: float, time_stop: float) -> pd.DataFrame:
    """Hold every partial of a snapshot constant over one time span.

    Parameters
    ----------
    snapshot : pandas.DataFrame
        A v1 ``snapshot`` table (validated), e.g. from ``find_peaks``.
    time_start, time_stop : float
        Span in seconds shared by every event, with ``0 <= time_start <=
        time_stop``.

    Returns
    -------
    pandas.DataFrame
        A validated ``events`` table, one row per snapshot row in the same
        order, with ``freq_start == freq_stop == freq`` and ``amp_start ==
        amp_stop == amp``. Other snapshot columns are dropped; a copy of
        ``attrs["meta"]`` (analysis provenance) is kept.

    Raises
    ------
    SchemaError
        Invalid snapshot.
    ValueError
        Times that are not finite or not ``0 <= time_start <= time_stop``.
    """
    table = validate(snapshot, "snapshot")
    if not (np.isfinite(time_start) and np.isfinite(time_stop) and 0 <= time_start <= time_stop):
        raise ValueError(
            f"need finite 0 <= time_start <= time_stop; got {time_start!r}, {time_stop!r}"
        )
    count = len(table)
    freq, amp = table["freq"].to_numpy(), table["amp"].to_numpy()
    events = validate(pd.DataFrame({
        "time_start": np.full(count, time_start, dtype=np.float64),
        "time_stop": np.full(count, time_stop, dtype=np.float64),
        "freq_start": freq, "freq_stop": freq.copy(),
        "amp_start": amp, "amp_stop": amp.copy(),
    }), "events")
    if "meta" in table.attrs:
        events.attrs["meta"] = dict(table.attrs["meta"])
    return events


def f0_to_tracks(contour: F0Contour) -> pd.DataFrame:
    """Turn the voiced frames of an f0 contour into a tracks table.

    Parameters
    ----------
    contour : F0Contour
        Result of ``analysis.f0.pyin``.

    Returns
    -------
    pandas.DataFrame
        A validated ``tracks`` table with one row per voiced frame, in time
        order: ``time``, ``freq`` = f0, ``amp`` = the fundamental's D4
        amplitude (0 where the fundamental is missing, so those frames draw
        and synthesise silent), and ``track_id`` counting voiced runs from 0
        (an unvoiced frame ends a run). Extra columns ``voiced_prob`` and
        ``refined`` copy the contour's. ``attrs["meta"]`` records the method
        and analysis settings. No voiced frame gives an empty table.
    """
    voiced = np.asarray(contour.voiced, dtype=bool)
    previous = np.zeros_like(voiced)
    previous[1:] = voiced[:-1]
    track_id = np.cumsum(voiced & ~previous) - 1
    rows = np.flatnonzero(voiced)
    tracks = validate(pd.DataFrame({
        "track_id": track_id[rows].astype(np.int64),
        "time": contour.times[rows],
        "freq": contour.f0[rows],
        "amp": contour.amp[rows],
        "voiced_prob": contour.voiced_prob[rows],
        "refined": np.asarray(contour.refined, dtype=bool)[rows],
    }), "tracks")
    tracks.attrs["meta"] = {
        "method": "pyin", "sr": contour.sr, "frame_length": contour.frame_length,
        "hop": contour.hop, "fmin": contour.fmin, "fmax": contour.fmax,
    }
    return tracks


def tracks_to_events(tracks: pd.DataFrame) -> pd.DataFrame:
    """Turn each step between two points of a track into one events row.

    Every pair of consecutive rows of one track (in row order, which the
    schema makes increasing in time) becomes a row from the first point to the
    second, so frequency and amplitude move linearly between the points.
    Neighbouring rows of a track share their boundary time, frequency and
    amplitude, and each keeps its ``track_id``, so ``synthesize`` in its
    default ``'continuous'`` mode plays a track as one phase-continuous voice
    with no amplitude ramps inside it, and never links two tracks.

    Parameters
    ----------
    tracks : pandas.DataFrame
        A v1 ``tracks`` table (validated), e.g. from ``f0_to_tracks``.

    Returns
    -------
    pandas.DataFrame
        A validated ``events`` table plus an int64 ``track_id`` column, one
        row per consecutive pair of points, in the input row order of each
        pair's first point. A track with a single point gives no row. The
        ``phase`` column and other extra columns are dropped (``'continuous'``
        integrates the phase from the frequencies); a copy of
        ``attrs["meta"]`` is kept.

    Raises
    ------
    SchemaError
        Invalid tracks table.

    Notes
    -----
    A track starts and stops at the amplitude of its first and last point.
    Unless those are 0, pass ``synthesize`` a ``fade`` of a few milliseconds,
    or each track end clicks.
    """
    table = validate(tracks, "tracks")
    ids = table["track_id"].to_numpy(dtype=np.int64)
    following = pd.Series(np.arange(len(table))).groupby(ids, sort=False).shift(-1).to_numpy()
    rows = np.flatnonzero(~np.isnan(following))
    nxt = following[rows].astype(np.int64)
    time, freq, amp = (table[name].to_numpy() for name in ("time", "freq", "amp"))
    events = validate(pd.DataFrame({
        "time_start": time[rows], "time_stop": time[nxt],
        "freq_start": freq[rows], "freq_stop": freq[nxt],
        "amp_start": amp[rows], "amp_stop": amp[nxt],
        "track_id": ids[rows],
    }), "events")
    if "meta" in table.attrs:
        events.attrs["meta"] = dict(table.attrs["meta"])
    return events


def tracks_to_chords(
    tracks: pd.DataFrame,
    windows: float | ArrayLike,
    *,
    merge_cents: float | None = 50.0,
    min_db: float | None = None,
    max_notes: int | None = None,
) -> pd.DataFrame:
    """Reduce tracks to one chord per time window, for notation.

    Every track that sounds in a window gives one note held over the whole
    window. Its frequency is the amplitude-weighted median of the track's
    frequencies in the window, and its amplitude the track's time average over
    the window, so a track heard for part of the window comes out that much
    quieter. In each window the loudest note is kept, then each next loudest
    unless it lies within ``merge_cents`` of a note already kept: this drops
    the copies of one partial that an analysis split over several tracks.
    ``min_db`` and ``max_notes`` thin the chords further. ``events_to_mei``
    writes the result as one chord per window.

    Parameters
    ----------
    tracks : pandas.DataFrame
        A v1 ``tracks`` table (validated), e.g. from ``read_partiels`` or
        ``f0_to_tracks``.
    windows : float or array_like
        A step in seconds, for windows ``[0, step)``, ``[step, 2 step)``, ...
        up to the table's last point; or increasing boundary times ``>= 0``,
        outside which points are ignored. Windows are half-open, except the
        last, which includes its end.
    merge_cents : float or None, default 50
        Drop a note within this many cents of a louder kept note of its
        window. None keeps one note per track.
    min_db : float or None, default None
        Drop notes more than ``-min_db`` dB below the window's loudest note
        (``min_db <= 0``).
    max_notes : int or None, default None
        Keep at most this many of the loudest notes per window.

    Returns
    -------
    pandas.DataFrame
        A validated ``events`` table plus an int64 ``track_id`` column (the
        track each note comes from): one row per note, ``time_start`` and
        ``time_stop`` the window's bounds, frequency and amplitude constant,
        sorted by window, then frequency. A window where no track sounds gives
        no row. Other columns are dropped; a copy of ``attrs["meta"]`` is kept.

    Raises
    ------
    SchemaError
        Invalid tracks table.
    ValueError
        Invalid ``windows`` or options.

    Notes
    -----
    A track's amplitude in a window is the mean amplitude of its points there
    times the share of the window they cover: their number times the frame
    hop, at most 1. The hop is the median time step between consecutive
    points of a track, over the whole table. The median frequency is the lower
    weighted median, the frequency of one of the track's points (the plain
    median where all amplitudes are 0); points at 0 Hz are ignored. Where an
    analysis jumps an octave for a few frames, the median stays on the line,
    while a mean would write a pitch between the two.
    """
    table = validate(tracks, "tracks")
    if merge_cents is not None and not (np.isfinite(merge_cents) and merge_cents >= 0):
        raise ValueError(f"merge_cents must be a number >= 0 or None, got {merge_cents!r}")
    if min_db is not None and not (np.isfinite(min_db) and min_db <= 0):
        raise ValueError(f"min_db must be a number <= 0 or None, got {min_db!r}")
    if max_notes is not None and (isinstance(max_notes, bool) or not isinstance(max_notes, Integral)
                                  or max_notes < 1):
        raise ValueError(f"max_notes must be an int >= 1 or None, got {max_notes!r}")
    ids = table["track_id"].to_numpy(dtype=np.int64)
    time, freq, amp = (table[name].to_numpy() for name in ("time", "freq", "amp"))
    edges = _window_edges(windows, time)
    last = len(edges) - 2
    window = np.searchsorted(edges, time, side="right") - 1
    window[time == edges[-1]] = last
    rows = np.flatnonzero((freq > 0) & (window >= 0) & (window <= last))
    # Points grouped by (window, track), each group in rising frequency.
    rows = rows[np.lexsort((freq[rows], ids[rows], window[rows]))]
    win, tid, f, a = window[rows], ids[rows], freq[rows], amp[rows]
    new = np.ones(len(rows), dtype=bool)
    new[1:] = (win[1:] != win[:-1]) | (tid[1:] != tid[:-1])
    start = np.flatnonzero(new)
    count = np.diff(np.append(start, len(rows)))
    group = np.repeat(np.arange(len(start)), count)
    total = np.bincount(group, weights=a, minlength=len(start))
    weight = np.where(total[group] > 0, a, 1.0)
    within = pd.Series(weight).groupby(group).cumsum().to_numpy()
    half = within[start + count - 1] / 2
    position = np.where(within >= half[group], np.arange(len(rows)), len(rows))
    median = np.minimum.reduceat(position, start) if len(start) else start
    note_win, note_id, note_freq = win[start], tid[start], f[median]
    note_amp = total / np.maximum(count, 1)
    hop = _median_hop(ids, time)
    span = edges[note_win + 1] - edges[note_win]
    if hop is not None:
        timed = span > 0
        note_amp[timed] *= np.minimum(1.0, count[timed] * hop / span[timed])
    cents = 1200 * np.log2(note_freq)

    # Loudest first in each window (ties: lower frequency, then track).
    kept: list[int] = []
    chosen: list[float] = []
    current, floor = -1, 0.0
    for k in np.lexsort((note_id, note_freq, -note_amp, note_win)):
        if note_win[k] != current:
            current, chosen = note_win[k], []
            floor = note_amp[k] * 10 ** (min_db / 20) if min_db is not None else 0.0
        if (max_notes is not None and len(chosen) == max_notes) or note_amp[k] < floor:
            continue
        if merge_cents is not None and any(abs(cents[k] - c) <= merge_cents for c in chosen):
            continue
        chosen.append(cents[k])
        kept.append(k)
    keep = np.asarray(kept, dtype=np.int64)
    keep = keep[np.lexsort((note_id[keep], note_freq[keep], note_win[keep]))]
    w = note_win[keep]
    events = validate(pd.DataFrame({
        "time_start": edges[w], "time_stop": edges[w + 1],
        "freq_start": note_freq[keep], "freq_stop": note_freq[keep].copy(),
        "amp_start": note_amp[keep], "amp_stop": note_amp[keep].copy(),
        "track_id": note_id[keep],
    }), "events")
    if "meta" in table.attrs:
        events.attrs["meta"] = dict(table.attrs["meta"])
    return events


def tracks_at(tracks: pd.DataFrame, time: float) -> pd.DataFrame:
    """Read the partials of a tracks table at one time, as a snapshot.

    Every track whose points span ``time`` gives one row, its frequency and
    amplitude interpolated linearly between its points around ``time`` (as
    ``tracks_to_events`` and ``synthesize`` play it). A track with a point at
    ``time`` gives that point's values.

    Parameters
    ----------
    tracks : pandas.DataFrame
        A v1 ``tracks`` table (validated).
    time : float
        Time in seconds.

    Returns
    -------
    pandas.DataFrame
        A validated ``snapshot`` table (``freq``, ``amp``) plus an int64
        ``track_id`` column, sorted by frequency, then track. ``phase`` and
        other columns are dropped; a copy of ``attrs["meta"]`` is kept. No
        track at ``time`` gives an empty table.

    Raises
    ------
    SchemaError
        Invalid tracks table.
    ValueError
        ``time`` is not a finite number.
    """
    table = validate(tracks, "tracks")
    if isinstance(time, bool) or not isinstance(time, Real) or not np.isfinite(time):
        raise ValueError(f"time must be a finite number, got {time!r}")
    ids = table["track_id"].to_numpy(dtype=np.int64)
    t, freq, amp = (table[name].to_numpy() for name in ("time", "freq", "amp"))
    following = pd.Series(np.arange(len(table))).groupby(ids, sort=False).shift(-1).to_numpy()
    final = np.isnan(following)
    nxt = np.where(final, np.arange(len(table)), following).astype(np.int64)
    inside = ~final & (t <= time) & (time < t[nxt])
    rows = np.flatnonzero(inside | (final & (t == time)))
    after = nxt[rows]
    frac = np.zeros(len(rows))
    mid = inside[rows]
    frac[mid] = (time - t[rows[mid]]) / (t[after[mid]] - t[rows[mid]])
    order = np.lexsort((ids[rows], freq[rows] + frac * (freq[after] - freq[rows])))
    rows, after, frac = rows[order], after[order], frac[order]
    snapshot = validate(pd.DataFrame({
        "freq": freq[rows] + frac * (freq[after] - freq[rows]),
        "amp": amp[rows] + frac * (amp[after] - amp[rows]),
        "track_id": ids[rows],
    }), "snapshot")
    if "meta" in table.attrs:
        snapshot.attrs["meta"] = dict(table.attrs["meta"])
    return snapshot


def _window_edges(windows: float | ArrayLike, time: np.ndarray) -> np.ndarray:
    """Window boundaries from a step (grid from 0 to the last time) or boundary times."""
    if np.ndim(windows) == 0:
        step = float(windows) if isinstance(windows, Real) and not isinstance(windows, bool) else np.nan
        if not (np.isfinite(step) and step > 0):
            raise ValueError(f"windows must be a step > 0 in seconds or boundary times, got {windows!r}")
        end = float(time.max()) if time.size else 0.0
        # A last window shorter than a billionth of a step is float noise in end / step.
        count = max(1, int(np.ceil(end / step - 1e-9)))
        return np.append(step * np.arange(count), end)
    try:
        edges = np.asarray(windows, dtype=np.float64)
    except (TypeError, ValueError):
        edges = np.zeros(0)
    if (edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all() or edges[0] < 0
            or (np.diff(edges) <= 0).any()):
        raise ValueError("windows must be a step > 0 in seconds or at least two increasing "
                         f"boundary times >= 0, got {windows!r}")
    return edges


def _median_hop(ids: np.ndarray, time: np.ndarray) -> float | None:
    """Median time step between consecutive points of a track; None without one."""
    steps = time - pd.Series(time).groupby(ids, sort=False).shift().to_numpy()
    steps = steps[~np.isnan(steps)]
    return float(np.median(steps)) if steps.size else None
