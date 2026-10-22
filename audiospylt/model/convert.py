"""Conversions between v1 table kinds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from audiospylt.model.schema import validate

if TYPE_CHECKING:
    from audiospylt.analysis.f0 import F0Contour

__all__ = ["snapshot_to_events", "f0_to_tracks", "tracks_to_events"]


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
