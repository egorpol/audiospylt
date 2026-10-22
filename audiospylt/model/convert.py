"""Conversions between v1 table kinds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from audiospylt.model.schema import validate

if TYPE_CHECKING:
    from audiospylt.analysis.f0 import F0Contour

__all__ = ["snapshot_to_events", "f0_to_tracks"]


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
