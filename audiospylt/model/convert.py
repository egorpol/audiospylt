"""Conversions between v1 table kinds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audiospylt.model.schema import validate

__all__ = ["snapshot_to_events"]


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
