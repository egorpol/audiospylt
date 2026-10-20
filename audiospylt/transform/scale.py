"""Pitch and time transforms of validated v1 tables."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_number

from audiospylt.model.schema import TABLE_CONTRACTS, infer_kind, validate


def _finite_scalar(value: float, name: str) -> float:
    """Accept finite real numeric scalars, excluding booleans and strings."""
    try:
        if (np.ndim(value) != 0 or not is_number(value)
                or isinstance(value, (bool, np.bool_, complex, np.complexfloating))):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


def _table(table: pd.DataFrame, kind: str | None) -> tuple[pd.DataFrame, str]:
    if kind is None:
        hint = getattr(table, "attrs", {}).get("audiospylt_table")
        kind = hint if isinstance(hint, str) and hint in TABLE_CONTRACTS else infer_kind(table)
    return validate(table, kind), kind


def _factors(value, name: str, kind: str, count: int, *, positive: bool) -> np.ndarray:
    def check(item):
        number = _finite_scalar(item, name)
        if positive and number <= 0:
            raise ValueError(f"{name} must be > 0; got {item!r}")
        return number

    if np.ndim(value) == 0:
        return np.full(count, check(value), dtype=np.float64)
    if kind != "events":
        raise ValueError(f"{name} sequences are only supported for events; got {kind}")
    if np.ndim(value) != 1:
        raise ValueError(f"{name} must be a scalar or a 1-D sequence")
    if len(value) != count:
        raise ValueError(f"{name} length ({len(value)}) must match number of time cues ({count})")
    return np.array([check(item) for item in value], dtype=np.float64)


def transpose(
    table: pd.DataFrame, factor=1.0, *, cents=0.0, fmin=None, fmax=None,
    out_of_range="clip", kind=None,
) -> pd.DataFrame:
    """Scale every frequency column by ``factor * 2**(cents/1200)``.

    Parameters
    ----------
    table : pandas.DataFrame
        Snapshot, events, or tracks table. Extra columns and attrs are kept.
    factor, cents : float or sequence of float
        Positive finite multiplier and finite pitch offset. Events also accept
        one value per distinct time_start, in ascending cue order.
    fmin, fmax : float or None
        Optional finite, nonnegative frequency bounds in Hz.
    out_of_range : {'clip', 'drop', 'keep'}
        Clamp columns, remove rows with any overshoot, or retain scaled values.
    kind : str or None
        Explicit kind; otherwise use a registered attrs hint, then infer_kind.

    Returns
    -------
    pandas.DataFrame
        Validated independent table, in input row order except dropped rows.

    Warns
    -----
    UserWarning
        Once when scaled frequencies exceed bounds, with row counts and range.

    Raises
    ------
    SchemaError
        Invalid input or resulting table.
    ValueError
        Invalid factors, cue lengths, bounds, or out_of_range policy.
    """
    result, kind = _table(table, kind)
    if not isinstance(out_of_range, str) or out_of_range not in ("clip", "drop", "keep"):
        raise ValueError("out_of_range must be one of 'clip', 'drop', 'keep'")
    bounds = []
    for value, name in ((fmin, "fmin"), (fmax, "fmax")):
        number = None if value is None else _finite_scalar(value, name)
        if number is not None and number < 0:
            raise ValueError(f"{name} must be >= 0")
        bounds.append(number)
    fmin, fmax = bounds
    if fmin is not None and fmax is not None and fmin > fmax:
        raise ValueError("fmin must be <= fmax")
    cues = np.unique(result.time_start) if kind == "events" else None
    count = len(cues) if cues is not None else 1
    factors = _factors(factor, "factor", kind, count, positive=True)
    offsets = _factors(cents, "cents", kind, count, positive=False)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        multipliers = factors * 2 ** (offsets / 1200)
    if np.any(~np.isfinite(multipliers) | (multipliers <= 0)):
        raise ValueError("combined pitch factors must be finite and > 0")
    row_factor = (multipliers[np.searchsorted(cues, result.time_start)]
                  if cues is not None else multipliers[0])
    columns = ["freq_start", "freq_stop"] if kind == "events" else ["freq"]
    with np.errstate(over="ignore"):
        for column in columns:
            result[column] = result[column] * row_factor
    below = (result[columns].lt(fmin).any(axis=1) if fmin is not None
             else pd.Series(False, index=result.index))
    above = (result[columns].gt(fmax).any(axis=1) if fmax is not None
             else pd.Series(False, index=result.index))
    outside = below | above
    if outside.any():
        warnings.warn(
            f"{int(outside.sum())} rows outside pitch bounds "
            f"[{fmin if fmin is not None else '-inf'}, {fmax if fmax is not None else 'inf'}]."
            f" (below: {int(below.sum())}, above: {int(above.sum())}; "
            f"observed min/max: {result[columns].min().min():g}/{result[columns].max().max():g})",
            UserWarning, stacklevel=2,
        )
    if out_of_range == "clip" and (fmin is not None or fmax is not None):
        result[columns] = result[columns].clip(lower=fmin, upper=fmax)
    elif out_of_range == "drop":
        result = result.loc[~outside]
    return validate(result, kind)


def stretch(table: pd.DataFrame, factor, *, kind=None) -> pd.DataFrame:
    """Stretch time about the earliest cue without reversing events.

    Parameters
    ----------
    table : pandas.DataFrame
        Events or tracks table, with extra columns and attrs preserved.
    factor : float or sequence of float
        Finite and > 0. Events accept one factor per ascending distinct start
        cue: each gap uses the preceding cue's factor, each duration its own.
        Tracks accept a scalar, anchored at the earliest time in the table.
    kind : str or None
        Explicit kind; otherwise use a registered attrs hint, then infer_kind.

    Returns
    -------
    pandas.DataFrame
        Validated independent table in input row order.

    Raises
    ------
    SchemaError
        Invalid input or resulting table.
    ValueError
        Snapshot input (no time axis), invalid factors, or cue length mismatch.
    """
    result, kind = _table(table, kind)
    if kind == "snapshot":
        raise ValueError("snapshot has no time axis to stretch")
    if kind == "tracks":
        scale = _factors(factor, "factor", kind, 1, positive=True)[0]
        if len(result):
            t0 = result.time.min()
            result["time"] = t0 + (result.time - t0) * scale
    else:
        cues = np.unique(result.time_start)
        factors = _factors(factor, "factor", kind, len(cues), positive=True)
        new_cues = cues.copy()
        # Preserve 0.6's subtraction, multiplication, then cumulative addition.
        for i in range(1, len(cues)):
            delta = cues[i] - cues[i - 1]
            new_delta = delta * factors[i - 1]
            new_cues[i] = new_cues[i - 1] + new_delta
        indices = np.searchsorted(cues, result.time_start)
        durations = (result.time_stop - result.time_start) * factors[indices]
        result["time_start"] = new_cues[indices]
        result["time_stop"] = result.time_start + durations
    return validate(result, kind)


def split_events(events: pd.DataFrame, times) -> pd.DataFrame:
    """Split events at strictly internal cuts, interpolating frequency and gain.

    Parameters
    ----------
    events : pandas.DataFrame
        Events table; extra columns are copied to every piece, attrs preserved.
    times : sequence of float
        Finite cuts in seconds. Duplicates and cuts outside rows are ignored.

    Returns
    -------
    pandas.DataFrame
        Validated events, stably sorted by time_start then freq_start.

    Raises
    ------
    SchemaError
        Invalid events table.
    ValueError
        Nonfinite or nonnumeric cuts, or times is not a 1-D sequence.
    """
    table = validate(events, "events")
    if np.ndim(times) != 1:
        raise ValueError("times must be a 1-D sequence of finite cuts")
    cuts = np.unique([_finite_scalar(value, "times") for value in times])
    pieces, positions = [], []
    for position, row in table.iterrows():
        start, stop = row.time_start, row.time_stop
        inside = cuts[(start < cuts) & (cuts < stop)]
        boundaries = [start, *inside, stop]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            piece = {"time_start": left, "time_stop": right}
            for prefix in ("freq", "amp"):
                a, b = row[f"{prefix}_start"], row[f"{prefix}_stop"]
                # Uncut rows (including zero-duration events) remain exact.
                piece[f"{prefix}_start"] = a if not len(inside) else a + (left - start) / (stop - start) * (b - a)
                piece[f"{prefix}_stop"] = b if not len(inside) else a + (right - start) / (stop - start) * (b - a)
            positions.append(position)
            pieces.append(piece)
    # Repeat the original rows first to retain extra-column dtypes as well as values.
    result = table.iloc[positions].copy().reset_index(drop=True)
    if pieces:
        for column in pieces[0]:
            result[column] = [piece[column] for piece in pieces]
    result = result.sort_values(["time_start", "freq_start"], kind="stable")
    return validate(result, "events")
