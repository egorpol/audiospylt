"""Column contracts, validation, and legacy upgrades for plain DataFrames."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_number,
    is_numeric_dtype,
    is_object_dtype,
    is_unsigned_integer_dtype,
)


class SchemaError(ValueError):
    """A table violates one or more schema or file-format rules."""


class LegacyFormatWarning(FutureWarning):
    """Legacy column names were upgraded to their v1 equivalents."""


@dataclass(frozen=True)
class ColumnContract:
    """Describe a column's name, normalized dtype, and lower-bound rule."""

    name: str
    dtype: str = "float64"
    nonnegative: bool = False


@dataclass(frozen=True)
class TableContract:
    """Describe required and optional columns in canonical order."""

    kind: str
    required: tuple[ColumnContract, ...]
    optional: tuple[ColumnContract, ...] = ()


TABLE_CONTRACTS: Mapping[str, TableContract] = MappingProxyType({
    "snapshot": TableContract(
        "snapshot",
        (ColumnContract("freq", nonnegative=True), ColumnContract("amp", nonnegative=True)),
        (ColumnContract("phase"), ColumnContract("bin", "Int64", True),
         ColumnContract("prominence", nonnegative=True)),
    ),
    "events": TableContract(
        "events",
        tuple(ColumnContract(name, nonnegative=True) for name in (
            "time_start", "time_stop", "freq_start", "freq_stop", "amp_start", "amp_stop",
        )),
    ),
    "tracks": TableContract(
        "tracks",
        (ColumnContract("track_id", "int64"), ColumnContract("time", nonnegative=True),
         ColumnContract("freq", nonnegative=True), ColumnContract("amp", nonnegative=True)),
        (ColumnContract("phase"),),
    ),
})

_LEGACY_NAMES = {
    "Frequency (Hz)": "freq", "Amplitude": "amp",
    "amp_min": "amp_start", "amp_max": "amp_stop",
}
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _contract(kind: str) -> TableContract:
    if not isinstance(kind, str) or kind not in TABLE_CONTRACTS:
        raise SchemaError(f"Unknown table kind {kind!r}; expected snapshot, events, or tracks")
    return TABLE_CONTRACTS[kind]


def _mixed_names(df: pd.DataFrame) -> list[str]:
    return [f"ambiguous legacy/v1 columns: {old!r} and {new!r}"
            for old, new in _LEGACY_NAMES.items() if old in df.columns and new in df.columns]


def _problem(problems: list[str], column: str, rule: str, mask: np.ndarray) -> None:
    rows = np.flatnonzero(mask)
    if rows.size:
        problems.append(
            f"{column}: {rule} ({rows.size} row(s); first row positions {rows[:5].tolist()})"
        )


def _array_column(
    series: pd.Series, column: ColumnContract, required: bool, problems: list[str],
) -> tuple[np.ndarray, np.ndarray | pd.arrays.IntegerArray, np.ndarray | None]:
    """Validate native numeric/nullable arrays without visiting Python scalars."""
    missing = series.isna().to_numpy(dtype=bool)
    real = (is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)
            and not is_complex_dtype(series.dtype))
    if not real:
        _problem(problems, column.name, "numeric real values required (booleans rejected)", ~missing)
        if required:
            _problem(problems, column.name, "NaN/missing values forbidden", missing)
        floats = np.full(len(series), np.nan)
        if column.dtype == "float64":
            return floats, floats, None
        valid = np.zeros(len(series), dtype=bool)
        return floats, pd.arrays.IntegerArray(np.zeros(len(series), dtype=np.int64), ~valid), valid

    floats = series.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(floats)
    if required:
        _problem(problems, column.name, "NaN/missing values forbidden", missing)
    _problem(problems, column.name, "finite values required (infinity forbidden)", ~missing & ~finite)
    if column.nonnegative:
        _problem(problems, column.name, "must be >= 0", floats < 0)
    if column.dtype == "float64":
        return floats, floats, None

    if is_integer_dtype(series.dtype):
        unsigned = is_unsigned_integer_dtype(series.dtype)
        raw = series.to_numpy(dtype=np.uint64 if unsigned else np.int64, na_value=0)
        outside = raw > np.uint64(_INT64_MAX) if unsigned else np.zeros(len(series), dtype=bool)
        fractional = np.zeros(len(series), dtype=bool)
    else:
        raw = series.to_numpy(dtype=getattr(series.dtype, "numpy_dtype", series.dtype), na_value=np.nan)
        fractional = finite & (np.trunc(raw) != raw)
        # int64 max rounds UP in binary64; use an exclusive upper bound.
        outside = ~missing & ((raw < _INT64_MIN) | (raw >= 2**63))
    _problem(problems, column.name, "must be integer-valued", fractional)
    _problem(problems, column.name, "must be within int64 range", outside)
    valid = ~missing & finite & ~fractional & ~outside
    integers = np.zeros(len(series), dtype=np.int64)
    integers[valid] = raw[valid]
    return floats, pd.arrays.IntegerArray(integers, ~valid), valid


def _numeric_column(
    series: pd.Series, column: ColumnContract, required: bool, problems: list[str],
) -> tuple[np.ndarray, np.ndarray | pd.arrays.IntegerArray, np.ndarray | None]:
    """Check every scalar before casting, preserving exact integer bounds."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    if not is_object_dtype(series.dtype):
        return _array_column(series, column, required, problems)
    values = series.tolist()
    missing = series.isna().to_numpy(dtype=bool)
    numeric = np.array([
        is_number(value) and not isinstance(value, (bool, np.bool_, complex, np.complexfloating))
        for value in values
    ], dtype=bool)
    floats = np.full(len(values), np.nan, dtype=np.float64)
    for row, value in enumerate(values):
        if numeric[row] and not missing[row]:
            try:
                floats[row] = float(value)
            except OverflowError:
                floats[row] = -np.inf if value < 0 else np.inf
    _problem(problems, column.name, "numeric real values required (booleans rejected)",
             ~numeric & ~missing)
    if required:
        _problem(problems, column.name, "NaN/missing values forbidden", missing)
    _problem(problems, column.name, "finite values required (infinity forbidden)",
             numeric & ~missing & ~np.isfinite(floats))
    if column.nonnegative:
        _problem(problems, column.name, "must be >= 0", floats < 0)

    integers: list[int | None] = [None] * len(values)
    if column.dtype in ("int64", "Int64"):
        fractional = np.zeros(len(values), dtype=bool)
        outside = np.zeros(len(values), dtype=bool)
        for row, value in enumerate(values):
            if numeric[row] and not missing[row]:
                outside[row] = value < _INT64_MIN or value > _INT64_MAX
                if np.isfinite(floats[row]):
                    fractional[row] = int(value) != value
                    if not fractional[row] and not outside[row]:
                        integers[row] = int(value)
        _problem(problems, column.name, "must be integer-valued", fractional)
        _problem(problems, column.name, "must be within int64 range", outside)
        valid = np.array([value is not None for value in integers], dtype=bool)
        data = np.array([value if value is not None else 0 for value in integers], dtype=np.int64)
        return floats, pd.arrays.IntegerArray(data, ~valid), valid
    return floats, floats, None


def validate(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return a normalized copy, or report all detected schema violations.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to check. Extra columns and unrelated attributes are preserved.
    kind : str
        Explicit contract token; takes precedence over the frame's attrs hint.

    Returns
    -------
    pandas.DataFrame
        Canonically ordered copy with normalized dtypes, a RangeIndex, and
        ``attrs["audiospylt_table"]`` set to ``kind``.

    Raises
    ------
    SchemaError
        Any missing/duplicate column, invalid value, or violated relationship.
        Row positions in messages are zero-based, independent of index labels.
    """
    contract = _contract(kind)
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("table must be a pandas DataFrame")
    problems = _mixed_names(df)
    duplicates = df.columns[df.columns.duplicated(keep=False)]
    if len(duplicates):
        problems.append(f"duplicate column labels: {duplicates.tolist()!r}")
    required = {column.name for column in contract.required}
    for name in (column.name for column in contract.required):
        if name not in df.columns:
            problems.append(f"{name}: required column missing")

    checked: dict[str, np.ndarray] = {}
    normalized: dict[str, pd.Series] = {}
    track_ids: np.ndarray | None = None
    valid_ids: np.ndarray | None = None
    columns = contract.required + contract.optional
    for column in columns:
        name = column.name
        if name not in df.columns or name in duplicates:
            continue
        before = len(problems)
        floats, data, valid = _numeric_column(df[name], column, name in required, problems)
        checked[name] = floats
        if name == "track_id":
            track_ids = data.to_numpy(dtype=np.int64, na_value=0)
            valid_ids = valid
        if len(problems) == before:
            normalized[name] = pd.Series(data, dtype=column.dtype)

    if kind == "events" and {"time_start", "time_stop"} <= checked.keys():
        start, stop = checked["time_start"], checked["time_stop"]
        _problem(problems, "time_stop", "must be >= time_start",
                 np.isfinite(start) & np.isfinite(stop) & (stop < start))
    if kind == "tracks" and track_ids is not None and "time" in checked:
        time = checked["time"]
        positions = np.flatnonzero(valid_ids & np.isfinite(time))
        times = pd.Series(time[positions], copy=False)
        previous = times.groupby(track_ids[positions], sort=False).shift().to_numpy()
        bad_time = np.zeros(len(df), dtype=bool)
        bad_time[positions] = time[positions] <= previous
        _problem(problems, "time", "must be strictly increasing within each track in row order",
                 bad_time)
    if problems:
        raise SchemaError("Invalid table:\n- " + "\n- ".join(problems))

    canonical = [column.name for column in columns if column.name in df.columns]
    extras = [name for name in df.columns if name not in canonical]
    result = df.loc[:, canonical + extras].copy().reset_index(drop=True)
    for name, series in normalized.items():
        result[name] = series
    result.attrs["audiospylt_table"] = kind
    return result


def infer_kind(df: pd.DataFrame) -> str:
    """Infer the unique v1 kind from columns, ignoring attributes and values.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with canonical column names.

    Returns
    -------
    str
        Contract token. Complete tracks take precedence over snapshot columns.

    Raises
    ------
    SchemaError
        No contract matches, or multiple independent contracts match.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("table must be a pandas DataFrame")
    conflicts = _mixed_names(df)
    if conflicts:
        raise SchemaError("; ".join(conflicts))
    names = set(df.columns)
    matches = [kind for kind, contract in TABLE_CONTRACTS.items()
               if {column.name for column in contract.required} <= names]
    if "tracks" in matches:
        matches.remove("snapshot")
    if len(matches) != 1:
        state = f"ambiguous ({', '.join(matches)})" if matches else "unknown"
        raise SchemaError(f"Cannot infer table kind: {state}; columns {list(df.columns)!r}")
    return matches[0]


def _upgrade_legacy(
    df: pd.DataFrame, kind: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("table must be a pandas DataFrame")
    conflicts = _mixed_names(df)
    if conflicts:
        raise SchemaError("; ".join(conflicts))
    mapping = {old: new for old, new in _LEGACY_NAMES.items() if old in df.columns}
    renamed = df.rename(columns=mapping)
    # Attributes remain hints; an inferred contract is always validated again.
    selected = kind if kind is not None else renamed.attrs.get("audiospylt_table")
    return validate(renamed, selected if selected is not None else infer_kind(renamed)), mapping


def _legacy_message(source: str, mapping: Mapping[str, str]) -> str:
    changes = ", ".join(f"{old!r} -> {new!r}" for old, new in mapping.items())
    return f"{source}: upgraded AudioSpylt 0.6 columns: {changes}"


def upgrade(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 0.6 columns and validate a copy; canonical input is warning-free.

    Parameters
    ----------
    df : pandas.DataFrame
        Legacy or canonical frame. Old/new pairs in one frame are ambiguous.

    Returns
    -------
    pandas.DataFrame
        Normalized table. Amplitudes are renamed without reordering values.

    Warns
    -----
    LegacyFormatWarning
        Once per successful call that renames columns, pointing to the caller.

    Raises
    ------
    SchemaError
        Ambiguous names, unknown/ambiguous kind, or invalid table values.
    """
    result, mapping = _upgrade_legacy(df)
    if mapping:
        warnings.warn(_legacy_message("DataFrame", mapping), LegacyFormatWarning, stacklevel=2)
    return result
