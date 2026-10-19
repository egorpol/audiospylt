"""Versioned UTF-8 TSV tables and explicit upgrades of legacy files."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from decimal import Decimal, InvalidOperation
from io import StringIO
import os
from pathlib import Path
import re
from typing import TextIO
import warnings

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from audiospylt.model.schema import (
    LegacyFormatWarning,
    SchemaError,
    TABLE_CONTRACTS,
    _LEGACY_NAMES,
    _legacy_message,
    _upgrade_legacy,
    infer_kind,
    validate,
)

_TYPE_LINE = re.compile(r"# audiospylt-table: ([^\s]+) (v[^\s]+)")
_META_KEY = re.compile(r"[a-z][a-z0-9_]*")
_META_LINE = re.compile(r"# ([a-z][a-z0-9_]*): ([^\r\n\t]*)")
_META_START = re.compile(r"# [a-z][a-z0-9_]*: ")


def _metadata(meta: Mapping[str, object] | None) -> dict[str, str]:
    if meta is None:
        return {}
    if not isinstance(meta, Mapping):
        raise SchemaError("metadata must be a mapping")
    result = {}
    for key, value in meta.items():
        if not isinstance(key, str) or _META_KEY.fullmatch(key) is None:
            raise SchemaError(f"Invalid metadata key {key!r}; expected [a-z][a-z0-9_]*")
        text = str(value)
        if any(character in text for character in "\n\r\t"):
            raise SchemaError(f"Metadata value for {key!r} contains a newline, carriage return, or tab")
        result[key] = text
    return result


def _columns(payload: str) -> list[str]:
    """Check TSV records before pandas can mangle labels or infer an index."""
    try:
        reader = csv.reader(StringIO(payload, newline=""), delimiter="\t", strict=True)
        columns = next(reader, [])
        if not columns:
            raise SchemaError("Missing column-name row")
        if len(set(columns)) != len(columns):
            duplicates = pd.Index(columns)[pd.Index(columns).duplicated()].tolist()
            raise SchemaError(f"duplicate column labels: {duplicates!r}")
        for row in reader:
            if len(row) == 1 and row[0].startswith("#"):
                raise SchemaError(f"Comment lines are forbidden in table data (line {reader.line_num})")
            if len(row) != len(columns):
                raise SchemaError(
                    f"Wrong cell count at TSV line {reader.line_num}: "
                    f"expected {len(columns)}, found {len(row)}"
                )
        return columns
    except csv.Error as error:
        raise SchemaError(f"Malformed TSV: {error}") from error


def _number(cell: str, *, integer: bool = False) -> object:
    if cell == "":
        return float("nan")
    try:
        # Decimal prevents rounding int64 identifiers through binary64 first.
        return Decimal(cell) if integer else float(cell)
    except (ValueError, InvalidOperation):
        return cell  # Validation reports nonnumeric cells alongside other errors.


def _physical_line(content: str) -> tuple[str, str]:
    """Separate an LF/CRLF preamble line without altering quoted TSV cells."""
    line, separator, rest = content.partition("\n")
    return (line.removesuffix("\r") if separator else line), rest


def write_table(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    kind: str | None = None,
    *,
    meta: Mapping[str, object] | None = None,
) -> Path:
    """Validate and write a v1 TSV file, without creating parent directories.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table. The original frame is not modified.
    path : str or os.PathLike
        Destination file, whose parent must already exist.
    kind : str, optional
        Explicit kind; otherwise use the attrs hint, then column inference.
    meta : Mapping[str, object], optional
        Metadata in iteration order. Values are converted with ``str()``.
        If omitted, use ``df.attrs["meta"]`` when present.

    Returns
    -------
    pathlib.Path
        Destination as supplied, without resolving it to an absolute path.

    Raises
    ------
    SchemaError
        Invalid table or metadata. Validation and serialization complete before
        opening the destination, so these errors leave existing files intact.
    OSError
        The destination cannot be opened or written.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("table must be a pandas DataFrame")
    selected = kind if kind is not None else df.attrs.get("audiospylt_table")
    normalized = validate(df, selected if selected is not None else infer_kind(df))
    metadata = _metadata(df.attrs.get("meta") if meta is None else meta)
    csv_options = {
        "sep": "\t", "index": False, "lineterminator": "\n", "na_rep": "",
        "float_format": lambda value: repr(float(value)),
    }
    payload = normalized.to_csv(**csv_options)
    if "\r" in payload:
        # With an LF terminator, pandas' minimal quoting misses bare CR cells.
        payload = normalized.to_csv(quoting=csv.QUOTE_ALL, **csv_options)
    _columns(payload)
    preamble = f"# audiospylt-table: {normalized.attrs['audiospylt_table']} v1\n"
    preamble += "".join(f"# {key}: {value}\n" for key, value in metadata.items())
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        stream.write(preamble + payload)
    return destination


def read_table(
    path_or_buffer: str | os.PathLike[str] | TextIO, kind: str | None = None,
) -> pd.DataFrame:
    """Read and validate a v1 table or upgrade a headerless 0.6 table.

    Parameters
    ----------
    path_or_buffer : str, os.PathLike, or text file object
        UTF-8 path, or text stream read from its current position. Caller-owned
        streams are not closed. Unnamed streams are identified as ``<buffer>``.
    kind : str, optional
        Must agree with a v1 header. For headerless files, overrides inference.

    Returns
    -------
    pandas.DataFrame
        Validated table with ``audiospylt_table`` and ``meta`` attrs.

    Warns
    -----
    LegacyFormatWarning
        Once when legacy columns are renamed, naming the source and mapping.

    Raises
    ------
    SchemaError
        Invalid version, kind, metadata, TSV records, or table values.
    """
    if isinstance(path_or_buffer, (str, os.PathLike)):
        source = os.fspath(path_or_buffer)
        with open(path_or_buffer, encoding="utf-8", newline="") as stream:
            content = stream.read()
    else:
        source = str(getattr(path_or_buffer, "name", "<buffer>"))
        content = path_or_buffer.read()
    if not isinstance(content, str):
        raise TypeError("read_table requires a text stream, not a binary stream")

    content = content.removeprefix("\ufeff")
    first, remainder = _physical_line(content)
    metadata: dict[str, str] = {}
    versioned = first.startswith("# audiospylt-table")
    if versioned:
        match = _TYPE_LINE.fullmatch(first)
        if match is None:
            raise SchemaError(f"Malformed audiospylt-table type line: {first!r}")
        header_kind, version = match.groups()
        if version != "v1":
            raise SchemaError(f"Unsupported table version {version!r}; only v1 is supported")
        if header_kind not in TABLE_CONTRACTS:
            raise SchemaError(f"Unknown table kind {header_kind!r}")
        if kind is not None and kind != header_kind:
            raise SchemaError(f"Explicit kind {kind!r} disagrees with header kind {header_kind!r}")
        kind = header_kind
        while remainder.startswith("#"):
            line, rest = _physical_line(remainder)
            # A '#' in a tab-separated column name is data, not a comment.
            if "\t" in line and _META_START.match(line) is None:
                break
            match = _META_LINE.fullmatch(line)
            if match is None:
                raise SchemaError(f"Invalid metadata line/key (no other comments allowed): {line!r}")
            key, value = match.groups()
            if key in metadata:
                raise SchemaError(f"Duplicate metadata key {key!r}")
            metadata[key] = value
            remainder = rest
        payload = remainder
    else:
        if first.startswith("#") and "\t" not in first:
            raise SchemaError(f"Unexpected comment or malformed type line: {first!r}")
        payload = content

    # Only trim physical empty lines at EOF. A closing quote protects embedded
    # CR/LF in the last cell; interior empty records remain visible to _columns.
    payload = payload.rstrip("\r\n")
    columns = _columns(payload)
    options: dict[str, object] = {"float_precision": "round_trip"}
    if versioned:
        options.update({"keep_default_na": False, "na_values": [""]})
    # Native parsing is exact for ordinary integer columns, but nullable bins
    # and decimal/exponent IDs would round via binary64. Preserve those tokens.
    options["converters"] = {
        name: (lambda cell: _number(cell, integer=True))
        for name in ("track_id", "bin") if name in columns
    }
    try:
        frame = pd.read_csv(
            StringIO(payload), sep="\t", header=0, names=columns, skip_blank_lines=False, **options,
        )
    except (ValueError, pd.errors.ParserError) as error:
        raise SchemaError(f"Malformed TSV: {error}") from error
    selected = kind if kind is not None else infer_kind(frame.rename(columns=_LEGACY_NAMES))
    float_names = {column.name for column in
                   TABLE_CONTRACTS[selected].required + TABLE_CONTRACTS[selected].optional
                   if column.dtype == "float64"} if selected in TABLE_CONTRACTS else set()
    if not versioned:
        float_names.update(old for old, new in _LEGACY_NAMES.items() if new in float_names)
    for name in float_names.intersection(frame.columns):
        series = frame[name]
        if not is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
            # A bad cell makes pandas retain even valid numeric tokens as text.
            # Only these exceptional columns need scalar parsing for validation.
            frame[name] = series.map(lambda cell: _number(cell) if isinstance(cell, str) else cell)
    if versioned:
        result = validate(frame, kind)
    else:
        result, mapping = _upgrade_legacy(frame, kind)
        if mapping:
            warnings.warn(_legacy_message(source, mapping), LegacyFormatWarning, stacklevel=2)
    result.attrs["meta"] = metadata
    return result
