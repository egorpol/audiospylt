"""Read and write partial tracks in SDIF (AudioSculpt/PM2, Loris, SPEAR) and read AudioSculpt chord sequences."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import struct

import numpy as np
import pandas as pd

from audiospylt.model.schema import validate

__all__ = ["read_sdif", "write_sdif", "read_chord_seq"]

_PARTIALS = ("1TRC", "1HRM", "RBEP")
_COLUMNS = {"1TRC": 4, "1HRM": 4, "RBEP": 6}
_NAMES = "index, frequency, amplitude, phase"
_NUMERIC = {
    0x0004: ">f4", 0x0008: ">f8",
    0x0101: ">i1", 0x0102: ">i2", 0x0104: ">i4", 0x0108: ">i8",
    0x0201: ">u1", 0x0202: ">u2", 0x0204: ">u4", 0x0208: ">u8",
}
_TEXT = (0x0001, 0x0301)
# Header frames: NVT (name-value table), TYP (type declarations), IDS (stream IDs).
_INFO = (b"1NVT", b"1TYP", b"1IDS")
_FLOAT32_INDEX_MAX = 2**24  # float32 holds every integer up to here exactly


@dataclass(frozen=True)
class _Frame:
    sig: str
    time: float
    stream: int
    matrices: list[tuple[str, np.ndarray | str]]


@dataclass(frozen=True)
class _File:
    name: str
    nvt: list[dict[str, str]]
    frames: list[_Frame]


def _matrices(data: bytes, pos: int, count: int, where: str) -> tuple[list[tuple[str, np.ndarray | str]], int]:
    """Read ``count`` matrices from ``pos``; return them and the position after the last."""
    matrices: list[tuple[str, np.ndarray | str]] = []
    for _ in range(count):
        if pos + 16 > len(data):
            raise ValueError(f"{where}: truncated matrix header at byte {pos}")
        sig = data[pos:pos + 4].decode("latin-1")
        code, rows, cols = struct.unpack_from(">iii", data, pos + 4)
        if rows < 0 or cols < 0:
            raise ValueError(f"{where}: matrix {sig} at byte {pos} has {rows} x {cols} entries")
        value: np.ndarray | str
        if code in _TEXT:
            size = rows * cols
        elif code in _NUMERIC:
            dtype = np.dtype(_NUMERIC[code])
            size = rows * cols * dtype.itemsize
        else:
            raise ValueError(f"{where}: matrix {sig} at byte {pos} has unknown data type {code:#06x}")
        if pos + 16 + size > len(data):
            raise ValueError(f"{where}: truncated matrix {sig} at byte {pos}")
        if code in _TEXT:
            value = data[pos + 16:pos + 16 + size].decode("utf-8", "replace").rstrip("\0")
        else:
            value = np.frombuffer(data, dtype, rows * cols, pos + 16).reshape(rows, cols)
            value = value.astype(dtype.newbyteorder("="))
        matrices.append((sig, value))
        pos += 16 + size + (-size % 8)  # data padded to 8 bytes
    return matrices, pos


def _nvt(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition("\t")
        if key.strip():
            entries[key.strip()] = value.strip()
    return entries


def _read(path: str | os.PathLike[str]) -> _File:
    """Parse an SDIF file: its name-value tables and its data frames in file order.

    Data frames are read matrix by matrix from the frame header, ignoring the
    frame size (as the IRCAM library does; some writers store wrong sizes).
    The size only skips header frames (1NVT, 1TYP, 1IDS), which may be old
    ASCII chunks.
    """
    file = Path(path)
    data = file.read_bytes()
    if len(data) < 8 or data[:4] != b"SDIF":
        raise ValueError(f"{file}: not an SDIF file")
    (header,) = struct.unpack_from(">I", data, 4)
    pos = 8 + header
    nvt: list[dict[str, str]] = []
    frames: list[_Frame] = []
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError(f"{file}: truncated frame header at byte {pos}")
        raw = data[pos:pos + 4]
        sig = raw.decode("latin-1")
        where = f"{file}, frame {sig} at byte {pos}"
        if raw in _INFO:
            (size,) = struct.unpack_from(">I", data, pos + 4)
            end = pos + 8 + size
            if end > len(data):
                raise ValueError(f"{where}: truncated")
            body = data[pos + 8:end]
            if raw == b"1NVT" and not body.lstrip().startswith(b"{") and len(body) >= 16:
                count = struct.unpack_from(">i", body, 12)[0]
                matrices, _ = _matrices(body, 16, count, where)
                nvt += [_nvt(value) for _, value in matrices if isinstance(value, str)]
            pos = end
            continue
        if pos + 24 > len(data):
            raise ValueError(f"{where}: truncated")
        time, stream, count = struct.unpack_from(">dii", data, pos + 8)
        matrices, pos = _matrices(data, pos + 24, count, where)
        frames.append(_Frame(sig, time, stream, matrices))
    return _File(file.name, nvt, frames)


def _check_stream(stream: int | None) -> None:
    if stream is not None and (isinstance(stream, bool) or not isinstance(stream, (int, np.integer))):
        raise ValueError(f"stream must be an int or None, got {stream!r}")


def _pick_stream(stream: int | None, streams: list[int], what: str, name: str) -> int:
    if stream is None:
        return streams[0]
    if stream not in streams:
        raise ValueError(f"{name}: no {what} in stream {stream}; streams with {what}: {streams}")
    return int(stream)


def _is_chord_seq(sdif: _File) -> bool:
    return any(frame.sig == "1MRK" and any(sig == "1TRC" for sig, _ in frame.matrices) for frame in sdif.frames)


def _meta_nvt(meta: dict[str, object], nvt: list[dict[str, str]]) -> None:
    meta["nvt"] = nvt
    for key, field in (("TableName", "table_name"), ("WrittenBy", "written_by")):
        value = next((entries[key] for entries in nvt if key in entries), None)
        if value is not None:
            meta[field] = value


def read_sdif(
    path: str | os.PathLike[str],
    *,
    matrix: str | None = None,
    stream: int | None = None,
    max_step_cents: float | None = None,
) -> pd.DataFrame:
    """Read partial tracks from an SDIF file as a tracks table.

    Reads the partial-tracking matrices of SDIF: ``1TRC`` (sinusoidal tracks,
    written by AudioSculpt and PM2, Loris, the CNMAT tools), ``1HRM``
    (harmonic tracks, PM2) and ``RBEP`` (Loris and SPEAR breakpoints). Each
    row of such a matrix holds an index, a frequency, an amplitude and a
    phase; the index ties a frame's row to rows of other frames.

    Parameters
    ----------
    path : path
        The SDIF file.
    matrix : {'1TRC', '1HRM', 'RBEP'}, optional
        Matrix type to read. None (default) reads the one type the file holds
        and raises if it holds several.
    stream : int, optional
        SDIF stream to read. None (default) reads the lowest stream that holds
        ``matrix``. PM2 writes one stream per channel of a stereo analysis.
    max_step_cents : float, optional
        Also end a track where the frequency moves by more than this many
        cents from one point to the next of the same index. None (default)
        splits ``1TRC`` and ``1HRM`` at gaps only and never splits ``RBEP``.

    Returns
    -------
    pandas.DataFrame
        A validated ``tracks`` table (``track_id``, ``time``, ``freq``,
        ``amp``, ``phase``) plus an Int64 ``partial_id`` column, the SDIF
        index, and for ``RBEP`` a ``bandwidth`` column. Rows are grouped by
        track in time order, tracks numbered by start time (then start
        frequency). ``attrs["meta"]``: ``source`` ('sdif'), ``file``,
        ``matrix``, ``stream``, ``streams`` (all streams holding
        ``matrix``), ``frames`` (frames read), ``nvt`` (the file's name-value
        tables, a list of dicts), ``max_step_cents`` and, when an NVT has
        them, ``table_name`` and ``written_by``.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        Not an SDIF file, a truncated file, an unknown data type, no partial
        matrices (a chord sequence points to ``read_chord_seq``), several
        types and no ``matrix``, a missing ``stream``, a matrix with too few
        columns, an index that is not an integer or appears twice in one
        frame, or ``max_step_cents`` not positive.

    Notes
    -----
    A track is one index over consecutive frames of the stream:

    - ``1TRC`` and ``1HRM``: a run of frames holding the index. A frame
      without it (a gap) ends the track, and the index may start a new one
      later. PM2 numbers ``1TRC`` partials from 1 and never reuses a number,
      so gaps do not occur; in ``1HRM`` the index is the harmonic number, and
      a harmonic that drops out for a while returns as a new track. The CNMAT
      pm 1.2.2 reuses indices as slots.
    - ``RBEP``: all breakpoints of one index (a Loris or SPEAR partial). The
      point time is the frame time plus the breakpoint's offset (column 6),
      and the Loris noise share is kept as ``bandwidth`` (column 5).

    Frames are read in time order, empty frames count, and further columns
    are ignored. ``amp`` is the writer's amplitude (PM2 and Loris: a sine's
    peak amplitude, like ``spectrum``), ``phase`` its phase in radians.
    Times just below 0 (PM2 places its first frame half a sample early) are
    read as 0.
    """
    if matrix is not None and matrix not in _PARTIALS:
        raise ValueError(f"matrix must be one of {', '.join(_PARTIALS)} or None, got {matrix!r}")
    _check_stream(stream)
    if max_step_cents is not None and not (np.isfinite(max_step_cents) and max_step_cents > 0):
        raise ValueError(f"max_step_cents must be a positive number or None, got {max_step_cents!r}")
    sdif = _read(path)
    name = sdif.name
    present = sorted({frame.sig for frame in sdif.frames if frame.sig in _PARTIALS})
    if matrix is None and len(present) == 1:
        matrix = present[0]
    if matrix is None or matrix not in present:
        if not present and _is_chord_seq(sdif):
            raise ValueError(f"{name} is an AudioSculpt chord sequence (1MRK frames); read it with read_chord_seq()")
        if not present:
            types = sorted({frame.sig for frame in sdif.frames})
            raise ValueError(f"{name} holds no partial tracks ({', '.join(_PARTIALS)}); frame types: {types}")
        if matrix is None:
            raise ValueError(f"{name} holds {', '.join(present)}; choose one with matrix=")
        raise ValueError(f"{name} holds no {matrix} frames, only {', '.join(present)}")
    streams = sorted({frame.stream for frame in sdif.frames if frame.sig == matrix})
    stream = _pick_stream(stream, streams, f"{matrix} frames", name)
    frames = sorted((frame for frame in sdif.frames if frame.sig == matrix and frame.stream == stream),
                    key=lambda frame: frame.time)  # stable: equal times keep file order

    width = _COLUMNS[matrix]
    parts = []
    for number, frame in enumerate(frames):
        for sig, values in frame.matrices:
            if sig != matrix:
                continue
            if isinstance(values, str):
                raise ValueError(f"{name}: {matrix} matrix at {frame.time:g} s holds text, not numbers")
            if not len(values):
                continue  # the frame still counts: it ends every track
            if values.shape[1] < width:
                expected = _NAMES + (", bandwidth, offset" if matrix == "RBEP" else "")
                raise ValueError(f"{name}: {matrix} matrix at {frame.time:g} s has {values.shape[1]} columns; "
                                 f"expected {width} ({expected})")
            parts.append(np.column_stack([np.full(len(values), number), np.full(len(values), frame.time),
                                          values[:, :width].astype(np.float64)]))
    points = np.concatenate(parts) if parts else np.empty((0, 2 + width))
    frame_no: np.ndarray = points[:, 0].astype(np.int64)
    index_values = points[:, 2]
    bad = ~np.isfinite(index_values) | (np.trunc(index_values) != index_values)
    if bad.any():
        raise ValueError(f"{name}: {matrix} index must be an integer, found {index_values[bad][0]:g}")
    index: np.ndarray = index_values.astype(np.int64)
    time: np.ndarray = points[:, 1] + (points[:, 7] if matrix == "RBEP" else 0.0)
    freq, amp, phase = points[:, 3], points[:, 4], points[:, 5]

    order = np.lexsort((frame_no, index))
    twice = (index[order][1:] == index[order][:-1]) & (frame_no[order][1:] == frame_no[order][:-1])
    if twice.any():
        first = order[1:][twice][0]
        raise ValueError(f"{name}: {matrix} index {index[first]} appears twice in the frame at "
                         f"{frames[frame_no[first]].time:g} s")
    if matrix == "RBEP":
        order = np.lexsort((frame_no, time, index))
    frame_no, index, time, freq, amp, phase = (x[order] for x in (frame_no, index, time, freq, amp, phase))
    extra = {"bandwidth": points[order, 6]} if matrix == "RBEP" else {}

    # Runs: one index over consecutive frames (RBEP: over all its points), split at jumps.
    starts = np.ones(len(index), dtype=bool)
    starts[1:] = index[1:] != index[:-1]
    if matrix != "RBEP":
        starts[1:] |= frame_no[1:] != frame_no[:-1] + 1
    if max_step_cents is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.abs(1200 * np.log2(freq[1:] / freq[:-1]))
        starts[1:] |= ~(step <= max_step_cents)
    run: np.ndarray = np.cumsum(starts) - 1
    below = time < 0
    time = np.maximum(time, 0.0)
    # Two points clipped to 0 in one run: keep the later one.
    later = np.ones(len(time), dtype=bool)
    later[:-1] = (run[:-1] != run[1:]) | (time[1:] > 0) | ~below[:-1]
    run, index, time, freq, amp, phase = (x[later] for x in (run, index, time, freq, amp, phase))
    extra = {key: values[later] for key, values in extra.items()}
    same = (run[1:] == run[:-1]) & (time[1:] <= time[:-1])
    if same.any():
        first = np.flatnonzero(same)[0] + 1
        raise ValueError(f"{name}: {matrix} index {index[first]} has two points at {time[first]:g} s "
                         "(frames with equal times)")
    heads = np.flatnonzero(np.diff(run, prepend=-1))
    numbering = np.empty(heads.size, dtype=np.int64)
    numbering[np.lexsort((index[heads], freq[heads], time[heads]))] = np.arange(heads.size)
    track_id = numbering[run]
    rows = np.argsort(track_id, kind="stable")
    tracks = validate(pd.DataFrame({
        "track_id": track_id[rows], "time": time[rows], "freq": freq[rows], "amp": amp[rows],
        "phase": phase[rows], "partial_id": pd.array(index[rows], dtype="Int64"),
        **{key: values[rows] for key, values in extra.items()},
    }), "tracks")
    meta: dict[str, object] = {
        "source": "sdif", "file": name, "matrix": matrix, "stream": stream, "streams": streams,
        "frames": len(frames), "max_step_cents": max_step_cents,
    }
    _meta_nvt(meta, sdif.nvt)
    tracks.attrs["meta"] = meta
    return tracks


def _text_frame(sig: bytes, stream: int, text: str) -> bytes:
    data = text.encode("utf-8") + b"\0"
    matrix = sig + struct.pack(">iii", 0x0301, len(data), 1) + data + b"\0" * (-len(data) % 8)
    body = struct.pack(">dii", -np.finfo(np.float64).max, stream, 1) + matrix
    return sig + struct.pack(">I", len(body)) + body


def write_sdif(
    tracks: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    nvt: Mapping[str, object] | None = None,
) -> Path:
    """Write a tracks table as SDIF 1TRC partial tracks, laid out as PM2 and AudioSculpt write them.

    Parameters
    ----------
    tracks : pandas.DataFrame
        A ``tracks`` table; ``phase`` is optional.
    path : path
        The file to write (overwritten if it exists).
    nvt : mapping, optional
        Further name-value entries for the file's 1NVT table, after
        ``TableName SinusoidalTracks`` and ``WrittenBy audiospylt <version>``
        (an entry of the same name replaces those). Values are written with
        ``str()``; names and values may not contain tabs or line breaks.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    SchemaError
        ``tracks`` is not a valid tracks table (this includes two points of a
        track at one time).
    ValueError
        No points, a ``track_id`` outside 0 to 2**24 - 2 (the index must be
        exact in float32), or an NVT name or value with a tab or line break.

    Notes
    -----
    The file has one 1NVT frame and one 1TRC frame per distinct time of the
    table (stream 0, time in float64), holding a row per track present then:
    index ``track_id + 1`` (PM2 numbers partials from 1), frequency,
    amplitude and phase, as float32. ``read_sdif`` reads it back with
    ``partial_id`` = ``track_id + 1``.

    SDIF readers end a track at a frame that lacks its index. When tracks do
    not share one time grid (``RBEP`` breakpoints, merged tables), a track
    gets a point at every frame time inside its span: frequency and amplitude
    linear between its neighbours, phase integrated from the previous point,
    ``phase + pi * dt * (f_prev + f)``, wrapped to [-pi, pi). Missing phases
    (no ``phase`` column, or NaN) are integrated the same way, from 0 at a
    track's first point.
    """
    table = validate(tracks, "tracks")
    if not len(table):
        raise ValueError("tracks has no points to write")
    from audiospylt import __version__

    entries = {"TableName": "SinusoidalTracks", "WrittenBy": f"audiospylt {__version__}"}
    for key, value in (nvt or {}).items():
        text = str(value)
        if not isinstance(key, str) or not key.strip() or any(c in s for s in (key, text) for c in "\t\r\n"):
            raise ValueError(f"nvt names must be non-empty strings and names and values may not contain tabs "
                             f"or line breaks, got {key!r}: {text!r}")
        entries[key] = text
    track_ids = table["track_id"].to_numpy(np.int64)
    if track_ids.min() < 0 or track_ids.max() > _FLOAT32_INDEX_MAX - 2:
        raise ValueError(f"track_id must be in 0..{_FLOAT32_INDEX_MAX - 2} to be written as a float32 index, "
                         f"got {track_ids.min()}..{track_ids.max()}")
    order = np.lexsort((table["time"].to_numpy(np.float64), track_ids))
    ids = track_ids[order]
    time, freq, amp = (table[column].to_numpy(np.float64)[order] for column in ("time", "freq", "amp"))
    phase = table["phase"].to_numpy(np.float64)[order] if "phase" in table else np.full(len(ids), np.nan)

    # Every track gets a row at each grid time in its span: known points where it has them,
    # interpolated ones elsewhere.
    grid = np.unique(time)
    point_track = np.cumsum(np.diff(ids, prepend=ids[0] - 1) != 0) - 1
    heads = np.flatnonzero(np.diff(point_track, prepend=-1))
    tails = np.r_[heads[1:], len(ids)] - 1
    lo, hi = np.searchsorted(grid, time[heads]), np.searchsorted(grid, time[tails])
    counts = hi - lo + 1
    begin = np.cumsum(counts) - counts
    total = int(counts.sum())
    row_track = np.repeat(np.arange(heads.size), counts)
    row = np.arange(total)
    row_time = grid[lo[row_track] + row - begin[row_track]]
    at = begin[point_track] + np.searchsorted(grid, time) - lo[point_track]
    known = np.zeros(total, dtype=bool)
    known[at] = True
    left = np.maximum.accumulate(np.where(known, row, 0))
    right = np.minimum.accumulate(np.where(known, row, total)[::-1])[::-1]
    span = row_time[right] - row_time[left]
    weight = np.divide(row_time - row_time[left], span, out=np.zeros(total), where=span > 0)

    def spread(values: np.ndarray) -> np.ndarray:
        full = np.zeros(total)
        full[at] = values
        return full[left] + weight * (full[right] - full[left])

    row_freq, row_amp = spread(freq), spread(amp)
    known_phase = np.full(total, np.nan)
    known_phase[at] = phase
    step = np.zeros(total)
    step[1:] = np.pi * np.diff(row_time) * (row_freq[1:] + row_freq[:-1])
    step[begin] = 0.0
    integral = np.cumsum(step)
    has_phase = np.isfinite(known_phase)
    first = np.zeros(total, dtype=bool)
    first[begin] = True
    anchor = np.maximum.accumulate(np.where(has_phase | first, row, 0))
    integrated = np.where(has_phase, known_phase, 0.0)[anchor] + integral - integral[anchor]
    row_phase = np.where(has_phase, known_phase, (integrated + np.pi) % (2 * np.pi) - np.pi)

    row_index = ids[heads][row_track] + 1
    frames = np.lexsort((row_index, row_time))
    values = np.column_stack([row_index, row_freq, row_amp, row_phase])[frames].astype(">f4")
    row_time = row_time[frames]
    bounds = np.r_[0, np.flatnonzero(np.diff(row_time)) + 1, total]
    chunks = [b"SDIF", struct.pack(">III", 8, 3, 1),
              _text_frame(b"1NVT", -3, "".join(f"{key}\t{value}\n" for key, value in entries.items()))]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        data = values[start:stop].tobytes()
        body = (struct.pack(">dii", row_time[start], 0, 1) + b"1TRC"
                + struct.pack(">iii", 0x0004, stop - start, 4) + data + b"\0" * (-len(data) % 8))
        chunks.append(b"1TRC" + struct.pack(">I", len(body)) + body)
    target = Path(path)
    target.write_bytes(b"".join(chunks))
    return target


def read_chord_seq(path: str | os.PathLike[str], *, stream: int | None = None) -> pd.DataFrame:
    """Read an AudioSculpt chord sequence (SDIF ``1MRK`` frames) as an events table.

    AudioSculpt's chord-sequence analysis cuts a sound at markers and writes
    one ``1MRK`` frame per marker: a ``1BEG`` matrix with the indices of the
    partials that start there, a ``1TRC`` matrix with their frequency,
    amplitude and phase, and a ``1END`` matrix with the indices of partials
    that end there. Each partial holds one frequency and amplitude from its
    marker to the marker that ends it.

    Parameters
    ----------
    path : path
        The SDIF file (``TableName ChordSeq`` in its NVT).
    stream : int, optional
        SDIF stream to read. None (default) reads the lowest stream with
        chord-sequence frames.

    Returns
    -------
    pandas.DataFrame
        A validated ``events`` table, one row per partial (``freq_start`` =
        ``freq_stop``, ``amp_start`` = ``amp_stop``), plus an Int64
        ``partial_id`` column, the SDIF index. Rows are sorted by start time,
        then frequency. ``attrs["meta"]``: ``source`` ('sdif'), ``file``,
        ``stream``, ``streams``, ``markers`` (the marker times), ``nvt`` and,
        when the NVT has them, ``table_name`` and ``written_by``.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        Not an SDIF file, no chord-sequence frames, a missing ``stream``, or
        a partial that starts twice, ends without a start or never ends.

    Notes
    -----
    The partials start at the ``1TRC`` rows (``1BEG`` lists the same
    indices). Times below 0 (a first marker before the sound starts) are
    read as 0. ``events_to_mei`` writes the events as a chord per segment.
    """
    _check_stream(stream)
    sdif = _read(path)
    name = sdif.name
    if not _is_chord_seq(sdif):
        types = sorted({frame.sig for frame in sdif.frames})
        raise ValueError(f"{name} holds no chord sequence (1MRK frames with 1TRC partials); frame types: {types}")
    streams = sorted({frame.stream for frame in sdif.frames
                      if frame.sig == "1MRK" and any(sig == "1TRC" for sig, _ in frame.matrices)})
    stream = _pick_stream(stream, streams, "chord-sequence frames", name)
    markers = sorted((frame for frame in sdif.frames if frame.sig == "1MRK" and frame.stream == stream),
                     key=lambda frame: frame.time)
    started: dict[int, tuple[float, float, float]] = {}
    rows: list[tuple[float, float, float, float, int]] = []
    for frame in markers:
        found: dict[str, list[np.ndarray]] = {}
        for sig, values in frame.matrices:
            if not isinstance(values, str):
                found.setdefault(sig, []).append(values)
        ends = [int(value) for values in found.get("1END", []) for value in values[:, :1].ravel()]
        for index in ends:
            if index not in started:
                raise ValueError(f"{name}: partial {index} ends at {frame.time:g} s without a start")
            begin, freq, amp = started.pop(index)
            rows.append((begin, frame.time, freq, amp, index))
        for values in found.get("1TRC", []):
            if len(values) and values.shape[1] < 3:
                raise ValueError(f"{name}: 1TRC matrix at {frame.time:g} s has {values.shape[1]} columns; "
                                 f"expected index, frequency, amplitude")
            for index_value, freq, amp in values[:, :3].astype(np.float64):
                index = int(index_value)
                if index in started:
                    raise ValueError(f"{name}: partial {index} starts again at {frame.time:g} s before it ended")
                started[index] = (frame.time, freq, amp)
    if started:
        raise ValueError(f"{name}: {len(started)} partial(s) never end (no 1END), "
                         f"first index {min(started)}")
    columns = np.array([row[:4] for row in rows], dtype=np.float64).reshape(-1, 4)
    start, stop = np.maximum(columns[:, 0], 0.0), np.maximum(columns[:, 1], 0.0)
    order = np.lexsort((columns[:, 2], start))
    events = validate(pd.DataFrame({
        "time_start": start[order], "time_stop": stop[order],
        "freq_start": columns[order, 2], "freq_stop": columns[order, 2],
        "amp_start": columns[order, 3], "amp_stop": columns[order, 3],
        "partial_id": pd.array([rows[k][4] for k in order], dtype="Int64"),
    }), "events")
    meta: dict[str, object] = {
        "source": "sdif", "file": name, "stream": stream, "streams": streams,
        "markers": [max(frame.time, 0.0) for frame in markers],
    }
    _meta_nvt(meta, sdif.nvt)
    events.attrs["meta"] = meta
    return events
