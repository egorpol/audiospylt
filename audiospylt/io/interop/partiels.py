"""Read partial tracks exported by Partiels (Ircam PM2 plugins) as a tracks table."""

from __future__ import annotations

from collections.abc import Iterable
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from audiospylt.model.schema import validate

__all__ = ["read_partiels"]

_FORMATS = (".csv", ".json")


@dataclass(frozen=True)
class _Export:
    """One exported Partiels track: per channel an array (frames, 4) of time, freq, amp, phase."""

    name: str
    channels: list[np.ndarray]
    partial_id: int | None = None
    plugin: str | None = None


class _NotPartials(ValueError):
    """The file is a Partiels export, but not of a partial-tracking output."""


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _files(path: str | os.PathLike[str] | Iterable[str | os.PathLike[str]]) -> tuple[list[Path], bool]:
    """Return the export files and whether they came from a folder."""
    if isinstance(path, (str, os.PathLike)):
        root = Path(path)
        if root.is_dir():
            found = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _FORMATS),
                           key=_natural_key)
            if not found:
                raise ValueError(f"{root}: no .csv or .json files (subfolders are not searched)")
            return found, True
        files = [root]
    else:
        files = [Path(p) for p in path]
        if not files:
            raise ValueError("no files given")
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"{file}: no such file")
        if file.suffix.lower() not in _FORMATS:
            raise ValueError(f"{file}: expected a .csv or .json Partiels export")
    return files, False


def _number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _separator(line: str) -> str:
    """Column separator: the first character that cannot belong to the first field."""
    match = re.match(r"\s*(?:TIME|[-+0-9.eE]+)(.)", line)
    return match[1] if match else ","


def _read_csv(file: Path) -> _Export:
    lines = file.read_text(encoding="utf-8").splitlines()
    first = next((line for line in lines if line.strip()), None)
    if first is None:
        raise _NotPartials(f"{file}: empty file")
    separator = _separator(first)
    blocks: list[list[tuple[int, list[str]]]] = [[]]
    width = None
    for number, line in enumerate(lines, 1):
        if not line.strip():
            if blocks[-1]:
                blocks.append([])
            continue
        fields = next(csv.reader([line], delimiter=separator))
        if not _number(fields[0]):
            width = len(fields)  # a header line, repeated before every channel block
            continue
        blocks[-1].append((number, fields))
    if not blocks[-1]:
        blocks.pop()
    if width is None:
        # Without a header only rows with a value show the width (gap rows end after DURATION).
        width = max((len(f) for block in blocks for _, f in block if len(f) > 2 and f[2]), default=5)
    if width != 5:
        raise _NotPartials(f"{file}: {width} columns, not TIME, DURATION, frequency, amplitude, phase")
    channels: list[np.ndarray] = []
    for block in blocks:
        rows = np.full((len(block), 4), np.nan)
        for row, (number, fields) in enumerate(block):
            value = fields[2] if len(fields) > 2 else ""
            try:
                rows[row, 0] = float(fields[0])
                if value:
                    if len(fields) != 5:
                        raise ValueError(f"{len(fields)} fields")
                    rows[row, 1:] = [float(field) for field in fields[2:]]
            except ValueError as error:
                raise ValueError(f"{file}, line {number}: cannot read a partial point ({error})") from None
        channels.append(rows)
    return _Export(file.name, channels)


def _read_json(file: Path) -> _Export:
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
        results = document["results"]
    except (ValueError, TypeError, KeyError) as error:
        raise _NotPartials(f"{file}: not a Partiels JSON export ({error})") from None
    if not isinstance(results, list) or not all(isinstance(frames, list) for frames in results):
        raise _NotPartials(f"{file}: 'results' is not a list of channels")
    channels: list[np.ndarray] = []
    for frames in results:
        rows = np.full((len(frames), 4), np.nan)
        for row, frame in enumerate(frames):
            if not isinstance(frame, dict) or "value" not in frame:
                raise _NotPartials(f"{file}: frames without a single value")
            value, extra = frame["value"], frame.get("extra") or []
            if value is not None and (isinstance(value, (list, dict)) or not isinstance(extra, list)
                                      or len(extra) != 2):
                count = len(extra) if isinstance(extra, list) else "no list of"
                raise _NotPartials(f"{file}: {count} extra output(s), not amplitude and phase")
            try:
                rows[row, 0] = frame["time"]
                if value is not None:
                    rows[row, 1:] = [value, *extra]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{file}, channel {len(channels)}, frame {row}: "
                                 f"cannot read a partial point ({error!r})") from None
        channels.append(rows)
    track = document.get("track") if isinstance(document, dict) else None
    partial_id = plugin = None
    if isinstance(track, dict):
        key = track.get("key")
        if isinstance(key, dict) and isinstance(key.get("identifier"), str):
            plugin = key["identifier"]
        parameters = (track.get("state") or {}).get("parameters") or {}
        number = parameters.get("partialid") if isinstance(parameters, dict) else None
        if isinstance(number, (int, float)) and not isinstance(number, bool) and float(number).is_integer():
            partial_id = int(number)
    return _Export(file.name, channels, partial_id, plugin)


def read_partiels(
    path: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    channel: int = 0,
    max_step_cents: float | None = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Read Partiels exports of PM2 partial tracking as one tracks table.

    Partiels exports one file per analysis track, and the Ircam PM2 templates
    hold one track per partial ("slot"): every frame of the file gives that
    slot's frequency, amplitude and phase, or nothing (a gap). A gap ends a
    track, so each run of frames between gaps becomes one ``track_id``.

    Parameters
    ----------
    path : path, or iterable of paths
        A folder of exports (its ``.csv`` or ``.json`` files; not both, and
        subfolders are not searched), one exported file, or several. CSV may
        have any Partiels header (``none``, ``generic``, ``specific``) and
        holds one block per channel; JSON may come with or without
        ``--description``. In a folder, tracks that are not partial tracks
        (the F0 "Feature Scoring" track, an FFT) are skipped and listed in
        ``attrs["meta"]["skipped"]``; a named file of that kind raises.
    channel : int, default 0
        Channel to read. Partiels analyses every channel of the sound
        separately, so a stereo export holds two independent analyses.
    max_step_cents : float, optional
        Also end a track where the frequency moves by more than this many
        cents from one frame to the next (an inharmonic slot taken over by
        another partial, or a harmonic following an octave error of the f0).
        None (default) splits at gaps only.
    dedupe : bool, default True
        Drop points that appear in more than one slot (the same time,
        frequency and amplitude; inharmonic PM2 in Partiels writes such
        copies). Longer runs keep their points, so a partial keeps its
        longest continuous form, and runs that lose points split around them.

    Returns
    -------
    pandas.DataFrame
        A validated ``tracks`` table (``track_id``, ``time``, ``freq``,
        ``amp``, ``phase``) plus an Int64 ``partial_id`` column, rows grouped
        by track in time order, tracks numbered by start time (then start
        frequency). ``amp`` is PM2's amplitude, a sine's peak amplitude re full
        scale like ``spectrum``; ``phase`` is PM2's phase in radians, the sine
        phase at the frame time (the reference of ``synthesize``).
        ``partial_id`` is PM2's partial ID, read from JSON exported with
        ``--description`` (the harmonic number for harmonic tracking), and
        missing otherwise. ``attrs["meta"]``: ``source``, ``format``,
        ``channel``, ``channels``, ``files``, ``skipped``,
        ``duplicates_dropped``, ``max_step_cents`` and, from JSON with a
        description, ``plugin``.

    Raises
    ------
    FileNotFoundError
        A named file does not exist.
    ValueError
        No partial-track files, mixed CSV and JSON, a named file that is not a
        partial track, a value that cannot be read, a missing ``channel``,
        or ``max_step_cents`` not positive.

    Notes
    -----
    Times are frame times as exported; a time just below 0 (PM2 can place the
    first frame half a sample early) is read as 0. CSV keeps 6 significant
    digits, JSON the full precision. Partiels exports from the command line
    with, for example, ``Partiels --export -i SOUND.wav -t TEMPLATE.ptldoc
    -o FOLDER -f json --description``.
    """
    if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)) or channel < 0:
        raise ValueError(f"channel must be an int >= 0, got {channel!r}")
    if max_step_cents is not None and not (np.isfinite(max_step_cents) and max_step_cents > 0):
        raise ValueError(f"max_step_cents must be a positive number or None, got {max_step_cents!r}")
    files, from_folder = _files(path)
    formats = {file.suffix.lower() for file in files}
    if len(formats) > 1:
        where = f"{path}: " if from_folder else ""
        raise ValueError(f"{where}mixed CSV and JSON exports; read one format at a time "
                         "(export each format to its own folder, or pass a list of files)")
    reader = _read_csv if formats == {".csv"} else _read_json
    exports, skipped = [], []
    for file in files:
        try:
            exports.append(reader(file))
        except _NotPartials:
            if not from_folder:
                raise
            skipped.append(file.name)
    if not exports:
        raise ValueError(f"no partial tracks among {[file.name for file in files]}")
    for export in exports:
        if channel >= len(export.channels):
            raise ValueError(f"{export.name} has {len(export.channels)} channel(s); "
                             f"channel={channel} does not exist")

    # One row per exported point: slot (file), frame, run (between gaps and jumps), values.
    parts = []
    run_count = 0
    for number, export in enumerate(exports):
        frames = export.channels[channel]
        present = ~np.isnan(frames[:, 1])
        starts = present.copy()
        starts[1:] &= ~present[:-1]
        if max_step_cents is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.abs(1200 * np.log2(frames[1:, 1] / frames[:-1, 1]))
            starts[1:] |= present[1:] & present[:-1] & ~(step <= max_step_cents)
        runs = run_count + np.cumsum(starts) - 1
        run_count += int(starts.sum())
        index = np.flatnonzero(present)
        parts.append(np.column_stack([np.full(index.size, number), index, runs[index], frames[index]]))
    points = np.concatenate(parts) if parts else np.empty((0, 7))
    columns: list[np.ndarray] = [points[:, k].astype(np.int64) for k in range(3)] + list(points[:, 3:].T)
    slot, frame, run, time, freq, amp, phase = columns

    duplicates = 0
    if dedupe:
        # Rank runs longest first (then slot, start frame); a point found in several runs stays in
        # the best-ranked one. Run ids count 0, 1, ... in row order.
        heads = np.flatnonzero(np.diff(run, prepend=-1))
        length = np.diff(np.r_[heads, len(run)])
        rank = np.empty(heads.size, dtype=np.int64)
        rank[np.lexsort((frame[heads], slot[heads], -length))] = np.arange(heads.size)
        order = np.lexsort((rank[run], amp, freq, time))
        t, f, a = time[order], freq[order], amp[order]
        repeat = np.zeros(len(order), dtype=bool)
        repeat[1:] = (t[1:] == t[:-1]) & (f[1:] == f[:-1]) & (a[1:] == a[:-1])
        keep = np.ones(len(time), dtype=bool)
        keep[order[repeat]] = False
        duplicates = int(repeat.sum())
        slot, frame, run, time, freq, amp, phase = (x[keep] for x in (slot, frame, run, time, freq, amp, phase))

    # Pieces: runs, split again where dedupe took points out. Rows stay in slot, frame order.
    begins = np.ones(len(run), dtype=bool)
    begins[1:] = (run[1:] != run[:-1]) | (frame[1:] != frame[:-1] + 1)
    piece = np.cumsum(begins) - 1
    time = np.maximum(time, 0.0)
    # Two frames clipped to 0 in one piece: keep the later one.
    later = np.ones(len(time), dtype=bool)
    later[:-1] = (piece[:-1] != piece[1:]) | (time[1:] > 0)
    piece, slot, frame, time, freq, amp, phase = (x[later] for x in (piece, slot, frame, time, freq, amp, phase))
    heads = np.flatnonzero(np.diff(piece, prepend=-1))
    numbering = np.empty(heads.size, dtype=np.int64)
    numbering[np.lexsort((frame[heads], slot[heads], freq[heads], time[heads]))] = np.arange(heads.size)
    track_id = numbering[piece]
    rows = np.argsort(track_id, kind="stable")
    partial_id = pd.array([export.partial_id for export in exports], dtype="Int64")[slot[rows]]
    tracks = validate(pd.DataFrame({
        "track_id": track_id[rows], "time": time[rows], "freq": freq[rows], "amp": amp[rows],
        "phase": phase[rows], "partial_id": partial_id,
    }), "tracks")
    meta: dict[str, object] = {
        "source": "partiels", "format": formats.pop().lstrip("."), "channel": int(channel),
        "channels": max(len(export.channels) for export in exports),
        "files": [export.name for export in exports], "skipped": skipped,
        "duplicates_dropped": duplicates, "max_step_cents": max_step_cents,
    }
    plugins = sorted({export.plugin for export in exports if export.plugin is not None})
    if plugins:
        meta["plugin"] = ", ".join(plugins)
    tracks.attrs["meta"] = meta
    return tracks
