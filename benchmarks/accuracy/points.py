"""Raw runner output shared by all venvs (numpy only).

Each row (tool + setting) writes OUT/raw/<row>/<signal>.csv with columns
time, freq, amp, track (amp = D4 peak amplitude re full scale; track -1 for
untracked per-frame peaks) and OUT/raw/<row>/meta.json with the parameters
and the nominal hop in seconds.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def write_points(folder: pathlib.Path, name: str, time, freq, amp, track) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([time, freq, amp, track]).astype(float)
    np.savetxt(folder / f"{name}.csv", rows, delimiter=",", header="time,freq,amp,track", comments="",
               fmt=["%.9f", "%.6f", "%.9g", "%d"])


def write_meta(folder: pathlib.Path, **meta) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))


def read_points(folder: pathlib.Path, name: str) -> np.ndarray:
    """(n, 4) array: time, freq, amp, track."""
    data = np.loadtxt(folder / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
    return data.reshape(-1, 4)


def frames_to_tracks(freq: np.ndarray) -> np.ndarray:
    """Track ids for a (frames, slots) array where 0 means empty: runs between gaps per slot."""
    ids = np.full(freq.shape, -1, dtype=int)
    next_id = 0
    for slot in range(freq.shape[1]):
        on = freq[:, slot] > 0
        starts = on & ~np.concatenate([[False], on[:-1]])
        run = np.cumsum(starts) - 1 + next_id
        ids[on, slot] = run[on]
        next_id += int(starts.sum())
    return ids
