"""Case-study excerpts and the raw render format (numpy + soundfile only, imported by every runner).

Renders go to OUT/raw/<row>/<excerpt>.wav (float32, the tool's own level) with OUT/raw/<row>/meta.json.
Track tables that another synthesizer replays go to OUT/tracks/<row>/<excerpt>.csv in the format of
`benchmarks/accuracy/points.py` (time, freq, amp, track).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "accuracy"))
import points  # noqa: E402,F401  (re-exported for the runners)

SR = 44100
# PM2 needs a harmonic or inharmonic template; the flute is the only harmonic excerpt.
EXCERPTS = {"bayle": "inharmonic", "de_natura_son": "inharmonic", "saariaho_flute": "harmonic"}


def load(name: str) -> np.ndarray:
    """The excerpt from samples/ as float64 mono (mean of the channels)."""
    x, sr = sf.read(REPO / "samples" / f"{name}.wav", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{name}: {sr} Hz, expected {SR}")
    return np.ascontiguousarray(x.mean(axis=1))


def write_render(out: pathlib.Path, row: str, name: str, y) -> None:
    folder = out / "raw" / row
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / f"{name}.wav", np.asarray(y, dtype=np.float32), SR, subtype="FLOAT")


def write_meta(out: pathlib.Path, row: str, **meta) -> None:
    folder = out / "raw" / row
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
