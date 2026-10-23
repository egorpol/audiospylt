"""Loris rows (loristrck): documented rule of thumb and matched window.

Usage: python run_loris.py OUT   (venv with loristrck + soundfile)
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys

import numpy as np
import soundfile as sf
import loristrck as lt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import points  # noqa: E402
import truth  # noqa: E402

# Hann 2048 at 44.1 kHz: main lobe 4 bins zero to zero = 86.13 Hz; Loris resolution = windowsize / 2
MATCHED = {"resolution": 2 * truth.SR / 2048, "windowsize": 4 * truth.SR / 2048, "hoptime": 256 / truth.SR,
           "ampfloor": -80.0}


def run(out: pathlib.Path, row: str, params_for) -> None:
    folder = out / "raw" / row
    used, steps = {}, []
    for name in truth.SIGNALS:
        x, sr = sf.read(out / "suite" / f"{name}.wav")
        p = params_for(name)
        used[name] = p
        parts = lt.analyze(np.ascontiguousarray(x, dtype=float), sr, **p)
        rows = [np.column_stack([q[:, 0], q[:, 1], q[:, 2], np.full(len(q), i)]) for i, q in enumerate(parts)]
        steps += [np.diff(q[:, 0]) for q in parts if len(q) > 1]
        data = np.vstack(rows) if rows else np.zeros((0, 4))
        points.write_points(folder, name, *data.T)
    hop = float(np.median(np.concatenate(steps)))
    points.write_meta(folder, kind="tracks", hop_s=hop, version="loristrck " + importlib.metadata.version("loristrck"),
                      params=used, hop_note="median breakpoint step within partials")
    print(row, "done, hop", round(hop, 5))


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    # loristrck docs: resolution ~= 0.9 * min(f0); everything else at its default
    run(out, "loris/default", lambda name: {"resolution": 0.9 * truth.lowest_freq(name)})
    run(out, "loris/matched", lambda name: dict(MATCHED))
