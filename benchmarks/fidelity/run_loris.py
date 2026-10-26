"""Loris row (loristrck): matched analysis, Loris's own synthesis, tracks for asp.synthesize.

Usage: python run_loris.py OUT   (venv with loristrck + soundfile)
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys

import loristrck as lt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402
from excerpts import points  # noqa: E402

ROW = "loris/matched"
# Hann 2048 at 44.1 kHz: main lobe 86.13 Hz zero to zero; Loris resolution = windowsize / 2 (as in 4.1)
MATCHED = {"resolution": 2 * excerpts.SR / 2048, "windowsize": 4 * excerpts.SR / 2048, "hoptime": 256 / excerpts.SR,
           "ampfloor": -80.0}


def main(out: pathlib.Path) -> None:
    version = "loristrck " + importlib.metadata.version("loristrck")
    for name in excerpts.EXCERPTS:
        x = excerpts.load(name)
        parts = lt.analyze(x, excerpts.SR, **MATCHED)
        # synthesize starts at the first breakpoint by default (83 ms late on bayle); render from 0 instead
        excerpts.write_render(out, ROW, name, lt.synthesize(parts, excerpts.SR, start=0.0, end=x.size / excerpts.SR))
        rows = [np.column_stack([q[:, 0], q[:, 1], q[:, 2], np.full(len(q), i)]) for i, q in enumerate(parts)]
        points.write_points(out / "tracks" / ROW, name, *np.vstack(rows).T)
        print(ROW, name, len(parts), "partials", flush=True)
    excerpts.write_meta(out, ROW, version=version, params=MATCHED, synth="loristrck.synthesize")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
