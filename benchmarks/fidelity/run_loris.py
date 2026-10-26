"""Loris rows (loristrck): matched and home analyses, Loris's own synthesis, tracks for asp.synthesize.

Usage: python run_loris.py OUT [--sets case,icmc2000,tools]   (venv with loristrck + soundfile)
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

# Hann 2048 at 44.1 kHz: main lobe 86.13 Hz zero to zero; Loris resolution = windowsize / 2 (as in 4.1). The values
# are in Hz and seconds, so they hold at every rate.
MATCHED = {"resolution": 2 / excerpts.FRAME_S, "windowsize": 4 / excerpts.FRAME_S, "hoptime": excerpts.HOP_S,
           "ampfloor": -80.0}
# Loris 1.8 test/morphtest.py (lines 59-61, 116-117): Analyzer(res) = configure(res, 2 res) (src/Analyzer.C
# 236-239), whose defaults (configure, ~400-420) give hop 1/windowsize and sidelobe 90 dB; then setFreqDrift(30),
# and setAmpFloor(-80) for the clarinet (the flute keeps the default -90). Every value is passed explicitly.
HOME = {"loris-clarinet": {"resolution": 390.0, "windowsize": 780.0, "hoptime": 1 / 780, "sidelobe": 90.0,
                           "freqdrift": 30.0, "ampfloor": -80.0},
        "loris-flute": {"resolution": 270.0, "windowsize": 540.0, "hoptime": 1 / 540, "sidelobe": 90.0,
                        "freqdrift": 30.0, "ampfloor": -90.0}}


def run(out: pathlib.Path, row: str, e: excerpts.Excerpt, params: dict) -> dict:
    x = excerpts.load(e)
    parts = lt.analyze(x, e.sr, **params)
    # synthesize starts at the first breakpoint by default (83 ms late on bayle); render from 0 instead
    excerpts.write_render(out, row, e, lt.synthesize(parts, e.sr, start=0.0, end=x.size / e.sr))
    rows = [np.column_stack([q[:, 0], q[:, 1], q[:, 2], np.full(len(q), i)]) for i, q in enumerate(parts)]
    points.write_points(out / "tracks" / row, e.name, *np.vstack(rows).T)
    print(row, e.name, len(parts), "partials", flush=True)
    return {"set": e.set, "sr": e.sr, "partials": len(parts)}


def main(out: pathlib.Path, sets) -> None:
    version = "loristrck " + importlib.metadata.version("loristrck")
    chosen = excerpts.excerpts(out, sets)
    done = {e.name: run(out, "loris/matched", e, MATCHED) for e in chosen}
    excerpts.write_meta(out, "loris/matched", version=version, params=MATCHED, synth="loristrck.synthesize",
                        excerpts=done)
    home = [e for e in chosen if e.name in HOME]
    if home:
        done = {e.name: {**run(out, "loris/home", e, HOME[e.name]), "params": HOME[e.name]} for e in home}
        excerpts.write_meta(out, "loris/home", version=version, synth="loristrck.synthesize", excerpts=done,
                            source="Loris 1.8 test/morphtest.py settings for its test sounds")


if __name__ == "__main__":
    args, sets = excerpts.cli(sys.argv[1:])
    main(pathlib.Path(args[0]), sets)
