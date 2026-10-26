"""Excerpt sets and the raw render format (numpy + soundfile only, imported by every runner).

Sets: `case` (the case-study excerpts in samples/, always present) and the reference sets of `reference.json`
(`icmc2000`, `tools`), present once `fetch_reference.py OUT` has put them in OUT/reference/. Every excerpt keeps
its own sampling rate; renders are written at that rate.

Renders go to OUT/raw/<row>/<excerpt>.wav (float32, the tool's own level) with OUT/raw/<row>/meta.json.
Track tables that another synthesizer replays go to OUT/tracks/<row>/<excerpt>.csv in the format of
`benchmarks/accuracy/points.py` (time, freq, amp, track).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
import sys

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "accuracy"))
import points  # noqa: E402,F401  (re-exported for the runners)

SR = 44100  # rate of the case-study excerpts; the matched window and hop are defined there, in seconds
FRAME_S, HOP_S = 2048 / SR, 256 / SR  # 46.44 ms, 5.805 ms
# PM2 needs a harmonic or inharmonic template; the flute is the only harmonic case-study excerpt. Reference
# sounds get both templates.
CASE = {"bayle": "inharmonic", "de_natura_son": "inharmonic", "saariaho_flute": "harmonic"}
SETS = ("case", "icmc2000", "tools")


@dataclasses.dataclass(frozen=True)
class Excerpt:
    name: str
    set: str
    path: pathlib.Path
    pm2: tuple[str, ...]  # PM2 templates to run

    @property
    def sr(self) -> int:
        return sf.info(self.path).samplerate


@functools.cache
def manifest() -> dict:
    return json.loads((HERE / "reference.json").read_text())


def excerpts(out: pathlib.Path, sets=SETS) -> list[Excerpt]:
    """The excerpts of `sets` in manifest order; a reference set counts only when every input is fetched."""
    found = [Excerpt(n, "case", REPO / "samples" / f"{n}.wav", (k,)) for n, k in CASE.items()] if "case" in sets else []
    for s, inputs in manifest()["inputs"].items():
        paths = {n: out / "reference" / p for n, p in inputs.items()}
        if s in sets and all(p.is_file() for p in paths.values()):
            found += [Excerpt(n, s, p, ("harmonic", "inharmonic")) for n, p in paths.items()]
    return found


def by_name(out: pathlib.Path) -> dict[str, Excerpt]:
    return {e.name: e for e in excerpts(out)}


def cli(argv: list[str]) -> tuple[list[str], tuple[str, ...]]:
    """The positional arguments and the sets named by `--sets a,b` (default: every set)."""
    args, sets = list(argv), SETS
    if "--sets" in args:
        i = args.index("--sets")
        sets = tuple(args[i + 1].split(","))
        del args[i: i + 2]
        if unknown := set(sets) - set(SETS):
            raise SystemExit(f"unknown sets {sorted(unknown)}; known: {', '.join(SETS)}")
    return args, sets


def samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def load(e: Excerpt) -> np.ndarray:
    """The excerpt as float64 mono (mean of the channels) at its own rate."""
    x, _ = sf.read(e.path, always_2d=True)
    return np.ascontiguousarray(x.mean(axis=1))


def write_render(out: pathlib.Path, row: str, e: Excerpt, y) -> None:
    folder = out / "raw" / row
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / f"{e.name}.wav", np.asarray(y, dtype=np.float32), e.sr, subtype="FLOAT")


def write_meta(out: pathlib.Path, row: str, **meta) -> None:
    """OUT/raw/<row>/meta.json. Its per-excerpt entries (`excerpts`: name -> dict) are merged into those of an
    existing file, so that a run on some sets keeps the entries of the others."""
    path = out / "raw" / row / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if "excerpts" in meta and path.is_file():
        meta["excerpts"] = {**json.loads(path.read_text()).get("excerpts", {}), **meta["excerpts"]}
    path.write_text(json.dumps(meta, indent=1))
