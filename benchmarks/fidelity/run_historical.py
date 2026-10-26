"""Historical rows: the resyntheses the ICMC 2000 panel's authors made of its sounds (Wright et al. 2000).

Usage: python run_historical.py OUT   (numpy, scipy, soundfile; after fetch_reference.py OUT)

The renders are read as archived (reference.json `renders`). Each is resampled to its input's rate when the two
differ (Loris rendered the 22.05 and 48 kHz inputs at 44.1 kHz), then delayed by the shift within +-MAX_LAG_S that
gives the lowest bin LSD (evaluate.py's), searched on a 2 ms grid and refined to 0.1 ms. Zero is on the grid, so
the shift never raises the score. An envelope cross-correlation was tried first and dropped: on sustained sounds
it chose shifts that raised the bin LSD by up to 0.9 dB (SMS shaku, +14 ms). Rates, shift and lengths go to
meta.json; the evaluator cuts or pads the render to the input's length, as for every other row.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402

# benchmarks/accuracy/ (first on sys.path, via excerpts) has an evaluate.py too, so load this one by path
_spec = importlib.util.spec_from_file_location("fidelity_evaluate", excerpts.HERE / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)

MAX_LAG_S = 0.25  # the Loris webern_choral render starts 0.23 s into the input
COARSE_S, FINE_S = 0.002, 0.0001
SOURCES = {
    "ircam-2000/author": "IRCAM (Rodet, Schwarz): additive (*.synt.aiff) or hmm (*.hmm.synt.aiff); bongo_roll plain "
                         ".synt, applebite2 noise+trans, xylo m6.synth",
    "loris-2000/author": "Loris (Fitz, Haken): *.loris.aiff",
    "loris-2000/author-rs": "Loris (Fitz, Haken): *.rs.loris.aiff (file names as archived; 'rs' is not explained)",
    "sms-2000/author": "SMS (Serra, de Boer): *.sms.wav, analysis settings in the matching .anl",
}


def delay(y: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.zeros(k), y]) if k >= 0 else y[-k:]


def best_delay(x: np.ndarray, y: np.ndarray, sr: int) -> tuple[int, float, float]:
    """The delay (samples) of y with the lowest bin LSD against x, that LSD, and the LSD at delay 0."""
    _, sx = evaluate.magnitudes(x, sr)
    floor = sx.max() * evaluate.BIN_FLOOR
    ref_db = 20 * np.log10(np.maximum(sx, floor))

    def lsd(k: int) -> float:  # the bin term of evaluate.score
        z = delay(y, k)
        z = z[: x.size] if z.size >= x.size else np.pad(z, (0, x.size - z.size))
        return float(np.sqrt(((ref_db - 20 * np.log10(np.maximum(evaluate.magnitudes(z, sr)[1], floor))) ** 2).mean()))

    def grid(centre: int, half: float, step: float) -> list[int]:
        return sorted({centre + excerpts.samples(t, sr) for t in np.arange(-half, half + step / 2, step)})

    k = min(grid(0, MAX_LAG_S, COARSE_S), key=lsd)
    k = min(grid(k, COARSE_S, FINE_S), key=lsd)
    return k, lsd(k), lsd(0)


def main(out: pathlib.Path) -> None:
    found = excerpts.by_name(out)
    if not any(e.set == "icmc2000" for e in found.values()):
        raise SystemExit(f"no ICMC 2000 sounds in {out / 'reference'}: run fetch_reference.py {out} first")
    for row, files in excerpts.manifest()["renders"].items():
        done = {}
        for name, rel in files.items():
            e = found[name]
            x = excerpts.load(e)
            y, sr = sf.read(out / "reference" / rel, always_2d=True)
            y = y.mean(axis=1)
            if sr != e.sr:
                g = math.gcd(sr, e.sr)
                y = ss.resample_poly(y, e.sr // g, sr // g)
            k, best, zero = best_delay(x, y, e.sr)
            y = delay(y, k)
            excerpts.write_render(out, row, e, y)
            done[name] = {"set": e.set, "sr": e.sr, "file": pathlib.PurePath(rel).name, "file_sr": sr,
                          "delay_ms": round(1e3 * k / e.sr, 2), "bin_lsd_db_zero_delay": round(zero, 3),
                          "input_s": round(x.size / e.sr, 4), "render_s": round(y.size / e.sr, 4)}
            print(row, name, f"{sr} -> {e.sr} Hz, delay {1e3 * k / e.sr:+.1f} ms, bin LSD {zero:.2f} -> {best:.2f}",
                  flush=True)
        excerpts.write_meta(out, row, version="author render, 2000", source=SOURCES[row],
                            align={"criterion": "lowest bin LSD", "max_lag_s": MAX_LAG_S, "coarse_s": COARSE_S,
                                   "fine_s": FINE_S, "resample": "scipy.signal.resample_poly"},
                            excerpts=done)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
