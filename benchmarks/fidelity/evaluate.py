"""Score every render in OUT/raw against its excerpt; write results/ and RMS-matched listening WAVs.

Usage: python evaluate.py OUT RESULTS   (numpy, scipy, soundfile; no AudioSpylt import)
"""

from __future__ import annotations

import csv
import pathlib
import re
import sys

import numpy as np
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402

SR = excerpts.SR
NPERSEG, HOP = 2048, 512
MIN_BINS = 8  # band LSD: third-octave bands merged upward until each holds this many STFT bins
LISTEN_PEAK_DB = -1.0
ORDER = ["audiospylt-v1/default", "audiospylt-v1/matched", "audiospylt-v1/matched-fade", "audiospylt-0.6/matched",
         "loris/matched", "loris/matched+asp", "sms-tools/matched", "sms-tools/matched+asp",
         "pm2-harmonic/matched+asp", "pm2-inharmonic/matched+asp"]


def band_edges() -> np.ndarray:
    upper = 1000 * 2 ** (np.arange(-14, 14) / 3 + 1 / 6)
    edges = [0.0]
    for hi in [*upper[upper < SR / 2], SR / 2 + 1]:
        if (hi - edges[-1]) / (SR / NPERSEG) >= MIN_BINS:
            edges.append(float(hi))
    edges[-1] = SR / 2 + 1  # the tail joins the last band
    return np.array(edges)


EDGES = band_edges()


def magnitudes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, _, s = ss.stft(x, fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
    return f, np.abs(s)


def score(ref: np.ndarray, y: np.ndarray) -> dict:
    """Bin and band LSD (dB) over the whole excerpt and per band, plus the RMS gain re the reference."""
    y = y[: ref.size] if y.size >= ref.size else np.pad(y, (0, ref.size - y.size))
    f, sx = magnitudes(ref)
    _, sy = magnitudes(y)
    floor = sx.max() * 1e-4  # bin LSD as in the 2026-10-10 audit: floor 80 dB below the reference peak
    d_bin = (20 * np.log10(np.maximum(sx, floor)) - 20 * np.log10(np.maximum(sy, floor))) ** 2
    sel = [(f >= lo) & (f < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    px = np.array([(sx[s] ** 2).sum(axis=0) for s in sel])
    py = np.array([(sy[s] ** 2).sum(axis=0) for s in sel])
    bfloor = px.max() * 1e-8  # 80 dB below the loudest band power
    d_band = (10 * np.log10(np.maximum(px, bfloor)) - 10 * np.log10(np.maximum(py, bfloor))) ** 2
    return {"bin": float(np.sqrt(d_bin.mean())), "band": float(np.sqrt(d_band.mean())),
            "gain": float(10 * np.log10(np.mean(y ** 2) / np.mean(ref ** 2))),
            "bands": [(float(np.sqrt(d_bin[s].mean())), float(np.sqrt(d_band[i].mean()))) for i, s in enumerate(sel)]}


def noise_floor() -> dict:
    """Two independent white-noise realisations of equal level: the distance a perfect noise model still gets."""
    rng = np.random.default_rng(42)
    a, b = (0.1 * rng.standard_normal(10 * SR) for _ in range(2))
    return score(a, b)


def group(row: str) -> str:
    return re.sub(r"/r\d+$", "", row)


def main(out: pathlib.Path, results: pathlib.Path) -> None:
    raw = out / "raw"
    rows = sorted(p.relative_to(raw).as_posix() for p in raw.glob("**/") if any(p.glob("*.wav")))
    refs = {name: excerpts.load(name) for name in excerpts.EXCERPTS}
    scores = {}
    for row in rows:
        for name, ref in refs.items():
            wav = raw / row / f"{name}.wav"
            if wav.is_file():
                y, sr = sf.read(wav)
                assert sr == SR
                scores[row, name] = score(ref, y)
        print(row, "scored", flush=True)
    floor = noise_floor()
    write_tables(results, rows, scores, floor)
    write_listening(out, rows, refs)


def write_tables(results: pathlib.Path, rows, scores, floor) -> None:
    results.mkdir(parents=True, exist_ok=True)
    names = list(excerpts.EXCERPTS)
    with open(results / "per_excerpt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "excerpt", "bin_lsd_db", "band_lsd_db", "gain_db"])
        for (row, name), s in scores.items():
            w.writerow([row, name, f"{s['bin']:.3f}", f"{s['band']:.3f}", f"{s['gain']:.3f}"])
        w.writerow(["noise-pair", "white noise", f"{floor['bin']:.3f}", f"{floor['band']:.3f}", f"{floor['gain']:.3f}"])
    with open(results / "per_band.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "excerpt", "band_lo_hz", "band_hi_hz", "bin_lsd_db", "band_lsd_db"])
        for (row, name), s in [*scores.items(), (("noise-pair", "white noise"), floor)]:
            for (lo, hi), (b, bb) in zip(zip(EDGES[:-1], EDGES[1:]), s["bands"]):
                w.writerow([row, name, f"{lo:.1f}", f"{min(hi, SR / 2):.1f}", f"{b:.3f}", f"{bb:.3f}"])
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(group(row), []).append(row)
    ordered = [g for g in ORDER if g in groups] + sorted(set(groups) - set(ORDER))
    with open(results / "repeats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "excerpt", "runs", "metric", "mean", "min", "max"])
        for g in ordered:
            if len(groups[g]) > 1:
                for name in names:
                    got = [scores[r, name] for r in groups[g] if (r, name) in scores]
                    for m in ("bin", "band", "gain"):
                        v = [s[m] for s in got]
                        if v:
                            w.writerow([g, name, len(v), m, f"{np.mean(v):.3f}", f"{min(v):.3f}", f"{max(v):.3f}"])

    def cell(g: str, name: str, m: str) -> str:
        v = [scores[r, name][m] for r in groups[g] if (r, name) in scores]
        if not v:
            return "–"
        if len(v) == 1:
            return f"{v[0]:.2f}"
        return f"{np.mean(v):.2f} ({min(v):.2f}–{max(v):.2f})"

    lines = ["# Resynthesis fidelity: results", "",
             "Generated by `evaluate.py`. LSD in dB, lower is better; gain is the RMS level of the render re the "
             "excerpt. \"–\": the row has no render of that excerpt. Rows with several runs show the mean and "
             "(min–max).", "",
             f"Noise-pair floor (two independent white-noise realisations, 10 s): bin LSD {floor['bin']:.2f} dB, "
             f"band LSD {floor['band']:.2f} dB.", ""]
    for m, title in (("bin", "Bin LSD (dB)"), ("band", f"Band LSD (dB, {len(EDGES) - 1} bands)"),
                     ("gain", "Gain (dB re excerpt RMS)")):
        lines += [f"**{title}**", "", "| row | " + " | ".join(names) + " |",
                  "| --- |" + " ---: |" * len(names)]
        for g in ordered:
            label = g + (f" ({len(groups[g])} runs)" if len(groups[g]) > 1 else "")
            lines.append(f"| {label} | " + " | ".join(cell(g, n, m) for n in names) + " |")
        lines.append("")
    lines += ["Bands (Hz): " + ", ".join(f"{lo:.0f}–{min(hi, SR / 2):.0f}" for lo, hi in zip(EDGES[:-1], EDGES[1:])),
              ""]
    (results / "summary.md").write_text("\n".join(lines))


def write_listening(out: pathlib.Path, rows, refs) -> None:
    """Per excerpt: the mono original and every render (first run of repeated rows), equal RMS, 16-bit."""
    raw = out / "raw"
    picked = []
    for row in rows:
        if group(row) == row or row.endswith("/r1"):
            picked.append(row)
    picked.sort(key=lambda r: ORDER.index(group(r)) if group(r) in ORDER else len(ORDER))
    for name, ref in refs.items():
        target = np.sqrt(np.mean(ref ** 2))
        clips = [("00_original", ref)]
        for i, row in enumerate(picked, start=1):
            wav = raw / row / f"{name}.wav"
            if wav.is_file():
                y, _ = sf.read(wav)
                y = y[: ref.size] if y.size >= ref.size else np.pad(y, (0, ref.size - y.size))
                clips.append((f"{i:02d}_{group(row).replace('/', '_')}", y * target / np.sqrt(np.mean(y ** 2))))
        peak = max(np.abs(c).max() for _, c in clips)
        scale = min(1.0, 10 ** (LISTEN_PEAK_DB / 20) / peak)  # one factor per excerpt keeps the levels equal
        folder = out / "listening" / name
        folder.mkdir(parents=True, exist_ok=True)
        for stem, c in clips:
            sf.write(folder / f"{stem}.wav", c * scale, SR, subtype="PCM_16")
        print("listening", name, len(clips), "files, scale", round(20 * np.log10(scale), 2), "dB", flush=True)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
