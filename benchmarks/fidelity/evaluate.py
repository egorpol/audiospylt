"""Score every render in OUT/raw against its excerpt; write results/ and RMS-matched listening WAVs.

Usage: python evaluate.py OUT RESULTS   (numpy, scipy, soundfile; no AudioSpylt import)

A row is a folder OUT/raw/<row>/ with a meta.json; its renders are the excerpts listed there (`excerpts`). Every
excerpt is scored at its own rate, with the STFT frame and hop fixed in seconds.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

import numpy as np
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402

FRAME_S, HOP_S = 2048 / excerpts.SR, 512 / excerpts.SR  # 46.4 and 11.6 ms: 2048 and 512 samples at 44.1 kHz
BIN_FLOOR = 1e-4  # bin LSD as in the 2026-10-10 audit: floor 80 dB below the reference peak
MIN_BINS = 8  # band LSD: third-octave bands merged upward until each holds this many STFT bins
LISTEN_PEAK_DB = -1.0
COVER, QUIET_DB = 0.9, -10.0  # summary notes: historical renders shorter than this share of their input, or quieter
ORDER = ["audiospylt-v1/default", "audiospylt-v1/matched", "audiospylt-v1/matched-fade", "audiospylt-0.6/matched",
         "loris/matched", "loris/matched+asp", "loris/home", "loris/home+asp",
         "sms-tools/matched", "sms-tools/matched+asp", "sms-tools/home", "sms-tools/home+asp",
         "pm2-harmonic/matched+asp", "pm2-inharmonic/matched+asp",
         "ircam-2000/author", "loris-2000/author", "loris-2000/author-rs", "sms-2000/author"]
TITLES = {"case": "Case-study excerpts (`case`)", "icmc2000": "ICMC 2000 panel sounds (`icmc2000`)",
          "tools": "Test sounds of Loris and sms-tools (`tools`)"}
METRICS = (("bin", "Bin LSD (dB)"), ("band", "Band LSD (dB)"), ("gain", "Gain (dB re excerpt RMS)"))


def stft_size(sr: int) -> tuple[int, int]:
    return excerpts.samples(FRAME_S, sr), excerpts.samples(HOP_S, sr)


def band_edges(sr: int) -> np.ndarray:
    upper = 1000 * 2 ** (np.arange(-14, 14) / 3 + 1 / 6)
    edges = [0.0]
    for hi in [*upper[upper < sr / 2], sr / 2 + 1]:
        if (hi - edges[-1]) / (sr / stft_size(sr)[0]) >= MIN_BINS:
            edges.append(float(hi))
    edges[-1] = sr / 2 + 1  # the tail joins the last band
    return np.array(edges)


def magnitudes(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg, hop = stft_size(sr)
    f, _, s = ss.stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, boundary=None, padded=False)
    return f, np.abs(s)


def score(ref: np.ndarray, y: np.ndarray, sr: int) -> dict:
    """Bin and band LSD (dB) over the whole excerpt and per band, plus the RMS gain re the reference."""
    y = y[: ref.size] if y.size >= ref.size else np.pad(y, (0, ref.size - y.size))
    f, sx = magnitudes(ref, sr)
    _, sy = magnitudes(y, sr)
    floor = sx.max() * BIN_FLOOR
    d_bin = (20 * np.log10(np.maximum(sx, floor)) - 20 * np.log10(np.maximum(sy, floor))) ** 2
    edges = band_edges(sr)
    sel = [(f >= lo) & (f < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    px = np.array([(sx[s] ** 2).sum(axis=0) for s in sel])
    py = np.array([(sy[s] ** 2).sum(axis=0) for s in sel])
    bfloor = px.max() * 1e-8  # 80 dB below the loudest band power
    d_band = (10 * np.log10(np.maximum(px, bfloor)) - 10 * np.log10(np.maximum(py, bfloor))) ** 2
    return {"bin": float(np.sqrt(d_bin.mean())), "band": float(np.sqrt(d_band.mean())),
            "gain": float(10 * np.log10(np.mean(y ** 2) / np.mean(ref ** 2))),
            "bands": [(float(np.sqrt(d_bin[s].mean())), float(np.sqrt(d_band[i].mean()))) for i, s in enumerate(sel)]}


def noise_floor(sr: int) -> dict:
    """Two independent white-noise realisations of equal level: the distance a perfect noise model still gets."""
    rng = np.random.default_rng(42)
    a, b = (0.1 * rng.standard_normal(10 * sr) for _ in range(2))
    return score(a, b, sr)


def group(row: str) -> str:
    return re.sub(r"/r\d+$", "", row)


def historical(row: str) -> bool:
    return row.endswith("-2000/author") or row.endswith("-2000/author-rs")


def find_rows(raw: pathlib.Path) -> dict[str, dict]:
    """Row -> its meta.json."""
    return {p.parent.relative_to(raw).as_posix(): json.loads(p.read_text()) for p in sorted(raw.glob("**/meta.json"))}


def main(out: pathlib.Path, results: pathlib.Path) -> None:
    raw = out / "raw"
    known = excerpts.excerpts(out)
    by_name = {e.name: e for e in known}
    metas = find_rows(raw)
    refs: dict[str, np.ndarray] = {}
    scores = {}
    for row, meta in metas.items():
        names = [n for n in meta["excerpts"] if n in by_name]  # a set not fetched here is skipped
        for name in names:
            e = by_name[name]
            if name not in refs:
                refs[name] = excerpts.load(e)
            y, sr = sf.read(raw / row / f"{name}.wav")
            assert sr == e.sr, (row, name, sr, e.sr)
            scores[row, name] = score(refs[name], y, sr)
        print(row, len(names), "scored", flush=True)
    floors = {sr: noise_floor(sr) for sr in sorted({e.sr for e in known})}
    write_tables(results, known, list(metas), scores, floors, metas)
    write_listening(out, known, list(metas), refs)


def write_tables(results: pathlib.Path, known, rows, scores, floors, metas) -> None:
    results.mkdir(parents=True, exist_ok=True)
    sets = {s: [e for e in known if e.set == s] for s in excerpts.SETS if any(e.set == s for e in known)}
    srs = {e.name: e.sr for e in known}
    floor_rows = [("noise-pair", f"white noise, {sr} Hz", sr, f) for sr, f in floors.items()]
    with open(results / "per_excerpt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "row", "excerpt", "sr", "bin_lsd_db", "band_lsd_db", "gain_db"])
        for s, members in sets.items():
            for row in rows:
                for e in members:
                    if (row, e.name) in scores:
                        sc = scores[row, e.name]
                        w.writerow([s, row, e.name, e.sr, f"{sc['bin']:.3f}", f"{sc['band']:.3f}", f"{sc['gain']:.3f}"])
        for row, name, sr, f in floor_rows:
            w.writerow(["", row, name, sr, f"{f['bin']:.3f}", f"{f['band']:.3f}", f"{f['gain']:.3f}"])
    with open(results / "per_band.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "row", "excerpt", "band_lo_hz", "band_hi_hz", "bin_lsd_db", "band_lsd_db"])
        cells = [(s, row, e.name, e.sr, scores[row, e.name]) for s, members in sets.items() for row in rows
                 for e in members if (row, e.name) in scores]
        for s, row, name, sr, sc in [*cells, *(("", r, n, sr, f) for r, n, sr, f in floor_rows)]:
            edges = band_edges(sr)
            for (lo, hi), (b, bb) in zip(zip(edges[:-1], edges[1:]), sc["bands"]):
                w.writerow([s, row, name, f"{lo:.1f}", f"{min(hi, sr / 2):.1f}", f"{b:.3f}", f"{bb:.3f}"])
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(group(row), []).append(row)
    ordered = [g for g in ORDER if g in groups] + sorted(set(groups) - set(ORDER))
    with open(results / "repeats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "row", "excerpt", "runs", "metric", "mean", "min", "max"])
        for s, members in sets.items():
            for g in ordered:
                for e in members:
                    got = [scores[r, e.name] for r in groups[g] if (r, e.name) in scores]
                    if len(got) > 1:
                        for m, _ in METRICS:
                            v = [sc[m] for sc in got]
                            w.writerow([s, g, e.name, len(v), m, f"{np.mean(v):.3f}", f"{min(v):.3f}", f"{max(v):.3f}"])

    def runs(g: str, name: str, m: str) -> list[float]:
        return [scores[r, name][m] for r in groups[g] if (r, name) in scores]

    def cell(g: str, name: str, m: str) -> str:
        v = runs(g, name, m)
        if not v:
            return "–"
        if len(v) == 1:
            return f"{v[0]:.2f}"
        return f"{np.mean(v):.2f} ({min(v):.2f}–{max(v):.2f})"

    def bands_line(sr: int) -> str:
        edges = band_edges(sr)
        return (f"Bands at {sr} Hz ({len(edges) - 1}): "
                + ", ".join(f"{lo:.0f}–{min(hi, sr / 2):.0f}" for lo, hi in zip(edges[:-1], edges[1:])))

    lines = ["# Resynthesis fidelity: results", "",
             "Generated by `evaluate.py`. LSD in dB, lower is better; gain is the RMS level of the render re the "
             "excerpt. \"–\": the row has no render of that excerpt. Rows with several runs show the mean and "
             "(min–max). Every excerpt is scored at its own rate (STFT frame 46.4 ms, hop 11.6 ms).", "",
             "Noise-pair floor (two independent white-noise realisations, 10 s), bin / band LSD: "
             + "; ".join(f"{sr} Hz {f['bin']:.2f} / {f['band']:.2f} dB" for sr, f in floors.items()) + ".", ""]
    for s, members in sets.items():
        names = [e.name for e in members]
        present = [g for g in ordered if any(runs(g, n, "bin") for n in names)]
        lines += [f"## {TITLES[s]}", ""]
        if s != "icmc2000":
            for m, title in METRICS:
                lines += [f"**{title}**", "", "| row | " + " | ".join(names) + " |",
                          "| --- |" + " ---: |" * len(names)]
                for g in present:
                    n_runs = max(len(runs(g, n, m)) for n in names)  # PM2 inharmonic ran 5 times on `case` only
                    label = g + (f" ({n_runs} runs)" if n_runs > 1 else "")
                    lines.append(f"| {label} | " + " | ".join(cell(g, n, m) for n in names) + " |")
                lines.append("")
        else:
            hist = [g for g in present if historical(g)]
            common = [n for n in names if all(runs(g, n, "bin") for g in hist)]
            lines += [f"{len(names)} sounds. Per row: number of sounds it rendered, mean and median over those, and "
                      f"mean and median over the {len(common)} sounds that every historical row (`*-2000`) "
                      "rendered. Per-sound values are in `per_excerpt.csv`.", ""]
            for m, title in METRICS:
                lines += [f"**{title}**", "", f"| row | n | mean | median | mean ({len(common)}) | "
                          f"median ({len(common)}) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
                for g in present:
                    per = {n: float(np.mean(runs(g, n, m))) for n in names if runs(g, n, m)}
                    on = [per[n] for n in common if n in per]
                    agg = f"{np.mean(on):.2f} | {np.median(on):.2f}" if len(on) == len(common) else "– | –"
                    lines.append(f"| {g} | {len(per)} | {np.mean(list(per.values())):.2f} | "
                                 f"{np.median(list(per.values())):.2f} | {agg} |")
                lines.append("")
            short = [(g, n, d) for g in hist for r in groups[g] for n, d in metas[r]["excerpts"].items()
                     if n in names and d["render_s"] < COVER * d["input_s"]]
            if short:
                lines += [f"Historical renders shorter than {COVER:.0%} of their input (scored as rendered, the rest "
                          "zero-padded): " + "; ".join(f"{g} {n} {d['render_s']:.2f} of {d['input_s']:.2f} s"
                                                       for g, n, d in short) + ".", ""]
            quiet = [(g, n, scores[g, n]["gain"]) for g in hist for n in names
                     if (g, n) in scores and scores[g, n]["gain"] < QUIET_DB]
            if quiet:
                lines += [f"Historical renders more than {-QUIET_DB:g} dB below their input (scored as archived): "
                          + "; ".join(f"{g} {n} {v:.1f} dB" for g, n, v in quiet) + ".", ""]
        lines += [bands_line(sr) for sr in sorted({srs[n] for n in names})] + [""]
    (results / "summary.md").write_text("\n".join(lines))


def write_listening(out: pathlib.Path, known, rows, refs) -> None:
    """Per excerpt: the mono original and every render (first run of repeated rows), equal RMS, 16-bit."""
    raw = out / "raw"
    picked = [row for row in rows if group(row) == row or row.endswith("/r1")]
    picked.sort(key=lambda r: ORDER.index(group(r)) if group(r) in ORDER else len(ORDER))
    for e in known:
        if e.name not in refs:
            continue
        ref = refs[e.name]
        target = np.sqrt(np.mean(ref ** 2))
        clips = [("00_original", ref)]
        for i, row in enumerate(picked, start=1):
            wav = raw / row / f"{e.name}.wav"
            if wav.is_file():
                y, _ = sf.read(wav)
                y = y[: ref.size] if y.size >= ref.size else np.pad(y, (0, ref.size - y.size))
                clips.append((f"{i:02d}_{group(row).replace('/', '_')}", y * target / np.sqrt(np.mean(y ** 2))))
        peak = max(np.abs(c).max() for _, c in clips)
        scale = min(1.0, 10 ** (LISTEN_PEAK_DB / 20) / peak)  # one factor per excerpt keeps the levels equal
        folder = out / "listening" / e.name
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("[0-9][0-9]_*.wav"):  # numbering follows ORDER, which may have changed
            old.unlink()
        for stem, c in clips:
            sf.write(folder / f"{stem}.wav", c * scale, e.sr, subtype="PCM_16")
        print("listening", e.name, len(clips), "files, scale", round(20 * np.log10(scale), 2), "dB", flush=True)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
