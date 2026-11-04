"""Tune only on OUT/dev/suite; never use the published suite.

Usage: python tune_tracker.py OUT RESULTS [--max-cents C]
Fixed analysis and the parameter grid are independent of track_partials defaults.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evaluate  # noqa: E402
import truth  # noqa: E402

import audiospylt as asp  # noqa: E402

FRAME_LENGTH = 2048
HOP = 256
ZERO_PAD = 4
WINDOW = "hann"
CENTER = True
MIN_DB = -80.0
MAX_CENTS = 50.0
PROMINENCES = (None, 10, 15, 20, 25, 30)
DURATIONS = (0.02, 0.05, 0.1)
GAPS = (0, 1, 2, 4)
DEFAULT = (20, 0.1, 0)


def filter_span(tracks: pd.DataFrame, duration: float) -> pd.DataFrame:
    """Keep measured spans >= duration; IDs need not be renumbered for scoring."""
    spans = tracks.groupby("track_id")["time"].agg(["min", "max"])
    ids = spans.index[spans["max"] - spans["min"] >= duration]
    return tracks[tracks["track_id"].isin(ids)]


def pareto(macro: pd.DataFrame) -> pd.DataFrame:
    """Nondominated recall (max), false/frame (min), tracks/segment (min); retain ties."""
    values = macro[["recall", "false_per_frame", "tracks_per_segment"]].to_numpy() * [-1, 1, 1]
    keep = np.isfinite(values).all(axis=1)
    for i, value in enumerate(values):
        if keep[i] and np.any(np.all(values <= value, axis=1) & np.any(values < value, axis=1)):
            keep[i] = False
    return macro[keep].sort_values("recall", ascending=False, kind="stable")


def report(macro: pd.DataFrame, max_cents: float = MAX_CENTS) -> str:
    front = pareto(macro)
    cols = {
        "prominence_db": ("prominence", "num"), "min_duration": ("min_duration", "num"),
        "max_gap": ("max_gap", "int"), "recall": ("recall %", "pct"),
        "false_per_frame": ("false / frame", "num"), "tracks_per_segment": ("tracks / segment", "num"),
        "switch_frac": ("switch %", "pct"), "bridged_frac": ("bridged %", "pct"),
        "onset_abs_p95_ms": ("onset \\|err\\| p95 (ms)", "num"),
        "offset_abs_p95_ms": ("offset \\|err\\| p95 (ms)", "num"),
        "freq_abs_p95_c": ("\\|Δf\\| p95 (c)", "num"),
    }

    def line(row):
        return "| " + " | ".join("None" if m == "prominence_db" and pd.isna(row[m])
                                 else evaluate.cell(row[m], kind) for m, (_, kind) in cols.items()) + " |"

    header = "| " + " | ".join(h for h, _ in cols.values()) + " |"
    rule = "|" + "---:|" * len(cols)
    lines = ["# Tracker tuning (development suite only)", "",
             f"Linking used `max_cents={max_cents:g}`. "
             "Macro means give each family equal weight, skipping NaN metrics. "
             "The Pareto set maximizes recall and minimizes false / frame and tracks / segment; ties remain.", "",
             header, rule]
    lines.extend(line(row) for _, row in front.iterrows())
    if max_cents == MAX_CENTS:
        chosen = ((macro["prominence_db"] == DEFAULT[0]) & (macro["min_duration"] == DEFAULT[1])
                  & (macro["max_gap"] == DEFAULT[2]))
        if not macro.index[chosen].isin(front.index).any():
            lines += ["", "Default (outside the Pareto set):", "", header, rule, line(macro[chosen].iloc[0])]
    return "\n".join(lines) + "\n"


def main(out: pathlib.Path, results: pathlib.Path, max_cents: float = MAX_CENTS) -> None:
    started = time.perf_counter()
    folder = out / "dev" / "suite"
    if not folder.is_dir():
        raise SystemExit(f"Missing {folder}; run suite.py OUT --dev first (OUT={out}).")
    grid = list(itertools.product(PROMINENCES, DURATIONS, GAPS))
    scored = {key: {} for key in grid}
    for name in truth.DEV_SIGNALS:
        x, sr = sf.read(folder / f"{name}.wav")
        if sr != truth.SR:
            raise SystemExit(f"{name}: expected {truth.SR} Hz, got {sr}")
        spec = asp.spectrogram(x, sr, frame_length=FRAME_LENGTH, hop=HOP, zero_pad=ZERO_PAD,
                               window=WINDOW, center=CENTER)
        for prominence in PROMINENCES:
            peaks = asp.spectrogram_peaks(spec, min_db=MIN_DB, prominence_db=prominence)
            for gap in GAPS:
                linked = asp.link_peaks(peaks, max_cents=max_cents, max_gap=gap, min_duration=0)
                for duration in DURATIONS:
                    tracks = filter_span(linked, duration)
                    pts = tracks[["time", "freq", "amp", "track_id"]].to_numpy()
                    acc = evaluate.accuracy(pts, name)
                    det = evaluate.detection(pts, name, HOP / truth.SR)
                    scored[prominence, duration, gap][name] = (acc, det, evaluate.tracking(pts, name, det))
            print(f"{name}: prominence={prominence} done ({time.perf_counter() - started:.1f} s)", flush=True)
    rows = []
    families = tuple(dict.fromkeys(s["family"] for s in truth.DEV_SIGNALS.values()))
    for key in grid:
        params = dict(zip(("prominence_db", "min_duration", "max_gap"), key))
        pooled = []
        for family in families:
            sel = [scored[key][n] for n, s in truth.DEV_SIGNALS.items() if s["family"] == family]
            metrics = {"signals": len(sel), **evaluate.stats([s[0] for s in sel], [s[1] for s in sel],
                                                            [s[2] for s in sel])}
            pooled.append(metrics)
            rows.append({**params, "family": family, **metrics})
        rows.append({**params, "family": "macro", **pd.DataFrame(pooled).mean().to_dict()})
    results.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(results / "tuning.csv", index=False, float_format="%.8g")
    (results / "tuning.md").write_text(report(table[table["family"] == "macro"], max_cents=max_cents))
    print(f"Total: {time.perf_counter() - started:.2f} s", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="python tune_tracker.py OUT RESULTS [--max-cents C]")
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--max-cents", type=float, default=MAX_CENTS, dest="max_cents")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args.out, args.results, max_cents=args.max_cents)
