"""Score every row in OUT/raw against the analytic truth.

Usage: python evaluate.py OUT RESULTS   (numpy + pandas)
Writes per_signal.csv, per_family.csv, repeats.csv (mean/min/max over repeat runs r1, r2, ...) and summary.md.

Accuracy: every found point inside the guard is compared with the true partials at its own time. Points that
share a time (one frame) pair one-to-one, loudest first: each takes the nearest true partial in cents if it lies
within min(50 c, half the gap to a neighbouring partial) and no louder point of the frame took it. Frequency
error in cents, amplitude error in dB re the true D4 amplitude. A point pairs only if its amplitude is at most
20 dB below the true one (GATE_DB), so the side lobe of a missed partial is not a hit. Unpaired points are
extra: "dup" when a louder point of the frame took the partial, "far" otherwise (no partial within tolerance,
or the point is gated out).
Detection, on a 10 ms grid: a track covers [first - hop/2, last + hop/2] and is interpolated linearly inside
(values held at its ends); an untracked frame peak is a one-point track, so it covers its frame +- hop/2.
At each grid time true and covering points pair one-to-one, closest in cents first, within the tolerance
and the amplitude gate.
Missed = unpaired true partials; false = unpaired points at or above -60 dBFS.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import points  # noqa: E402
import truth  # noqa: E402

GRID = 0.01
FALSE_DB = -60.0
GATE_DB: float | None = -20.0  # a point pairs only if at most this far below the true amp
REPEAT = r"/r\d+$"
# summary.md columns: metric -> (header, kind)
SUMMARY = {
    "points": ("points", "int"), "matched_frac": ("paired %", "pct"), "dup_frac": ("dup %", "pct"),
    "far_frac": ("far %", "pct"), "freq_abs_p50_c": ("\\|Δf\\| p50 (c)", "num"),
    "freq_abs_p95_c": ("\\|Δf\\| p95 (c)", "num"), "amp_bias_db": ("ΔA median (dB)", "num"),
    "amp_abs_p95_db": ("\\|ΔA\\| p95 (dB)", "num"), "recall": ("recall %", "pct"),
    "false_per_frame": ("false / frame", "num"),
}
SPREAD = ("matched_frac", "freq_abs_p95_c", "amp_bias_db", "recall", "false_per_frame")


def accuracy(pts: np.ndarray, name: str) -> dict:
    t, f, a = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = (t >= truth.GUARD) & (t <= truth.DURATION - truth.GUARD) & (f > 0) & (a > 0)
    zero_amp = int((pts[:, 2] <= 0).sum())
    if not keep.any():
        return {"cents": np.zeros(0), "db": np.zeros(0), "n": 0, "dup": 0, "far": 0, "zero_amp": zero_amp}
    order = np.lexsort((-a[keep], t[keep]))
    t, f, a = t[keep][order], f[keep][order], a[keep][order]
    F, A = truth.partials(name, t)
    cents = 1200 * np.log2(f[None, :] / F)
    col = np.arange(len(t))
    # tolerances never overlap (each <= half a gap), so the nearest partial is the only candidate
    k = np.argmin(np.abs(cents), axis=0)
    near = np.abs(cents[k, col]) <= truth.tolerance_cents(F)[k, col]
    if GATE_DB is not None:
        near &= a >= A[k, col] * 10 ** (GATE_DB / 20)
    _, first, group = np.unique(t, return_index=True, return_inverse=True)
    rank = col - first[group]
    taken = np.zeros((len(first), F.shape[0]), bool)
    paired = np.zeros(len(t), bool)
    for r in range(int(rank.max()) + 1):
        sel = np.flatnonzero((rank == r) & near)
        sel = sel[~taken[group[sel], k[sel]]]
        taken[group[sel], k[sel]] = True
        paired[sel] = True
    k, col = k[paired], col[paired]
    return {"cents": cents[k, col], "db": 20 * np.log10(a[paired] / A[k, col]), "n": len(t),
            "dup": int((near & ~paired).sum()), "far": int((~near).sum()), "zero_amp": zero_amp}


def covering(pts: np.ndarray, grid: np.ndarray, hop: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grid index, freq, amp) of every point covering a grid time."""
    gi, gf, ga = [], [], []
    tracked = pts[pts[:, 3] >= 0]
    for _, idx in pd.Series(np.arange(len(tracked))).groupby(tracked[:, 3]):
        tr = tracked[idx.to_numpy()]
        tr = tr[np.argsort(tr[:, 0], kind="stable")]
        lo = np.searchsorted(grid, tr[0, 0] - hop / 2, "left")
        hi = np.searchsorted(grid, tr[-1, 0] + hop / 2, "right")
        if hi > lo:
            g = grid[lo:hi]
            gi.append(np.arange(lo, hi))
            gf.append(np.interp(g, tr[:, 0], tr[:, 1]))
            ga.append(np.interp(g, tr[:, 0], tr[:, 2]))
    peaks = pts[pts[:, 3] < 0]
    if len(peaks):
        frames = np.unique(peaks[:, 0])
        j = np.clip(np.searchsorted(frames, grid), 1, len(frames) - 1)
        j = np.where(np.abs(frames[j - 1] - grid) <= np.abs(frames[j] - grid), j - 1, j)
        near = np.abs(frames[j] - grid) <= hop / 2 + 1e-12
        order = np.argsort(peaks[:, 0], kind="stable")
        starts = np.searchsorted(peaks[order, 0], frames, "left")
        stops = np.searchsorted(peaks[order, 0], frames, "right")
        for i in np.flatnonzero(near):
            rows = peaks[order[starts[j[i]]:stops[j[i]]]]
            gi.append(np.full(len(rows), i))
            gf.append(rows[:, 1])
            ga.append(rows[:, 2])
    if not gi:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    return np.concatenate(gi), np.concatenate(gf), np.concatenate(ga)


def detection(pts: np.ndarray, name: str, hop: float) -> dict[str, int]:
    grid = np.arange(truth.GUARD, truth.DURATION - truth.GUARD + 1e-9, GRID)
    F, A = truth.partials(name, grid)
    tol = truth.tolerance_cents(F)
    gi, gf, ga = covering(pts, grid, hop)
    paired = false = 0
    order = np.argsort(gi, kind="stable")
    bounds = np.searchsorted(gi[order], np.arange(len(grid) + 1))
    for i in range(len(grid)):
        sel = order[bounds[i]:bounds[i + 1]]
        f, a = gf[sel], ga[sel]
        used = np.zeros(len(f), bool)
        if len(f):
            ok = f > 0
            cents = np.full((F.shape[0], len(f)), np.inf)
            cents[:, ok] = np.abs(1200 * np.log2(f[None, ok] / F[:, i:i + 1]))
            if GATE_DB is not None:
                cents[a[None, :] < A[:, i:i + 1] * 10 ** (GATE_DB / 20)] = np.inf
            cand = np.argwhere(cents <= tol[:, i:i + 1])
            free = np.ones(F.shape[0], bool)
            for k, m in cand[np.argsort(cents[cand[:, 0], cand[:, 1]], kind="stable")]:
                if free[k] and not used[m]:
                    free[k] = False
                    used[m] = True
                    paired += 1
        false += int(np.sum(~used & (20 * np.log10(np.maximum(a, 1e-300)) >= FALSE_DB)))
    return {"true": F.shape[0] * len(grid), "paired": paired, "false": false, "frames": len(grid)}


def q(v: np.ndarray, p: float) -> float:
    return float(np.percentile(v, p)) if len(v) else np.nan


def stats(acc: list[dict], det: list[dict]) -> dict[str, float]:
    c = np.concatenate([x["cents"] for x in acc])
    d = np.concatenate([x["db"] for x in acc])
    n = sum(x["n"] for x in acc)
    return {
        "points": n, "matched_frac": len(c) / n if n else np.nan,
        "dup_frac": sum(x["dup"] for x in acc) / n if n else np.nan,
        "far_frac": sum(x["far"] for x in acc) / n if n else np.nan,
        "freq_bias_c": q(c, 50), "freq_abs_p50_c": q(np.abs(c), 50), "freq_abs_p95_c": q(np.abs(c), 95),
        "amp_bias_db": q(d, 50), "amp_abs_p95_db": q(np.abs(d), 95),
        "recall": sum(x["paired"] for x in det) / sum(x["true"] for x in det),
        "false_per_frame": sum(x["false"] for x in det) / sum(x["frames"] for x in det),
        "zero_amp_points": sum(x["zero_amp"] for x in acc),
    }


def repeats(fam: pd.DataFrame) -> pd.DataFrame:
    """Mean, min and max of every metric over the repeat runs of a row, per family."""
    rep = fam[fam["row"].str.contains(REPEAT)].assign(row=lambda d: d["row"].str.replace(REPEAT, "", regex=True))
    g = rep.groupby(["row", "family"], sort=False)[list(SUMMARY)]
    out = pd.concat({"mean": g.mean(), "min": g.min(), "max": g.max()}, axis=1)
    out.columns = [f"{m}_{s}" for s, m in out.columns]
    return out.reset_index().assign(runs=g.size().to_numpy())


def cell(v: float, kind: str) -> str:
    if not np.isfinite(v):
        return "–"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "pct":
        return f"{100 * v:.1f}"
    return np.format_float_positional(v, precision=3, unique=False, fractional=False, trim="-")


def summary(fam: pd.DataFrame, rep: pd.DataFrame) -> str:
    means = rep[["row", "family", "runs"] + [f"{m}_mean" for m in SUMMARY]]
    means = means.rename(columns={f"{m}_mean": m for m in SUMMARY})
    table = pd.concat([fam[~fam["row"].str.contains(REPEAT)],
                       means.assign(row=means["row"] + " (mean of " + means["runs"].astype(str) + ")")])
    lines = ["# Accuracy benchmark results", "", "Generated by `evaluate.py`; see README.md for the rules and "
             "parameters. Rows with repeat runs show the mean over the runs; their spread follows at the end.", ""]
    head = "| row | " + " | ".join(h for h, _ in SUMMARY.values()) + " |"
    rule = "|---|" + "---:|" * len(SUMMARY)
    for family in dict.fromkeys(fam["family"]):
        lines += [f"## {family}", "", head, rule]
        for _, r in table[table["family"] == family].sort_values("row", kind="stable").iterrows():
            lines.append(f"| {r['row']} | " + " | ".join(cell(r[m], k) for m, (_, k) in SUMMARY.items()) + " |")
        lines.append("")
    lines += ["## Spread over repeat runs (min – max)", "",
              "| row | family | runs | " + " | ".join(SUMMARY[m][0] for m in SPREAD) + " |",
              "|---|---|---:|" + "---:|" * len(SPREAD)]
    for _, r in rep.iterrows():
        spread = [f"{cell(r[m + '_min'], SUMMARY[m][1])} – {cell(r[m + '_max'], SUMMARY[m][1])}" for m in SPREAD]
        lines.append(f"| {r['row']} | {r['family']} | {r['runs']} | " + " | ".join(spread) + " |")
    return "\n".join(lines) + "\n"


def main(out: pathlib.Path, results: pathlib.Path) -> None:
    per_signal, per_family = [], []
    for meta_file in sorted((out / "raw").rglob("meta.json")):
        folder = meta_file.parent
        row = folder.relative_to(out / "raw").as_posix()
        meta = json.loads(meta_file.read_text())
        names = meta.get("signals", list(truth.SIGNALS))
        scored = {}
        for name in names:
            pts = points.read_points(folder, name)
            scored[name] = (accuracy(pts, name), detection(pts, name, meta["hop_s"]))
            per_signal.append({"row": row, "signal": name, "family": truth.SIGNALS[name]["family"],
                               **stats([scored[name][0]], [scored[name][1]])})
        for family in dict.fromkeys(truth.SIGNALS[n]["family"] for n in names):
            sel = [scored[n] for n in names if truth.SIGNALS[n]["family"] == family]
            per_family.append({"row": row, "family": family, "signals": len(sel),
                               **stats([s[0] for s in sel], [s[1] for s in sel])})
        print(row, "scored", flush=True)
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_signal).to_csv(results / "per_signal.csv", index=False, float_format="%.4g")
    fam = pd.DataFrame(per_family)
    fam.to_csv(results / "per_family.csv", index=False, float_format="%.4g")
    rep = repeats(fam)
    rep.to_csv(results / "repeats.csv", index=False, float_format="%.4g")
    (results / "summary.md").write_text(summary(fam, rep))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
