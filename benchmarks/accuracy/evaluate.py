"""Score every row in OUT/raw against the analytic truth.

Usage: python evaluate.py OUT RESULTS   (numpy + pandas)
Writes per_signal.csv, per_family.csv, repeats.csv (mean/min/max over repeat runs r1, r2, ...) and summary.md.

Accuracy compares guarded points at their own times, loudest first within a frame, to the nearest
non-absent fixed partial identity. Tolerance is min(50 cents, half the nearest non-absent gap) at that time.
Points within a neutral partial's tolerance are excluded before the amplitude gate. Otherwise a match needs
an amplitude within GATE_DB of truth; unpaired points are duplicates or far points as before.
Detection uses a 10 ms grid. Tracks cover [first - hop/2, last + hop/2], linearly interpolated with held ends;
frame peaks cover their frame +- hop/2. Greedy closest-cents pairing includes present and neutral partials,
with tolerance and amplitude gates. Neutral pairs count as neither hits nor false points; absent partials
cannot pair. Recall counts present cells only; false points must reach -60 dBFS. File fades are not transitions.
Tracking uses only present pairs: distinct tracks per present segment, tracks that switch partial identity,
onset/offset errors from raw track endpoints, and identity continuity across dropouts. Transition zones
are neutral within 50 ms. Counts and samples pool across files before computing family statistics.
Meetings count tracks instead of the switch rate, which a merged peak inflates: a track is eligible if its raw
points start before and end after MEET_ZONE around the meeting and it has a present pair outside the zone, so false
tracks stay in false / frame alone. Those pairs vote on each side; a side needs MEET_MIN votes and one partial
strictly ahead. Same partial on both sides is right, different is a swap, anything else (a tie, too few votes) is
unscorable.
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
MEET_ZONE = 0.28  # seconds either side of the meeting; the partials are too close to tell apart inside
MEET_MIN = 3      # votes a side needs before its majority counts
# summary.md columns: metric -> (header, kind)
SUMMARY = {
    "points": ("points", "int"), "matched_frac": ("paired %", "pct"), "dup_frac": ("dup %", "pct"),
    "far_frac": ("far %", "pct"), "freq_abs_p50_c": ("\\|Δf\\| p50 (c)", "num"),
    "freq_abs_p95_c": ("\\|Δf\\| p95 (c)", "num"), "amp_bias_db": ("ΔA median (dB)", "num"),
    "amp_abs_p95_db": ("\\|ΔA\\| p95 (dB)", "num"), "recall": ("recall %", "pct"),
    "false_per_frame": ("false / frame", "num"),
}
SPREAD = ("matched_frac", "freq_abs_p95_c", "amp_bias_db", "recall", "false_per_frame")


# Appended CSV columns, in the specified order. Counts are NaN when inapplicable.
TRACK_METRICS = (
    "segments", "tracks_per_segment", "switch_tracks", "switch_frac",
    "onset_n", "onset_missed", "onset_med_ms", "onset_abs_p95_ms",
    "offset_n", "offset_missed", "offset_med_ms", "offset_abs_p95_ms",
    "bridge_n", "bridge_undet", "bridged_frac",
)
# Written after every older column, here and in repeats.csv, so existing columns keep their position.
COUNT_METRICS = ("bridged", "meet_eligible", "meet_right", "meet_swap", "meet_unscorable")
TRACK_SUMMARY = {
    "tracks_per_segment": ("tracks / segment", "num"), "switch_frac": ("switch %", "pct"),
    "onset_med_ms": ("onset med (ms)", "num"), "onset_abs_p95_ms": (r"onset \|err\| p95 (ms)", "num"),
    "onset_missed": ("onset missed", "int"), "offset_med_ms": ("offset med (ms)", "num"),
    "offset_abs_p95_ms": (r"offset \|err\| p95 (ms)", "num"), "offset_missed": ("offset missed", "int"),
    "bridged_frac": ("bridged %", "pct"), "bridge_undet": ("bridge undet.", "int"),
}


def signal_specs() -> dict:
    return {**truth.SIGNALS, **truth.DEV_SIGNALS, **truth.DEV_MOVING}


def accuracy(pts: np.ndarray, name: str) -> dict:
    t, f, a = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = (t >= truth.GUARD) & (t <= truth.DURATION - truth.GUARD) & (f > 0) & (a > 0)
    zero_amp = int((pts[:, 2] <= 0).sum())
    if not keep.any():
        return {"cents": np.zeros(0), "db": np.zeros(0), "n": 0, "dup": 0, "far": 0, "zero_amp": zero_amp}
    order = np.lexsort((-a[keep], t[keep]))
    t, f, a = t[keep][order], f[keep][order], a[keep][order]
    F, A = truth.partials(name, t)
    state = truth.states(name, t)
    tol = truth.tolerance_cents(F, state)
    cents = 1200 * np.log2(f[None, :] / F)
    cents[state == 0] = np.inf
    neutral = np.any((state == -1) & (np.abs(cents) <= tol), axis=0)
    col = np.arange(len(t))
    # tolerances never overlap (each <= half a gap), so the nearest partial is the only candidate
    k = np.argmin(np.abs(cents), axis=0)
    near = (np.abs(cents[k, col]) <= tol[k, col]) & ~neutral
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
    return {"cents": cents[k, col], "db": 20 * np.log10(a[paired] / A[k, col]), "n": int((~neutral).sum()),
            "dup": int((near & ~paired).sum()), "far": int((~near & ~neutral).sum()), "zero_amp": zero_amp}


def covering(pts: np.ndarray, grid: np.ndarray, hop: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(grid index, freq, amp, source track id) of every covering point; -1 for peaks."""
    gi, gf, ga, gt = [], [], [], []
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
            gt.append(np.full(hi - lo, int(tr[0, 3]), dtype=int))
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
            gt.append(np.full(len(rows), -1, dtype=int))
    if not gi:
        return np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0, int)
    return np.concatenate(gi), np.concatenate(gf), np.concatenate(ga), np.concatenate(gt)


def detection(pts: np.ndarray, name: str, hop: float) -> dict:
    grid = np.arange(truth.GUARD, truth.DURATION - truth.GUARD + 1e-9, GRID)
    F, A = truth.partials(name, grid)
    state = truth.states(name, grid)
    tol = truth.tolerance_cents(F, state)
    gi, gf, ga, gt = covering(pts, grid, hop)
    pairs = []
    paired = false = 0
    false_grid = np.zeros(len(grid), dtype=int)
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
                    if state[k, i] == 1:
                        paired += 1
                        pairs.append((i, k, gt[sel[m]]))
        false_grid[i] = int(np.sum(~used & (20 * np.log10(np.maximum(a, 1e-300)) >= FALSE_DB)))
        false += int(false_grid[i])
    return {"true": int((state == 1).sum()), "paired": paired, "false": false, "frames": len(grid),
            "false_grid": false_grid, "pairs": np.asarray(pairs, dtype=int).reshape(-1, 3), "grid": grid, "state": state}


def present_segments(state: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive grid bounds of maximal present runs for one fixed partial."""
    edges = np.diff(np.r_[False, state == 1, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def majority(partials: np.ndarray) -> int | None:
    """The partial strictly ahead among at least MEET_MIN votes, else None."""
    if len(partials) < MEET_MIN:
        return None
    votes = np.bincount(partials)
    top = np.flatnonzero(votes == votes.max())
    return int(top[0]) if len(top) == 1 else None


def meeting(pts: np.ndarray, det: dict) -> dict:
    """Eligible, right, swap and unscorable track counts over the meeting at DURATION / 2."""
    centre = truth.DURATION / 2
    grid_time = det["grid"][det["pairs"][:, 0]]
    outside = np.abs(grid_time - centre) > MEET_ZONE + 1e-9
    result = dict.fromkeys(COUNT_METRICS[1:], 0)
    for tid in np.unique(pts[pts[:, 3] >= 0, 3]):
        times = pts[pts[:, 3] == tid, 0]
        mine = outside & (det["pairs"][:, 2] == tid)
        if times.min() >= centre - MEET_ZONE or times.max() <= centre + MEET_ZONE or not mine.any():
            continue
        result["meet_eligible"] += 1
        before = majority(det["pairs"][mine & (grid_time < centre), 1])
        after = majority(det["pairs"][mine & (grid_time > centre), 1])
        verdict = "unscorable" if before is None or after is None else "right" if before == after else "swap"
        result["meet_" + verdict] += 1
    return result


def tracking(pts: np.ndarray, name: str, det: dict) -> dict:
    """Poolable tracking samples and transition totals, from present detection pairs only."""
    pairs, grid, state = det["pairs"], det["grid"], det["state"]
    segments = [present_segments(row) for row in state]
    sizes, switches = [], []
    tracked = pairs[pairs[:, 2] >= 0]
    for k, runs in enumerate(segments):
        for lo, hi in runs:
            ids = tracked[(tracked[:, 1] == k) & (tracked[:, 0] >= lo) & (tracked[:, 0] <= hi), 2]
            if len(ids):
                sizes.append(len(np.unique(ids)))
    for tid in np.unique(tracked[:, 2]):
        tr = tracked[tracked[:, 2] == tid]
        if len(np.unique(tr[:, 0])) >= 2:
            switches.append(len(np.unique(tr[:, 1])) > 1)
    raw_ends = {tid: (tr[:, 0].min(), tr[:, 0].max())
                for tid in np.unique(pts[pts[:, 3] >= 0, 3]) if len(tr := pts[pts[:, 3] == tid])}

    def edge_track(k, tau, after):
        # Select the adjacent present segment, even when it contains no pairs.
        runs = [run for run in segments[k] if (grid[run[0]] > tau if after else grid[run[1]] < tau)]
        if not runs:
            return None
        lo, hi = runs[0] if after else runs[-1]
        tr = tracked[(tracked[:, 1] == k) & (tracked[:, 0] >= lo) & (tracked[:, 0] <= hi)]
        if not len(tr):
            return None
        edge = tr[np.argmin(tr[:, 0]) if after else np.argmax(tr[:, 0])]
        if abs(grid[edge[0]] - grid[lo if after else hi]) > 0.1 + 1e-12:
            return None
        return int(edge[2])

    spec = signal_specs()[name]
    result = {"sizes": sizes, "switches": switches}
    for key, after in (("on", True), ("off", False)):
        times = spec.get(key, [])
        errors = []
        for k, tau in enumerate(times):
            if 0 < tau < truth.DURATION:
                tid = edge_track(k, tau, after)
                if tid is not None:
                    errors.append(raw_ends[tid][0 if after else 1] - tau)
        result[key + "_errors"] = errors
        result[key + "_total"] = sum(0 < tau < truth.DURATION for tau in times)
    bridges = []
    for k, (centre, width) in enumerate(spec.get("drop", [])):
        before = edge_track(k, centre - width / 2, False)
        after = edge_track(k, centre + width / 2, True)
        if before is not None and after is not None:
            bridges.append(before == after)
    result.update(bridges=bridges, drop_total=len(spec.get("drop", [])),
                  meeting=meeting(pts, det) if truth.meets(name) else None)
    return result


def track_stats(samples: list[dict] | None) -> dict:
    if samples is None:
        return dict.fromkeys(TRACK_METRICS + COUNT_METRICS, np.nan)
    sizes = [v for s in samples for v in s["sizes"]]
    switches = [v for s in samples for v in s["switches"]]
    out = {"segments": len(sizes), "tracks_per_segment": np.mean(sizes) if sizes else np.nan,
           "switch_tracks": len(switches), "switch_frac": np.mean(switches) if switches else np.nan}
    for key, label in (("on", "onset"), ("off", "offset")):
        total = sum(s[key + "_total"] for s in samples)
        errors = np.array([v for s in samples for v in s[key + "_errors"]]) * 1000
        out.update({label + "_n": len(errors) if total else np.nan,
                    label + "_missed": total - len(errors) if total else np.nan,
                    label + "_med_ms": q(errors, 50), label + "_abs_p95_ms": q(np.abs(errors), 95)})
    total = sum(s["drop_total"] for s in samples)
    bridges = [v for s in samples for v in s["bridges"]]
    out.update(bridge_n=len(bridges) if total else np.nan,
               bridge_undet=total - len(bridges) if total else np.nan,
               bridged_frac=np.mean(bridges) if bridges else np.nan,
               bridged=sum(bridges) if total else np.nan)
    meetings = [s["meeting"] for s in samples if s.get("meeting")]
    for m in COUNT_METRICS[1:]:
        out[m] = sum(v[m] for v in meetings) if meetings else np.nan
    return {m: out[m] for m in TRACK_METRICS + COUNT_METRICS}


def q(v: np.ndarray, p: float) -> float:
    return float(np.percentile(v, p)) if len(v) else np.nan


def stats(acc: list[dict], det: list[dict], tracks: list[dict] | None = None) -> dict[str, float]:
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
        **track_stats(tracks),
    }


def repeats(fam: pd.DataFrame) -> pd.DataFrame:
    """Mean, min and max of every metric over the repeat runs of a row, per family."""
    rep = fam[fam["row"].str.contains(REPEAT)].assign(row=lambda d: d["row"].str.replace(REPEAT, "", regex=True))
    groups = rep.groupby(["row", "family"], sort=False)

    def spread(metrics) -> pd.DataFrame:
        g = groups[list(metrics)]
        block = pd.concat({"mean": g.mean(), "min": g.min(), "max": g.max()}, axis=1)
        block.columns = [f"{m}_{s}" for s, m in block.columns]
        return block

    out = spread(SUMMARY).reset_index().assign(runs=groups.size().to_numpy())
    return pd.concat([out, *[spread(m).reset_index(drop=True) for m in (TRACK_METRICS, COUNT_METRICS)]], axis=1)


def cell(v: float, kind: str) -> str:
    if not np.isfinite(v):
        return "–"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "pct":
        return f"{100 * v:.1f}"
    return np.format_float_positional(v, precision=3, unique=False, fractional=False, trim="-")


def count_tables(fam: pd.DataFrame, sig: pd.DataFrame) -> list[str]:
    """Meetings and holes as counts: a percentage of 2 or 12 tracks says less than the tracks themselves."""
    meets = fam[fam["meet_eligible"].notna()]
    holes = sig[(sig["family"] == truth.HOLES) & sig["bridged"].notna()]
    if meets.empty and holes.empty:
        return []
    lines = ["## Meetings and holes", "", f"Counts, pooled over files. Meetings: tracks spanning +-{MEET_ZONE} s of "
             f"the meeting with a present pair outside it; identity by majority of at least {MEET_MIN} present pairs "
             "each side. Holes: bridged of determined (held across the hole by one track), undetermined holes left "
             "out. Repeat runs are listed one by one, as a mean of a few tracks is no longer a count; repeats.csv "
             "has their mean, min and max.", ""]
    if not meets.empty:
        lines += ["### Meetings", "", "| row | family | files | eligible | right | swap | unscorable |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for _, r in meets.sort_values(["family", "row"], kind="stable").iterrows():
            values = [cell(r[m], "int") for m in ("signals", *COUNT_METRICS[1:])]
            lines.append(f"| {r['row']} | {r['family']} | " + " | ".join(values) + " |")
        lines.append("")
    if not holes.empty:
        shapes = holes["signal"].map(truth.shape)
        names = list(dict.fromkeys(holes["signal"]))

        def ratio(rows: pd.DataFrame) -> str:
            return f"{rows['bridged'].sum():.0f}/{rows['bridge_n'].sum():.0f}" if len(rows) else "–"

        lines += ["### Holes in moving partials", "", "| row | all | undet. | vibrato | glide | " + " | ".join(names) + " |",
                  "|---|" + "---:|" * (4 + len(names))]
        for row, g in holes.groupby("row", sort=False):
            cells = [ratio(g), f"{g['bridge_undet'].sum():.0f}", ratio(g[shapes[g.index] == "vibrato"]),
                     ratio(g[shapes[g.index] == "glide"]), *[ratio(g[g["signal"] == n]) for n in names]]
            lines.append(f"| {row} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def summary(fam: pd.DataFrame, rep: pd.DataFrame, sig: pd.DataFrame) -> str:
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
    # Keep the entire accuracy and spread prefix byte-identical for cached rows.
    lines += ["", "## Tracking", ""]
    metrics = list(TRACK_METRICS)
    means = rep[["row", "family", "runs"] + [f"{m}_mean" for m in metrics]]
    means = means.rename(columns={f"{m}_mean": m for m in metrics})
    tracks = pd.concat([fam[~fam["row"].str.contains(REPEAT)],
                        means.assign(row=means["row"] + " (mean of " + means["runs"].astype(str) + ")")])
    for family in dict.fromkeys(fam["family"]):
        rows = tracks[(tracks["family"] == family) & tracks["segments"].notna()]
        if rows.empty:
            continue
        cols = {m: spec for m, spec in TRACK_SUMMARY.items()
                if any(cell(v, spec[1]) != "–" for v in rows[m])}
        lines += [f"### {family}", "", "| " + " | ".join(["row", *[h for h, _ in cols.values()]]) + " |",
                  "|---|" + "---:|" * len(cols)]
        for _, r in rows.sort_values("row", kind="stable").iterrows():
            values = [r["row"], *[cell(r[m], k) for m, (_, k) in cols.items()]]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines += count_tables(fam, sig)
    return "\n".join(lines) + "\n"


def main(out: pathlib.Path, results: pathlib.Path) -> None:
    per_signal, per_family = [], []
    for meta_file in sorted((out / "raw").rglob("meta.json")):
        folder = meta_file.parent
        row = folder.relative_to(out / "raw").as_posix()
        meta = json.loads(meta_file.read_text())
        names = meta.get("signals", [n for n in truth.SIGNALS if (folder / f"{n}.csv").exists()])
        specs = signal_specs()
        tracked = meta.get("kind") == "tracks"
        track_samples = {}
        scored = {}
        for name in names:
            pts = points.read_points(folder, name)
            scored[name] = (accuracy(pts, name), detection(pts, name, meta["hop_s"]))
            if tracked:
                track_samples[name] = tracking(pts, name, scored[name][1])
            per_signal.append({"row": row, "signal": name, "family": specs[name]["family"],
                               **stats([scored[name][0]], [scored[name][1]], [track_samples[name]] if tracked else None)})
        for family in dict.fromkeys(specs[n]["family"] for n in names):
            sel = [scored[n] for n in names if specs[n]["family"] == family]
            per_family.append({"row": row, "family": family, "signals": len(sel),
                               **stats([s[0] for s in sel], [s[1] for s in sel],
                                       [track_samples[n] for n in names if specs[n]["family"] == family]
                                       if tracked else None)})
        print(row, "scored", flush=True)
    results.mkdir(parents=True, exist_ok=True)
    sig = pd.DataFrame(per_signal)
    sig.to_csv(results / "per_signal.csv", index=False, float_format="%.4g")
    fam = pd.DataFrame(per_family)
    fam.to_csv(results / "per_family.csv", index=False, float_format="%.4g")
    rep = repeats(fam)
    rep.to_csv(results / "repeats.csv", index=False, float_format="%.4g")
    (results / "summary.md").write_text(summary(fam, rep, sig))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
