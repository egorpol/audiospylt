"""Tracking benchmark truth and scores, using analytic hand-built points, not analysis audio."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import audiospylt as asp

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "accuracy"

# Frozen original specifications: edits to the published truth must not rewrite this baseline.
ORIGINAL_SPECS = {
    "stat_harm": {"family": "stationary", "kind": "f0", "f0": ("const", 220.0), "count": 12, "base": 0.25},
    "stat_inharm": {"family": "stationary", "kind": "fixed", "freqs": (223.37, 457.81, 911.29, 1843.43, 3299.61),
                    "amps": (0.12, 0.10, 0.08, 0.06, 0.04)},
    "gliss_600": {"family": "glissando", "kind": "f0", "f0": ("exp", 220.0, 440.0), "count": 6, "base": 0.25},
    "gliss_1200": {"family": "glissando", "kind": "f0", "f0": ("exp", 220.0, 880.0), "count": 6, "base": 0.25},
    "vib_50": {"family": "vibrato", "kind": "f0", "f0": ("vib", 440.0, 50.0, 5.5), "count": 8, "base": 0.25},
    "vib_100": {"family": "vibrato", "kind": "f0", "f0": ("vib", 440.0, 100.0, 5.5), "count": 8, "base": 0.25},
    "pair_5": {"family": "close pair", "kind": "fixed", "freqs": (440.0, 445.0), "amps": (0.4, 0.4)},
    "pair_10": {"family": "close pair", "kind": "fixed", "freqs": (440.0, 450.0), "amps": (0.4, 0.4)},
    "pair_20": {"family": "close pair", "kind": "fixed", "freqs": (440.0, 460.0), "amps": (0.4, 0.4)},
    "pair_40": {"family": "close pair", "kind": "fixed", "freqs": (440.0, 480.0), "amps": (0.4, 0.4)},
    "noise_20": {"family": "noise", "kind": "f0", "f0": ("const", 220.0), "count": 12, "base": 0.1, "snr_db": 20.0},
    "noise_10": {"family": "noise", "kind": "f0", "f0": ("const", 220.0), "count": 12, "base": 0.1, "snr_db": 10.0},
    "noise_0": {"family": "noise", "kind": "f0", "f0": ("const", 220.0), "count": 12, "base": 0.1, "snr_db": 0.0},
}
PUBLISHED = tuple(ORIGINAL_SPECS) + ("cross_fast", "cross_slow", "stagger_harm", "stagger_inharm",
                                      "drop_30", "drop_80", "trem_20", "trem_40")
DEV = ("dev_stat_harm", "dev_stat_inharm", "dev_gliss_900", "dev_gliss_down", "dev_vib_75", "dev_vib_150",
       "dev_pair_15", "dev_pair_30", "dev_noise_15", "dev_noise_5", "dev_cross", "dev_stagger",
       "dev_drop_40", "dev_drop_60", "dev_trem_30")


@pytest.fixture
def bench():
    """Avoid clashes with fidelity's evaluate/excerpts and restore all affected import state."""
    names = ("truth", "points", "evaluate", "tracking_evaluate", "tracking_tuner")
    saved_path = list(sys.path)
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    try:
        modules = []
        for filename, name in (("evaluate.py", "tracking_evaluate"), ("tune_tracker.py", "tracking_tuner")):
            spec = importlib.util.spec_from_file_location(name, BENCH / filename)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            modules.append(mod)
            if filename == "evaluate.py":
                sys.modules["evaluate"] = mod
        yield modules[0], modules[0].truth, modules[1]
    finally:
        sys.path[:] = saved_path
        for n in names:
            sys.modules.pop(n, None)
        for n, prev in saved.items():
            if prev is not None:
                sys.modules[n] = prev


def perfect(truth, name):
    t = np.arange(int(truth.SR * truth.DURATION / 256) + 1) * 256 / truth.SR
    f, a = truth.partials(name, t)
    gate = truth.gates(name, t)
    rows = []
    tid = 0
    for k in range(len(f)):
        for lo, hi in _runs(gate[k] > 0.5):
            sel = slice(lo, hi + 1)
            rows.append(np.column_stack([t[sel], f[k, sel], a[k, sel], np.full(hi - lo + 1, tid)]))
            tid += 1
    return np.vstack(rows)


def _runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1)


def scored(e, pts, name, tracks=True):
    acc = e.accuracy(pts, name)
    det = e.detection(pts, name, 256 / e.truth.SR)
    return e.stats([acc], [det], [e.tracking(pts, name, det)] if tracks else None), det


def test_specs_and_seeds(bench):
    _, tr, _ = bench
    assert tr.ORIGINAL == tuple(ORIGINAL_SPECS)
    assert {n: tr.SIGNALS[n] for n in tr.ORIGINAL} == ORIGINAL_SPECS
    assert tuple(tr.SIGNALS) == PUBLISHED
    assert tuple(tr.DEV_SIGNALS) == DEV
    assert all(n.startswith("dev_") for n in DEV)
    for names, base in ((tr.ORIGINAL, 0), (PUBLISHED[13:], 100), (DEV, 1000)):
        assert {n: tr.seed(n) for n in names} == {n: base + i for i, n in enumerate(sorted(names))}
    json.dumps(tr.SIGNALS)
    json.dumps(tr.DEV_SIGNALS)


def test_state_zones(bench):
    _, tr, _ = bench
    np.testing.assert_array_equal(tr.states("stagger_harm", np.array([0.14, 0.18, 0.2, 0.24, 0.26]))[0],
                                  [0, -1, -1, -1, 1])
    np.testing.assert_array_equal(tr.states("drop_80", np.array([0.25, 0.29, 0.35, 0.41, 0.45]))[0],
                                  [1, -1, -1, -1, 1])
    # The long dropout has an absent middle only if neutral zones are shorter; here they overlap.
    assert np.all(tr.states("trem_40", np.array([1 / 12])) == 1)
    _, amp = tr.partials("trem_40", np.array([0.0, 1 / 12]))
    assert amp[0, 1] > 0


@pytest.mark.parametrize("name", PUBLISHED + DEV)
def test_truth_grid(bench, name):
    _, tr, _ = bench
    t = np.linspace(0, tr.DURATION, 4001)
    state, gate = tr.states(name, t), tr.gates(name, t)
    assert state.dtype == np.int8
    assert np.all((gate[state != -1] == 0) | (gate[state != -1] == 1))
    for times in tr.transitions(name):
        for tau in times:
            assert tr.GUARD + tr.NEUTRAL - 1e-12 <= tau <= tr.DURATION - tr.GUARD - tr.NEUTRAL + 1e-12
    f, _ = tr.partials(name, t)
    tol = tr.tolerance_cents(f, state)
    assert np.isnan(tol[state == 0]).all()
    cents = 1200 * np.log2(f)
    for i in range(len(f)):
        for j in range(i):
            keep = (state[i] != 0) & (state[j] != 0)
            assert np.all(np.abs(cents[i, keep] - cents[j, keep]) + 1e-9 >= tol[i, keep] + tol[j, keep])
    if name in tr.ORIGINAL:
        gap = np.full_like(cents, np.inf)
        d = np.diff(cents, axis=0)
        gap[:-1] = np.minimum(gap[:-1], d)
        gap[1:] = np.minimum(gap[1:], d)
        np.testing.assert_array_equal(tr.tolerance_cents(f), np.minimum(50.0, gap / 2))
    assert np.max(np.abs(tr.render(name))) <= 0.9


@pytest.mark.parametrize("name", PUBLISHED)
def test_perfect_tracks(bench, name):
    e, tr, _ = bench
    pts = perfect(tr, name)
    result, det = scored(e, pts, name)
    assert result["tracks_per_segment"] == 1
    assert result["switch_frac"] == 0
    if name.startswith("cross_"):
        hit = np.zeros_like(det["state"], dtype=bool)
        hit[det["pairs"][:, 1], det["pairs"][:, 0]] = True
        missed = np.any((det["state"] == 1) & ~hit, axis=0)
        assert np.all(np.abs(det["grid"][missed | (det["false_grid"] > 0)] - 1) <= 0.01 + 1e-12)
    else:
        assert result["recall"] == 1
        assert result["false_per_frame"] == 0
    if name.startswith("drop_"):
        assert result["bridge_n"] == 6
        assert result["bridged_frac"] == 0


def test_swaps_and_fragmentation(bench):
    e, tr, _ = bench
    pts = perfect(tr, "cross_fast")
    sel = pts[:, 0] > 1
    pts[sel, 3] = 1 - pts[sel, 3]
    result, _ = scored(e, pts, "cross_fast")
    assert result["switch_tracks"] == 2
    assert result["switch_frac"] == 1
    pts = perfect(tr, "stat_harm")
    pts = pts[pts[:, 3] == 0]
    pts[pts[:, 0] > 1, 3] = 1
    result, _ = scored(e, pts, "stat_harm")
    assert result["segments"] == 1
    assert result["tracks_per_segment"] == 2


@pytest.mark.parametrize("bridged", [False, True])
def test_dropout_identity(bench, bridged):
    e, tr, _ = bench
    pts = perfect(tr, "drop_80")
    if bridged:
        pts[:, 3] = pts[:, 3] // 2
    result, _ = scored(e, pts, "drop_80")
    assert result["bridge_n"] == 6
    assert result["bridge_undet"] == 0
    assert result["bridged_frac"] == int(bridged)


def test_onset_offset_and_missed(bench):
    e, _, _ = bench
    t = np.r_[0.188, np.arange(0.26, 1.74, 0.01), 1.812]
    pts = np.column_stack([t, np.full(len(t), 220), np.full(len(t), 0.25), np.zeros(len(t))])
    result, _ = scored(e, pts, "stagger_harm")
    assert result["onset_n"] == result["offset_n"] == 1
    assert result["onset_med_ms"] == pytest.approx(-12)
    assert result["offset_med_ms"] == pytest.approx(12)
    result, _ = scored(e, pts[pts[:, 0] >= 0.37], "stagger_harm")
    assert result["onset_n"] == 0
    assert result["onset_missed"] == 6
    assert np.isnan(result["onset_med_ms"])
    result, _ = scored(e, pts[pts[:, 0] <= 1.63], "stagger_harm")
    assert result["offset_n"] == 0
    assert result["offset_missed"] == 6


def test_frame_neutral_and_absent(bench):
    e, tr, _ = bench
    result, _ = scored(e, perfect(tr, "stat_harm"), "stat_harm", tracks=False)
    assert all(np.isnan(result[m]) for m in e.TRACK_METRICS)
    for amp in (0.25, 1e-8):  # neutrality ignores the amplitude gate for accuracy
        pts = np.array([[0.2, 220, amp, -1]])
        acc = e.accuracy(pts, "stagger_harm")
        assert acc["n"] == acc["dup"] == acc["far"] == len(acc["cents"]) == 0
        det = e.detection(pts, "stagger_harm", 0.001)
        assert det["paired"] == det["false"] == 0
    pts = np.array([[0.1, 220, 0.25, -1]])
    acc = e.accuracy(pts, "stagger_harm")
    det = e.detection(pts, "stagger_harm", 0.001)
    assert acc["far"] == det["false"] == 1
    assert acc["n"] == 1
    assert det["paired"] == 0
    assert det["true"] == np.sum(det["state"] == 1)


def test_absent_tolerance_and_covering(bench):
    e, tr, _ = bench
    freq = np.array([[440.0], [441.0], [880.0]])
    state = np.array([[1], [0], [-1]], dtype=np.int8)
    np.testing.assert_array_equal(tr.tolerance_cents(freq, state)[[0, 2]], [[50], [50]])
    assert np.isnan(tr.tolerance_cents(freq, state)[1, 0])
    pts = np.array([[0.2, 220, 0.1, 7], [0.3, 230, 0.1, 7], [0.25, 440, 0.2, -1]])
    gi, _, _, ids = e.covering(pts, np.array([0.25]), 0.01)
    np.testing.assert_array_equal(gi, [0, 0])
    np.testing.assert_array_equal(ids, [7, -1])


def test_row_selection(bench, tmp_path):
    e, tr, _ = bench
    folder = tmp_path / "raw" / "test" / "frames"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"kind": "frames", "hop_s": 256 / tr.SR}))
    for name in ("stat_harm", "cross_slow"):
        pts = perfect(tr, name)
        e.points.write_points(folder, name, pts[:, 0], pts[:, 1], pts[:, 2], -np.ones(len(pts)))
    e.main(tmp_path, tmp_path / "results")
    result = pd.read_csv(tmp_path / "results" / "per_signal.csv")
    assert list(result["signal"]) == ["stat_harm", "cross_slow"]
    assert result[list(e.TRACK_METRICS)].isna().all().all()


def test_span_filtering(bench):
    _, _, tuner = bench
    rows = [(i, i * 0.01, freq, 0.2) for freq, stop in ((220, 15), (440, 3), (880, 8))
            for i in range(stop)]
    peaks = pd.DataFrame(rows, columns=["frame", "time", "freq", "amp"])
    peaks["bin"] = pd.array([1] * len(peaks), dtype="Int64")
    peaks["prominence"] = 30.0

    def sets(tracks):
        return {frozenset(map(tuple, g[["time", "freq", "amp"]].to_numpy()))
                for _, g in tracks.groupby("track_id")}

    for gap in tuner.GAPS:
        linked = asp.link_peaks(peaks, min_duration=0, max_gap=gap)
        for duration in tuner.DURATIONS:
            assert sets(tuner.filter_span(linked, duration)) == sets(asp.link_peaks(
                peaks, min_duration=duration, max_gap=gap))


def test_pareto(bench):
    _, _, tuner = bench
    table = pd.DataFrame([(1, 1, 1), (0.9, 2, 2), (1, 1, 1), (0.8, 0, 1), (1, 2, 0.5), (1, 1, 2)],
                         columns=["recall", "false_per_frame", "tracks_per_segment"])
    assert list(tuner.pareto(table).index) == [0, 2, 4, 3]


def test_tuner_requires_dev(bench, tmp_path):
    _, _, tuner = bench
    (tmp_path / "suite").mkdir()
    with pytest.raises(SystemExit, match="suite.py OUT --dev"):
        tuner.main(tmp_path, tmp_path / "results")


def test_pooling_and_repeat_columns(bench):
    e, _, _ = bench
    samples = [
        {"sizes": [1], "switches": [False], "on_errors": [-0.012], "on_total": 2,
         "off_errors": [], "off_total": 0, "bridges": [False], "drop_total": 2},
        {"sizes": [2, 3], "switches": [True, True], "on_errors": [0.02, 0.04], "on_total": 2,
         "off_errors": [], "off_total": 0, "bridges": [True, True], "drop_total": 2},
    ]
    result = e.track_stats(samples)
    assert result["segments"] == result["switch_tracks"] == 3
    assert result["tracks_per_segment"] == 2
    assert result["switch_frac"] == result["bridged_frac"] == pytest.approx(2 / 3)
    assert result["onset_n"] == 3
    assert result["onset_missed"] == result["bridge_undet"] == 1
    assert result["onset_med_ms"] == 20
    assert result["onset_abs_p95_ms"] == pytest.approx(38)
    assert np.isnan(result["offset_n"])
    fam = pd.DataFrame([{"row": f"tool/r{i}", "family": "test", "signals": 1,
                         **dict.fromkeys(e.SUMMARY, 0.5), **e.track_stats([sample])}
                        for i, sample in enumerate(samples, 1)])
    rep = e.repeats(fam)
    runs = rep.columns.get_loc("runs")
    assert list(rep.columns[runs + 1:]) == [f"{m}_{stat}" for stat in ("mean", "min", "max")
                                          for m in e.TRACK_METRICS]
    assert rep.iloc[0]["tracks_per_segment_mean"] == 1.75
    text = e.summary(fam, rep)
    assert "### test" in text
    assert "tool (mean of 2)" in text
    assert "offset med (ms)" not in text


def test_endpoints_outside_guard_and_no_pairs(bench):
    e, _, _ = bench
    t = np.r_[0.02, np.arange(0.26, 1.74, 0.01), 1.99]
    pts = np.column_stack([t, np.full(len(t), 220), np.full(len(t), 0.25), np.zeros(len(t))])
    result, _ = scored(e, pts, "stagger_harm")
    assert result["onset_med_ms"] == pytest.approx(-180)
    assert result["offset_med_ms"] == pytest.approx(190)
    result, _ = scored(e, np.empty((0, 4)), "drop_80")
    assert result["segments"] == result["bridge_n"] == 0
    assert result["bridge_undet"] == 6
    assert np.isnan(result["bridged_frac"])
    result, _ = scored(e, np.empty((0, 4)), "stat_harm")
    fam = pd.DataFrame([{"row": "empty/tracked", "family": "stationary", "signals": 1, **result}])
    assert "### stationary" in e.summary(fam, e.repeats(fam))


def test_neutral_detection_keeps_amplitude_gate(bench):
    e, _, _ = bench
    pts = np.array([[0.23, 220, 0.002, -1]])  # neutral, but above FALSE_DB and below the pairing gate
    acc = e.accuracy(pts, "stagger_harm")
    det = e.detection(pts, "stagger_harm", 0.001)
    assert acc["n"] == acc["far"] == acc["dup"] == 0
    assert det["paired"] == 0
    assert det["false"] == 1
