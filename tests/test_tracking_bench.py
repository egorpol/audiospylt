"""Tracking benchmark truth and scores, using analytic hand-built points, not analysis audio."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

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
PAIRS = ("320_720", "280_840", "450_800", "260_640", "550_880", "380_1140")
DEV_PAIRS = ("350_700", "300_600", "440_880", "250_1000", "500_750", "400_500")
MOVING = (*[f"{kind}_{pair}" for pair in PAIRS for kind in ("turn", "cross")],
          "hole_vib_330", "hole_vib_262", "hole_vib_494", "hole_vib_208",
          "hole_gliss_165", "hole_gliss_147", "hole_gliss_660", "hole_gliss_990")
DEV_MOVING = (*[f"dev_{kind}_{pair}" for pair in DEV_PAIRS for kind in ("turn", "x")],
              "dev_hole_vib_60", "dev_hole_gliss_60", "dev_hole_vib_100", "dev_hole_gliss_100")


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
    assert tr.BASE == PUBLISHED
    assert tuple(tr.SIGNALS) == PUBLISHED + MOVING
    assert tuple(tr.MOVING) == MOVING
    assert tuple(tr.DEV_SIGNALS) == DEV
    assert tuple(tr.DEV_MOVING) == DEV_MOVING
    assert all(n.startswith("dev_") for n in DEV + DEV_MOVING)
    for names, base in ((tr.ORIGINAL, 0), (PUBLISHED[13:], 100), (MOVING, 200), (DEV, 1000), (DEV_MOVING, 1100)):
        assert {n: tr.seed(n) for n in names} == {n: base + i for i, n in enumerate(sorted(names))}
    assert len({tr.seed(n) for n in PUBLISHED + MOVING + DEV + DEV_MOVING}) == 72
    json.dumps(tr.SIGNALS)
    json.dumps(tr.DEV_SIGNALS)
    json.dumps(tr.DEV_MOVING)


def test_moving_files(bench):
    _, tr, _ = bench
    assert [tr.seed(n) for n in ("cross_fast", "trem_40", "dev_cross", "dev_vib_75")] == [100, 107, 1000, 1014]
    for kind_a, kind_b, table in (("turn", "cross", tr.MOVING), ("dev_turn", "dev_x", tr.DEV_MOVING)):
        for name, spec in table.items():
            if name.startswith(kind_a + "_"):
                other = table[kind_b + name[len(kind_a):]]
                assert (spec["family"], other["family"]) == ("turn-back", "matched crossing")
                assert (spec["kind"], other["kind"]) == ("turn", "sweeps")
                assert spec["sweeps"] == other["sweeps"] and spec["amps"] == other["amps"]
    edges = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(tr.partials("turn_320_720", edges)[0][:, 1], [480, 480])
    f = tr.partials("turn_320_720", np.array([0.01, 0.5, 1.5, 1.99]))[0]
    np.testing.assert_allclose(f[:, :2], f[:, :1:-1])
    assert np.all(f[0] < 480) and np.all(f[1] > 480)
    first_half = np.linspace(0.0, 1.0, 101)
    assert np.array_equal(tr.partials("turn_320_720", first_half)[0], tr.partials("cross_320_720", first_half)[0])
    for turn, cross in (("turn_450_800", "cross_450_800"), ("dev_turn_400_500", "dev_x_400_500")):
        a, b = tr.render(turn), tr.render(cross)
        half = len(a) // 2
        assert np.array_equal(a[:half], b[:half]) and not np.array_equal(a[half + 2000:], b[half + 2000:])
    crossing = tr.partials("cross_320_720", np.array([0.5, 1.5]))[0]
    assert crossing[0, 0] < 480 < crossing[0, 1]
    holes = [tr.MOVING[n] for n in MOVING if "hole" in n]
    assert sum(len(h["drop"]) for h in holes) == 48
    assert [h["drop"][0][1] for h in holes] == [0.07, 0.11] * 4
    firsts = [h["drop"][0][0] for h in holes]
    np.testing.assert_allclose(firsts, [0.4, 1.5, 1.5, 0.4] * 2)
    assert [tr.shape(n) for n in MOVING if "hole" in n] == ["vibrato"] * 4 + ["glide"] * 4
    assert tr.meets("cross_fast") and tr.meets("turn_320_720") and not tr.meets("hole_vib_330")


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


@pytest.mark.parametrize("name", PUBLISHED + MOVING + DEV + DEV_MOVING)
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


@pytest.mark.parametrize("name", PUBLISHED + MOVING + DEV_MOVING)
def test_perfect_tracks(bench, name):
    e, tr, _ = bench
    pts = perfect(tr, name)
    result, det = scored(e, pts, name)
    assert result["tracks_per_segment"] == 1
    assert result["switch_frac"] == 0
    if tr.meets(name):
        assert [result[m] for m in e.COUNT_METRICS[1:]] == [2, 2, 0, 0]
        hit = np.zeros_like(det["state"], dtype=bool)
        hit[det["pairs"][:, 1], det["pairs"][:, 0]] = True
        missed = np.any((det["state"] == 1) & ~hit, axis=0)
        assert np.all(np.abs(det["grid"][missed | (det["false_grid"] > 0)] - 1) <= 0.01 + 1e-12)
    else:
        assert result["recall"] == 1
        assert result["false_per_frame"] == 0
        assert np.isnan(result["meet_eligible"])
    if "drop" in tr._spec(name):
        assert result["bridge_n"] == 6
        assert result["bridged"] == result["bridged_frac"] == 0
        assert [result[m] for m in e.EDGE_METRICS] == [0, 6, 0]
    else:
        assert np.isnan(result["bridged"])
        assert all(np.isnan(result[m]) for m in e.EDGE_METRICS)


def meeting_of(e, before, after, start=0.2, stop=1.8, inside=()):
    """One track with these partial votes on each side of the zone, plus any votes inside it."""
    grid = np.arange(e.truth.GUARD, e.truth.DURATION - e.truth.GUARD + 1e-9, e.GRID)  # unrounded, as in detection
    index = {round(float(t), 2): i for i, t in enumerate(grid)}
    votes = [*zip(np.round(0.3 + 0.01 * np.arange(len(before)), 2), before),
             *zip(np.round(1.4 + 0.01 * np.arange(len(after)), 2), after), *inside]
    pairs = np.array([(index[round(float(t), 2)], k, 7) for t, k in votes], dtype=int).reshape(-1, 3)
    pts = np.array([[start, 400.0, 0.1, 7], [stop, 400.0, 0.1, 7]])
    result = e.meeting(pts, {"grid": grid, "pairs": pairs})
    assert result["meet_eligible"] == result["meet_right"] + result["meet_swap"] + result["meet_unscorable"]
    return [result[m] for m in e.COUNT_METRICS[1:]]


@pytest.mark.parametrize("before, after, verdict", [
    ([0, 0, 0], [0, 0, 0], [1, 1, 0, 0]),
    ([1, 1, 1], [1, 1, 1], [1, 1, 0, 0]),
    ([0, 0, 0], [1, 1, 1], [1, 0, 1, 0]),
    ([0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]),
    ([0, 0, 1, 1], [0, 0, 0], [1, 0, 0, 1]),
    ([0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
    ([0, 0], [0, 0, 0], [1, 0, 0, 1]),
    ([0, 0, 0], [], [1, 0, 0, 1]),
    ([], [], [0, 0, 0, 0]),
])
def test_meeting_verdicts(bench, before, after, verdict):
    assert meeting_of(bench[0], before, after) == verdict


def test_meeting_zone(bench):
    e = bench[0]
    three = [0, 0, 0]
    # Only tracks reaching past both edges are eligible.
    assert meeting_of(e, three, three, start=0.72) == [0, 0, 0, 0]
    assert meeting_of(e, three, three, start=0.73) == [0, 0, 0, 0]
    assert meeting_of(e, three, three, stop=1.28) == [0, 0, 0, 0]
    assert meeting_of(e, three, three, start=0.71, stop=1.29) == [1, 1, 0, 0]
    # Inside the zone nothing votes, whichever partial the merged peak pairs with.
    inside = [(t, 1) for t in (0.8, 0.9, 1.0, 1.1, 1.2, 0.73, 1.27)]
    assert meeting_of(e, [0, 0], three, inside=inside) == [1, 0, 0, 1]
    assert meeting_of(e, three, three, inside=inside) == [1, 1, 0, 0]
    # A track with pairs only inside the zone has no identity to judge: it is not eligible.
    assert meeting_of(e, [], [], inside=inside) == [0, 0, 0, 0]
    # Neither do the two grid points on the zone edges.
    assert meeting_of(e, [0, 0], three, inside=[(0.72, 0)]) == [1, 0, 0, 1]
    assert meeting_of(e, three, [0, 0], inside=[(1.28, 0)]) == [1, 0, 0, 1]
    assert meeting_of(e, [0, 0], three, inside=[(0.71, 0)]) == [1, 1, 0, 0]


@pytest.mark.parametrize("name", ["turn_320_720", "cross_320_720", "dev_turn_400_500", "cross_fast"])
def test_planted_swap_at_the_meeting(bench, name):
    e, tr, _ = bench
    pts = perfect(tr, name)
    swapped = pts.copy()
    swapped[:, 3] = np.where(pts[:, 0] < 1, pts[:, 3], 1 - pts[:, 3])
    result, _ = scored(e, swapped, name)
    assert [result[m] for m in e.COUNT_METRICS[1:]] == [2, 0, 2, 0]
    cut = pts.copy()
    cut[:, 3] = pts[:, 3] + 2 * (pts[:, 0] >= 1)
    result, _ = scored(e, cut, name)
    assert [result[m] for m in e.COUNT_METRICS[1:]] == [0, 0, 0, 0]
    assert result["tracks_per_segment"] == 2


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
    assert [result[m] for m in e.EDGE_METRICS] == [6 * bridged, 6, 0]


@pytest.mark.parametrize("side", ["after", "before"])
def test_reclaim_from_a_stub(bench, side):
    """A held track takes the partial back from a stub past the neutral zone: bridged, but not at the edge."""
    e, tr, _ = bench
    name = "dev_hole_vib_60"
    centre, width = tr._spec(name)["drop"][0]
    lo, hi = centre - width / 2, centre + width / 2
    pts = perfect(tr, name)
    held = pts[:, 3] <= 1  # partial 0 on both sides of its hole
    pts[held, 3] = 0
    hop = 256 / tr.SR
    stub = held & ((pts[:, 0] > hi) & (pts[:, 0] < hi + tr.NEUTRAL - hop) if side == "after"
                   else (pts[:, 0] < lo) & (pts[:, 0] > lo - tr.NEUTRAL + hop))
    assert stub.sum() >= 5
    pts[stub, 3] = 99
    result, _ = scored(e, pts, name)
    assert (result["bridged"], result["bridge_n"]) == (1, 6)  # the registered count takes it as bridged
    assert [result[m] for m in e.EDGE_METRICS] == [0, 6, 0]
    pts[stub, 3] = 0  # the same frames on one track are bridged by both counts
    result, _ = scored(e, pts, name)
    assert (result["bridged"], result["bridge_n"]) == (1, 6)
    assert [result[m] for m in e.EDGE_METRICS] == [1, 6, 0]


def hole_frames(tr, frames, name="dev_hole_vib_60"):
    """Points around partial 0's hole, one per tuple of `frame`'s arguments."""
    centre, width = tr._spec(name)["drop"][0]

    def frame(after, offset, track, cents=0.0, amp=0.1, partial=0):
        """`offset` seconds from the upper edge if `after`, else from the lower one."""
        t = centre + (width / 2 if after else -width / 2) + offset
        return t, tr.partials(name, np.array([t]))[0][partial, 0] * 2 ** (cents / 1200), amp, track

    return np.array([frame(*f) for f in frames], dtype=float).reshape(-1, 4)


BEFORE, AFTER = (False, -0.01, 0), (True, 0.01, 0)


@pytest.mark.parametrize("frames, verdicts", [
    ([BEFORE, AFTER], [True]),
    ([BEFORE, (True, 0.01, 1)], [False]),
    ([(False, -0.099, 0), (True, 0.099, 0)], [True]),                       # inside the 0.1 s window
    ([(False, -0.1 - 1e-13, 0), (True, 0.1 + 1e-13, 0)], [True]),           # rounding at the window's end
    ([(False, -0.101, 0), AFTER], []),                                      # too far before
    ([BEFORE, (True, 0.101, 0)], []),                                       # too far after
    ([BEFORE], []),                                                         # one side is not determined
    ([(False, 0.0, 0), AFTER], []),                                         # an edge frame belongs to neither side
    ([BEFORE, (True, 0.0, 0)], []),
    ([(False, 0.01, 0), AFTER], []),                                        # inside the hole
    ([BEFORE, (True, -0.01, 0)], []),
    ([BEFORE, (True, -0.01, 1), AFTER], [True]),                            # a hole frame changes nothing
    ([(False, -0.05, 1), BEFORE, AFTER, (True, 0.05, 1)], [True]),          # the nearest frame decides
    ([(False, -0.01, 1), (False, -0.05, 0), AFTER, (True, 0.05, 1)], [False]),
    ([(False, -0.01, 0, 49.0), (True, 0.01, 0, -49.0)], [True]),            # tolerance, either way
    ([(False, -0.01, 0, 51.0), AFTER], []),
    ([BEFORE, (True, 0.01, 0, -51.0)], []),
    ([(False, -0.01, 1, 51.0), BEFORE, AFTER], [True]),                     # off the partial is not read
    ([(False, -0.01, 0, 0.0, 1e-9), (True, 0.01, 0, 0.0, 1e-9)], [True]),   # no amplitude gate
    ([(False, -0.01, 0, 0.0, 0.1, 1), AFTER], []),                          # another partial's frame
    ([(False, -0.01, 0), (False, -0.01, 1), AFTER], [True]),                # a same-time tie: the first row
    ([(False, -0.01, 1), (False, -0.01, 0), AFTER], [False]),
    ([BEFORE, (True, 0.01, 1), AFTER], [False]),
    ([(False, -0.01, -1), AFTER], []),                                      # an untracked peak
    ([BEFORE, (True, 0.01, -1)], []),
])
def test_edge_rule(bench, frames, verdicts):
    e, tr, _ = bench
    assert e.edge_bridges(hole_frames(tr, frames), "dev_hole_vib_60") == verdicts


def test_edge_rule_skips_zero_frequency(bench):
    e, tr, _ = bench
    pts = hole_frames(tr, [BEFORE, AFTER])
    pts[0, 1] = 0
    with np.errstate(all="raise"):
        assert e.edge_bridges(pts, "dev_hole_vib_60") == []


def test_edge_rule_reads_each_hole_on_its_own_partial(bench):
    e, tr, _ = bench
    name = "dev_hole_vib_60"
    pts = perfect(tr, name)
    pts[pts[:, 3] <= 3, 3] = pts[pts[:, 3] <= 3, 3] // 2  # bridge partials 0 and 1
    assert e.edge_bridges(pts, name) == [True, True, False, False, False, False]
    assert e.edge_bridges(pts, "dev_turn_400_500") == []  # no holes, whatever the points


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
    assert result[list(e.TRACK_METRICS + e.COUNT_METRICS)].isna().all().all()
    assert "Meetings and holes" not in (tmp_path / "results" / "summary.md").read_text()


def test_count_tables(bench, tmp_path):
    e, tr, _ = bench
    folder = tmp_path / "raw" / "test" / "tracks"
    names = ["dev_turn_400_500", "dev_x_400_500", "dev_hole_vib_60", "dev_hole_gliss_60", "dev_drop_40"]
    (folder).mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"kind": "tracks", "hop_s": 256 / tr.SR, "signals": names}))
    for name in names:
        pts = perfect(tr, name)
        if name == "dev_hole_gliss_60":  # bridge the holes of the first two partials
            pts[:, 3] = np.where(pts[:, 3] < 4, pts[:, 3] // 2 * 2, pts[:, 3])
        e.points.write_points(folder, name, *pts.T)
    e.main(tmp_path, tmp_path / "results")
    columns = list(pd.read_csv(tmp_path / "results" / "per_family.csv").columns)
    assert columns[columns.index("bridged_frac") + 1:] == list(e.COUNT_METRICS + e.EDGE_METRICS)
    text = (tmp_path / "results" / "summary.md").read_text()
    head = text.index("## Meetings and holes")
    assert head > text.index("## Tracking")
    assert "| test/tracks | turn-back | 1 | 2 | 2 | 0 | 0 |" in text
    assert "| test/tracks | matched crossing | 1 | 2 | 2 | 0 | 0 |" in text
    assert "| test/tracks | 2/12 | 0 | 0/6 | 2/6 | 0/6 | 2/6 |" in text
    assert "Repeat runs are listed one by one" in text[head:]
    assert "dev_drop_40" not in text[head:]  # a hole in a steady partial is not a moving hole
    edge = text.index("### Holes at the hole edge (exploratory)")
    assert text.index("### Holes in moving partials") < edge
    assert text.count("| test/tracks | 2/12 | 0 | 0/6 | 2/6 | 0/6 | 2/6 |") == 2
    note = text[edge:]
    assert "added after the published run" in note
    assert "does not prove one track through the gap" in note
    assert "not the registered score" in note


def test_edge_table_keeps_its_own_count(bench):
    e, tr, _ = bench
    sig = pd.DataFrame([{"row": "test/tracks", "signal": "dev_hole_vib_60", "family": tr.HOLES,
                         "bridged": 5, "bridge_n": 6, "bridge_undet": 0,
                         "bridged_edge": 2, "bridge_edge_n": 4, "bridge_edge_undet": 2}])
    registered, edge = "\n".join(e.count_tables(pd.DataFrame({"meet_eligible": [np.nan]}), sig)).split(
        "### Holes at the hole edge (exploratory)")
    assert "| test/tracks | 5/6 | 0 | 5/6 | – | 5/6 |" in registered
    assert "| test/tracks | 2/4 | 2 | 2/4 | – | 2/4 |" in edge


def load_runner(monkeypatch, filename, **stubs):
    """Import a runner from the benchmark folder, with stand-ins for tools this venv lacks."""
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location(f"bench_{filename[:-3]}", BENCH / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_suite(monkeypatch, tr, folder):
    """Three old files and three new ones; new files step 0.5 s, so a pooled median would differ."""
    old = {n: {"family": "x", "kind": "f0"} for n in ("old_a", "old_b", "old_c")}
    new = {"new_hole": {"family": "y", "kind": "f0"}, "new_turn": {"family": "y", "kind": "turn"},
           "new_cross": {"family": "y", "kind": "sweeps"}}
    monkeypatch.setattr(tr, "BASE", tuple(old))
    monkeypatch.setattr(tr, "MOVING", new)
    (folder / "suite").mkdir(parents=True)
    for name in {**old, **new}:
        sf.write(folder / "suite" / f"{name}.wav", np.full(64, 0.5 if name in new else 0.0), tr.SR, subtype="FLOAT")
    return old, new


def breakpoints(step):
    return np.column_stack([np.arange(5) * step, np.full(5, 440.0), np.full(5, 0.1)])


def test_loris_new_run_agrees_with_full_run(bench, tmp_path, monkeypatch):
    _, tr, _ = bench
    seen = []

    def analyze(x, sr, **p):
        seen.append(float(x[0]))
        return [breakpoints(0.5 if x[0] > 0 else 0.01)]

    run_loris = load_runner(monkeypatch, "run_loris.py", loristrck=SimpleNamespace(analyze=analyze))
    monkeypatch.setattr(run_loris.importlib.metadata, "version", lambda _: "0")
    full, split = tmp_path / "full", tmp_path / "split"
    old, new = tiny_suite(monkeypatch, tr, full)
    (split / "suite").parent.mkdir()
    (split / "suite").symlink_to(full / "suite")

    def params(name):
        return {"resolution": name}

    monkeypatch.setattr(tr, "SIGNALS", old)
    run_loris.run(split, "loris/x", params)
    monkeypatch.setattr(tr, "SIGNALS", {**old, **new})
    seen.clear()
    run_loris.run(split, "loris/x", params, new=True)
    assert seen == [0.5] * len(new)  # the old files are not analysed again
    run_loris.run(full, "loris/x", params)
    metas = [json.loads((f / "raw" / "loris/x" / "meta.json").read_text()) for f in (full, split)]
    assert metas[0] == metas[1]
    assert metas[0]["hop_s"] == 0.01
    assert set(metas[0]["params"]) == set(old) | set(new)
    assert sorted(p.name for p in (split / "raw" / "loris/x").glob("*.csv")) == sorted(f"{n}.csv" for n in {**old, **new})
    with pytest.raises(SystemExit, match="full run"):
        run_loris.run(tmp_path / "none", "loris/x", params, new=True)
    monkeypatch.setattr(run_loris.importlib.metadata, "version", lambda _: "1")
    with pytest.raises(SystemExit, match="same tool"):
        run_loris.run(split, "loris/x", params, new=True)


def test_sms_new_run_extends_a_full_run(bench, tmp_path, monkeypatch):
    _, tr, _ = bench

    def analyse(x, sr, w, N, H, t, maxnSines, minSineDur):
        return np.full((4, 1), 440.0), np.full((4, 1), -20.0), None

    models = SimpleNamespace(sineModel=SimpleNamespace(sineModelAnal=analyse))
    run_sms = load_runner(monkeypatch, "run_sms.py", smstools=SimpleNamespace(models=models), **{"smstools.models": models})
    monkeypatch.setattr(run_sms.importlib.metadata, "version", lambda _: "0")
    old, new = tiny_suite(monkeypatch, tr, tmp_path)
    with pytest.raises(SystemExit, match="full run"):
        run_sms.main(tmp_path, new=True)
    assert not (tmp_path / "raw").exists()
    monkeypatch.setattr(tr, "SIGNALS", old)
    run_sms.main(tmp_path)
    folder = tmp_path / "raw" / "sms-tools/default"
    assert sorted(p.stem for p in folder.glob("*.csv")) == sorted(old)
    run_sms.main(tmp_path, new=True)
    assert sorted(p.stem for p in folder.glob("*.csv")) == sorted({**old, **new})
    monkeypatch.setattr(run_sms.importlib.metadata, "version", lambda _: "1")
    with pytest.raises(SystemExit, match="same tool"):
        run_sms.main(tmp_path, new=True)


def test_partiels_new_run_agrees_with_full_run(bench, tmp_path, monkeypatch):
    _, tr, _ = bench
    run_partiels = load_runner(monkeypatch, "run_partiels.py")
    full, split = tmp_path / "full", tmp_path / "split"
    old, new = tiny_suite(monkeypatch, tr, full)
    (split / "suite").parent.mkdir()
    (split / "suite").symlink_to(full / "suite")
    templates = tmp_path / "templates"
    templates.mkdir()
    for fname in run_partiels.TEMPLATES.values():
        (templates / fname).touch()
    monkeypatch.setattr(run_partiels, "matched_template", lambda src, dst: dst.touch())

    seen = set()

    def run_one(partiels, wav, template, tmp):
        seen.add(wav.stem)
        step = 0.5 if wav.stem in new else 0.01
        return pd.DataFrame({"time": np.arange(5) * step, "freq": 440.0, "amp": 0.1, "track_id": 0})

    monkeypatch.setattr(run_partiels, "run_one", run_one)
    monkeypatch.setattr(tr, "SIGNALS", old)
    run_partiels.main(split, templates, "x")
    monkeypatch.setattr(tr, "SIGNALS", {**old, **new})
    seen.clear()
    run_partiels.main(split, templates, "x", new=True)
    assert seen == set(new)  # old PM2 inharmonic runs are not reproducible: they must stay
    run_partiels.main(split, templates, "x", new=True)  # a rerun must not list a file twice
    run_partiels.main(full, templates, "x")
    for row, names in (("pm2-inharmonic/matched/r3", [*old, *new]), ("pm2-harmonic/template", [*old, "new_hole"])):
        metas = [json.loads((f / "raw" / row / "meta.json").read_text()) for f in (full, split)]
        assert metas[0] == metas[1]
        assert metas[0]["signals"] == names
        assert metas[0]["hop_s"] == 0.01


def test_tracked_rows(bench, tmp_path, monkeypatch):
    _, tr, _ = bench
    runner = load_runner(monkeypatch, "run_audiospylt.py")
    assert list(runner.TRACKED) == [f"audiospylt-v1/tracked{s}" for s in
                                    ("", "-bh", "-lp", "-gap24", "-lp-gap24", "-gap60", "-lp-gap60")]
    (tmp_path / "suite").mkdir()
    name = "dev_hole_gliss_60"
    sf.write(tmp_path / "suite" / f"{name}.wav", tr.render(name).astype(np.float32), tr.SR, subtype="FLOAT")
    runner.tracked_rows(tmp_path, [name])
    defaults = json.loads((tmp_path / "raw" / "audiospylt-v1/tracked" / "meta.json").read_text())
    assert (defaults["params"]["prediction"], defaults["params"]["max_gap"]) == ("hold", 0)
    meta = json.loads((tmp_path / "raw" / "audiospylt-v1/tracked-lp-gap60" / "meta.json").read_text())
    assert (meta["params"]["prediction"], meta["params"]["max_gap"], meta["signals"]) == ("lp", 60, [name])
    assert (tmp_path / "raw" / "audiospylt-v1/tracked-gap24" / f"{name}.csv").is_file()


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
         "off_errors": [], "off_total": 0, "bridges": [False], "edge_bridges": [False, True], "drop_total": 2},
        {"sizes": [2, 3], "switches": [True, True], "on_errors": [0.02, 0.04], "on_total": 2,
         "off_errors": [], "off_total": 0, "bridges": [True, True], "edge_bridges": [], "drop_total": 2},
    ]
    result = e.track_stats(samples)
    assert result["segments"] == result["switch_tracks"] == 3
    assert result["tracks_per_segment"] == 2
    assert result["switch_frac"] == result["bridged_frac"] == pytest.approx(2 / 3)
    assert result["onset_n"] == 3
    assert result["onset_missed"] == result["bridge_undet"] == 1
    assert [result[m] for m in e.EDGE_METRICS] == [1, 2, 2]  # its own denominator, not bridge_n
    assert result["onset_med_ms"] == 20
    assert result["onset_abs_p95_ms"] == pytest.approx(38)
    assert np.isnan(result["offset_n"])
    fam = pd.DataFrame([{"row": f"tool/r{i}", "family": "test", "signals": 1,
                         **dict.fromkeys(e.SUMMARY, 0.5), **e.track_stats([sample])}
                        for i, sample in enumerate(samples, 1)])
    rep = e.repeats(fam)
    runs = rep.columns.get_loc("runs")
    assert list(rep.columns[runs + 1:]) == [f"{m}_{stat}" for metrics in (e.TRACK_METRICS, e.COUNT_METRICS, e.EDGE_METRICS)
                                          for stat in ("mean", "min", "max") for m in metrics]
    assert rep.iloc[0]["tracks_per_segment_mean"] == 1.75
    text = e.summary(fam, rep, pd.DataFrame({"family": [], "bridged": []}))
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
    assert [result[m] for m in e.EDGE_METRICS] == [0, 0, 6]
    result, _ = scored(e, np.empty((0, 4)), "stat_harm")
    fam = pd.DataFrame([{"row": "empty/tracked", "family": "stationary", "signals": 1, **result}])
    assert "### stationary" in e.summary(fam, e.repeats(fam), pd.DataFrame({"family": [], "bridged": []}))


def test_neutral_detection_keeps_amplitude_gate(bench):
    e, _, _ = bench
    pts = np.array([[0.23, 220, 0.002, -1]])  # neutral, but above FALSE_DB and below the pairing gate
    acc = e.accuracy(pts, "stagger_harm")
    det = e.detection(pts, "stagger_harm", 0.001)
    assert acc["n"] == acc["far"] == acc["dup"] == 0
    assert det["paired"] == 0
    assert det["false"] == 1


def test_tuner_default_matches_track_partials(bench):
    import inspect
    _, _, tuner = bench
    params = inspect.signature(asp.track_partials).parameters
    assert tuner.DEFAULT == (
        params["prominence_db"].default,
        params["min_duration"].default,
        params["max_gap"].default,
    )


def test_tuner_max_cents_reaches_link_peaks(bench, tmp_path, monkeypatch):
    import soundfile as sf
    _, tr, tuner = bench
    seen: list[float] = []
    real = tuner.asp.link_peaks

    def spy(peaks, **kwargs):
        seen.append(kwargs["max_cents"])
        return real(peaks, **kwargs)

    monkeypatch.setattr(tuner.asp, "link_peaks", spy)
    monkeypatch.setattr(tuner, "PROMINENCES", (20,))
    monkeypatch.setattr(tuner, "DURATIONS", (0.1,))
    monkeypatch.setattr(tuner, "GAPS", (0,))
    name = "dev_stat_harm"
    monkeypatch.setattr(tuner.truth, "DEV_SIGNALS", {name: tr.DEV_SIGNALS[name]})
    folder = tmp_path / "dev" / "suite"
    folder.mkdir(parents=True)
    sf.write(folder / f"{name}.wav", tr.render(name), tr.SR)
    tuner.main(tmp_path, tmp_path / "results", max_cents=400.0)
    assert seen and all(c == 400.0 for c in seen)
    assert tuner.parse_args(["OUT", "RESULTS", "--max-cents", "400"]).max_cents == 400.0
    assert tuner.parse_args(["OUT", "RESULTS"]).max_cents == tuner.MAX_CENTS
