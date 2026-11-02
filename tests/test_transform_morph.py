"""0.6 pairing characterization, transport plans, full/primary duplicates, D13 levels and curved glides."""

import warnings

import numpy as np
import pandas as pd
import pytest

from audiospylt import df_transforms as legacy
from audiospylt.model.schema import SchemaError, TABLE_CONTRACTS, validate
from audiospylt.synth.additive import synthesize
from audiospylt.synth.additive import _link, _sample_index
from audiospylt.transform.morph import METHODS, _glide_plan, morph, pair, parameter_range


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def snapshot(freq, amp=None):
    return validate(pd.DataFrame({"freq": freq, "amp": np.ones(len(freq)) if amp is None else amp}),
                    "snapshot")


STRATEGIES = [("equal", {}), ("power", {}), ("power", {"power": 1.5}),
              ("power", {"power": -1.5}), ("power", {"power": 50.5}),
              ("power", {"power": -50.5}), ("sigmoid", {}),
              ("sigmoid", {"sigmoid_range": 3}), ("sigmoid", {"sigmoid_range": 300}),
              ("sigmoid", {"sigmoid_range": -10})]
RENDER_STRATEGIES = [("equal", {}), ("power", {"power": 1.5}),
                     ("sigmoid", {"sigmoid_range": 3}), ("transport", {})]


@pytest.mark.parametrize("sizes", [(1, 1), (1, 7), (7, 1), (4, 4), (3, 11), (11, 3), (8, 13), (13, 8)])
@pytest.mark.parametrize("method,kwargs", STRATEGIES)
def test_pair_matches_legacy_row_for_row(sizes, method, kwargs):
    rng = np.random.default_rng(sum(sizes))
    a, b = [snapshot(np.sort(rng.uniform(30, 3000, size)), rng.random(size)) for size in sizes]
    before = [table.copy(deep=True) for table in (a, b)]
    functions = {"equal": legacy.merge_equal_split, "power": legacy.merge_cdf,
                 "sigmoid": legacy.merge_sigmoid}
    old_kwargs = {"power_factor" if k == "power" else k: v for k, v in kwargs.items()}
    expected = functions[method](a, b, **old_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = pair(a, b, method, **kwargs)
    np.testing.assert_array_equal(result[["freq_start", "freq_stop"]],
                                  expected[["freq_data1", "freq_data2"]])
    np.testing.assert_array_equal(result[["amp_source", "amp_target"]],
                                  expected[["amp_data1", "amp_data2"]])
    expected_parameter = expected.attrs.get("power_factor_used", expected.attrs.get("sigmoid_range_used"))
    assert result.attrs["morph"] == {"method": method, "parameter": expected_parameter, "weight": None}
    assert list(result) == ["src", "dst", "freq_start", "freq_stop", "amp_source", "amp_target", "mass",
                            "primary_start", "primary_stop", "amp_start", "amp_stop"]
    assert result.src.dtype == result.dst.dtype == np.dtype("int64")
    assert result.mass.dtype == np.dtype("float64")
    assert result.primary_start.dtype == result.primary_stop.dtype == np.dtype("bool")
    np.testing.assert_array_equal(result.mass, np.full(max(sizes), 1 / max(sizes)))
    assert "audiospylt_table" not in result.attrs
    assert set(TABLE_CONTRACTS) == {"snapshot", "events", "tracks"}
    for actual, original in zip((a, b), before):
        pd.testing.assert_frame_equal(actual, original)


@pytest.mark.parametrize("sizes", [(1, 8), (8, 1), (4, 4), (3, 9), (9, 3)])
@pytest.mark.parametrize("method", ["power", "sigmoid"])
def test_parameter_range_matches_legacy(sizes, method):
    a, b = [snapshot(np.arange(1., size + 1)) for size in sizes]
    analysis = legacy.merge_cdf_analysis if method == "power" else legacy.merge_sigmoid_analysis
    assert parameter_range(a, b, method) == tuple(analysis(a, b))


@pytest.mark.parametrize("method,fallback,mapping", [
    ("power", (1., 1.), "_cdf_mapping_indices"),
    ("sigmoid", (6., 6.), "_sigmoid_mapping_indices"),
])
def test_no_covering_parameter_fallback(monkeypatch, method, fallback, mapping):
    # Force the otherwise hard-to-reach candidate-grid fallback, without plotting.
    monkeypatch.setattr(f"audiospylt.transform.morph.{mapping}", lambda n, m, p: np.zeros(m, dtype=int))
    with pytest.warns(UserWarning, match="No fully covering"):
        assert parameter_range(snapshot([100, 200]), snapshot([300, 400, 500]), method) == fallback


def assert_d13(table, a, b):
    for identity, flag, gain, original in (("src", "primary_start", "amp_start", a),
                                           ("dst", "primary_stop", "amp_stop", b)):
        for position, group in table.groupby(identity):
            assert group[flag].sum() == 1
            np.testing.assert_array_equal(group[gain], np.where(group[flag], original.amp.iloc[position], 0))


@pytest.mark.parametrize("method,kwargs", STRATEGIES + [("transport", {}), ("transport", {"weight": "power"})])
@pytest.mark.parametrize("reverse", [False, True])
def test_d13_gains_energy_order_and_unsorted(method, kwargs, reverse):
    a = snapshot([800., 200., 400.], [.5, .2, .4])
    b = snapshot([150., 100., 1600., 300., 900., 500., 700.], [.2, .3, .4, .5, .6, .7, .8])
    if reverse:
        a, b = b, a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plan = pair(a, b, method, **kwargs)
        events = morph(a, b, 2.3, method, time_start=1.2, duplicates="primary", **kwargs)
        sorted_events = morph(a.sort_values("freq", kind="stable"), b.sort_values("freq", kind="stable"),
                              2.3, method, time_start=1.2, duplicates="primary", **kwargs)
    assert_d13(plan, a, b)
    assert events.amp_start.pow(2).sum() == pytest.approx(a.amp.iloc[plan.src.unique()].pow(2).sum())
    assert events.amp_stop.pow(2).sum() == pytest.approx(b.amp.iloc[plan.dst.unique()].pow(2).sum())
    # Recover identities from the actual emitted frequency pairs (unique here),
    # independently of the implementation's topological ordering helper.
    ordered = plan.set_index(["freq_start", "freq_stop"]).loc[
        pd.MultiIndex.from_frame(events[["freq_start", "freq_stop"]])]
    for identity, flag in (("src", "primary_start"), ("dst", "primary_stop")):
        assert ordered.groupby(identity, sort=False)[flag].first().all()
    np.testing.assert_array_equal(events[["amp_start", "amp_stop"]], ordered[["amp_start", "amp_stop"]])
    pd.testing.assert_frame_equal(events, sorted_events)
    assert events.attrs == {"audiospylt_table": "events",
                            "morph": {**plan.attrs["morph"], "duplicates": "primary", "freq_space": "hz",
                                      "freq_curve": 0.0, "amp_curve": 0.0, "max_cents": 1.0}}
    np.testing.assert_array_equal(events.time_start, 1.2)
    np.testing.assert_array_equal(events.time_stop, 1.2 + 2.3)


@pytest.mark.parametrize("weight", ["amp", "power"])
@pytest.mark.parametrize("seed", range(5))
def test_transport_marginals_and_staircase(weight, seed):
    rng = np.random.default_rng(seed)
    a = snapshot(np.arange(1., 12), rng.random(11))
    b = snapshot(np.arange(1., 18), rng.random(17))
    result = pair(a, b, "transport", weight=weight)
    for identity, table in (("src", a), ("dst", b)):
        expected = table.amp.to_numpy() ** (1 if weight == "amp" else 2)
        expected /= expected.sum()
        actual = result.groupby(identity).mass.sum().reindex(range(len(table)), fill_value=0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        assert (np.diff(result[identity]) >= 0).all()
    assert len(result) <= len(a) + len(b) - 1
    assert (result.mass > 1e-12).all()
    assert result.attrs["morph"] == {"method": "transport", "parameter": None, "weight": weight}
    assert_d13(result, a, b)


@pytest.mark.parametrize("weight", ["amp", "power"])
def test_transport_degenerate_zero_partials_and_silent_rows(weight):
    a, b = snapshot([100., 200., 300.], [1., 0., 1.]), snapshot([110., 210., 310.], [1., 0., 1.])
    result = pair(a, b, "transport", weight=weight)
    assert result.src.tolist() == result.dst.tolist() == [0, 2]
    assert result.mass.tolist() == [.5, .5]
    assert len(result) < len(a) + len(b) - 1
    # Edges (0,1) and (1,2) lose the election at both ends.
    a, b = snapshot([100., 200.], [.5, .5]), snapshot([110., 150., 210.], [.4, .2, .4])
    if weight == "power":
        a["amp"], b["amp"] = np.sqrt(a.amp), np.sqrt(b.amp)
    result = pair(a, b, "transport", weight=weight)
    assert len(result) == 4
    assert (~result.primary_start & ~result.primary_stop).sum() == 1
    assert len(morph(a, b, 1, "transport", weight=weight)) == 3


def test_primary_mass_distance_amplitude_and_row_ties():
    a = snapshot([400.], [.5])
    # Equal mass: nearest cents wins over an earlier row.
    b = snapshot([100., 200., 800.], [.5, .2, .7])
    result = pair(a, b)
    assert result.loc[result.primary_start, "dst"].tolist() == [2]  # 200 and 800 equally near; louder wins
    b.amp = [.5, .7, .7]
    assert pair(a, b).loc[lambda x: x.primary_start, "dst"].tolist() == [1]  # exact earlier-row tie
    b = snapshot([410., 1000.], [.1, .9])
    assert pair(a, b, "transport").loc[lambda x: x.primary_start, "dst"].tolist() == [1]  # mass before cents
    b = snapshot([100., 410., 800.], [0., .5, .5])
    assert pair(a, b, "transport").loc[lambda x: x.primary_start, "dst"].tolist() == [1]
    # Duplicate frequencies remain distinct partial identities, even with duplicate labels.
    a = snapshot([400., 400.], [.2, .6])
    a.index = [7, 7]
    result = pair(a, snapshot([450., 450., 600., 600.]))
    assert_d13(result, a.reset_index(drop=True), snapshot([450., 450., 600., 600.]))
    assert result.groupby("src").primary_start.sum().tolist() == [1, 1]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("side", ["source", "target"])
def test_snapshot_validation(method, side):
    valid = snapshot([100., 200.])
    for invalid, exception, match in [(snapshot([]), ValueError, "empty"),
        (snapshot([0., 100., 0.]), ValueError, r"2 partial.*positions \[0, 2\].*filter DC first"),
        (pd.DataFrame({"freq": [-1.], "amp": [.5]}), SchemaError, "freq"),
        (pd.DataFrame({"freq": [1.]}), SchemaError, "amp")]:
        with pytest.raises(exception, match=match):
            pair(invalid if side == "source" else valid, invalid if side == "target" else valid, method)


@pytest.mark.parametrize("kwargs", [{"method": "bad"}, {"power": 2}, {"sigmoid_range": 3},
    {"method": "transport", "power": 2}, {"method": "power", "sigmoid_range": 3},
    {"method": "power", "power": True}, {"method": "power", "power": np.inf},
    {"method": "sigmoid", "sigmoid_range": [3]}, {"method": "sigmoid", "sigmoid_range": np.nan},
    {"weight": "power"}, {"method": "transport", "weight": "unknown"}])
def test_pair_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        pair(snapshot([100.]), snapshot([200.]), **kwargs)


@pytest.mark.parametrize("method", ["equal", "transport", "bad"])
def test_parameter_range_invalid_method(method):
    with pytest.raises(ValueError, match="power.*sigmoid"):
        parameter_range(snapshot([100.]), snapshot([200.]), method)


@pytest.mark.parametrize("weight", ["amp", "power"])
@pytest.mark.parametrize("reverse", [False, True])
def test_zero_transport_mass(weight, reverse):
    a, b = snapshot([100., 200.], [0., 0.]), snapshot([300.])
    if reverse:
        a, b = b, a
    with pytest.raises(ValueError, match="zero total"):
        pair(a, b, "transport", weight=weight)


@pytest.mark.parametrize("method,kwargs", STRATEGIES + [("transport", {}), ("transport", {"weight": "power"})])
@pytest.mark.parametrize("reverse", [False, True])
def test_full_duplicates_keep_every_row_at_full_amplitude(method, kwargs, reverse):
    a = snapshot([800., 200., 400.], [.5, .2, .4])
    b = snapshot([150., 100., 1600., 300., 900., 500., 700.], [.2, .3, .4, .5, .6, .7, .8])
    if reverse:
        a, b = b, a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plan = pair(a, b, method, **kwargs)
        full = morph(a, b, 2.3, method, time_start=1.2, duplicates="full", **kwargs)
        default = morph(a, b, 2.3, method, time_start=1.2, **kwargs)
    # Every pairing row sounds, each copy with its partials' full amplitudes.
    columns = ["freq_start", "freq_stop", "amp_source", "amp_target"]
    expected = plan[columns].sort_values(columns).to_numpy()
    emitted = full[["freq_start", "freq_stop", "amp_start", "amp_stop"]].sort_values(
        ["freq_start", "freq_stop", "amp_start", "amp_stop"]).to_numpy()
    np.testing.assert_array_equal(emitted, expected)
    # Primaries still come first in each group, so continuous linking picks them.
    ordered = plan.set_index(["freq_start", "freq_stop"]).loc[
        pd.MultiIndex.from_frame(full[["freq_start", "freq_stop"]])]
    for identity, flag in (("src", "primary_start"), ("dst", "primary_stop")):
        assert ordered.groupby(identity, sort=False)[flag].first().all()
    assert full.attrs["morph"] == {**plan.attrs["morph"], "duplicates": "full", "freq_space": "hz",
                                   "freq_curve": 0.0, "amp_curve": 0.0, "max_cents": 1.0}
    # None resolves per method: 0.6 methods keep 0.6 copies, transport keeps primaries.
    if method == "transport":
        expected_default = morph(a, b, 2.3, method, time_start=1.2, duplicates="primary", **kwargs)
    else:
        expected_default = full
    pd.testing.assert_frame_equal(default, expected_default)
    assert default.attrs["morph"] == expected_default.attrs["morph"]


@pytest.mark.parametrize("sizes", [(3, 11), (11, 3), (4, 4)])
@pytest.mark.parametrize("method,old", [("equal", "merge_equal_split"), ("power", "merge_cdf"),
                                        ("sigmoid", "merge_sigmoid")])
def test_full_duplicates_match_06_transition_rows(sizes, method, old):
    rng = np.random.default_rng(sum(sizes))
    a, b = [snapshot(np.sort(rng.uniform(30, 3000, size)), rng.random(size)) for size in sizes]
    expected = legacy.create_time_transition_df(getattr(legacy, old)(a, b), 2.0, scale_amp=False)
    result = morph(a, b, 2.0, method)
    columns = ["freq_start", "freq_stop", "amp_start", "amp_stop"]
    np.testing.assert_array_equal(
        result[columns].sort_values(columns).to_numpy(),
        expected.rename(columns={"amp_min": "amp_start", "amp_max": "amp_stop"})[columns]
        .sort_values(columns).to_numpy())


def test_full_transport_keeps_rows_primary_at_neither_end():
    a, b = snapshot([100., 200.], [.5, .5]), snapshot([110., 150., 210.], [.4, .2, .4])
    assert len(pair(a, b, "transport")) == 4
    assert len(morph(a, b, 1, "transport")) == 3
    full = morph(a, b, 1, "transport", duplicates="full")
    assert len(full) == 4 and (full.amp_start > 0).all() and (full.amp_stop > 0).all()


@pytest.mark.parametrize("duplicates", ["copies", "", 1, True, ["full"]])
def test_bad_duplicates(duplicates):
    with pytest.raises(ValueError, match="duplicates"):
        morph(snapshot([100.]), snapshot([200.]), 1, duplicates=duplicates)


@pytest.mark.parametrize("kwargs", [{"duration": 0}, {"duration": -1}, {"duration": np.inf},
    {"duration": True}, {"duration": [1]}, {"duration": 1, "time_start": -1},
    {"duration": 1, "time_start": np.nan}, {"duration": 1, "time_start": True}])
def test_bad_morph_times(kwargs):
    with pytest.raises(ValueError):
        morph(snapshot([100.]), snapshot([200.]), **kwargs)


def test_unused_warning(capsys):
    a, b = snapshot(np.arange(100., 900., 100)), snapshot(np.arange(100., 1100., 100))
    for method, kwargs in (("power", {"power": 50.5}), ("sigmoid", {"sigmoid_range": 300})):
        with pytest.warns(UserWarning, match=r"\d+ smaller-side partial\(s\) unused") as caught:
            pair(a, b, method, **kwargs)
        assert len(caught) == 1
    assert capsys.readouterr().out == ""


def endpoint_levels(events, source, target, onset, duration):
    """50 ms full Hann windows, re linear endpoint power at window centres."""
    sr, window = 8000, .05
    audio = synthesize(events, sr=sr, phase="continuous")
    weights = np.hanning(round(window * sr))
    p0, p1 = source.amp.pow(2).sum() / 2, target.amp.pow(2).sum() / 2
    levels = []
    for start in (onset, onset + duration - window):
        offset = round(start * sr)
        segment = audio[offset:offset + len(weights)]
        assert len(segment) == len(weights)
        measured = np.sum(weights * segment**2) / weights.sum()
        ideal = p0 + (p1 - p0) * (start + window / 2 - onset) / duration
        levels.append(10 * np.log10(measured / ideal))
    return np.array(levels)


def hold(table, start, stop):
    return pd.DataFrame({"time_start": start, "time_stop": stop,
                         "freq_start": table.freq, "freq_stop": table.freq,
                         "amp_start": table.amp, "amp_stop": table.amp})


def test_render_random_diverging_converging_and_chained():
    rng = np.random.default_rng(13)
    ranges = {"15s": [], "2s": [], "chain": []}
    for k in (2, 4, 8):
        for draw in range(5):
            while True:
                freq = np.sort(rng.uniform(460., 1400., k))
                if np.diff(freq).min() >= 40:
                    break
            amp = rng.uniform(.3, 1., k)
            amp *= .5 / np.linalg.norm(amp)
            a, b = snapshot([440.], [.5]), snapshot(freq, amp)
            for method, kwargs in RENDER_STRATEGIES:
                for source, target in ((a, b), (b, a)):
                    for duration, tolerance in ((15., .1), (2., .25)):
                        events = morph(source, target, duration, method, duplicates="primary", **kwargs)
                        levels = endpoint_levels(events, source, target, 0., duration)
                        ranges[f"{duration:g}s"].extend(levels)
                        assert np.abs(levels).max() <= tolerance, (k, draw, method, duration, levels)
                    middle = morph(source, target, 2., method, time_start=1., duplicates="primary", **kwargs)
                    chain = pd.concat([hold(source, 0., 1.), middle, hold(target, 3., 4.)], ignore_index=True)
                    levels = endpoint_levels(chain, source, target, 1., 2.)
                    ranges["chain"].extend(levels)
                    assert np.abs(levels).max() <= .25, (k, draw, method, "chain", levels)
    for name, values in ranges.items():
        print(f"render {name} (8000 Hz): {min(values):+.4f} .. {max(values):+.4f} dB")


@pytest.mark.parametrize("weight", ["amp", "power"])
def test_transport_render_three_to_seven_both_directions(weight):
    a = snapshot([200., 600., 1000.], [.3, .4, .5])
    b = snapshot([240., 390., 550., 710., 900., 1150., 1400.], [.2, .3, .2, .4, .3, .2, .5])
    for source, target in ((a, b), (b, a)):
        events = morph(source, target, 15., "transport", weight=weight)
        levels = endpoint_levels(events, source, target, 0., 15.)
        print(f"render transport {weight} {len(source)} -> {len(target)}: {levels} dB")
        assert np.abs(levels).max() <= .1, levels


def test_transport_handles_extreme_finite_amplitudes():
    for amp in ([1e308, 1e308], [1e-300, 1e-300]):
        for weight in ("amp", "power"):
            result = pair(snapshot([100., 200.], amp), snapshot([300., 400.], amp), "transport", weight=weight)
            np.testing.assert_array_equal(result.mass, [.5, .5])


# Curved glides (step 1.11, D34): linear rows on w(x) = expm1(c x) / expm1(c).

WIDE_A = ([180., 260., 410., 700., 1300.], [.3, .5, .2, .4, .6])
WIDE_B = ([95., 330., 520., 640., 900., 1500., 2400., 3100.], [.6, .2, .4, .3, .5, .1, .4, .2])


def w_ref(x, c):
    """Overflow-safe reference for w(x) = expm1(c x) / expm1(c); not ``_warp``'s form."""
    x = np.asarray(x, dtype=float)
    if c == 0:
        return x
    if abs(c) < 1e-6:
        return x + 0.5 * c * x * (x - 1.0)
    if c > 0:
        # exp(c (x - 1)) * (1 - exp(-c x)) / (1 - exp(-c)) stays finite for large c.
        return np.exp(c * (x - 1.0)) * (1.0 - np.exp(-c * x)) / (1.0 - np.exp(-c))
    return np.expm1(c * x) / np.expm1(c)


def exact_freq(f0, f1, x, space, c):
    return f0 + (f1 - f0) * w_ref(x, c) if space == "hz" else f0 * (f1 / f0) ** w_ref(x, c)


def glides(events, onset=0.0):
    """Rows of each pair, in emission order (every pair's first row starts at the onset)."""
    pid = np.cumsum(events.time_start.to_numpy() == onset) - 1
    return [events.iloc[np.flatnonzero(pid == k)] for k in range(pid.max() + 1)]


def breakpoints(rows, column):
    return np.r_[rows[f"{column}_start"].to_numpy(), rows[f"{column}_stop"].iloc[-1]]


def worst_errors(rows, onset, duration, space, freq_curve, amp_curve, n=4001):
    """Max cents and amp error re the pair's peak, dense grid per emitted segment."""
    f0, f1 = rows.freq_start.iloc[0], rows.freq_stop.iloc[-1]
    a0, a1 = rows.amp_start.iloc[0], rows.amp_stop.iloc[-1]
    times = breakpoints(rows, "time")
    freqs = breakpoints(rows, "freq")
    amps = breakpoints(rows, "amp")
    xs = (times - onset) / duration
    u = np.linspace(0., 1., n)
    cents = amp = 0.0
    peak = max(a0, a1)
    for i in range(len(xs) - 1):
        x = xs[i] + (xs[i + 1] - xs[i]) * u
        line_f = freqs[i] + (freqs[i + 1] - freqs[i]) * u
        line_a = amps[i] + (amps[i + 1] - amps[i]) * u
        exact_f = exact_freq(f0, f1, x, space, freq_curve)
        exact_a = a0 + (a1 - a0) * w_ref(x, amp_curve)
        assert np.isfinite(exact_f).all() and np.isfinite(exact_a).all()
        seg_cents = np.abs(1200 * np.log2(line_f / exact_f))
        seg_amp = np.abs(line_a - exact_a)
        assert np.isfinite(seg_cents).all() and np.isfinite(seg_amp).all()
        cents = max(cents, float(np.max(seg_cents)))
        if peak:
            amp = max(amp, float(np.max(seg_amp)) / peak)
    return cents, amp


CURVED = [(space, fc, ac) for space in ("hz", "pitch") for fc in (-8, -2, 0, 2, 8, 30) for ac in (0, -4, 4)
          if (space, fc, ac) != ("hz", 0, 0)]


@pytest.mark.parametrize("method,duplicates", [("equal", "full"), ("transport", "primary")])
@pytest.mark.parametrize("space,fc,ac", CURVED)
def test_curved_rows_within_tolerance_and_fewest(method, duplicates, space, fc, ac):
    a, b = snapshot(*WIDE_A), snapshot(*WIDE_B)
    onset, duration, tolerance = .4, 2.5, 1.
    straight = morph(a, b, duration, method, time_start=onset, duplicates=duplicates)
    events = morph(a, b, duration, method, time_start=onset, duplicates=duplicates,
                   freq_space=space, freq_curve=fc, amp_curve=ac, max_cents=tolerance)
    groups = glides(events, onset)
    assert len(groups) == len(straight)
    for rows, (f0, f1, a0, a1) in zip(groups, straight[["freq_start", "freq_stop", "amp_start", "amp_stop"]]
                                      .itertuples(index=False)):
        # Ends are the pair's values bit for bit; interior joins share one value.
        assert (rows.freq_start.iloc[0], rows.freq_stop.iloc[-1]) == (f0, f1)
        assert (rows.amp_start.iloc[0], rows.amp_stop.iloc[-1]) == (a0, a1)
        assert (rows.time_start.iloc[0], rows.time_stop.iloc[-1]) == (onset, onset + duration)
        for column in ("time", "freq", "amp"):
            np.testing.assert_array_equal(rows[f"{column}_stop"].to_numpy()[:-1],
                                          rows[f"{column}_start"].to_numpy()[1:])
        assert np.all(np.diff(breakpoints(rows, "time")) > 0)
        cents, amp = worst_errors(rows, onset, duration, space, fc, ac)
        assert cents <= tolerance * (1 + 1e-3) and amp <= 1e-3 * tolerance * (1 + 1e-3), (len(rows), cents, amp)
        plan = _glide_plan(f0, f1, a0, a1, space, float(fc), float(ac), tolerance)
        if plan is None:
            assert len(rows) == 1
        elif len(rows) > 1:
            grid, errors, _ = plan
            # Fewest: emitted N is the smallest m whose refined check passes.
            assert errors(grid(len(rows)))[0] <= 1
            assert all(errors(grid(m))[0] > 1 for m in range(1, len(rows)))


def test_default_and_straight_options_are_byte_identical():
    a, b = snapshot(*WIDE_A), snapshot([180., 330., 410., 640., 1300.], [.3, .2, .2, .3, .6])
    for method in METHODS:
        default = morph(a, b, 2., method, time_start=.5)
        explicit = morph(a, b, 2., method, time_start=.5, freq_space="hz", freq_curve=0, amp_curve=0, max_cents=5)
        pd.testing.assert_frame_equal(default, explicit, check_exact=True)
        assert explicit.attrs["morph"] == {**default.attrs["morph"], "max_cents": 5.0}
    # Pairs equal in freq (pitch space) and amp take the curved code path but stay one row.
    a, b = snapshot([200., 500.], [.4, .3]), snapshot([200., 500.], [.4, .3])
    curved = morph(a, b, 2., "equal", freq_space="pitch", freq_curve=6, amp_curve=-3)
    pd.testing.assert_frame_equal(curved, morph(a, b, 2., "equal"), check_exact=True)
    mixed = morph(snapshot([200., 500.], [.4, .3]), snapshot([200., 800.], [.4, .3]), 2., "equal",
                  freq_space="pitch", amp_curve=5)
    assert len(mixed) > 2 and (mixed.freq_start == 200.).sum() == 1


def test_flat_pitch_pair_with_curved_amp_keeps_its_frequency():
    # exp(log f) differs from f by 1-4 ulp for these; the segments must carry f itself.
    for f in (440., 123.456, 999.999):
        events = morph(snapshot([f], [.5]), snapshot([f], [.1]), 1., "equal", freq_space="pitch", amp_curve=5)
        assert len(events) > 1
        np.testing.assert_array_equal(events[["freq_start", "freq_stop"]].to_numpy(), f)


@pytest.mark.parametrize("space", ["hz", "pitch"])
def test_curve_sign_is_late_progress_for_rising_and_falling(space):
    for f0, f1, a0, a1 in ((200., 800., .2, .8), (800., 200., .8, .2)):
        for c in (4., -4.):
            events = morph(snapshot([f0], [a0]), snapshot([f1], [a1]), 2., "equal",
                           freq_space=space, freq_curve=c, amp_curve=c)
            freq = np.interp(1., breakpoints(events, "time"), breakpoints(events, "freq"))
            amp = np.interp(1., breakpoints(events, "time"), breakpoints(events, "amp"))
            progress = ((freq - f0) / (f1 - f0) if space == "hz" else np.log(freq / f0) / np.log(f1 / f0))
            for value in (progress, (amp - a0) / (a1 - a0)):
                assert (value < .5) if c > 0 else (value > .5), (f0, c, value)
    # Pitch space with curve 0 is the geometric glide: constant cents per second.
    events = morph(snapshot([100.]), snapshot([1600.]), 4., "equal", freq_space="pitch")
    x = breakpoints(events, "time") / 4.
    np.testing.assert_allclose(breakpoints(events, "freq"), 100. * 16. ** x, rtol=1e-12)
    assert len(events) > 1


def own_pair_links(events, onset=0.0, sr=44100):
    pid = np.cumsum(events.time_start.to_numpy() == onset) - 1
    t0, t1 = events.time_start.to_numpy(), events.time_stop.to_numpy()
    f0, f1 = events.freq_start.to_numpy(), events.freq_stop.to_numpy()
    a0, a1 = events.amp_start.to_numpy(), events.amp_stop.to_numpy()
    pred = _link(_sample_index(t0, sr), _sample_index(t1, sr), f0, f1, None, 100., t0, t1, a0, a1)
    inner = np.flatnonzero(pred >= 0)
    return int(np.sum(pid[pred[inner]] != pid[inner])), len(events) - len(inner) - (pid.max() + 1), pred


@pytest.mark.parametrize("method,kwargs", [("equal", {}), ("transport", {}), ("power", {"power": -1.5}),
                                           ("sigmoid", {"sigmoid_range": -10})])
def test_segments_link_to_their_own_pair(method, kwargs):
    a, b = snapshot(*WIDE_A), snapshot(*WIDE_B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for options in ({"freq_curve": 6, "amp_curve": -3}, {"freq_space": "pitch", "freq_curve": -5}):
            events = morph(a, b, 1.5, method, **kwargs, **options)
            wrong, missing, _ = own_pair_links(events)
            assert (wrong, missing) == (0, 0), (method, options)


def test_chained_curved_morphs_link_at_the_shared_snapshot():
    a, b, c = snapshot(*WIDE_A), snapshot(*WIDE_B), snapshot([150., 450., 1000.], [.5, .4, .3])
    first = morph(a, b, 2., "transport", freq_curve=4, amp_curve=2)
    second = morph(b, c, 2., "transport", time_start=2., freq_space="pitch", freq_curve=-3)
    chain = pd.concat([first, second], ignore_index=True)
    _, _, pred = own_pair_links(chain)
    heads = np.flatnonzero(chain.time_start.to_numpy() == 2.)
    linked = heads[pred[heads] >= 0]
    # Every second-morph row that leaves a partial still sounding at 2 s continues one that arrives there.
    assert len(linked) == min(len(heads), int((chain.time_stop.to_numpy() == 2.).sum()))
    np.testing.assert_array_equal(chain.freq_stop.to_numpy()[pred[linked]], chain.freq_start.to_numpy()[linked])


def test_extreme_and_tiny_curves_stay_finite():
    a, b = snapshot([100., 400.], [.5, .2]), snapshot([3000., 150.], [.1, .7])
    for c in (500., -500., 1e-300, -1e-300):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            events = morph(a, b, 1., "equal", freq_space="pitch", freq_curve=c, amp_curve=c)
        assert np.isfinite(events.to_numpy(dtype=float)).all()
        assert len(events) <= 2 * 256
        if abs(c) < 1:
            # A subnormal curve is the straight (here geometric) glide, not a broken one.
            assert len(events) == len(morph(a, b, 1., "equal", freq_space="pitch"))


@pytest.mark.parametrize("amp_curve,db", [(0., r"-inf dB"), (8., r"-\d+\.\d dB")])
def test_cap_warns_and_emits_256_segments(amp_curve, db):
    # amp_curve=8: the cents check fails first, so the amp error is refined without its sample pass.
    a, b = snapshot([100.], [.5]), snapshot([3000.], [.1])
    with pytest.warns(UserWarning, match=r"1 pair\(s\) exceed max_cents=0.0001 at 256 segments; "
                                         rf"worst error [\d.e+-]+ c, {db} re the pair's peak"):
        events = morph(a, b, 1., "equal", freq_curve=8, amp_curve=amp_curve, max_cents=1e-4)
    assert len(events) == 256


@pytest.mark.parametrize("kwargs,match", [
    ({"freq_space": "Hz"}, "freq_space"), ({"freq_space": "log"}, "freq_space"), ({"freq_space": None}, "freq_space"),
    ({"freq_space": 1}, "freq_space"), ({"freq_curve": np.nan}, "freq_curve"), ({"freq_curve": np.inf}, "freq_curve"),
    ({"freq_curve": True}, "freq_curve"), ({"freq_curve": "2"}, "freq_curve"), ({"amp_curve": [1.]}, "amp_curve"),
    ({"amp_curve": -np.inf}, "amp_curve"), ({"max_cents": 0}, "max_cents"), ({"max_cents": -1}, "max_cents"),
    ({"max_cents": np.inf}, "max_cents"), ({"max_cents": True}, "max_cents"),
    ({"freq_curve": 1.0001e4}, "freq_curve"), ({"freq_curve": -1e5}, "freq_curve"),
    ({"amp_curve": 1.0001e4}, "amp_curve"), ({"amp_curve": -1e5}, "amp_curve")])
def test_bad_curve_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        morph(snapshot([100.]), snapshot([200.]), 1, **kwargs)


def test_curve_bound_accepts_1e4():
    for option in ("freq_curve", "amp_curve"):
        for value in (1e4, -1e4):
            events = morph(snapshot([100.], [.5]), snapshot([3000.], [.1]), 1., "equal", **{option: value})
            assert len(events) >= 1


def _phase_integral_rows(events, sr=44100):
    t0, t1 = events.time_start.to_numpy(), events.time_stop.to_numpy()
    f0, f1 = events.freq_start.to_numpy(), events.freq_stop.to_numpy()
    a0, a1 = events.amp_start.to_numpy(), events.amp_stop.to_numpy()
    n0, n1 = _sample_index(t0, sr), _sample_index(t1, sr)
    y = np.zeros(int(n1.max()) if len(events) else 0)
    cycles = 0.0
    order = np.argsort(t0, kind="stable")
    for k in order:
        span = t1[k] - t0[k]
        if n1[k] > n0[k]:
            tau = np.arange(n0[k], n1[k]) / sr - t0[k]
            frac = tau / span
            y[n0[k]:n1[k]] += (a0[k] + (a1[k] - a0[k]) * frac) * np.sin(
                2 * np.pi * (cycles + tau * (f0[k] + 0.5 * (f1[k] - f0[k]) * frac))
            )
        cycles += span * 0.5 * (f0[k] + f1[k])
    return y


def test_short_curved_morph_bridges_to_one_voice():
    events = morph(snapshot([100.], [.5]), snapshot([3000.], [.5]), .001, "equal", freq_curve=4)
    n0 = _sample_index(events.time_start.to_numpy(), 44100)
    n1 = _sample_index(events.time_stop.to_numpy(), 44100)
    wrong, missing, pred = own_pair_links(events)
    assert (wrong, missing) == (0, 0)
    assert int(((pred < 0) & (n1 > n0)).sum()) == 1
    y = synthesize(events, 44100, fade=0)
    np.testing.assert_allclose(y, _phase_integral_rows(events), rtol=0, atol=1e-9)


@pytest.mark.parametrize("duration", [0.002, 0.001])
def test_docs_morph_short_durations_one_voice_per_pair(duration):
    from pathlib import Path
    from audiospylt.io.tables import read_table
    root = Path(__file__).resolve().parents[1]
    speech = read_table(root / "data/tables/speech1.tsv")
    conga = read_table(root / "data/tables/conga3.tsv")
    events = morph(speech, conga, duration, "transport", freq_space="pitch", freq_curve=4, amp_curve=-2)
    n0 = _sample_index(events.time_start.to_numpy(), 44100)
    n1 = _sample_index(events.time_stop.to_numpy(), 44100)
    _, _, pred = own_pair_links(events)
    assert int(((pred < 0) & (n1 > n0)).sum()) == 41


def test_fewest_segments_is_global_minimum_on_reviewer_case():
    from audiospylt.transform.morph import _glide_breakpoints
    case = (11000., 80., .2, .25, "pitch", 19., -16., 1.)
    points, _ = _glide_breakpoints(*case)
    emitted = points.size - 1
    grid, errors, _ = _glide_plan(*case)
    smallest = min(m for m in range(1, 257) if errors(grid(m))[0] <= 1)
    assert emitted == smallest == 78


@pytest.mark.parametrize("case,tol", [
    ((40., 12000., .6, .5, "pitch", 3300., 2000.), 1.),
    ((12000., 200., 1., 1., "pitch", 8000., 0.), 6000.),
])
def test_refined_acceptance_keeps_dense_error_within_tolerance(case, tol):
    from audiospylt.transform.morph import _glide_breakpoints
    f0, f1, a0, a1, space, fc, ac = case
    points, _ = _glide_breakpoints(f0, f1, a0, a1, space, fc, ac, tol)
    freqs = exact_freq(f0, f1, points, space, fc)
    freqs[0], freqs[-1] = f0, f1
    amps = a0 + (a1 - a0) * w_ref(points, ac)
    amps[0], amps[-1] = a0, a1
    assert np.isfinite(freqs).all() and np.isfinite(amps).all()
    # Per-segment dense grid (uniform-over-pair can miss peaks inside short segments).
    u = np.linspace(0., 1., 4001)
    cents = amp = 0.0
    for i in range(points.size - 1):
        x = points[i] + (points[i + 1] - points[i]) * u
        line_f = freqs[i] + (freqs[i + 1] - freqs[i]) * u
        line_a = amps[i] + (amps[i + 1] - amps[i]) * u
        exact_f = exact_freq(f0, f1, x, space, fc)
        exact_a = a0 + (a1 - a0) * w_ref(x, ac)
        seg_cents = np.abs(1200 * np.log2(line_f / exact_f))
        seg_amp = np.abs(line_a - exact_a)
        assert np.isfinite(exact_f).all() and np.isfinite(exact_a).all()
        assert np.isfinite(seg_cents).all() and np.isfinite(seg_amp).all()
        cents = max(cents, float(np.max(seg_cents)))
        amp = max(amp, float(np.max(seg_amp)))
    assert cents <= tol * (1 + 1e-3)
    assert amp <= 1e-3 * tol * max(a0, a1) * (1 + 1e-3)


def _dense_abs_max_error(points, f0, f1, a0, a1, space, fc, ac, n=200001):
    """Independent dense |cents| / |amp| max; written here, not imported from morph.py."""
    if space == "hz" or f0 == f1:
        freqs = f0 + (f1 - f0) * w_ref(points, fc)
    else:
        freqs = f0 * (f1 / f0) ** w_ref(points, fc)
    freqs = np.asarray(freqs, dtype=float).copy()
    freqs[0], freqs[-1] = f0, f1
    amps = np.asarray(a0 + (a1 - a0) * w_ref(points, ac), dtype=float).copy()
    amps[0], amps[-1] = a0, a1
    u = np.linspace(0.0, 1.0, n)
    cents = amp = 0.0
    for i in range(points.size - 1):
        x = points[i] + (points[i + 1] - points[i]) * u
        line_f = freqs[i] + (freqs[i + 1] - freqs[i]) * u
        line_a = amps[i] + (amps[i + 1] - amps[i]) * u
        if space == "hz" or f0 == f1:
            exact_f = f0 + (f1 - f0) * w_ref(x, fc)
        else:
            exact_f = f0 * (f1 / f0) ** w_ref(x, fc)
        exact_a = a0 + (a1 - a0) * w_ref(x, ac)
        seg_cents = np.abs(1200 * np.log2(line_f / exact_f))
        seg_amp = np.abs(line_a - exact_a)
        assert np.isfinite(seg_cents).all() and np.isfinite(seg_amp).all()
        cents = max(cents, float(np.max(seg_cents)))
        amp = max(amp, float(np.max(seg_amp)))
    return cents, amp


# Pitch opposite-sign hump cases: one segment passes the probes but the true
# maximum exceeds the tolerance (dense/tol 1.33 / 1.08 / 1.56 / 1.51 without the
# inflection path); each must split or warn at the cap.
_OPPOSITE_HUMP_CASES = (
    (20000.0, 2.0, 0.5, 0.5, "pitch", 32.0, 0.0, 3400.0),
    (1.0301990780713464, 45896.41233732215, 0.5, 0.5, "pitch", -71.35846280359516, 0.0, 5009.9477876555875),
    (1.1555756857936812, 50004.2915900016, 0.5, 0.5, "pitch", -38.96137042183536, 0.0, 4120.658474701769),
    (41661.35269687521, 1.3961108870208012, 0.5, 0.5, "pitch", 42.11687824329343, 0.0, 3786.562244542034),
)

# Case table: signed-extrema miss, both spaces × curves, amp cases, and pitch
# opposite-sign curvature (falling c>0 / rising c<0) over 20 Hz–20 kHz.
_DENSE_TRUE_MAX_CASES = (
    [(20000.0, 4.0, 0.5, 0.5, "pitch", 24.0, 0.0, 3000.0)]
    + list(_OPPOSITE_HUMP_CASES)
    + [
        (20000.0, 4.0, 0.5, 0.5, space, fc, 0.0, tol)
        for space in ("hz", "pitch")
        for fc in (8.0, 24.0, -24.0)
        for tol in (1.0, 10.0, 100.0, 3000.0)
    ]
    + [
        (440.0, 880.0, 0.8, 0.2, "hz", 8.0, 8.0, 10.0),
        (200.0, 800.0, 0.2, 0.9, "pitch", 0.0, -24.0, 100.0),
    ]
    + [
        (f0, f1, 0.5, 0.5, "pitch", fc, 0.0, tol)
        for f0, f1, sign in ((20000.0, 20.0, 1.0), (20.0, 20000.0, -1.0))
        for mag in (8.0, 32.0, 200.0, 1e4)
        for tol in (1.0, 100.0, 3000.0, 6000.0)
        for fc in (sign * mag,)
    ]
)


@pytest.mark.parametrize("case", _DENSE_TRUE_MAX_CASES)
def test_accepted_error_is_true_maximum_dense(case):
    from audiospylt.transform.morph import _glide_breakpoints
    f0, f1, a0, a1, space, fc, ac, tol = case
    points, worst = _glide_breakpoints(f0, f1, a0, a1, space, fc, ac, tol)
    if case in _OPPOSITE_HUMP_CASES:
        assert points.size - 1 > 1 or worst is not None
    n = 200001 if points.size <= 32 else max(4001, 200001 // max(points.size - 1, 1))
    cents, amp = _dense_abs_max_error(points, f0, f1, a0, a1, space, fc, ac, n=n)
    amp_tol = 1e-3 * tol * max(a0, a1)
    assert cents / tol <= 1.0 + 1e-9, (case, points.size - 1, cents)
    if amp_tol > 0 and ac != 0 and a0 != a1:
        assert amp / amp_tol <= 1.0 + 1e-9, (case, points.size - 1, amp)


def test_pitch_inflection_roots_find_both_g_zeros():
    from audiospylt.transform.morph import _freq_path, _log_abs_w2, _pitch_inflection_roots, _warp
    f0, f1, c = 12255.828018993558, 31.57021109597755, 86.2772504577835
    span = np.log(f1 / f0)
    la, lb = _freq_path(np.array([0.0, 1.0]), f0, f1, "pitch", c)[0]
    roots = _pitch_inflection_roots(np.array([0.0]), np.array([1.0]), np.array([la]), np.array([lb]), span, c)[0]
    assert np.isfinite(roots).all() and roots[0] < roots[1]
    m = (lb - la)

    def g(x):
        return (2 * np.log(abs(m)) - 2 * np.log(la + m * x)
                - np.log(abs(span)) - float(_log_abs_w2(np.asarray(x), c)))

    def e_prime(x):
        return m / (la + m * x) - span * float(_warp(np.asarray([x]), c)[1][0])

    h = 1e-8
    for r in roots:
        assert abs(g(r)) < 1e-10
        left = (e_prime(r) - e_prime(r - h)) / h
        right = (e_prime(r + h) - e_prime(r)) / h
        assert np.sign(left) != np.sign(right)
