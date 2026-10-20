"""0.6 pairing characterization, transport plans, and D13 table/render levels."""

import warnings

import numpy as np
import pandas as pd
import pytest

from audiospylt import df_transforms as legacy
from audiospylt.model.schema import SchemaError, TABLE_CONTRACTS, validate
from audiospylt.synth.additive import synthesize
from audiospylt.transform.morph import METHODS, morph, pair, parameter_range


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
        events = morph(a, b, 2.3, method, time_start=1.2, **kwargs)
        sorted_events = morph(a.sort_values("freq", kind="stable"), b.sort_values("freq", kind="stable"),
                              2.3, method, time_start=1.2, **kwargs)
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
    assert events.attrs == {"audiospylt_table": "events", "morph": plan.attrs["morph"]}
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
                        events = morph(source, target, duration, method, **kwargs)
                        levels = endpoint_levels(events, source, target, 0., duration)
                        ranges[f"{duration:g}s"].extend(levels)
                        assert np.abs(levels).max() <= tolerance, (k, draw, method, duration, levels)
                    middle = morph(source, target, 2., method, time_start=1., **kwargs)
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
