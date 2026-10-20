"""v1 scale contracts, DSP-9 fixes, and 0.6 numerical characterization."""

from contextlib import redirect_stdout
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from audiospylt import df_transforms as legacy
from audiospylt.model.schema import SchemaError, validate
from audiospylt.transform.scale import split_events, stretch, transpose
from tests import _cases as cases


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def canonical(table):
    return validate(table.rename(columns={"Frequency (Hz)": "freq", "Amplitude": "amp",
                                         "amp_min": "amp_start", "amp_max": "amp_stop"}),
                    "events" if "time_start" in table else "snapshot")


@pytest.mark.parametrize("case", ["single", "multi", "clip", "threshold", "allow", "cues"])
def test_legacy_cases(case):
    snapshot, _ = cases.spectra()
    with redirect_stdout(io.StringIO()):
        expected = canonical(cases.transform_case(case))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if case == "single":
            actual = transpose(canonical(snapshot), 1.25, cents=1200)
        elif case == "multi":
            actual = stretch(transpose(canonical(cases.events()), [1., 2.], cents=[1200., -1200.]),
                             [2., 0.5])
        elif case == "cues":
            actual = split_events(canonical(cases.events()), [0., .03, .03, .06, .09, .12, 1.])
        else:
            actual = transpose(canonical(snapshot), 2, fmin=250, fmax=500,
                               out_of_range={"threshold": "drop", "allow": "keep"}.get(case, case))
    pd.testing.assert_frame_equal(actual, expected, atol=1e-12, rtol=0)


def random_events(seed=16):
    rng = np.random.default_rng(seed)
    # Shuffled rows, repeated cues, unequal durations, and a nonzero anchor.
    cues = np.array([.3, .8, 1.3, 2.7, 3.2])
    starts = rng.permutation(np.repeat(cues, 4))
    return validate(pd.DataFrame({
        "time_start": starts, "time_stop": starts + rng.uniform(.01, 2, len(starts)),
        "freq_start": rng.uniform(10, 1000, len(starts)),
        "freq_stop": rng.uniform(10, 1000, len(starts)),
        "amp_start": rng.random(len(starts)), "amp_stop": rng.random(len(starts)),
        "tag": np.arange(len(starts)),
    }), "events")


@pytest.mark.parametrize("seed", range(5))
def test_random_per_cue_legacy(seed):
    table = random_events(seed)
    rng = np.random.default_rng(seed)
    factors, cents, times = rng.uniform(.1, 3, (3, 5))
    old = table.rename(columns={"amp_start": "amp_min", "amp_stop": "amp_max"})
    expected = canonical(legacy.process_spectral_dataframe(
        old, stretch_factor=times, pitch_factor=factors, pitch_cents=cents * 1200))
    actual = stretch(transpose(table, factors, cents=cents * 1200), times)
    pd.testing.assert_frame_equal(actual, expected, atol=1e-12, rtol=0)
    cuts = rng.uniform(0, 6, 10).tolist()
    # 0.6 iterrows coerces integer extras to floats; v1 keeps their input dtype.
    pd.testing.assert_frame_equal(split_events(table, cuts), canonical(legacy.add_time_cues(old, cuts)),
                                  atol=1e-12, rtol=0, check_dtype=False)


@pytest.mark.parametrize("kind", ["snapshot", "events", "tracks"])
def test_all_columns_metadata_order_and_no_mutation(kind, capsys):
    if kind == "snapshot":
        table = canonical(cases.peaks())
    elif kind == "events":
        table = random_events()
    else:
        table = validate(pd.DataFrame({"track_id": [2, 1, 2, 1], "time": [2., 1., 3., 4.],
                                       "freq": [200., 100., 300., 400.], "amp": [.2] * 4}), kind)
    table["extra"] = pd.Categorical(["x"] * len(table))
    table.attrs["custom"] = {"label": ["retained"]}
    before = table.copy(deep=True)
    result = transpose(table, 1.5, cents=1200)
    columns = ["freq_start", "freq_stop"] if kind == "events" else ["freq"]
    np.testing.assert_allclose(result[columns], table[columns] * 3)
    pd.testing.assert_series_equal(result.extra, table.extra)
    assert result.attrs == table.attrs
    if kind != "snapshot":
        stretched = stretch(table, 2)
        time = "time_start" if kind == "events" else "time"
        t0 = table[time].min()
        np.testing.assert_allclose(stretched[time], t0 + (table[time] - t0) * 2)
        if kind == "events":
            np.testing.assert_allclose(stretched.time_stop - stretched.time_start,
                                       (table.time_stop - table.time_start) * 2)
        assert stretched.attrs == table.attrs
    pd.testing.assert_frame_equal(table, before)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("mode", ["clip", "drop", "keep"])
def test_bounds_warning_and_each_endpoint(mode, capsys):
    table = canonical(cases.events())
    table["freq_start"] = [10., 50., 100.]
    table["freq_stop"] = [100., 5., 60.]
    with pytest.warns(UserWarning, match=r"3 rows.*below: 2, above: 2; observed min/max: 10/200") as caught:
        result = transpose(table, 2, fmin=25, fmax=150, out_of_range=mode)
    assert len(caught) == 1
    if mode == "drop":
        assert result.empty
    else:
        expected = table[["freq_start", "freq_stop"]] * 2
        if mode == "clip":
            expected = expected.clip(25, 150)
        np.testing.assert_array_equal(result[["freq_start", "freq_stop"]], expected)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(False), "2", 1j])
@pytest.mark.parametrize("operation", [transpose, stretch])
def test_bad_factors(value, operation):
    with pytest.raises(ValueError, match="factor"):
        operation(canonical(cases.events()), value)
    with pytest.raises(ValueError, match="factor"):
        operation(canonical(cases.events()), [1., value])


@pytest.mark.parametrize("value", [True, "1200", np.inf, np.nan, 1j])
def test_bad_cents(value):
    with pytest.raises(ValueError, match="cents"):
        transpose(canonical(cases.events()), cents=value)


@pytest.mark.parametrize("operation,kwargs", [(transpose, {"factor": [1.]}),
                                             (transpose, {"cents": [1.]}),
                                             (stretch, {"factor": [1.]})])
def test_sequence_restrictions_and_lengths(operation, kwargs):
    with pytest.raises(ValueError, match=r"length \(1\).*cues \(2\)"):
        operation(canonical(cases.events()), **kwargs)
    tracks = pd.DataFrame({"track_id": [0], "time": [0.], "freq": [100.], "amp": [.5]})
    with pytest.raises(ValueError, match="sequences"):
        operation(tracks, **kwargs)
    if operation is transpose:
        with pytest.raises(ValueError, match="sequences"):
            operation(canonical(cases.peaks()), **kwargs)


@pytest.mark.parametrize("kwargs", [{"fmin": -1}, {"fmax": -1}, {"fmin": True}, {"fmax": np.inf},
                                    {"fmin": 100, "fmax": 99}, {"out_of_range": "allow"},
                                    {"out_of_range": "threshold"}, {"out_of_range": None},
                                    {"out_of_range": np.array(["clip", "drop"])}])
def test_bad_bounds_and_policy(kwargs):
    with pytest.raises(ValueError):
        transpose(canonical(cases.peaks()), **kwargs)


def test_kind_resolution_and_validation():
    table = canonical(cases.events())
    table["freq"], table["amp"] = 100., .2  # ambiguous columns without a valid hint
    assert transpose(table).attrs["audiospylt_table"] == "events"
    assert transpose(table, kind="snapshot").attrs["audiospylt_table"] == "snapshot"
    table.attrs["audiospylt_table"] = "unregistered"
    with pytest.raises(SchemaError, match="ambiguous"):
        transpose(table)
    table = canonical(cases.peaks())
    table.attrs["audiospylt_table"] = ["invalid"]
    assert transpose(table).attrs["audiospylt_table"] == "snapshot"
    with pytest.raises(ValueError, match="no time axis"):
        stretch(table, 1)
    for operation in (transpose, stretch):
        with pytest.raises(SchemaError):
            operation(pd.DataFrame({"freq": [-1.], "amp": [.5]}), 1)
        with pytest.raises(SchemaError):
            operation(table, 1, kind="bad")


def test_split_linear_stable_extras_no_mutation():
    table = validate(pd.DataFrame({"time_start": [0., 0., .5], "time_stop": [1., 1., .5],
        "freq_start": [100., 100., 30.], "freq_stop": [200., 300., 40.],
        "amp_start": [.2, .4, .6], "amp_stop": [.6, 0., .7],
        "label": pd.Categorical(["first", "second", "zero"]), "id": [2**62, 2**62 + 1, 7]}), "events")
    table.attrs["custom"] = "kept"
    before = table.copy(deep=True)
    result = split_events(table, [-1., 0., .5, .5, 1., 3.])
    assert result.label.tolist() == ["first", "second", "zero", "first", "second"]
    np.testing.assert_allclose(result.freq_start, [100., 100., 30., 150., 200.])
    np.testing.assert_allclose(result.amp_stop, [.4, .2, .7, .6, 0.])
    assert result.id.tolist() == [2**62, 2**62 + 1, 7, 2**62, 2**62 + 1]
    assert result.label.dtype == table.label.dtype
    assert result.attrs == table.attrs
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("times", [[np.nan], [np.inf], [True], [".5"], [[.5]], .5])
def test_invalid_cuts(times):
    with pytest.raises(ValueError, match="times"):
        split_events(canonical(cases.events()), times)


@pytest.mark.parametrize("kind", ["snapshot", "events", "tracks"])
def test_empty_tables(kind):
    from audiospylt.model.schema import TABLE_CONTRACTS
    empty = validate(pd.DataFrame(columns=[c.name for c in TABLE_CONTRACTS[kind].required]), kind)
    pd.testing.assert_frame_equal(transpose(empty), empty)
    if kind != "snapshot":
        pd.testing.assert_frame_equal(stretch(empty, 2), empty)
    if kind == "events":
        pd.testing.assert_frame_equal(split_events(empty, []), empty)


def test_mixed_and_multidimensional_factors():
    table = canonical(cases.events())
    for value in ([1., True], [[1., 2.]]):
        with pytest.raises(ValueError):
            transpose(table, value)
        with pytest.raises(ValueError):
            stretch(table, value)
    with pytest.raises(ValueError, match="combined pitch"):
        transpose(table, cents=1e308)
