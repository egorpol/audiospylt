"""Contracts and validation for the additive v1 table API."""

import ast
from dataclasses import FrozenInstanceError
import importlib
import io
from pathlib import Path
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from audiospylt.model.schema import (
    LegacyFormatWarning,
    SchemaError,
    TABLE_CONTRACTS,
    infer_kind,
    upgrade,
    validate,
)
import audiospylt.model.schema as schema


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def _frame(kind):
    values = {
        "snapshot": {"freq": [0.0, 440.0], "amp": [1.0, 0.5]},
        "events": {"time_start": [0.0, 1.0], "time_stop": [0.0, 2.0],
                   "freq_start": [0.0, 440.0], "freq_stop": [0.0, 400.0],
                   "amp_start": [0.0, 1.0], "amp_stop": [0.0, 0.1]},
        "tracks": {"track_id": [1, 1], "time": [0.0, 1.0],
                   "freq": [0.0, 440.0], "amp": [1.0, 0.5]},
    }
    return pd.DataFrame(values[kind])


def test_package_skeleton_and_absolute_imports():
    root = Path(__file__).resolve().parents[1]
    packages = ("io", "io/interop", "analysis", "model", "transform", "synth",
                "notation", "viz", "nb")
    for package in packages:
        path = root / "audiospylt" / package / "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.Expr)
        assert isinstance(tree.body[0].value.value, str)
        assert "\n" not in tree.body[0].value.value
        importlib.import_module("audiospylt." + package.replace("/", "."))
    for path in (root / "audiospylt/model/schema.py", root / "audiospylt/io/tables.py"):
        assert all(node.level == 0 for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, ast.ImportFrom))
    assert importlib.import_module("io") is io
    assert io.StringIO("stdlib").read() == "stdlib"
    assert importlib.import_module("audiospylt.io") is not io
    # Existing modules still resolve absolute stdlib imports correctly.
    assert importlib.import_module("audiospylt.audio_utils").tempfile._io is io


def test_contracts_are_frozen_and_mapping_is_immutable():
    assert set(TABLE_CONTRACTS) == {"snapshot", "events", "tracks"}
    with pytest.raises(TypeError):
        TABLE_CONTRACTS["new"] = TABLE_CONTRACTS["snapshot"]
    with pytest.raises(FrozenInstanceError):
        TABLE_CONTRACTS["snapshot"].kind = "new"
    with pytest.raises(FrozenInstanceError):
        TABLE_CONTRACTS["snapshot"].required[0].dtype = "int64"
    assert [column.name for column in TABLE_CONTRACTS["events"].required] == [
        "time_start", "time_stop", "freq_start", "freq_stop", "amp_start", "amp_stop",
    ]


@pytest.mark.parametrize("kind", TABLE_CONTRACTS)
def test_normalization_copy_order_dtypes_index_and_attrs(kind):
    frame = _frame(kind)
    contract = TABLE_CONTRACTS[kind]
    for column in contract.optional:
        frame[column.name] = [1.0, np.nan]
    frame["z_extra"] = ["left", "right"]
    frame["a_extra"] = [2, 3]
    frame = frame[frame.columns[::-1]]
    frame.index = pd.Index([10, 20], name="old")
    frame.attrs = {"audiospylt_table": "wrong", "custom": {"keep": "me"}}
    original = frame.copy(deep=True)
    result = validate(frame, kind)
    canonical = [column.name for column in contract.required + contract.optional]
    assert result.columns.tolist() == canonical + ["a_extra", "z_extra"]
    assert result.index.equals(pd.RangeIndex(2))
    assert result.index.name is None
    for column in contract.required + contract.optional:
        assert str(result[column.name].dtype) == column.dtype
    assert result.attrs == {"audiospylt_table": kind, "custom": {"keep": "me"}}
    assert_frame_equal(frame, original, check_exact=True)
    result.loc[0, canonical[0]] = 100
    result.attrs["custom"]["keep"] = "changed"
    assert_frame_equal(frame, original, check_exact=True)
    assert frame.attrs == original.attrs


@pytest.mark.parametrize("kind", TABLE_CONTRACTS)
def test_empty_tables_valid(kind):
    contract = TABLE_CONTRACTS[kind]
    result = validate(pd.DataFrame(columns=[column.name for column in
                                           contract.required + contract.optional]), kind)
    assert result.empty
    assert result.index.equals(pd.RangeIndex(0))
    assert [str(dtype) for dtype in result.dtypes] == [
        column.dtype for column in contract.required + contract.optional
    ]


REQUIRED = [(kind, column.name) for kind, contract in TABLE_CONTRACTS.items()
            for column in contract.required]
ALL_COLUMNS = [(kind, column.name) for kind, contract in TABLE_CONTRACTS.items()
               for column in contract.required + contract.optional]
NONNEGATIVE = [(kind, column.name) for kind, contract in TABLE_CONTRACTS.items()
               for column in contract.required + contract.optional if column.nonnegative]


@pytest.mark.parametrize("kind,column", REQUIRED)
def test_required_columns_cannot_be_replaced_by_extras(kind, column):
    frame = _frame(kind).drop(columns=column)
    frame[column + "_extra"] = [1, 2]
    with pytest.raises(SchemaError, match=rf"{column}: required column missing"):
        validate(frame, kind)


@pytest.mark.parametrize("kind,column", ALL_COLUMNS)
@pytest.mark.parametrize("bad,rule", [
    ("not a number", "numeric"), (True, "booleans rejected"),
    (1 + 2j, "numeric"), (np.inf, "finite"), (-np.inf, "finite"),
])
def test_numeric_and_finite_rules(kind, column, bad, rule):
    frame = _frame(kind)
    frame[column] = pd.Series([bad, 1], dtype=object)
    with pytest.raises(SchemaError) as caught:
        validate(frame, kind)
    assert f"{column}:" in str(caught.value)
    assert rule in str(caught.value)
    assert "1 row(s); first row positions [0]" in str(caught.value)


@pytest.mark.parametrize("kind,column", REQUIRED)
@pytest.mark.parametrize("missing", [np.nan, pd.NA, None])
def test_required_missing_rejected(kind, column, missing):
    frame = _frame(kind)
    frame[column] = [missing, 1]
    with pytest.raises(SchemaError, match=rf"{column}: NaN/missing"):
        validate(frame, kind)


@pytest.mark.parametrize("kind,column", NONNEGATIVE)
def test_nonnegative_rules(kind, column):
    frame = _frame(kind)
    frame[column] = [-1, 1]
    with pytest.raises(SchemaError, match=rf"{column}: must be >= 0"):
        validate(frame, kind)


@pytest.mark.parametrize("column", ["freq", "extra"])
def test_duplicate_labels_rejected(column):
    frame = pd.DataFrame([[1, 2, 3, 4]], columns=["freq", "amp", column, column])
    with pytest.raises(SchemaError, match="duplicate column labels"):
        validate(frame, "snapshot")


@pytest.mark.parametrize("kind,column", [("snapshot", "bin"), ("tracks", "track_id")])
@pytest.mark.parametrize("bad,rule", [
    (0.5, "integer-valued"), (2**63, "within int64 range"), (-(2**63) - 1, "within int64 range"),
])
def test_integer_rules(kind, column, bad, rule):
    frame = _frame(kind)
    frame[column] = pd.Series([bad, 1], dtype=object)
    with pytest.raises(SchemaError, match=rf"{column}: must be {rule}"):
        validate(frame, kind)


def test_optional_missing_and_integral_values_normalized():
    frame = _frame("snapshot")
    frame["bin"] = [3.0, np.nan]
    frame["phase"] = pd.Series([pd.NA, -2.0], dtype="Float64")
    frame["prominence"] = [np.nan, 0]
    result = validate(frame, "snapshot")
    assert result["bin"].tolist() == [3, pd.NA]
    assert str(result["bin"].dtype) == "Int64"
    assert str(result["phase"].dtype) == "float64"
    assert np.isnan(result.loc[0, "phase"])
    tracks = _frame("tracks")
    tracks["track_id"] = [1.0, 1.0]
    assert str(validate(tracks, "tracks")["track_id"].dtype) == "int64"


def test_int64_boundaries_are_checked_before_float_conversion():
    frame = pd.DataFrame({"track_id": [-(2**63), 2**63 - 1, 2**53 + 1],
                          "time": [0, 0, 0], "freq": [0, 0, 0], "amp": [0, 0, 0]})
    assert validate(frame, "tracks")["track_id"].tolist() == frame["track_id"].tolist()
    snapshot = pd.DataFrame({"freq": [0, 0], "amp": [0, 0],
                             "bin": pd.Series([2**63 - 1, pd.NA], dtype="Int64")})
    assert validate(snapshot, "snapshot")["bin"].tolist() == snapshot["bin"].tolist()


def test_events_decreasing_ramps_unordered_rows_and_zero_duration_allowed():
    frame = _frame("events").iloc[::-1]
    result = validate(frame, "events")
    assert result.loc[0, "amp_start"] > result.loc[0, "amp_stop"]
    assert result.loc[1, "time_start"] == result.loc[1, "time_stop"] == 0


def test_events_stop_before_start_rejected():
    frame = _frame("events")
    frame["time_stop"] = [0, 0.5]
    with pytest.raises(SchemaError, match="time_stop: must be >= time_start.*positions \\[1\\]"):
        validate(frame, "events")


@pytest.mark.parametrize("times", [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 2.0]])
def test_tracks_may_interleave(times):
    frame = pd.DataFrame({"track_id": [-1, 10, -1, 10], "time": times,
                          "freq": [0, 1, 2, 3], "amp": [1, 1, 1, 1]})
    result = validate(frame, "tracks")
    assert result["track_id"].tolist() == [-1, 10, -1, 10]
    assert result["time"].tolist() == times


@pytest.mark.parametrize("time", [0.0, -0.5])
def test_tracks_repeated_or_decreasing_time_rejected_without_sorting(time):
    frame = pd.DataFrame({"track_id": [1, 2, 1], "time": [0, 0, time],
                          "freq": [1, 1, 1], "amp": [1, 1, 1]})
    original = frame.copy()
    with pytest.raises(SchemaError, match="time: must be strictly increasing.*positions \\[2\\]"):
        validate(frame, "tracks")
    assert_frame_equal(frame, original)


def test_all_problems_reported_together_with_counts_and_row_positions():
    frame = pd.DataFrame({"freq": [-1, np.inf, np.nan, False],
                          "bin": pd.Series([-1, 0.5, np.inf, 2**63], dtype=object),
                          "prominence": [-2, np.inf, 0, 0]})
    frame["extra"] = [1] * 4
    frame.insert(len(frame.columns), "extra", [1] * 4, allow_duplicates=True)
    frame.index = [100, 200, 300, 400]
    with pytest.raises(SchemaError) as caught:
        validate(frame, "snapshot")
    message = str(caught.value)
    for problem in ("amp: required column missing", "duplicate column labels", "freq: numeric",
                    "freq: NaN/missing", "freq: finite", "freq: must be >= 0",
                    "bin: must be integer-valued", "bin: finite", "bin: must be >= 0",
                    "bin: must be within int64 range", "prominence: must be >= 0",
                    "prominence: finite"):
        assert problem in message
    assert "first row positions [3]" in message
    assert "100" not in message
    frame = pd.DataFrame({"track_id": [1, 1, 1], "time": [2, 1, 1],
                          "freq": [-1, np.nan, 1], "amp": [1, 1, 1]})
    with pytest.raises(SchemaError) as caught:
        validate(frame, "tracks")
    assert "2 row(s); first row positions [1, 2]" in str(caught.value)
    assert "freq: NaN/missing" in str(caught.value)
    assert "freq: must be >= 0" in str(caught.value)


@pytest.mark.parametrize("kind", TABLE_CONTRACTS)
def test_inference_ignores_attrs_and_tracks_beat_snapshot(kind):
    frame = _frame(kind)
    frame.attrs["audiospylt_table"] = "wrong"
    assert infer_kind(frame) == kind


def test_unknown_and_ambiguous_inference():
    with pytest.raises(SchemaError, match="unknown"):
        infer_kind(pd.DataFrame(columns=["freq", "amplitude"]))
    frame = _frame("events").assign(freq=1, amp=1)
    with pytest.raises(SchemaError, match="ambiguous"):
        infer_kind(frame)
    assert validate(frame, "events").attrs["audiospylt_table"] == "events"
    tracks_and_events = _frame("events").assign(freq=1, amp=1, track_id=1, time=[0, 1])
    with pytest.raises(SchemaError, match="ambiguous.*events, tracks"):
        infer_kind(tracks_and_events)


def test_invalid_kind_and_non_dataframe_errors():
    with pytest.raises(SchemaError, match="Unknown table kind"):
        validate(_frame("snapshot"), "wrong")
    for function in (infer_kind, upgrade):
        with pytest.raises(SchemaError, match="pandas DataFrame"):
            function({"freq": [1], "amp": [1]})
    with pytest.raises(SchemaError, match="pandas DataFrame"):
        validate(None, "snapshot")


@pytest.mark.parametrize("kind,mapping", [
    ("snapshot", {"freq": "Frequency (Hz)", "amp": "Amplitude"}),
    ("events", {"amp_start": "amp_min", "amp_stop": "amp_max"}),
])
def test_upgrade_warning_mapping_stacklevel_and_idempotence(kind, mapping):
    expected = validate(_frame(kind), kind)
    legacy = _frame(kind).rename(columns=mapping)
    original = legacy.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        upgraded = upgrade(legacy)
    assert len(caught) == 1
    assert caught[0].category is LegacyFormatWarning
    assert issubclass(LegacyFormatWarning, FutureWarning)
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert "DataFrame" in str(caught[0].message)
    for new, old in mapping.items():
        assert f"{old!r} -> {new!r}" in str(caught[0].message)
    assert_frame_equal(upgraded, expected, check_exact=True)
    assert_frame_equal(legacy, original, check_exact=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = upgrade(upgraded)
    assert caught == []
    assert_frame_equal(again, upgraded, check_exact=True)
    assert again.attrs == upgraded.attrs


@pytest.mark.parametrize("old,new", [("Frequency (Hz)", "freq"), ("Amplitude", "amp"),
                                       ("amp_min", "amp_start"), ("amp_max", "amp_stop")])
def test_mixed_legacy_and_v1_names_rejected(old, new):
    frame = _frame("events" if old.startswith("amp_") else "snapshot")
    frame[old] = frame[new]
    for function in (upgrade, infer_kind):
        with pytest.raises(SchemaError, match="ambiguous legacy/v1"):
            function(frame)
    with pytest.raises(SchemaError, match="ambiguous legacy/v1"):
        validate(frame, "events" if old.startswith("amp_") else "snapshot")


@pytest.mark.parametrize("dtype,values", [
    ("float64", [np.nan, np.inf, -np.inf, -1.0, 0.5, float(2**63)]),
    ("Float64", [pd.NA, np.inf, -np.inf, -1.0, 0.5, float(2**63)]),
    ("float32", [np.nan, np.inf, -np.inf, -1.0, 0.5, float(2**63)]),
    ("uint64", [0, 2**63 - 1, 2**63, 2**64 - 1]),
    ("UInt64", [0, pd.NA, 2**63, 2**64 - 1]),
    ("Int64", [0, pd.NA, -1, 2**63 - 1]),
    ("bool", [True, False]),
    ("boolean", [True, pd.NA, False]),
    ("complex128", [1 + 2j, complex("nan"), 1 + 0j]),
])
@pytest.mark.parametrize("kind,column", [("snapshot", "freq"), ("snapshot", "bin"),
                                       ("tracks", "track_id")])
def test_native_dtype_matches_object_fallback(dtype, values, kind, column):
    series = pd.Series(values, dtype=dtype)
    frame = pd.concat([_frame(kind).iloc[:1]] * len(series), ignore_index=True)
    frame[column] = series
    if kind == "tracks":
        frame["time"] = np.arange(len(frame), dtype=float)
    try:
        native = validate(frame, kind)
    except SchemaError as error:
        native = str(error)
    frame[column] = frame[column].astype(object)
    if isinstance(native, str):
        with pytest.raises(SchemaError) as fallback:
            validate(frame, kind)
        assert native == str(fallback.value)
    else:
        assert_frame_equal(native, validate(frame, kind), check_exact=True)


@pytest.mark.parametrize("dtype,values", [
    ("int64", [-(2**63), 2**63 - 1, 2**53 + 1]),
    ("uint64", [0, 2**63 - 1, 2**53 + 1]),
    ("Int64", [-(2**63), 2**63 - 1, 2**53 + 1]),
    ("float64", [-1.0, 1.0, float(2**53)]),
])
def test_native_integer_results_identical_to_object_fallback(dtype, values):
    frame = pd.DataFrame({"track_id": pd.Series(values, dtype=dtype), "time": [0, 0, 0],
                          "freq": [1, 1, 1], "amp": [1, 1, 1]})
    expected = validate(frame.astype(object), "tracks")
    result = validate(frame, "tracks")
    assert_frame_equal(result, expected, check_exact=True)
    assert result.track_id.tolist() == values


def test_large_numeric_tracks_use_vectorized_path(monkeypatch):
    def scalar_check_forbidden(value):
        raise AssertionError("numeric dtype validation visited a Python scalar")

    monkeypatch.setattr(schema, "is_number", scalar_check_forbidden)
    rows = 200_000
    frame = pd.DataFrame({"track_id": np.tile(np.arange(1000), rows // 1000),
                          "time": np.repeat(np.arange(rows // 1000, dtype=float), 1000),
                          "freq": np.full(rows, 440.0), "amp": np.full(rows, 0.5),
                          "phase": np.zeros(rows)})
    start = perf_counter()
    result = validate(frame, "tracks")
    elapsed = perf_counter() - start
    # The scalar-path guard pins the implementation; allow ample CI headroom.
    assert elapsed < 2.0, f"200k-row numeric validation took {elapsed:.3f} s"
    assert_frame_equal(result, frame, check_exact=True)


def test_extended_float_ids_keep_precision_before_integer_cast():
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("longdouble has no additional precision on this platform")
    values = np.array([-(2**63), 2**63 - 1, 2**53 + 1], dtype=np.longdouble)
    frame = pd.DataFrame({"track_id": values, "time": [0.0] * 3,
                          "freq": [1.0] * 3, "amp": [1.0] * 3})
    native = validate(frame, "tracks")
    assert native.track_id.tolist() == [-(2**63), 2**63 - 1, 2**53 + 1]
    assert_frame_equal(native, validate(frame.astype(object), "tracks"), check_exact=True)
