"""Versioned TSV interoperability, exact floats, and real legacy inputs."""

from io import BytesIO, StringIO
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from audiospylt.io import tables
from audiospylt.io.tables import read_table, write_table
from audiospylt.model.schema import LegacyFormatWarning, SchemaError, validate


DATA = Path(__file__).resolve().parents[1] / "data" / "tables"


def _frame(kind):
    difficult = [0.1 + 0.2, 1e-300, np.nextafter(1.0, 2.0), 71.42857142857142]
    if kind == "snapshot":
        frame = pd.DataFrame({"freq": difficult, "amp": difficult,
                              "phase": [-0.0, np.nan, -np.pi, np.nextafter(0.0, 1.0)],
                              "bin": pd.Series([0, pd.NA, 2**63 - 1, 2**53 + 1], dtype="Int64"),
                              "prominence": [np.nan, 0, 1e-300, 1]})
    elif kind == "events":
        frame = pd.DataFrame({"time_start": [0, 1e-300, 0.1 + 0.2, 1],
                              "time_stop": [0, 1, np.nextafter(1.0, 2.0), 71.42857142857142],
                              "freq_start": difficult, "freq_stop": difficult[::-1],
                              "amp_start": difficult, "amp_stop": difficult[::-1]})
    else:
        frame = pd.DataFrame({"track_id": [2**63 - 1, -(2**63), 2**53 + 1, 2**63 - 1],
                              "time": [0, 1e-300, 0.1 + 0.2, 71.42857142857142],
                              "freq": difficult, "amp": difficult,
                              "phase": [np.nan, -0.0, -np.pi, np.nextafter(0.0, 1.0)]})
    frame["note"] = ["# cell", 'tab\tquote"', "line\r\nbreak", "NA — 音"]
    frame["extra_float"] = difficult
    return validate(frame, kind)


@pytest.mark.parametrize("kind", ["snapshot", "events", "tracks"])
@pytest.mark.parametrize("empty", [False, True])
def test_exact_roundtrip_all_kinds_optional_extras_metadata_empty(kind, empty, tmp_path):
    frame = _frame(kind)
    if empty:
        frame = frame.iloc[:0].copy()
        # TSV carries no extra-column dtype declaration for an empty table.
        frame["note"] = pd.Series([], dtype=object)
        frame["extra_float"] = pd.Series([], dtype=object)
    frame.attrs["meta"] = {"source": "FFTimbre", "note": "  keep spaces: café  ", "empty": ""}
    original = frame.copy(deep=True)
    destination = tmp_path / f"{kind}.tsv"
    shuffled = frame[list(frame.columns[:-2][::-1]) + list(frame.columns[-2:])]
    assert write_table(shuffled, destination) == destination
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_table(destination)
    assert caught == []
    assert_frame_equal(result, frame, check_exact=True)
    assert result.attrs == frame.attrs
    assert_frame_equal(frame, original, check_exact=True)
    # Include signed zero and subnormal checks: equality alone treats +/-0 as equal.
    for column in ("phase", "freq", "extra_float"):
        if column in frame and not empty:
            np.testing.assert_array_equal(result[column].to_numpy().view("uint64"),
                                          frame[column].to_numpy().view("uint64"))


def test_many_binary64_values_roundtrip_bit_for_bit(tmp_path):
    random = np.random.default_rng(1234)
    bits = random.integers(0, 2**64 - 1, size=1000, dtype=np.uint64)
    bits &= np.uint64(0x7FEFFFFFFFFFFFFF)
    values = bits.view(np.float64)
    frame = pd.DataFrame({"freq": values, "amp": values[::-1]})
    path = write_table(frame, tmp_path / "binary64.tsv")
    result = read_table(path)
    for column in frame:
        np.testing.assert_array_equal(result[column].to_numpy().view("uint64"),
                                      frame[column].to_numpy().view("uint64"))


def test_byte_header_metadata_column_order_and_no_index(tmp_path):
    frame = pd.DataFrame({"extra": ["# inside"], "amp": [0.1 + 0.2], "freq": [440]})
    path = write_table(frame, tmp_path / "bytes.tsv", meta={"source": "café", "take": 2})
    assert path.read_bytes() == (
        "# audiospylt-table: snapshot v1\n# source: café\n# take: 2\n"
        "freq\tamp\textra\n440.0\t0.30000000000000004\t# inside\n"
    ).encode("utf-8")
    assert read_table(path).attrs["meta"] == {"source": "café", "take": "2"}
    assert frame.attrs == {}


@pytest.mark.parametrize("input_kind", ["str", "path", "file", "buffer"])
def test_paths_and_text_streams(input_kind, tmp_path):
    frame = _frame("snapshot")
    frame.attrs["meta"] = {}
    path = write_table(frame, tmp_path / "input.tsv")
    if input_kind == "str":
        result = read_table(str(path))
    elif input_kind == "path":
        result = read_table(path)
    else:
        stream = (path.open(encoding="utf-8", newline="") if input_kind == "file"
                  else StringIO(path.read_bytes().decode("utf-8")))
        try:
            result = read_table(stream)
            assert not stream.closed
            assert stream.read() == ""
        finally:
            stream.close()
    assert_frame_equal(result, frame, check_exact=True)
    assert result.attrs == frame.attrs


def test_buffer_current_position_and_binary_rejection():
    stream = StringIO("skip\n# audiospylt-table: snapshot v1\nfreq\tamp\n1\t2\n")
    stream.readline()
    assert read_table(stream).loc[0, "freq"] == 1
    with pytest.raises(TypeError, match="text stream"):
        read_table(BytesIO(b"freq\tamp\n1\t2\n"))


def test_kind_argument_attrs_and_inference_precedence(tmp_path):
    frame = _frame("snapshot")
    frame.attrs["audiospylt_table"] = "events"
    with pytest.raises(SchemaError, match="required column missing"):
        write_table(frame, tmp_path / "bad.tsv")
    assert not (tmp_path / "bad.tsv").exists()
    path = write_table(frame, tmp_path / "good.tsv", "snapshot")
    assert read_table(path, "snapshot").attrs["audiospylt_table"] == "snapshot"
    with pytest.raises(SchemaError, match="disagrees"):
        read_table(path, "events")
    frame.attrs.clear()
    assert read_table(write_table(frame, tmp_path / "inferred.tsv")).attrs["audiospylt_table"] == "snapshot"


def test_metadata_override_and_attrs_fallback(tmp_path):
    frame = _frame("snapshot")
    frame.attrs["meta"] = {"source": "attrs"}
    assert read_table(write_table(frame, tmp_path / "attrs.tsv")).attrs["meta"] == {"source": "attrs"}
    assert read_table(write_table(frame, tmp_path / "override.tsv", meta={})).attrs["meta"] == {}
    result = read_table(write_table(frame, tmp_path / "values.tsv", meta={"x": None, "y": True}))
    assert result.attrs["meta"] == {"x": "None", "y": "True"}


@pytest.mark.parametrize("meta", [
    {"Bad": "x"}, {"1bad": "x"}, {"bad-key": "x"}, {"": "x"}, {1: "x"},
    {"good": "bad\nvalue"}, {"good": "bad\rvalue"}, {"good": "bad\tvalue"},
])
def test_bad_writer_metadata_does_not_touch_destination(meta, tmp_path):
    path = tmp_path / "existing.tsv"
    path.write_bytes(b"keep me")
    with pytest.raises(SchemaError, match="[Mm]etadata"):
        write_table(_frame("snapshot"), path, meta=meta)
    assert path.read_bytes() == b"keep me"


@pytest.mark.parametrize("existing", [False, True])
def test_validation_completes_before_opening_file(existing, tmp_path):
    path = tmp_path / "invalid.tsv"
    if existing:
        path.write_bytes(b"keep")
    frame = _frame("snapshot")
    frame["amp"] = -1
    with pytest.raises(SchemaError, match="amp: must be >= 0"):
        write_table(frame, path)
    assert path.read_bytes() == b"keep" if existing else not path.exists()
    with pytest.raises(SchemaError, match="pandas DataFrame"):
        write_table(None, path)


def test_no_parent_directory_created(tmp_path):
    path = tmp_path / "missing" / "file.tsv"
    with pytest.raises(FileNotFoundError):
        write_table(_frame("snapshot"), path)
    assert not path.parent.exists()


@pytest.mark.parametrize("preamble,match", [
    ("# audiospylt-table: snapshot v2\n", "Unsupported.*v2"),
    ("# audiospylt-table: snapshot v0\n", "Unsupported.*v0"),
    ("# audiospylt-table: snapshot v1.1\n", "Unsupported.*v1.1"),
    ("# audiospylt-table: unknown v1\n", "Unknown table kind.*unknown"),
    ("# audiospylt-table:  snapshot v1\n", "Malformed.*type line"),
    ("# audiospylt-table snapshot v1\n", "Malformed.*type line"),
    ("# audiospylt-table: snapshot\n", "Malformed.*type line"),
    ("# audiospylt-table: snapshot v1 \n", "Malformed.*type line"),
    ("# audiospylt-table: snapshot\tv1\n", "Malformed.*type line"),
    ("# audiospylt-table: snapshot v1\n# a: x\n# a: y\n", "Duplicate metadata key.*a"),
    ("# audiospylt-table: snapshot v1\n# Bad: x\n", "Invalid metadata line/key"),
    ("# audiospylt-table: snapshot v1\n# bad-key: x\n", "Invalid metadata line/key"),
    ("# audiospylt-table: snapshot v1\n# good: x\ty\n", "Invalid metadata line/key"),
    ("# audiospylt-table: snapshot v1\n# good: x\ry\n", "Invalid metadata line/key"),
    ("# audiospylt-table: snapshot v1\n# good:no space\n", "Invalid metadata line/key"),
    ("# audiospylt-table: snapshot v1\n# arbitrary comment\n", "no other comments"),
    ("# arbitrary comment\n", "Unexpected comment"),
])
def test_header_errors(preamble, match):
    with pytest.raises(SchemaError, match=match):
        read_table(StringIO(preamble + "freq\tamp\n1\t2\n"))


@pytest.mark.parametrize("payload,match", [
    ("freq\tamp\n1\t2\n# note: late\n", "Comment lines"),
    ("freq\tamp\n1\t2\t3\n", "cell count"),
    ("freq\tamp\n1\n", "cell count"),
    ("freq\tamp\n\n1\t2\n", "cell count"),
    ("freq\tamp\tamp\n1\t2\t3\n", "duplicate column labels"),
    ('freq\tamp\n"unterminated\t2\n', "Malformed TSV"),
    ("", "Missing column-name row"),
])
def test_malformed_tsv_rejected(payload, match):
    with pytest.raises(SchemaError, match=match):
        read_table(StringIO("# audiospylt-table: snapshot v1\n" + payload))


@pytest.mark.parametrize("label", ["#label", "# label"])
@pytest.mark.parametrize("versioned", [False, True])
def test_hash_in_data_is_not_a_comment_even_in_first_cell(label, versioned):
    prefix = "# audiospylt-table: snapshot v1\n" if versioned else ""
    result = read_table(StringIO(
        prefix + "extra\tfreq\tamp\n# comment-like data\t1\t2\n"
    ))
    assert result.loc[0, "extra"] == "# comment-like data"
    assert read_table(StringIO(prefix + f"{label}\tfreq\tamp\nx\t1\t2\n"))[label].tolist() == ["x"]


def test_v1_values_are_validated_and_never_legacy_upgraded():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SchemaError, match="freq: required column missing"):
            read_table(StringIO("# audiospylt-table: snapshot v1\nFrequency (Hz)\tAmplitude\n1\t2\n"))
    assert caught == []
    with pytest.raises(SchemaError, match="amp: must be >= 0"):
        read_table(StringIO("# audiospylt-table: snapshot v1\nfreq\tamp\n1\t-1\n"))
    with pytest.raises(SchemaError, match="bin: must be integer-valued"):
        read_table(StringIO("# audiospylt-table: snapshot v1\nfreq\tamp\tbin\n1\t1\t0.5\n"))


def test_headerless_canonical_input_is_silent_and_accepts_explicit_kind():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_table(StringIO("freq\tamp\n0\t1\n"))
    assert caught == []
    assert result.attrs == {"audiospylt_table": "snapshot", "meta": {}}
    with pytest.raises(SchemaError, match="required column missing"):
        read_table(StringIO("freq\tamp\n0\t1\n"), "events")


def test_legacy_buffer_warning_stacklevel_and_name():
    stream = StringIO("Frequency (Hz)\tAmplitude\n1\t2\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_table(stream)
    assert len(caught) == 1
    assert caught[0].category is LegacyFormatWarning
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert "<buffer>" in str(caught[0].message)
    assert "'Frequency (Hz)' -> 'freq'" in str(caught[0].message)
    assert "'Amplitude' -> 'amp'" in str(caught[0].message)


@pytest.mark.parametrize("versioned", [False, True])
def test_mixed_old_and_new_file_names_rejected(versioned):
    prefix = "# audiospylt-table: snapshot v1\n" if versioned else ""
    with pytest.raises(SchemaError, match="ambiguous legacy/v1"):
        read_table(StringIO(prefix + "freq\tamp\tAmplitude\n1\t2\t3\n"))


LEGACY = [
    ("bayle1.tsv", "snapshot", 64, None, None),
    ("cello_single.tsv", "snapshot", 14, None, None),
    ("conga3.tsv", "snapshot", 10, None, None),
    ("speech1.tsv", "snapshot", 41, None, None),
    ("conga_multidft.tsv", "events", 87, 0, 0),
    ("conga_multidft2_dropped.tsv", "events", 70, 0, 1),
    ("sigmoid_high.tsv", "events", 41, 32, 0),
    ("sigmoid_low.tsv", "events", 41, 26, 0),
]


def test_legacy_fixture_inventory_is_complete():
    assert {path.name for path in DATA.glob("*.tsv")} == {row[0] for row in LEGACY}

@pytest.mark.parametrize("version,expected", [
    ("2.2.3", {"float_precision": "round_trip"}),
    ("3.0.3", {"float_precision": "round_trip"}),
    ("3.1.0", {}),
    ("3.1.0rc0", {}),
    ("3.2.0.dev0+12.gabc", {}),
    ("10.0.0", {}),
    ("unknown", {"float_precision": "round_trip"}),
])
def test_float_precision_only_before_pandas_3_1(version, expected):
    assert tables._float_precision_options(version) == expected


@pytest.mark.parametrize("version", ["3.0.3", "3.1.0"])
def test_reader_passes_float_options_for_both_pandas_branches(version, monkeypatch, tmp_path):
    # Each branch runs whatever pandas is installed; the argument is checked
    # at the call, then dropped so a newer pandas does not warn about it.
    options = tables._float_precision_options(version)
    monkeypatch.setattr(tables, "_FLOAT_OPTIONS", options)
    real_read_csv = pd.read_csv
    seen = []

    def recording_read_csv(*args, **kwargs):
        seen.append(kwargs.pop("float_precision", None))
        return real_read_csv(*args, **kwargs)

    values = np.array([0.1, 1 / 3, 5e-324, -0.0, 1.7976931348623157e308])
    frame = pd.DataFrame({"freq": values, "amp": values[::-1]})
    path = write_table(frame, tmp_path / "branch.tsv")
    monkeypatch.setattr(tables.pd, "read_csv", recording_read_csv)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = read_table(path)
    assert seen == [options.get("float_precision")]
    for column in frame:
        np.testing.assert_array_equal(result[column].to_numpy().view("uint64"),
                                      frame[column].to_numpy().view("uint64"))


@pytest.mark.parametrize("filename,kind,rows,decays,zero_length", LEGACY)
def test_real_legacy_fixtures_and_versioned_roundtrip(filename, kind, rows, decays, zero_length, tmp_path):
    source = DATA / filename
    raw = pd.read_csv(source, sep="\t", **tables._FLOAT_OPTIONS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        upgraded = read_table(source)
    assert len(caught) == 1
    assert caught[0].category is LegacyFormatWarning
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert str(source) in str(caught[0].message)
    assert upgraded.attrs == {"audiospylt_table": kind, "meta": {}}
    assert len(upgraded) == rows
    mapping = {"Frequency (Hz)": "freq", "Amplitude": "amp",
               "amp_min": "amp_start", "amp_max": "amp_stop"}
    for column in raw:
        assert_series_equal(upgraded[mapping.get(column, column)], raw[column],
                            check_names=False, check_exact=True)
        if column in mapping:
            assert f"{column!r} -> {mapping[column]!r}" in str(caught[0].message)
    if kind == "events":
        assert int((upgraded.amp_start > upgraded.amp_stop).sum()) == decays
        assert int((upgraded.time_start == upgraded.time_stop).sum()) == zero_length
    path = write_table(upgraded, tmp_path / filename)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reread = read_table(path)
    assert caught == []
    assert_frame_equal(reread, upgraded, check_exact=True)
    assert reread.attrs == upgraded.attrs


@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("bom", ["", "\ufeff"])
@pytest.mark.parametrize("trailing", [0, 1, 3])
@pytest.mark.parametrize("input_kind", ["path", "buffer"])
def test_reader_tolerates_bom_line_endings_and_trailing_empty_lines(
    versioned, newline, bom, trailing, input_kind, tmp_path,
):
    preamble = (f"# audiospylt-table: snapshot v1{newline}"
                f"# source: FFTimbre{newline}# spaces:  café  {newline}") if versioned else ""
    payload = f"freq\tamp\tnote{newline}1\t0.30000000000000004\t\"CR\r\nLF\nCR\rend\"{newline}"
    content = bom + preamble + payload + newline * trailing
    if input_kind == "path":
        source = tmp_path / "tolerant.tsv"
        source.write_bytes(content.encode("utf-8"))
    else:
        source = StringIO(content)
    result = read_table(source)
    assert result.freq.tolist() == [1.0]
    assert result.amp.tolist() == [0.1 + 0.2]
    assert result.note.tolist() == ["CR\r\nLF\nCR\rend"]
    assert result.attrs == {"audiospylt_table": "snapshot", "meta":
                            {"source": "FFTimbre", "spaces": " café  "} if versioned else {}}
    destination = write_table(result, tmp_path / "strict.tsv")
    written = destination.read_bytes()
    assert written.startswith(b"# audiospylt-table: snapshot v1\n")
    assert not written.startswith(b"\xef\xbb\xbf")
    assert not written.endswith(b"\n\n")
    # Record terminators remain LF; CR/LF inside a quoted value is content.
    assert written.count(b"\r\n") == 1
    assert_frame_equal(read_table(destination), result, check_exact=True)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("preamble", ["", "# audiospylt-table: snapshot v1\n",
                                    "# audiospylt-table: snapshot v1\n# key: value\n"])
@pytest.mark.parametrize("position", ["before_header", "between_rows"])
def test_interior_empty_lines_still_rejected(newline, preamble, position):
    preamble = preamble.replace("\n", newline)
    payload = ("\nfreq\tamp\n1\t2\n" if position == "before_header"
               else "freq\tamp\n1\t2\n\n3\t4\n")
    with pytest.raises(SchemaError, match="column-name row|cell count"):
        read_table(StringIO(preamble + payload.replace("\n", newline)))


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_metadata_tabs_rejected_before_column_header_detection(newline):
    content = newline.join(["# audiospylt-table: snapshot v1", "# source: bad\tvalue",
                            "freq\tamp", "1\t2", ""])
    with pytest.raises(SchemaError, match="metadata"):
        read_table(StringIO(content))


@pytest.mark.parametrize("versioned", [False, True])
def test_bad_numeric_cells_aggregate_after_native_float_parsing(versioned):
    prefix = "# audiospylt-table: snapshot v1\n" if versioned else ""
    with pytest.raises(SchemaError) as caught:
        read_table(StringIO(prefix + "freq\tamp\n1\t-1\nbad\tinf\n-2\t3\n"))
    message = str(caught.value)
    assert "freq: numeric real values required (booleans rejected) (1 row(s); first row positions [1])" in message
    assert "freq: must be >= 0 (1 row(s); first row positions [2])" in message
    assert "amp: finite" in message
    assert "amp: must be >= 0" in message


@pytest.mark.parametrize("versioned", [False, True])
def test_integer_file_tokens_never_round_through_binary64(versioned):
    prefix = "# audiospylt-table: tracks v1\n" if versioned else ""
    result = read_table(StringIO(prefix + "track_id\ttime\tfreq\tamp\n"
                                "9007199254740993\t0\t1\t1\n"
                                "9223372036854775807.0\t0\t1\t1\n"
                                "-9223372036854775808e0\t0\t1\t1\n"))
    assert result.track_id.tolist() == [2**53 + 1, 2**63 - 1, -(2**63)]
    assert str(result.track_id.dtype) == "int64"


@pytest.mark.parametrize("bom", ["", "\ufeff"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("input_kind", ["path", "buffer"])
def test_legacy_names_upgrade_with_tolerated_file_variants(bom, newline, input_kind, tmp_path):
    content = bom + newline.join(["Frequency (Hz)\tAmplitude", "1\t2", "", ""])
    if input_kind == "path":
        source = tmp_path / "legacy.tsv"
        source.write_bytes(content.encode("utf-8"))
    else:
        source = StringIO(content)
    with pytest.warns(LegacyFormatWarning) as caught:
        result = read_table(source)
    assert len(caught) == 1
    assert_frame_equal(result, pd.DataFrame({"freq": [1.0], "amp": [2.0]}), check_exact=True)


def test_mixed_lf_and_crlf_record_endings():
    result = read_table(StringIO("\ufeff# audiospylt-table: snapshot v1\r\n"
                                "# first: keep  \n# second: value\r\n"
                                "freq\tamp\r\n1\t2\n\r\n\n"))
    assert result.attrs["meta"] == {"first": "keep  ", "second": "value"}
    assert result.loc[0, "freq"] == 1.0
