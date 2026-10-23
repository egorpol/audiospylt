"""SDIF partial tracks: read 1TRC/1HRM/RBEP, write 1TRC, read AudioSculpt chord sequences (D24)."""

from pathlib import Path
import struct

import numpy as np
import pandas as pd
import pytest

import audiospylt
from audiospylt.io.interop.partiels import read_partiels
from audiospylt.io.interop.sdif import read_chord_seq, read_sdif, write_sdif
from audiospylt.model.schema import SchemaError
from audiospylt.notation.mei import events_to_mei

DATA = Path(__file__).resolve().parent / "data"
SDIF = DATA / "sdif"
KRAFT = SDIF / "kraft_snare_mono-trc_45hz.sdif"
PHEEK = SDIF / "pheek_bdrum-trc.sdif"
CHORDS = SDIF / "kraft_snare_mono-cs.sdif"
LORIS = SDIF / "loris_flute_rbep.sdif"
NEG_MAX = -np.finfo(np.float64).max
DTYPES = {0x0004: ">f4", 0x0008: ">f8", 0x0104: ">i4"}


# Independent low-level writer for synthetic files.
def text_matrix(sig, text):
    data = text.encode() + b"\0"
    return sig.encode() + struct.pack(">iii", 0x0301, len(data), 1) + data + b"\0" * (-len(data) % 8)


def matrix(sig, rows, code=0x0004, cols=4):
    values = np.asarray(rows, dtype=DTYPES[code]).reshape(len(rows), -1 if len(rows) else cols)
    data = values.tobytes()
    return sig.encode() + struct.pack(">iii", code, *values.shape) + data + b"\0" * (-len(data) % 8)


def frame(sig, time, *matrices, stream=0, size=None):
    body = struct.pack(">dii", time, stream, len(matrices)) + b"".join(matrices)
    return sig.encode() + struct.pack(">I", len(body) if size is None else size) + body


def sdif_file(path, *frames, nvt="TableName\tSinusoidalTracks\nWrittenBy\ttest\n"):
    head = b"SDIF" + struct.pack(">III", 8, 3, 1)
    if nvt is not None:
        head += frame("1NVT", NEG_MAX, text_matrix("1NVT", nvt), stream=-3)
    path.write_bytes(head + b"".join(frames))
    return path


def trc(time, rows, sig="1TRC", **kw):
    return frame(sig, time, matrix(sig, rows), **kw)


def parse(path):
    """Independent walk of a written file: [(sig, time, stream, [(msig, code, array or text)])]."""
    data = Path(path).read_bytes()
    assert data[:16] == b"SDIF" + struct.pack(">III", 8, 3, 1)
    pos, frames = 16, []
    while pos < len(data):
        sig = data[pos:pos + 4].decode()
        (size,) = struct.unpack_from(">I", data, pos + 4)
        time, stream, count = struct.unpack_from(">dii", data, pos + 8)
        at, matrices = pos + 24, []
        for _ in range(count):
            msig = data[at:at + 4].decode()
            code, rows, cols = struct.unpack_from(">iii", data, at + 4)
            raw = data[at + 16:at + 16 + rows * cols * (1 if code == 0x0301 else code & 0xFF)]
            value = raw.decode().rstrip("\0") if code == 0x0301 else np.frombuffer(raw, DTYPES[code]).reshape(rows, cols)
            matrices.append((msig, code, value))
            at += 16 + len(raw) + (-len(raw) % 8)
        assert at == pos + 8 + size
        frames.append((sig, time, stream, matrices))
        pos = at
    return frames


def points(table):
    """{track_id: [(time, freq, amp, phase)]} for comparisons."""
    return {k: list(map(tuple, g[["time", "freq", "amp", "phase"]].to_numpy()))
            for k, g in table.groupby("track_id", sort=True)}


# Real files

def test_audiosculpt_1trc_reads_pm2_partials():
    tracks = read_sdif(KRAFT)
    meta = tracks.attrs["meta"]
    # Counts from the D24 prototype reader (revision/step-2.3/sdif-facts.md); pm2 numbers partials from 1.
    assert tracks["track_id"].nunique() == 71 and len(tracks) == 320
    assert tracks.groupby("track_id")["partial_id"].nunique().eq(1).all()
    assert tracks["partial_id"].nunique() == 71
    assert meta["frames"] == 28  # including pm2's empty first frame at -1.1e-5 s
    assert {k: meta[k] for k in ("source", "file", "matrix", "stream", "streams", "table_name", "written_by")} == {
        "source": "sdif", "file": KRAFT.name, "matrix": "1TRC", "stream": 0, "streams": [0],
        "table_name": "SinusoidalTracks", "written_by": "Pm2_Version_unknown"}
    assert meta["nvt"] == [{"Date": "Sun Jul 22 01:43:16 2012", "TableName": "SinusoidalTracks",
                            "WrittenBy": "Pm2_Version_unknown"}]
    first = tracks[tracks["partial_id"] == 1]
    assert first["time"].iloc[:2].tolist() == [0.05552154195011338, 0.069421768707483]
    assert first[["freq", "amp", "phase"]].iloc[0].tolist() == [123.98333740234375, 0.030593613162636757,
                                                               -0.5646479725837708]
    assert first["freq"].iloc[1] == 122.62385559082031
    starts = tracks.groupby("track_id")["time"].first()
    assert starts.is_monotonic_increasing and starts.index.tolist() == list(range(71))
    assert tracks.groupby("track_id")["time"].apply(lambda t: t.is_monotonic_increasing).all()


def test_stereo_analysis_reads_one_stream():
    left = read_sdif(PHEEK)
    right = read_sdif(PHEEK, stream=1)
    assert (left["track_id"].nunique(), len(left)) == (69, 245)
    assert (right["track_id"].nunique(), len(right)) == (64, 243)
    assert left.attrs["meta"]["streams"] == right.attrs["meta"]["streams"] == [0, 1]
    assert right.attrs["meta"]["stream"] == 1
    with pytest.raises(ValueError, match=r"no 1TRC frames in stream 2; streams with 1TRC frames: \[0, 1\]"):
        read_sdif(PHEEK, stream=2)


def test_chord_sequence_reads_as_events():
    events = read_chord_seq(CHORDS)
    meta = events.attrs["meta"]
    assert len(events) == 235
    assert events.groupby(["time_start", "time_stop"]).ngroups == 18
    assert (events["freq_start"] == events["freq_stop"]).all() and (events["amp_start"] == events["amp_stop"]).all()
    assert meta["table_name"] == "ChordSeq" and meta["stream"] == 0 and len(meta["markers"]) == 19
    assert meta["markers"][:2] == [0.003765664529055, 0.01300198584795]
    one = events[events["partial_id"] == 1].iloc[0]
    assert one[["time_start", "time_stop", "freq_start", "amp_start"]].tolist() == [
        0.003765664529055, 0.01300198584795, 139.11026000976562, 0.010335189290344715]
    assert events.loc[events["partial_id"] == 16, "time_start"].item() == 0.01300198584795
    assert events["time_start"].is_monotonic_increasing
    assert "<mei" in events_to_mei(events)
    with pytest.raises(ValueError, match=r"chord sequence .*read_chord_seq\(\)"):
        read_sdif(CHORDS)
    with pytest.raises(ValueError, match="holds no chord sequence"):
        read_chord_seq(KRAFT)


def test_loris_rbep_point_time_is_frame_time_plus_offset():
    tracks = read_sdif(LORIS)
    meta = tracks.attrs["meta"]
    assert meta["matrix"] == "RBEP" and meta["nvt"] == [{"creator": "pysdif3"}] and "written_by" not in meta
    assert list(tracks.columns) == ["track_id", "time", "freq", "amp", "phase", "partial_id", "bandwidth"]
    # Breakpoints printed by tests/data/sdif/make_loris.py from loristrck's partial arrays.
    expected = {
        0: [(1.1064581222536378, 461.4333587574246, 0.0011034824897868448, -1.1117198136516144, 0.0),
            (1.1152464731642282, 462.40985488762493, 0.0010881126250313676, -0.7376869826025931, 0.0)],
        1: [(1.1059669177478326, 664.9021036480744, 0.024381430084143122, -2.0162714077911303, 0.0),
            (1.1150617190633008, 665.8301381681281, 0.023302386035634604, -1.6934873917567117, 0.0)],
        104: [(1.1371283191383474, 5259.436355763283, 7.005671254349595e-05, 3.3437447487298186,
               0.31074444448985394),
              (1.14155823301585, 5251.459464126235, 0.0001367936822925796, -1.1727242799999544,
               0.2553271402797403)],
    }
    for index, rows in expected.items():
        got = tracks.loc[tracks["partial_id"] == index, ["time", "freq", "amp", "phase", "bandwidth"]]
        np.testing.assert_allclose(got.to_numpy()[:2], rows, rtol=1e-15, atol=0)
    # RBEP breakpoints land in irregular frames: one track per index, never split at missing frames.
    assert tracks.groupby("partial_id")["track_id"].nunique().eq(1).all()
    assert tracks["track_id"].nunique() == tracks["partial_id"].nunique() <= 209
    assert len(tracks) < 778  # loristrck 1.7.2 never writes its last frame group
    split = read_sdif(LORIS, max_step_cents=1.0)
    assert split["track_id"].nunique() > tracks["track_id"].nunique() and len(split) == len(tracks)


# Synthetic files

def test_1hrm_splits_a_harmonic_at_gaps_and_counts_empty_frames(tmp_path):
    h = {1: [100.0, 101, 102, 103, 104, 105], 2: [200.0, 202, None, 206, 208, 210], 3: [None, None, 300, None, None, None]}
    frames = []
    times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    for k, t in enumerate(times):
        rows = [[n, f[k], 0.1 / n, 0.0] for n, f in h.items() if f[k] is not None]
        frames.append(trc(t, rows if k != 4 else [], sig="1HRM"))  # frame 4 is empty: a gap for all
    order = [0, 2, 1, 3, 5, 4]  # file order differs from time order
    tracks = read_sdif(sdif_file(tmp_path / "h.sdif", *[frames[k] for k in order],
                                 nvt="TableName\tHarmonicTracks\n"))
    got = {(int(g["partial_id"].iloc[0]), tuple(g["time"])) for _, g in tracks.groupby("track_id")}
    assert got == {(1, (0.0, 0.01, 0.02, 0.03)), (1, (0.05,)), (2, (0.0, 0.01)), (2, (0.03,)), (2, (0.05,)),
                   (3, (0.02,))}
    assert tracks.attrs["meta"]["frames"] == 6 and tracks.attrs["meta"]["table_name"] == "HarmonicTracks"
    assert tracks.groupby("track_id")["time"].first().is_monotonic_increasing


def test_max_step_cents_and_track_numbering(tmp_path):
    path = sdif_file(tmp_path / "j.sdif",
                     trc(0.0, [[1, 100.0, 0.1, 0], [2, 50.0, 0.1, 0]]),
                     trc(0.01, [[1, 101.0, 0.1, 0], [2, 50.0, 0.1, 0]]),
                     trc(0.02, [[1, 202.0, 0.1, 0], [2, 50.0, 0.1, 0]]))
    assert read_sdif(path)["track_id"].nunique() == 2
    split = read_sdif(path, max_step_cents=50)
    # Numbered by start time, then start frequency: 50 Hz before 100 Hz at t = 0.
    assert [tuple(g["freq"]) for _, g in split.groupby("track_id")] == [(50.0, 50.0, 50.0), (100.0, 101.0),
                                                                          (202.0,)]
    assert split.attrs["meta"]["max_step_cents"] == 50


def test_times_below_zero_read_as_zero_keeping_the_later_point(tmp_path):
    path = sdif_file(tmp_path / "n.sdif", trc(-3e-5, []), trc(-2e-5, [[1, 100.0, 0.1, 0]]),
                     trc(-1e-5, [[1, 110.0, 0.2, 0]]), trc(0.01, [[1, 120.0, 0.3, 0]]))
    tracks = read_sdif(path)
    assert tracks[["time", "freq"]].to_numpy().tolist() == [[0.0, 110.0], [0.01, 120.0]]


def test_frame_size_is_ignored_and_old_ascii_chunks_are_skipped(tmp_path):
    chunk = b"{\n  TableName Old;\n}\n" + b"\0" * 4
    path = sdif_file(tmp_path / "s.sdif", b"1NVT" + struct.pack(">I", len(chunk)) + chunk,
                     trc(0.0, [[1, 100.0, 0.1, 0]], size=4000),  # wrong size, as in IRCAM's fall.sdif
                     trc(0.01, [[1, 100.0, 0.1, 0]]), nvt=None)
    tracks = read_sdif(path)
    assert tracks["time"].tolist() == [0.0, 0.01] and tracks.attrs["meta"]["nvt"] == []
    assert "table_name" not in tracks.attrs["meta"]


def test_numeric_types_and_extra_columns(tmp_path):
    path = sdif_file(tmp_path / "d.sdif",
                     frame("1TRC", 0.0, matrix("1TRC", [[3, 100.0, 0.1, 0.5, 9.0]], code=0x0008)),
                     frame("1TRC", 0.01, matrix("1TRC", [[3, 100.0, 0.1, 0.5]]), matrix("XTRA", [[1]], 0x0104)),
                     frame("1FQ0", 0.0, matrix("1FQ0", [[100.0, 1.0]])))
    tracks = read_sdif(path)
    assert tracks["partial_id"].tolist() == [3, 3] and tracks["phase"].tolist() == [0.5, 0.5]
    empty = frame("1TRC", 0.005, b"1TRC" + struct.pack(">iii", 0x0004, 0, 0))  # a 0 x 0 matrix is a gap
    split = read_sdif(sdif_file(tmp_path / "z.sdif", trc(0.0, [[3, 100.0, 0.1, 0]]), empty,
                                trc(0.01, [[3, 100.0, 0.1, 0]])))
    assert split["track_id"].tolist() == [0, 1] and split.attrs["meta"]["frames"] == 3


@pytest.mark.parametrize(("frames", "kw", "message"), [
    ([trc(0.0, [[1, 100.0, 0.1, 0]]), trc(0.0, [[1, 100.0, 0.1, 0]], sig="1HRM")], {}, "holds 1HRM, 1TRC; choose one"),
    ([trc(0.0, [[1, 100.0, 0.1, 0]])], {"matrix": "RBEP"}, "holds no RBEP frames, only 1TRC"),
    ([frame("1FQ0", 0.0, matrix("1FQ0", [[100.0, 1.0]]))], {}, r"holds no partial tracks .*\['1FQ0'\]"),
    ([frame("1TRC", 0.0, matrix("1TRC", [[1, 100.0, 0.1]]))], {}, "has 3 columns; expected 4"),
    ([frame("RBEP", 0.0, matrix("RBEP", [[1, 100.0, 0.1, 0, 0]]))], {}, "has 5 columns; expected 6"),
    ([trc(0.0, [[1.5, 100.0, 0.1, 0]])], {}, "index must be an integer, found 1.5"),
    ([trc(0.0, [[1, 100.0, 0.1, 0], [1, 200.0, 0.1, 0]])], {}, "index 1 appears twice in the frame at 0 s"),
    ([trc(0.0, [[1, 100.0, 0.1, 0]]), trc(0.0, [[1, 100.0, 0.1, 0]])], {}, "two points at 0 s"),
    ([frame("1TRC", 0.0, matrix("1TRC", [[1, 100.0, 0.1, 0]]).replace(b"\0\0\0\x04", b"\0\0\x09\x99", 1))], {},
     "unknown data type 0x0999"),
    ([frame("1TRC", 0.0, text_matrix("1TRC", "x"))], {}, "holds text"),
])
def test_read_errors(tmp_path, frames, kw, message):
    with pytest.raises(ValueError, match=message):
        read_sdif(sdif_file(tmp_path / "e.sdif", *frames), **kw)


def test_file_and_argument_errors(tmp_path):
    good = sdif_file(tmp_path / "g.sdif", trc(0.0, [[1, 100.0, 0.1, 0]]), trc(0.01, [[1, 100.0, 0.1, 0]]))
    data = good.read_bytes()
    (tmp_path / "riff.sdif").write_bytes(b"RIFF" + data[4:])
    with pytest.raises(ValueError, match="not an SDIF file"):
        read_sdif(tmp_path / "riff.sdif")
    for cut in (len(data) - 4, len(data) - 30, len(data) - 46):
        (tmp_path / "cut.sdif").write_bytes(data[:cut])
        with pytest.raises(ValueError, match="truncated"):
            read_sdif(tmp_path / "cut.sdif")
    with pytest.raises(FileNotFoundError):
        read_sdif(tmp_path / "missing.sdif")
    for kw, message in [({"matrix": "1FQ0"}, "matrix must be one of"), ({"stream": "0"}, "stream must be an int"),
                        ({"stream": True}, "stream must be an int"), ({"max_step_cents": 0}, "positive")]:
        with pytest.raises(ValueError, match=message):
            read_sdif(good, **kw)
    with pytest.raises(ValueError, match="stream must be an int"):
        read_chord_seq(CHORDS, stream=0.0)


def mark(time, begin=(), end=(), rows=()):
    matrices = []
    if end:
        matrices.append(matrix("1END", [[k] for k in end], code=0x0104))
    if begin:
        matrices.append(matrix("1BEG", [[k] for k in begin], code=0x0104))
    if rows:
        matrices.append(matrix("1TRC", rows))
    return frame("1MRK", time, *matrices)


def test_chord_sequence_synthetic_and_errors(tmp_path):
    good = sdif_file(tmp_path / "c.sdif", mark(-0.01, [1, 2], rows=[[1, 100.0, 0.1, 0], [2, 300.0, 0.2, 0]]),
                     mark(0.5, [3], [1], rows=[[3, 200.0, 0.3, 0]]), mark(1.0, end=[2, 3]),
                     nvt="TableName\tChordSeq\n")
    events = read_chord_seq(good)
    np.testing.assert_allclose(events[["time_start", "time_stop", "freq_start", "amp_start"]], [
        [0.0, 0.5, 100.0, 0.1], [0.0, 1.0, 300.0, 0.2], [0.5, 1.0, 200.0, 0.3]], rtol=1e-7)
    assert events["partial_id"].tolist() == [1, 2, 3] and events.attrs["meta"]["markers"] == [0.0, 0.5, 1.0]
    cases = [
        ((mark(0.0, [1], rows=[[1, 100.0]]),), "has 2 columns; expected index, frequency, amplitude"),
        ((mark(0.0, [1], rows=[[1, 100.0, 0.1, 0]]),), "1 partial\\(s\\) never end"),
        ((mark(0.0, [1], rows=[[1, 100.0, 0.1, 0]]), mark(0.5, end=[1, 2])), "partial 2 ends at 0.5 s without"),
        ((mark(0.0, [1], rows=[[1, 100.0, 0.1, 0]]), mark(0.5, [1], rows=[[1, 100.0, 0.1, 0]])),
         "partial 1 starts again at 0.5 s"),
    ]
    for frames, message in cases:
        with pytest.raises(ValueError, match=message):
            read_chord_seq(sdif_file(tmp_path / "bad.sdif", *frames))


# Writer

def test_write_partiels_flute_round_trip_and_layout(tmp_path):
    tracks = read_partiels(DATA / "partiels" / "flute_json")
    path = write_sdif(tracks, tmp_path / "flute.sdif", nvt={"Date": "today", "Bins": 7})
    assert path == tmp_path / "flute.sdif"
    frames = parse(path)
    assert frames[0][:3] == ("1NVT", NEG_MAX, -3)
    assert frames[0][3][0][2] == (f"TableName\tSinusoidalTracks\nWrittenBy\taudiospylt {audiospylt.__version__}\n"
                                  "Date\ttoday\nBins\t7\n")
    data = frames[1:]
    times = sorted(tracks["time"].unique())
    assert [f[1] for f in data] == times and {f[0] for f in data} == {"1TRC"} and {f[2] for f in data} == {0}
    assert all(len(f[3]) == 1 and f[3][0][:2] == ("1TRC", 0x0004) and f[3][0][2].shape[1] == 4 for f in data)
    for sig, time, _, ((_, _, rows),) in data:
        here = tracks[tracks["time"] == time].sort_values("track_id")
        assert rows[:, 0].tolist() == (here["track_id"] + 1).tolist()
        assert np.array_equal(rows[:, 1:], here[["freq", "amp", "phase"]].to_numpy(np.float32))
    back = read_sdif(path)
    assert back.attrs["meta"]["written_by"] == f"audiospylt {audiospylt.__version__}"
    original = tracks.sort_values(["track_id", "time"]).reset_index(drop=True)
    again = back.assign(track_id=back["partial_id"].astype("int64") - 1).sort_values(["track_id", "time"])
    assert again["track_id"].tolist() == original["track_id"].tolist()
    assert again["time"].tolist() == original["time"].tolist()
    for column in ("freq", "amp", "phase"):
        assert np.array_equal(again[column].to_numpy(), original[column].to_numpy(np.float32).astype(np.float64))


def test_write_fills_frames_a_track_skips(tmp_path):
    tracks = pd.DataFrame({"track_id": [0, 0, 1, 1, 1], "time": [0.0, 0.2, 0.05, 0.1, 0.15],
                           "freq": [100.0, 300.0, 1003.0, 1003.0, 1003.0], "amp": [0.1, 0.3, 0.5, 0.5, 0.5],
                           "phase": [0.5, np.nan, np.nan, 1.0, np.nan]})
    back = read_sdif(write_sdif(tracks, tmp_path / "fill.sdif"))
    one = back[back["partial_id"] == 1]
    two = back[back["partial_id"] == 2]
    grid = [0.0, 0.05, 0.1, 0.15, 0.2]
    assert one["time"].tolist() == grid and two["time"].tolist() == [0.05, 0.1, 0.15]
    np.testing.assert_allclose(one["freq"], [100, 150, 200, 250, 300], rtol=1e-7)
    np.testing.assert_allclose(one["amp"], [0.1, 0.15, 0.2, 0.25, 0.3], rtol=1e-7)

    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    f = np.array([100, 150, 200, 250, 300.0])
    expected = 0.5 + np.r_[0, np.cumsum(np.pi * 0.05 * (f[1:] + f[:-1]))]
    expected[1:] = wrap(expected[1:])  # known phases are kept as given, integrated ones wrapped
    np.testing.assert_allclose(one["phase"], expected, atol=1e-6)
    # Track 1: no phase at its first point (0), the given 1.0 kept, then integrated from it.
    np.testing.assert_allclose(two["phase"], [0.0, 1.0, wrap(1.0 + np.pi * 0.05 * 2006)], atol=1e-5)
    np.testing.assert_allclose(two["phase"].iloc[2], 1.0 + 0.3 * np.pi, atol=1e-5)


def test_write_without_phase_integrates_from_zero(tmp_path):
    tracks = pd.DataFrame({"track_id": [5, 5, 5], "time": [0.0, 0.25, 0.5], "freq": [1.0, 1.0, 1.0], "amp": 0.5})
    back = read_sdif(write_sdif(tracks, tmp_path / "p.sdif"))
    np.testing.assert_allclose(back["phase"], [0.0, np.pi / 2, -np.pi], atol=1e-6)  # pi wraps to -pi
    assert back["partial_id"].tolist() == [6, 6, 6]


def test_write_loris_rbep_keeps_every_track_whole(tmp_path):
    loris = read_sdif(LORIS)
    back = read_sdif(write_sdif(loris, tmp_path / "loris.sdif"))
    assert back["track_id"].nunique() == loris["track_id"].nunique()
    assert len(back) > len(loris)
    merged = loris.assign(partial_id=loris["track_id"] + 1).merge(back, on=["partial_id", "time"], suffixes=("", "_w"))
    assert len(merged) == len(loris)  # every breakpoint is written at its own time
    assert np.array_equal(merged["freq_w"], merged["freq"].to_numpy(np.float32).astype(np.float64))


def test_write_errors(tmp_path):
    base = pd.DataFrame({"track_id": [0, 0], "time": [0.0, 0.1], "freq": 100.0, "amp": 0.1})
    with pytest.raises(ValueError, match="no points"):
        write_sdif(base.iloc[:0], tmp_path / "x.sdif")
    for ids in ([-1, -1], [2**24 - 1, 2**24 - 1]):
        with pytest.raises(ValueError, match="track_id must be in 0..16777214"):
            write_sdif(base.assign(track_id=ids), tmp_path / "x.sdif")
    for nvt in ({"a\tb": 1}, {"a": "b\nc"}, {"": 1}, {1: "x"}):
        with pytest.raises(ValueError, match="nvt names"):
            write_sdif(base, tmp_path / "x.sdif", nvt=nvt)
    with pytest.raises(SchemaError, match="strictly increasing"):
        write_sdif(base.assign(time=[0.1, 0.1]), tmp_path / "x.sdif")
    assert not (tmp_path / "x.sdif").exists()


def test_top_level_names():
    assert audiospylt.read_sdif is read_sdif
    assert audiospylt.write_sdif is write_sdif
    assert audiospylt.read_chord_seq is read_chord_seq
