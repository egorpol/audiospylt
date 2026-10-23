"""v1 tracks -> notation reductions: tracks_to_chords (chord per window) and tracks_at (snapshot), D23."""

from pathlib import Path

from lxml import etree
import numpy as np
import pandas as pd
import pytest

from audiospylt.io.interop.partiels import read_partiels
from audiospylt.model.convert import tracks_at, tracks_to_chords
from audiospylt.model.schema import SchemaError, validate
from audiospylt.notation.mei import events_to_mei, snapshot_to_mei

COLUMNS = ["time_start", "time_stop", "freq_start", "freq_stop", "amp_start", "amp_stop"]
FLUTE_JSON = Path(__file__).resolve().parent / "data" / "partiels" / "flute_json"
TIMES = np.arange(10) / 10  # hop 0.1 s


def _tracks(rows, **extra):
    table = pd.DataFrame(rows, columns=["track_id", "time", "freq", "amp"])
    for name, values in extra.items():
        table[name] = values
    return table


def _chords(rows):
    """Expected events: (time_start, time_stop, freq, amp, track_id) per note."""
    table = pd.DataFrame([(a, b, f, f, amp, amp) for a, b, f, amp, _ in rows], columns=COLUMNS, dtype=float)
    table["track_id"] = np.array([row[-1] for row in rows], dtype=np.int64)
    return table


def _cents(freq, cents):
    return freq * 2 ** (cents / 1200)


@pytest.fixture(scope="module")
def mei_schema():
    return etree.RelaxNG(etree.parse(str(Path(__file__).parent / "data" / "mei" / "mei-all.rng")))


# ---------------------------------------------------------------- tracks_to_chords


def test_one_note_per_track_and_window():
    # Track 0: 440 Hz at 0.5 over 0.0-0.9. Track 1: 0.0-0.2 at 200/220/1000 Hz, amps 0.1/0.3/0.1 ->
    # weighted median 220 Hz, mean amp 0.5/3 over 3 x 0.1 s of a 0.5 s window -> 0.1. Track 2: one point
    # at 0.9, in the closed last window [0.5, 0.9]: 0.4 x 0.1/0.4 -> 0.1. Track 3 at +20 cents of
    # track 0 and quieter: merged away. Track 4 at 0 Hz: ignored. Phase and extras are dropped.
    rows = [(0, t, 440.0, 0.5) for t in TIMES]
    rows += [(1, 0.0, 200.0, 0.1), (1, 0.1, 220.0, 0.3), (1, 0.2, 1000.0, 0.1), (2, 0.9, 300.0, 0.4)]
    rows += [(3, t, _cents(440.0, 20), 0.2) for t in TIMES[:5]] + [(4, 0.3, 0.0, 0.9)]
    tracks = validate(_tracks(rows, phase=0.5, note="x"), "tracks")
    tracks.attrs["meta"] = {"method": "pm2"}
    chords = tracks_to_chords(tracks, 0.5)
    expected = _chords([(0.0, 0.5, 220.0, 0.1, 1), (0.0, 0.5, 440.0, 0.5, 0),
                        (0.5, 0.9, 300.0, 0.1, 2), (0.5, 0.9, 440.0, 0.5, 0)])
    pd.testing.assert_frame_equal(chords, expected, rtol=1e-12)
    assert chords.attrs["audiospylt_table"] == "events" and chords.attrs["meta"] == {"method": "pm2"}
    chords.attrs["meta"]["method"] = "x"
    assert tracks.attrs["meta"] == {"method": "pm2"}
    assert list(tracks.columns) == ["track_id", "time", "freq", "amp", "phase", "note"]
    # merge_cents=None keeps a note per track.
    kept = tracks_to_chords(tracks, 0.5, merge_cents=None)
    assert kept.groupby("time_start")["track_id"].apply(list).to_dict() == {0.0: [1, 0, 3], 0.5: [2, 0]}


def test_weighted_median_stays_on_the_line():
    # 8 frames at 330 Hz and 2 octave errors at 660 Hz: the median keeps 330 Hz (a mean would not).
    freqs = [330.0] * 3 + [660.0] * 2 + [330.0] * 5
    chords = tracks_to_chords(_tracks([(0, t, f, 0.5) for t, f in zip(TIMES, freqs)]), 1.0)
    assert chords["freq_start"].tolist() == [330.0]
    # Lower weighted median: equal halves pick the lower frequency, the frequency of a real point.
    assert tracks_to_chords(_tracks([(0, 0.0, 300.0, 0.5), (0, 0.1, 400.0, 0.5)]), 1.0)["freq_start"].tolist() == [300.0]
    # Weights decide: a loud 400 Hz point outweighs three quiet 300 Hz points.
    rows = [(0, 0.0, 300.0, 0.1), (0, 0.1, 300.0, 0.1), (0, 0.2, 300.0, 0.1), (0, 0.3, 400.0, 0.4)]
    assert tracks_to_chords(_tracks(rows), 1.0)["freq_start"].tolist() == [400.0]
    # All amplitudes 0 (f0_to_tracks writes 0 for a missing fundamental): plain lower median.
    rows = [(0, t, f, 0.0) for t, f in zip(TIMES, (400.0, 100.0, 300.0, 200.0))]
    chords = tracks_to_chords(_tracks(rows), 1.0)
    assert chords["freq_start"].tolist() == [200.0] and chords["amp_start"].tolist() == [0.0]


def test_merge_keeps_the_louder_note():
    def notes(cents, **options):
        rows = [(0, t, 440.0, 0.2) for t in TIMES] + [(1, t, _cents(440.0, cents), 0.5) for t in TIMES]
        return tracks_to_chords(_tracks(rows), 1.0, **options)["track_id"].tolist()

    assert notes(49) == [1]
    assert notes(-49) == [1]
    assert notes(51) == [0, 1] and notes(-51) == [1, 0]
    assert notes(49, merge_cents=0) == [0, 1]
    assert notes(99, merge_cents=100) == [1]
    # Greedy, loudest first: B (0.5) drops A (+40 c, 0.4); C (+80 c, 0.3) is 80 c from B and stays,
    # although it lies within 50 c of the dropped A.
    rows = [(k, t, _cents(440.0, c), amp) for k, c, amp in ((0, 40, 0.4), (1, 0, 0.5), (2, 80, 0.3))
            for t in TIMES]
    assert tracks_to_chords(_tracks(rows), 1.0)["track_id"].tolist() == [1, 2]


def test_thinning_by_level_and_count():
    # Amplitudes 1, 0.1 (-20 dB), 0.01 (-40 dB) at 200, 300, 400 Hz; a second window holds only 0.01.
    rows = [(k, t, f, amp) for k, f, amp in ((0, 300.0, 1.0), (1, 200.0, 0.1), (2, 400.0, 0.01))
            for t in TIMES[:5]]
    rows += [(3, t, 500.0, 0.01) for t in TIMES[5:]]
    tracks = _tracks(rows)

    def kept(**options):
        chords = tracks_to_chords(tracks, [0.0, 0.5, 1.0], **options)
        return chords.groupby("time_start")["track_id"].apply(list).to_dict()

    assert kept() == {0.0: [1, 0, 2], 0.5: [3]}
    assert kept(min_db=-30) == {0.0: [1, 0], 0.5: [3]}  # relative to each window's loudest
    assert kept(min_db=-20) == {0.0: [1, 0], 0.5: [3]}
    assert kept(min_db=0) == {0.0: [0], 0.5: [3]}
    assert kept(max_notes=2) == {0.0: [1, 0], 0.5: [3]}
    assert kept(max_notes=1, min_db=-60) == {0.0: [0], 0.5: [3]}
    # max_notes counts kept notes: a merged note does not use up a place.
    rows = [(k, t, f, amp) for k, f, amp in ((0, 300.0, 1.0), (1, _cents(300.0, 10), 0.9), (2, 200.0, 0.1))
            for t in TIMES]
    assert tracks_to_chords(_tracks(rows), 1.0, max_notes=2)["track_id"].tolist() == [2, 0]


def test_amplitude_is_the_time_average_over_the_window():
    # A track present in 2 of the 5 frames of [0, 0.5) at 0.5 reads 0.5 x 2/5; a full track reads its
    # mean (coverage capped at 1, although 5 points x hop 0.1 cover [0, 0.5) exactly).
    rows = [(0, 0.1, 300.0, 0.5), (0, 0.2, 300.0, 0.5), (1, 0.0, 600.0, 0.2), (1, 0.1, 600.0, 0.6)]
    rows += [(1, t, 600.0, 0.4) for t in TIMES[2:5]]
    chords = tracks_to_chords(_tracks(rows), [0.0, 0.5])
    np.testing.assert_allclose(chords["amp_start"], [0.2, 0.4], rtol=1e-12)
    # Boundaries past the last point count as silence: [0, 1.0] halves both.
    np.testing.assert_allclose(tracks_to_chords(_tracks(rows), [0.0, 1.0])["amp_start"], [0.1, 0.2], rtol=1e-12)
    # The hop comes from consecutive points of each track, not from the union of their times: two
    # tracks on interleaved grids (hop 0.1, offset 0.05) keep hop 0.1.
    rows = [(0, t, 300.0, 0.5) for t in TIMES[:5]] + [(1, t + 0.05, 600.0, 0.5) for t in TIMES[:5]]
    np.testing.assert_allclose(tracks_to_chords(_tracks(rows), [0.0, 0.5])["amp_start"], [0.5, 0.5])
    # Without two points in any track the hop is unknown and each note keeps its mean amplitude.
    chords = tracks_to_chords(_tracks([(0, 0.1, 300.0, 0.5), (1, 0.7, 600.0, 0.3)]), 0.5)
    assert chords["amp_start"].tolist() == [0.5, 0.3]


def test_windows_from_a_step_or_boundaries():
    rows = [(0, t, 440.0, 0.5) for t in TIMES]
    # Step: grid from 0 to the last point; the last window is shorter and includes its end.
    chords = tracks_to_chords(_tracks(rows), 0.4)
    assert list(zip(chords["time_start"], chords["time_stop"])) == [(0.0, 0.4), (0.4, 0.8), (0.8, 0.9)]
    # Float noise in end / step makes no sliver window: 3 x 0.1 != 0.3.
    end = 3 * 0.1
    chords = tracks_to_chords(_tracks([(0, 0.0, 440.0, 0.5), (0, end, 440.0, 0.5)]), 0.1)
    # Windows [0, 0.1), [0.1, 0.2), [0.2, end]; the middle one is empty, so no row.
    assert list(zip(chords["time_start"], chords["time_stop"])) == [(0.0, 0.1), (0.2, end)]
    # Boundaries: half-open, last closed, points outside ignored, empty windows give no row.
    chords = tracks_to_chords(_tracks(rows), [0.2, 0.4, 0.45, 0.6])
    assert list(zip(chords["time_start"], chords["time_stop"])) == [(0.2, 0.4), (0.4, 0.45), (0.45, 0.6)]
    # One point in the 0.05 s window covers it (coverage capped at 1).
    np.testing.assert_allclose(chords["amp_start"], [0.5, 0.5, 0.5])
    assert tracks_to_chords(_tracks(rows), [0.31, 0.35, 0.39]).empty
    assert tracks_to_chords(_tracks(rows), np.array([0, 1])).shape == (1, 7)
    # All points at time 0: one window [0, 0].
    chords = tracks_to_chords(_tracks([(0, 0.0, 440.0, 0.5), (1, 0.0, 660.0, 0.25)]), 0.5)
    assert chords[["time_start", "time_stop"]].to_numpy().tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_tracks_to_chords_edge_cases():
    empty = tracks_to_chords(_tracks([]), 0.5)
    assert list(empty.columns) == COLUMNS + ["track_id"] and empty.empty
    assert empty["track_id"].dtype == np.int64
    assert tracks_to_chords(_tracks([(0, 0.2, 0.0, 0.5)]), 0.5).empty
    tracks = _tracks([(0, 0.0, 440.0, 0.5)])
    for windows in (0, -0.5, np.nan, np.inf, True, "0.5", None, [0.5], [0.3, 0.2], [0.1, 0.1], [-0.1, 0.2],
                    [[0.0, 1.0]], [0.0, np.nan], ["a", "b"]):
        with pytest.raises(ValueError, match="windows must be"):
            tracks_to_chords(tracks, windows)
    for options, name in (({"merge_cents": -1}, "merge_cents"), ({"merge_cents": np.nan}, "merge_cents"),
                          ({"min_db": 1}, "min_db"), ({"min_db": -np.inf}, "min_db"),
                          ({"max_notes": 0}, "max_notes"), ({"max_notes": 1.5}, "max_notes"),
                          ({"max_notes": True}, "max_notes")):
        with pytest.raises(ValueError, match=name):
            tracks_to_chords(tracks, 0.5, **options)
    assert len(tracks_to_chords(tracks, 0.5, max_notes=np.int64(1))) == 1
    with pytest.raises(SchemaError, match="strictly increasing"):
        tracks_to_chords(_tracks([(0, 0.1, 100.0, 0.5), (0, 0.1, 200.0, 0.5)]), 0.5)


def test_partiels_flute_chords_notate(mei_schema):
    # Four PM2 slots of the Saariaho flute (harmonics 1, 2, 3, 19; 1.097-1.400 s).
    tracks = read_partiels(FLUTE_JSON)
    chords = tracks_to_chords(tracks, 0.1)
    assert chords["time_start"].unique().tolist() == pytest.approx([1.0, 1.1, 1.2, 1.3])
    for _, chord in chords.groupby("time_start"):
        cents = np.sort(1200 * np.log2(chord["freq_start"].to_numpy()))
        assert (np.diff(cents) > 50).all()
    # Each note is a real point of its track, in its window.
    for row in chords.itertuples():
        points = tracks[(tracks["track_id"] == row.track_id) & (tracks["time"] >= row.time_start)
                        & (tracks["time"] <= row.time_stop)]
        assert row.freq_start in points["freq"].to_numpy()
    assert chords.attrs["meta"]["source"] == "partiels"
    mei = events_to_mei(chords, measure_seconds=0.1, labels=True)
    assert mei_schema.validate(etree.fromstring(mei.encode("utf-8"))), str(mei_schema.error_log)


# ---------------------------------------------------------------- tracks_at


def test_tracks_at_interpolates_each_track():
    # Track 0 spans 0.25 (linear in freq and amp), track 1 ends at 0.25, track 2 starts at 0.25,
    # track 3 ended before, track 4 is a single point elsewhere, track 5 has a gap around 0.25.
    rows = [(0, 0.0, 400.0, 0.2), (0, 1.0, 200.0, 0.4), (1, 0.1, 300.0, 0.3), (1, 0.25, 350.0, 0.35),
            (2, 0.25, 500.0, 0.5), (2, 0.5, 550.0, 0.1), (3, 0.0, 600.0, 0.6), (3, 0.2, 600.0, 0.6),
            (4, 0.5, 700.0, 0.7), (5, 0.0, 100.0, 0.1), (5, 0.5, 150.0, 0.2)]
    tracks = validate(_tracks(rows, phase=1.0), "tracks")
    tracks.attrs["meta"] = {"source": "partiels"}
    snap = tracks_at(tracks, 0.25)
    expected = pd.DataFrame({"freq": [125.0, 350.0, 350.0, 500.0], "amp": [0.15, 0.25, 0.35, 0.5],
                             "track_id": np.array([5, 0, 1, 2], dtype=np.int64)})
    pd.testing.assert_frame_equal(snap, expected, rtol=1e-12)
    assert snap.attrs["audiospylt_table"] == "snapshot" and snap.attrs["meta"] == {"source": "partiels"}
    snap.attrs["meta"]["source"] = "x"
    assert tracks.attrs["meta"] == {"source": "partiels"}
    assert tracks_at(tracks, 0.5)["track_id"].tolist() == [5, 0, 2, 4]
    assert tracks_at(tracks, 1)["track_id"].tolist() == [0]


def test_tracks_at_edge_cases(mei_schema):
    tracks = _tracks([(0, 0.1, 440.0, 0.5), (0, 0.3, 440.0, 0.5)])
    for time in (0.0, 0.31, -1.0):
        empty = tracks_at(tracks, time)
        assert list(empty.columns) == ["freq", "amp", "track_id"] and empty.empty
        assert empty["track_id"].dtype == np.int64
    assert tracks_at(_tracks([]), 0.0).empty
    for time in (np.nan, np.inf, True, "0.2", None):
        with pytest.raises(ValueError, match="time must be"):
            tracks_at(tracks, time)
    with pytest.raises(SchemaError, match="track_id"):
        tracks_at(pd.DataFrame({"time": [0.0], "freq": [1.0], "amp": [1.0]}), 0.0)
    # The flute fixture at 1.25 s: one note per slot, written as a chord.
    snap = tracks_at(read_partiels(FLUTE_JSON), 1.25)
    assert len(snap) == 4
    mei = snapshot_to_mei(snap)
    assert mei_schema.validate(etree.fromstring(mei.encode("utf-8"))), str(mei_schema.error_log)
