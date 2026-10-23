"""Partiels PM2 import: CSV and JSON exports -> tracks tables (D22)."""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from audiospylt.io.interop.partiels import read_partiels
from audiospylt.io.tables import read_table, write_table
from audiospylt.model.convert import tracks_to_events
from audiospylt.synth.additive import synthesize

DATA = Path(__file__).resolve().parent / "data" / "partiels"
FLUTE_JSON, FLUTE_CSV, CELLO_JSON = DATA / "flute_json", DATA / "flute_csv", DATA / "cello_json"
FLUTE_IDS = {"Group_1_Partial_1": 1, "Group_1_Partial_2": 2, "Group_1_Partial_3": 3, "Group_1_Partial_20": 19}


def raw(path, channel=0):
    """Independent read of one JSON export: list of (time, freq, amp, phase) or None per frame."""
    frames = json.loads(Path(path).read_text())["results"][channel]
    return [None if f["value"] is None else (f["time"], f["value"], *f["extra"]) for f in frames]


def runs(points, max_cents=None):
    """Runs of consecutive non-gap frames, cut where |step| > max_cents."""
    out, current = [], []
    for point in points:
        if point is None or (current and max_cents is not None
                             and abs(1200 * np.log2(point[1] / current[-1][1])) > max_cents):
            if current:
                out.append(current)
            current = [] if point is None else [point]
        else:
            current.append(point)
    if current:
        out.append(current)
    return out


def as_tracks(table):
    return [list(map(tuple, group[["time", "freq", "amp", "phase"]].to_numpy()))
            for _, group in table.groupby("track_id", sort=True)]


def write_csv(path, *blocks, header='TIME,DURATION,"Frequency","Amplitude","Phase"'):
    """Blocks of (time, freq, amp, phase) rows in Partiels' CSV layout; None is a gap at frame i."""
    parts = []
    for rows in blocks:
        lines = [header] if header else []
        lines += [f"{0.01 * (i + 1)!r},0.0000000000," if r is None else ",".join(map(repr, (r[0], 0.0, *r[1:])))
                  for i, r in enumerate(rows)]
        parts.append("\n".join(lines))
    Path(path).write_text("\n\n".join(parts) + "\n")
    return path


def test_flute_json_folder_reads_every_run_between_gaps():
    tracks = read_partiels(FLUTE_JSON)
    expected = []
    for stem, pid in FLUTE_IDS.items():
        expected += [(run, pid) for run in runs(raw(FLUTE_JSON / f"{stem}.json"))]
    expected.sort(key=lambda item: (item[0][0][0], item[0][0][1]))
    assert as_tracks(tracks) == [run for run, _ in expected]
    assert tracks.groupby("track_id")["partial_id"].first().tolist() == [pid for _, pid in expected]
    assert list(tracks.columns) == ["track_id", "time", "freq", "amp", "phase", "partial_id"]
    assert tracks.attrs["audiospylt_table"] == "tracks"
    assert tracks["partial_id"].dtype == "Int64"
    assert tracks.attrs["meta"] == {
        "source": "partiels", "format": "json", "channel": 0, "channels": 2,
        "files": [f"{stem}.json" for stem in FLUTE_IDS], "skipped": ["Group_1_Feature_Scoring.json"],
        "duplicates_dropped": 0, "max_step_cents": None, "plugin": "pm2:pm2harmonicpartialtracking",
    }
    # The cut has gaps, single-point runs and several runs per slot.
    sizes = tracks.groupby("track_id").size()
    assert len(sizes) > len(FLUTE_IDS) and (sizes == 1).any() and (sizes > 10).any()


def test_flute_csv_matches_json_to_six_significant_digits():
    from_csv, from_json = read_partiels(FLUTE_CSV), read_partiels(FLUTE_JSON)
    assert from_csv["track_id"].tolist() == from_json["track_id"].tolist()
    np.testing.assert_allclose(from_csv["time"], from_json["time"], rtol=0, atol=1e-9)
    for column in ("freq", "amp", "phase"):
        np.testing.assert_allclose(from_csv[column], from_json[column], rtol=5e-6, atol=0)
    assert from_csv["partial_id"].isna().all()
    assert "plugin" not in from_csv.attrs["meta"]
    assert from_csv.attrs["meta"]["skipped"] == ["Group_1_Feature_Scoring.csv"]


@pytest.mark.parametrize("variant", ["none", "generic", "tab"])
def test_csv_header_modes_and_separator(tmp_path, variant):
    for source in FLUTE_CSV.glob("Group_1_Partial_*.csv"):
        text = source.read_text()
        if variant == "none":
            text = "\n".join(line for line in text.split("\n") if not line.startswith("TIME"))
        elif variant == "generic":
            text = text.replace('"Frequency","Amplitude","Phase"', "VALUE,EXTRA0,EXTRA1")
        else:
            text = text.replace(",", "\t")
        (tmp_path / source.name).write_text(text)
    pd.testing.assert_frame_equal(read_partiels(tmp_path), read_partiels(FLUTE_CSV.glob("*Partial*")))


@pytest.mark.parametrize("folder", [FLUTE_JSON, FLUTE_CSV])
def test_channel_one_is_the_second_analysis(folder):
    first, second = read_partiels(folder), read_partiels(folder, channel=1)
    assert second.attrs["meta"]["channel"] == 1
    assert not np.array_equal(first["freq"].to_numpy(), second["freq"].to_numpy())
    if folder == FLUTE_JSON:
        expected = sorted((run for stem in FLUTE_IDS for run in runs(raw(folder / f"{stem}.json", 1))),
                          key=lambda run: (run[0][0], run[0][1]))
        assert as_tracks(second) == expected
    with pytest.raises(ValueError, match="channel=2 does not exist"):
        read_partiels(folder, channel=2)


def reference_dedupe(slots, max_cents=None):
    """Greedy rule: runs longest first (then slot, start), each claims the points nobody holds yet."""
    candidates = []
    for order, points in enumerate(slots):
        index = 0
        for run in runs(points, max_cents):
            while points[index] is None or points[index] != run[0]:
                index += 1
            candidates.append((-len(run), order, index, run))
            index += len(run)
    claimed, pieces = set(), []
    for *_, run in sorted(candidates, key=lambda c: c[:3]):
        piece = []
        for point in run:
            if point[:3] in claimed:
                if piece:
                    pieces.append(piece)
                piece = []
            else:
                piece.append(point)
        if piece:
            pieces.append(piece)
        claimed.update(point[:3] for point in run)
    return sorted(pieces, key=lambda p: (p[0][0], p[0][1]))


def test_cello_inharmonic_duplicates_are_dropped_longest_run_first():
    files = sorted(CELLO_JSON.glob("*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    slots = [raw(path) for path in files]
    every = [point for points in slots for point in points if point is not None]
    distinct = {point[:3] for point in every}
    assert len(every) - len(distinct) > 50  # the cut is full of copies

    tracks = read_partiels(CELLO_JSON)
    assert as_tracks(tracks) == reference_dedupe(slots)
    assert {tuple(p) for p in tracks[["time", "freq", "amp"]].to_numpy()} == distinct
    assert len(tracks) == len(distinct)
    assert tracks.attrs["meta"]["duplicates_dropped"] == len(every) - len(distinct)
    assert tracks.attrs["meta"]["plugin"] == "pm2:pm2inharmonicpartialtracking"
    assert tracks.attrs["meta"]["channels"] == 1
    assert sorted(tracks["partial_id"].unique()) == [8, 9, 10, 12, 15, 18]

    kept = read_partiels(CELLO_JSON, dedupe=False)
    assert len(kept) == len(every) and kept.attrs["meta"]["duplicates_dropped"] == 0
    assert as_tracks(kept) == sorted((run for points in slots for run in runs(points)),
                                     key=lambda run: (run[0][0], run[0][1]))
    # Slot reassignments: the cut holds steps over 100 cents inside a run.
    split = read_partiels(CELLO_JSON, max_step_cents=100)
    assert split["track_id"].nunique() > tracks["track_id"].nunique()
    assert as_tracks(split) == reference_dedupe(slots, max_cents=100)


def test_dedupe_keeps_the_longest_run_whole(tmp_path):
    a = [(0.01 * k, 440.0 + k, 0.1, 0.0) for k in range(1, 7)]
    short = [None, None, None, a[3], a[4], a[5], None]  # inside a, in the lower slot
    tail = [None, None, None, a[3], a[4], (0.07, 600.0, 0.1, 0.0), (0.08, 601.0, 0.1, 0.0)]
    write_csv(tmp_path / "Group_1_Partial_1.csv", short)
    write_csv(tmp_path / "Group_1_Partial_2.csv", tail)
    write_csv(tmp_path / "Group_1_Partial_3.csv", a)
    tracks = read_partiels(tmp_path)
    assert as_tracks(tracks) == [a, [(0.07, 600.0, 0.1, 0.0), (0.08, 601.0, 0.1, 0.0)]]
    assert tracks.attrs["meta"]["duplicates_dropped"] == 5


def test_max_step_cents_splits_inside_runs(tmp_path):
    rows = [(0.01, 440.0, 0.1, 0.0), (0.02, 445.0, 0.1, 0.0), (0.03, 900.0, 0.1, 0.0), (0.04, 905.0, 0.1, 0.0)]
    path = write_csv(tmp_path / "p.csv", rows)
    assert read_partiels(path)["track_id"].tolist() == [0, 0, 0, 0]
    assert read_partiels(path, max_step_cents=600)["track_id"].tolist() == [0, 0, 1, 1]
    assert read_partiels(path, max_step_cents=1300)["track_id"].tolist() == [0, 0, 0, 0]
    # The flute cut starts with an octave error inside the second harmonic's run.
    flute = read_partiels(FLUTE_JSON, max_step_cents=600)
    expected = sorted((run for stem in FLUTE_IDS for run in runs(raw(FLUTE_JSON / f"{stem}.json"), 600)),
                      key=lambda run: (run[0][0], run[0][1]))
    assert as_tracks(flute) == expected
    assert len(expected) > read_partiels(FLUTE_JSON)["track_id"].nunique()


def test_tracks_are_numbered_by_start_time_then_frequency(tmp_path):
    write_csv(tmp_path / "Group_1_Partial_1.csv", [None, (0.02, 880.0, 0.1, 0.0), (0.03, 880.0, 0.1, 0.0)])
    write_csv(tmp_path / "Group_1_Partial_2.csv", [None, (0.02, 440.0, 0.2, 0.0), None])
    write_csv(tmp_path / "Group_1_Partial_10.csv", [(0.01, 1000.0, 0.3, 0.0), None, None])
    tracks = read_partiels(tmp_path)
    assert tracks["freq"].tolist() == [1000.0, 440.0, 880.0, 880.0]
    assert tracks["track_id"].tolist() == [0, 1, 2, 2]
    assert tracks.attrs["meta"]["files"] == [f"Group_1_Partial_{k}.csv" for k in (1, 2, 10)]


def test_times_just_below_zero_read_as_zero(tmp_path):
    path = write_csv(tmp_path / "p.csv", [(-1.1338e-05, 440.0, 0.1, 0.5), (0.0058, 441.0, 0.1, 0.6)])
    assert read_partiels(path)["time"].tolist() == [0.0, 0.0058]
    path = write_csv(tmp_path / "q.csv", [(-2e-05, 439.0, 0.1, 0.4), (-1e-05, 440.0, 0.1, 0.5),
                                          (0.0058, 441.0, 0.1, 0.6)])
    tracks = read_partiels(path)
    assert tracks["time"].tolist() == [0.0, 0.0058] and tracks["freq"].tolist() == [440.0, 441.0]


def test_empty_slots_and_json_without_description(tmp_path):
    source = json.loads((FLUTE_JSON / "Group_1_Partial_1.json").read_text())
    plain = {"results": source["results"]}
    (tmp_path / "Group_1_Partial_1.json").write_text(json.dumps(plain))
    empty = {"results": [[{"duration": 0.0, "time": 0.01 * k, "value": None} for k in range(3)]] * 2}
    (tmp_path / "Group_1_Partial_2.json").write_text(json.dumps(empty))
    tracks = read_partiels(tmp_path)
    assert tracks["partial_id"].isna().all() and "plugin" not in tracks.attrs["meta"]
    assert tracks.attrs["meta"]["files"] == ["Group_1_Partial_1.json", "Group_1_Partial_2.json"]
    pd.testing.assert_frame_equal(tracks.drop(columns="partial_id"),
                                  read_partiels(FLUTE_JSON / "Group_1_Partial_1.json").drop(columns="partial_id"))
    nothing = read_partiels(tmp_path / "Group_1_Partial_2.json")
    assert nothing.empty and list(nothing.columns) == list(tracks.columns)
    assert nothing.attrs["audiospylt_table"] == "tracks"
    write_csv(tmp_path / "gaps.csv", [None, None], header=None)
    assert read_partiels(tmp_path / "gaps.csv").empty


def test_inputs_files_lists_and_errors(tmp_path):
    named = [FLUTE_JSON / f"{stem}.json" for stem in FLUTE_IDS]
    pd.testing.assert_frame_equal(read_partiels(named), read_partiels(FLUTE_JSON))
    pd.testing.assert_frame_equal(read_partiels(str(named[0])), read_partiels(iter(named[:1])))
    with pytest.raises(ValueError, match="1 extra output"):
        read_partiels(FLUTE_JSON / "Group_1_Feature_Scoring.json")
    with pytest.raises(ValueError, match="4 columns"):
        read_partiels([FLUTE_CSV / "Group_1_Partial_1.csv", FLUTE_CSV / "Group_1_Feature_Scoring.csv"])
    with pytest.raises(ValueError, match="mixed CSV and JSON"):
        read_partiels([FLUTE_CSV / "Group_1_Partial_1.csv", FLUTE_JSON / "Group_1_Partial_1.json"])
    both = tmp_path / "both"
    both.mkdir()
    shutil.copy(FLUTE_CSV / "Group_1_Partial_1.csv", both)
    shutil.copy(FLUTE_JSON / "Group_1_Partial_1.json", both)
    with pytest.raises(ValueError, match="mixed CSV and JSON"):
        read_partiels(both)
    with pytest.raises(ValueError, match="no .csv or .json files"):
        read_partiels(DATA)
    only_f0 = tmp_path / "f0"
    only_f0.mkdir()
    shutil.copy(FLUTE_CSV / "Group_1_Feature_Scoring.csv", only_f0)
    with pytest.raises(ValueError, match="no partial tracks"):
        read_partiels(only_f0)
    with pytest.raises(FileNotFoundError):
        read_partiels(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match=r"\.csv or \.json"):
        read_partiels(DATA / "trim.py")
    with pytest.raises(ValueError, match="no files given"):
        read_partiels([])
    bad = tmp_path / "bad.csv"
    bad.write_text("0.01,0,440,0.1,0\n0.02,0,441,oops,0\n")
    with pytest.raises(ValueError, match="line 2"):
        read_partiels(bad)
    for channel in (-1, True, 1.0):
        with pytest.raises(ValueError, match="channel"):
            read_partiels(FLUTE_JSON, channel=channel)
    for step in (0, -10, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_step_cents"):
            read_partiels(FLUTE_JSON, max_step_cents=step)


def test_table_round_trip_and_resynthesis(tmp_path):
    tracks = read_partiels(FLUTE_JSON)
    back = read_table(write_table(tracks, tmp_path / "flute.tsv"))
    pd.testing.assert_frame_equal(back, tracks, check_dtype=False, check_exact=True)
    assert back.attrs["meta"]["plugin"] == "pm2:pm2harmonicpartialtracking"
    events = tracks_to_events(tracks)
    sizes = tracks.groupby("track_id").size()
    assert len(events) == int((sizes - 1).sum())
    start = tracks["time"].min()
    shifted = events.assign(time_start=events["time_start"] - start, time_stop=events["time_stop"] - start)
    audio = synthesize(shifted, 44100, fade=0.005)
    assert abs(audio.size - (events["time_stop"].max() - start) * 44100) <= 1
    assert 0 < np.abs(audio).max() < tracks.groupby("time")["amp"].sum().max() + 1e-12
