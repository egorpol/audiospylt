"""MEI schema, notation semantics, lazy imports, and fd-level rendering tests."""

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from lxml import etree
import numpy as np
import pandas as pd
import pytest

from audiospylt.model.schema import SchemaError
from audiospylt.notation.mei import events_to_mei, snapshot_to_mei
from audiospylt.notation.pitch import cents_label, pitch_table
from audiospylt.notation.render import render_svg, render_svg_pages


NS = {"m": "http://www.music-encoding.org/ns/mei"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
ARROWS = {"nu", "nd", "su", "sd", "fu", "fd", "xu", "xd", "ffu", "ffd"}


def hz(midi, a4=440):
    return a4 * 2 ** ((np.asarray(midi) - 69) / 12)


def snapshot(midis=(48, 59.5, 60, 60.5, 69.12)):
    frequencies = hz(midis)
    return pd.DataFrame({"freq": frequencies, "amp": np.ones(len(frequencies))})


def events(rows):
    """Rows are (onset, stop, MIDI); freq_stop deliberately contains glides."""
    return pd.DataFrame([(start, stop, float(hz(midi)), float(hz(midi + 3)), 0.5, 0.25)
                         for start, stop, midi in rows], columns=[
                             "time_start", "time_stop", "freq_start", "freq_stop", "amp_start", "amp_stop"])


@pytest.fixture(scope="session")
def mei_schema():
    # The directory is a read-only symlink in delegated worktrees. Do not resolve.
    return etree.RelaxNG(etree.parse(str(Path(__file__).parent / "data" / "mei" / "mei-all.rng")))


@pytest.fixture
def make_mei(mei_schema):
    def make(table=None, *, kind="snapshot", **options):
        if table is None:
            table = snapshot() if kind == "snapshot" else events([(0, 0.5, 48), (0, 0.5, 69), (1, 2, 60)])
        builder = snapshot_to_mei if kind == "snapshot" else events_to_mei
        mei = builder(table, **options)
        root = etree.fromstring(mei.encode("utf-8"))
        assert mei_schema.validate(root), str(mei_schema.error_log)
        assert mei.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')
        assert root.tag == f"{{{NS['m']}}}mei" and root.get("meiversion") == "5.1"
        assert root.find("m:meiHead/m:fileDesc/m:titleStmt/m:title", NS) is not None
        assert root.find("m:meiHead/m:fileDesc/m:pubStmt", NS) is not None
        assert not root.xpath("//@pnum")
        assert not ARROWS.intersection(root.xpath("//@accid"))
        notes = root.xpath("//m:note", namespaces=NS)
        assert [n.get(XML_ID) for n in notes] == [f"n{i}" for i in range(1, len(notes) + 1)]
        by_id = {n.get(XML_ID): n for n in notes}
        for direction in root.xpath("//m:dir", namespaces=NS):
            note = by_id[direction.get("startid")[1:]]
            assert note.xpath("ancestor::m:staff", namespaces=NS)[0].get("n") == direction.get("staff")
            assert note.xpath("ancestor::m:measure", namespaces=NS)[0] is direction.getparent()
            assert direction.get("type") in {"cents", "onset"}
            # Labels are centred on the notehead: the text sits in one <rend halign="center">.
            assert not (direction.text or "").strip() and len(direction) == 1
            assert direction[0].tag == f"{{{NS['m']}}}rend" and direction[0].get("halign") == "center"
            assert direction[0].text
        assert all(m.get("right") == "invis" for m in root.xpath("//m:measure", namespaces=NS))
        assert not root.xpath("//m:fing", namespaces=NS)
        return mei, root
    return make


@pytest.mark.parametrize("layout", ["chord", "sequence"])
@pytest.mark.parametrize("clef", ["G", "F", "GF"])
@pytest.mark.parametrize("resolution", ["semitone", "quarter", "eighth"])
@pytest.mark.parametrize("spelling", ["sharp", "flat"])
def test_schema_determinism_layout_and_staves(make_mei, layout, clef, resolution, spelling):
    table = snapshot([69.12, 60, 48, 59.8, 60.5, 69.12])
    before = table.copy(deep=True)
    options = dict(layout=layout, clef=clef, resolution=resolution, spelling=spelling)
    mei, root = make_mei(table, **options)
    assert make_mei(table, **options)[0] == mei
    pd.testing.assert_frame_equal(table, before)
    measures = root.xpath("//m:measure", namespaces=NS)
    assert len(measures) == (1 if layout == "chord" else len(table))
    assert all(len(m.findall("m:staff", NS)) == len(clef) for m in measures)
    assert len(root.xpath("//m:note", namespaces=NS)) == len(table)
    assert len(root.xpath("//m:dir", namespaces=NS)) == (len(table) if resolution == "eighth" else 0)
    defs = root.xpath("//m:staffDef", namespaces=NS)
    assert [(d.get("n"), d.get("clef.shape"), d.get("clef.line")) for d in defs] == [
        (str(i), shape, "2" if shape == "G" else "4") for i, shape in enumerate(clef, 1)]
    if clef == "GF":
        assert root.xpath("//m:staffGrp/@symbol", namespaces=NS) == ["brace"]
    if layout == "chord":
        assert all(c.get("dur") == "4" and c.get("stem.visible") == "false"
                   for c in root.xpath("//m:chord", namespaces=NS))
    else:
        assert not root.xpath("//m:chord", namespaces=NS)
        assert all(n.get("dur") == "4" and n.get("stem.visible") == "false"
                   for n in root.xpath("//m:note", namespaces=NS))


def test_staff_split_uses_written_pitch_and_includes_middle_c(make_mei):
    _, root = make_mei(snapshot([59.74, 59.75, 60, 59.5]))
    assert root.xpath("//m:staff[@n='1']//m:note/@pname", namespaces=NS) == ["c", "c"]
    assert root.xpath("//m:staff[@n='2']//m:note/@pname", namespaces=NS) == ["b", "b"]
    # Flat B3 quarter-sharp spells C4 quarter-flat but stays on the lower staff.
    _, flat = make_mei(snapshot([59.5]), spelling="flat")
    assert flat.xpath("//m:staff[@n='2']//m:note/@oct", namespaces=NS) == ["4"]
    assert flat.xpath("//m:staff[@n='2']//m:note/@accid", namespaces=NS) == ["1qf"]


@pytest.mark.parametrize("layout, expected", [("chord", ["-10", "+10", "0"]),
                                             ("sequence", ["0", "-10", "+10"])])
def test_note_order_and_stable_written_unisons(make_mei, layout, expected):
    _, root = make_mei(snapshot([72, 60.4, 60.6]), clef="G", layout=layout, labels=True)
    assert root.xpath("//m:dir/m:rend/text()", namespaces=NS) == expected


@pytest.mark.parametrize("spelling", ["sharp", "flat"])
@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_labels_are_remainders_of_written_notes(make_mei, spelling, decimals):
    table = snapshot([60.12, 59.84, 48.043, 69.5])
    _, root = make_mei(table, layout="sequence", spelling=spelling,
                       resolution="eighth", decimals=decimals)
    pitches = pitch_table(table.freq, spelling=spelling, resolution="eighth")
    assert root.xpath("//m:dir/m:rend/text()", namespaces=NS) == [cents_label(r, decimals) for r in pitches.residual]
    for direction in root.xpath("//m:dir", namespaces=NS):
        assert direction.get("place") == ("above" if direction.get("staff") == "1" else "below")


@pytest.mark.parametrize("clef", ["GF", "F"])
def test_label_stacks_read_in_chord_order(make_mei, clef):
    # Verovio stacks dirs outward from the staff in document order: above the G staff
    # the lowest note's label comes first, below the F staff the highest note's.
    midis = [43.1, 38.2, 50.06, 57.3, 64.2, 62.08, 72.4]
    _, root = make_mei(snapshot(midis), labels=True, clef=clef)
    # Notes inside a chord are written low to high, so the position in the chord is the pitch rank.
    rank = {n.get(XML_ID): n.getparent().index(n) for n in root.xpath("//m:note", namespaces=NS)}
    for staff in {d.get("staff") for d in root.xpath("//m:dir", namespaces=NS)}:
        order = [rank[d.get("startid")[1:]] for d in root.xpath(f"//m:dir[@staff='{staff}']", namespaces=NS)]
        below = clef == "F" or staff == "2"
        assert order == sorted(order, reverse=below) and len(order) > 1


@pytest.mark.parametrize("clef, expected", [("GF", ["1", "2", "2", "2", "3"]),
                                            ("G", ["1", "1", "1", "2", "2"])])
def test_sequence_labels_share_height_within_a_third(make_mei, clef, expected):
    # B3 C4 D4 E4 F4 (E4 quarter-sharp keeps its step): a height group ends at a
    # staff change (B3 | C4 on the grand staff) or where it would span more than a third.
    _, root = make_mei(snapshot([59, 60, 62, 64.5, 65]), layout="sequence", labels=True, clef=clef)
    assert root.xpath("//m:dir/@vgrp", namespaces=NS) == expected


def test_height_groups_only_in_labelled_sequences(make_mei):
    for _, root in (make_mei(labels=True), make_mei(kind="events", labels=True, time_labels=True)):
        assert root.xpath("//m:dir", namespaces=NS) and not root.xpath("//m:dir/@vgrp", namespaces=NS)
    _, root = make_mei(layout="sequence", labels=False)
    assert not root.xpath("//m:dir", namespaces=NS)


@pytest.mark.parametrize("labels", [None, True, False])
@pytest.mark.parametrize("resolution", ["semitone", "quarter", "eighth"])
def test_label_defaults_and_eighth_requirement(make_mei, labels, resolution):
    if labels is False and resolution == "eighth":
        with pytest.raises(ValueError, match="requires cents labels"):
            make_mei(labels=labels, resolution=resolution)
    else:
        _, root = make_mei(labels=labels, resolution=resolution)
        enabled = resolution == "eighth" if labels is None else labels
        assert len(root.xpath("//m:dir", namespaces=NS)) == (5 if enabled else 0)


# Three notes on C4 (sharp) or D4 (flat): the third opens a second column, after D4
# (sharp) in document order, and still takes its sign from the whole staff's chord.
@pytest.mark.parametrize("spelling, midis, expected", [
    ("sharp", [60, 60.5, 60.5, 62, 48, 48.5], ["n", "1qs", None, "1qs", "n", "1qs"]),
    ("flat", [62, 61.5, 61.5, 60, 50, 49.5], [None, "1qf", "1qf", "n", "1qf", "n"]),
])
def test_same_step_naturals_and_exact_unisons(make_mei, spelling, midis, expected):
    _, root = make_mei(snapshot(midis), spelling=spelling, labels=True)
    notes = root.xpath("//m:note", namespaces=NS)
    assert [n.get("accid") for n in notes] == expected
    assert len(notes) == len(root.xpath("//m:dir", namespaces=NS)) == 6


@pytest.mark.parametrize("naturals", [False, True])
def test_naturals_option_and_no_cross_measure_cancellation(make_mei, naturals):
    _, root = make_mei(snapshot([60, 60.5, 62, 48]), layout="sequence", naturals=naturals)
    assert root.xpath("//m:note/@accid", namespaces=NS) == (["n", "1qs", "n", "n"] if naturals else ["1qs"])


def test_same_step_rule_does_not_cross_staff_or_octave(make_mei):
    # C4 quarter-flat and C4 lie on opposite staves, despite equal pname/oct.
    _, root = make_mei(snapshot([59.5, 60, 72.5]), spelling="flat")
    assert root.xpath("//m:staff[@n='1']//m:note/@accid", namespaces=NS) == ["3qf"]
    _, root = make_mei(snapshot([48, 60.5]))
    assert root.xpath("//m:staff[@n='2']//m:note/@accid", namespaces=NS) == []


# Shaped like the 2c chord (speech1's 12 strongest rows): four notes on D4, three on G3;
# every row has its own cents remainder.
CROWDED = [62.42, 63.12, 62.86, 63.46, 64.03, 64.21, 55.43, 56.05, 56.56, 57.39, 57.61, 44.02]


def test_crowded_step_continues_chord_in_columns(make_mei):
    _, root = make_mei(snapshot(CROWDED), labels=True)
    layers = root.xpath("//m:layer", namespaces=NS)
    shapes = [[[(n.get("pname"), n.get("accid")) for n in chord] for chord in layer] for layer in layers]
    assert shapes == [
        [[("d", "1qs"), ("d", "s"), ("e", None), ("e", None)], [("d", "s"), ("d", "3qs")]],
        [[("g", "s"), ("g", "1qs"), ("g", "s"), ("a", "1qs"), ("a", "1qs")], [("g", "3qs")]],
    ]
    dirs = root.xpath("//m:dir", namespaces=NS)
    labels = {d.get("startid")[1:]: d.findtext("m:rend", namespaces=NS) for d in dirs}
    assert labels == dict(zip([f"n{i}" for i in range(1, 13)], [
        "-8", "+12", "+3", "+21", "-14", "-4", "+2", "-7", "+5", "-11", "+11", "+6"]))
    # Stacks go column by column, each in the chord's order (high to low below the F staff).
    assert [d.get("startid") for d in dirs] == [
        f"#n{i}" for i in (1, 2, 3, 4, 5, 6, 11, 10, 9, 8, 7, 12)]
    # Pairs on a step fit one chord; the other staff pads the extra column.
    _, pairs = make_mei(snapshot([62.4, 63.1, 64, 64.2]), clef="GF")
    assert len(pairs.xpath("//m:chord", namespaces=NS)) == 1
    _, padded = make_mei(snapshot([62.4, 63.1, 62.9]))
    assert [[c.tag.split("}")[1] for c in layer] for layer in padded.xpath("//m:layer", namespaces=NS)] == [
        ["chord", "chord"], ["space", "space"]]


def test_crowded_chord_renders_every_notehead_apart(make_mei, capfd):
    mei, _ = make_mei(snapshot(CROWDED), labels=True)
    svg = ET.fromstring(render_svg(mei))
    svg_ns = "{http://www.w3.org/2000/svg}"
    heads = [g.find(f"{svg_ns}g[@class='notehead']/{svg_ns}use").get("transform")
             for g in svg.iter(f"{svg_ns}g") if g.get("class") == "note"]
    assert len(heads) == len(CROWDED) == len(set(heads))
    captured = capfd.readouterr()
    assert "[Warning]" not in captured.out + captured.err


@pytest.mark.parametrize("top, expected", [(63.5, ("d", "3qs", 1)), (64, ("e", None, 0))])
def test_crowded_onset_anchors_the_highest_note(make_mei, top, expected):
    # D4 quarter-sharp and D#4 twice put the second D#4 in a second column; the
    # highest note sits in that column (D4 three-quarter-sharp) or in the first (E4).
    table = events([(0, 1, 62.4), (0, 1, 63.1), (0, 1, 62.9), (0, 1, top), (1, 2, 55)])
    _, root = make_mei(table, kind="events", time_labels=True)
    first = root.xpath("//m:measure", namespaces=NS)[0]
    onset = first.xpath("m:dir[@type='onset']", namespaces=NS)[0]
    note = first.xpath(f"//m:note[@xml:id='{onset.get('startid')[1:]}']", namespaces=NS)[0]
    chord = note.getparent()
    assert (note.get("pname"), note.get("accid"), chord.getparent().index(chord)) == expected
    assert len(first.xpath("m:staff[@n='2']/m:layer/m:space", namespaces=NS)) == 2


@pytest.mark.parametrize("rows, count, occupied", [
    ([(0, .5, 48), (.5, 1, 60), (1, 1.5, 69)], 3, [1, 2, 3]),
    ([(0, .5, 48), (.5, 1, 60), (2, 2.5, 69)], 5, [1, 2, 5]),
    ([(0, 10, 48), (.5, .75, 60)], 2, [1, 2]),
    ([(1, 2, 69), (0, .5, 48), (1, 3, 60)], 6, [1, 3]),
    ([(0, 0, 60)], 1, [1]),
    ([(0, 1.25, 60)], 2, [1]),  # round(2.5) == 2
])
def test_temporal_span_counts_contiguous_gap_overlap_and_last(make_mei, rows, count, occupied):
    table = events(rows)
    before = table.copy(deep=True)
    mei, root = make_mei(table, kind="events", time_labels=True, labels=True)
    assert make_mei(table, kind="events", time_labels=True, labels=True)[0] == mei
    pd.testing.assert_frame_equal(table, before)
    measures = root.xpath("//m:measure", namespaces=NS)
    assert len(measures) == count
    assert [int(m.get("n")) for m in measures if m.xpath(".//m:note", namespaces=NS)] == occupied
    assert all(len(m.findall("m:staff", NS)) == 2 for m in measures)
    expected = pitch_table(table.freq_start)
    assert sorted(root.xpath("//m:note/@pname", namespaces=NS)) == sorted(expected.step)
    assert sorted(root.xpath("//m:note/@oct", namespaces=NS)) == sorted(expected.octave.astype(str))
    assert root.xpath("//m:dir/m:rend[contains(text(), ' s')]/text()", namespaces=NS) == [
        f"{onset:.3f} s" for onset in sorted(table.time_start.unique())]


def test_temporal_grouping_time_anchor_and_options(make_mei):
    table = events([(1, 1.5, 48), (0, 10, 69), (0, 1, 72), (0, .5, 60), (1, 2, 36)])
    _, root = make_mei(table, kind="events", measure_seconds=.25, time_labels=True,
                       time_format="onset={:.1f}", resolution="eighth", naturals=True)
    assert len(root.xpath("//m:measure", namespaces=NS)) == 8
    time_dirs = root.xpath("//m:dir[starts-with(m:rend, 'onset=')]", namespaces=NS)
    assert [d.findtext("m:rend", namespaces=NS) for d in time_dirs] == ["onset=0.0", "onset=1.0"]
    assert [d.get("staff") for d in time_dirs] == ["1", "2"]
    assert [d.get("place") for d in time_dirs] == ["above", "above"]
    assert [d.get("type") for d in time_dirs] == ["onset", "onset"]
    assert len(root.xpath("//m:dir[@type='cents']", namespaces=NS)) == len(table)
    by_id = {n.get(XML_ID): n for n in root.xpath("//m:note", namespaces=NS)}
    assert [(by_id[d.get("startid")[1:]].get("pname"), by_id[d.get("startid")[1:]].get("oct"))
            for d in time_dirs] == [("c", "5"), ("c", "3")]
    assert len(root.xpath("//m:dir", namespaces=NS)) == len(table) + 2


def test_title_is_escaped(make_mei):
    _, root = make_mei(title='A & B <"C"> café')
    assert root.find("m:meiHead/m:fileDesc/m:titleStmt/m:title", NS).text == 'A & B <"C"> café'


def test_import_and_serialization_preserve_elementtree_namespace_registry(make_mei):
    # This shared registry also controls the untouched 0.6 MEI goldens.
    import audiospylt.notation.mei as module

    before = ET.tostring(ET.Element(f"{{{NS['m']}}}mei"))
    importlib.reload(module)
    make_mei()
    make_mei(kind="events")
    assert ET.tostring(ET.Element(f"{{{NS['m']}}}mei")) == before


@pytest.mark.parametrize("kind", ["snapshot", "events"])
def test_table_errors_and_zero_hz(kind):
    build = snapshot_to_mei if kind == "snapshot" else events_to_mei
    valid = snapshot() if kind == "snapshot" else events([(0, 1, 69)])
    with pytest.raises(SchemaError):
        build(valid.drop(columns="amp" if kind == "snapshot" else "time_stop"))
    with pytest.raises(SchemaError):
        build([])
    with pytest.raises(ValueError, match="empty"):
        build(valid.iloc[:0])
    column = "freq" if kind == "snapshot" else "freq_start"
    zero = valid.copy()
    zero.loc[0, column] = 0
    with pytest.raises(ValueError, match="0 Hz.*filter"):
        build(zero)
    bad = valid.copy()
    bad.loc[0, column] = np.nan
    with pytest.raises(SchemaError, match="finite|missing"):
        build(bad)


@pytest.mark.parametrize("kind", ["snapshot", "events"])
@pytest.mark.parametrize("options, match", [
    ({"resolution": "sixteenth"}, "resolution"), ({"spelling": "mixed"}, "spelling"),
    ({"a4": True}, "a4"), ({"a4": 0}, "a4"), ({"clef": "C"}, "clef"),
    ({"clef": ["G"]}, "clef"), ({"labels": "yes"}, "labels"),
    ({"decimals": True}, "decimals"), ({"decimals": -1}, "decimals"),
    ({"resolution": "eighth", "labels": False}, "requires cents labels"),
])
def test_common_option_errors(kind, options, match):
    build = snapshot_to_mei if kind == "snapshot" else events_to_mei
    table = snapshot() if kind == "snapshot" else events([(0, 1, 69)])
    with pytest.raises(ValueError, match=match):
        build(table, **options)


@pytest.mark.parametrize("layout", ["frame", ["chord"], None])
def test_layout_errors(layout):
    with pytest.raises(ValueError, match="layout"):
        snapshot_to_mei(snapshot(), layout=layout)


@pytest.mark.parametrize("seconds", [0, -1, np.nan, np.inf, True, "0.5", [0.5]])
def test_measure_seconds_errors(seconds):
    with pytest.raises(ValueError, match="measure_seconds"):
        events_to_mei(events([(0, 1, 69)]), measure_seconds=seconds)


@pytest.mark.parametrize("spelling", ["sharp", "flat"])
@pytest.mark.parametrize("resolution", ["semitone", "quarter", "eighth"])
@pytest.mark.parametrize("a4", [440, 442])
def test_range_and_quantisation_boundaries(make_mei, spelling, resolution, a4):
    grid = 1 if resolution == "semitone" else .5
    low = 11.5 if spelling == "flat" and grid == .5 else 12
    high = 131.5 if spelling == "sharp" and grid == .5 else 131
    options = dict(spelling=spelling, resolution=resolution, a4=a4)
    valid = pd.DataFrame({"freq": hz([low - grid / 2, high + grid / 2 - .001], a4), "amp": [1, 1]})
    _, root = make_mei(valid, **options)
    assert sorted(root.xpath("//m:note/@oct", namespaces=NS)) == ["0", "9"]
    invalid = pd.DataFrame({"freq": hz([low - grid / 2 - .001, high + grid / 2], a4), "amp": [1, 1]})
    with pytest.raises(ValueError, match="2 row.*encodable frequency range") as caught:
        snapshot_to_mei(invalid, **options)
    message = str(caught.value)
    assert f"{spelling}/{resolution}" in message and f"a4={a4}" in message
    assert f"{float(hz(low - grid / 2, a4)):.12g}" in message
    assert f"{float(hz(high + grid / 2, a4)):.12g}" in message


@pytest.mark.parametrize("kind", ["snapshot", "events"])
def test_unencodable_realistic_frequencies(kind):
    table = snapshot([69, 69]) if kind == "snapshot" else events([(0, 1, 69), (0, 1, 69)])
    table["freq" if kind == "snapshot" else "freq_start"] = [10, 18000]
    build = snapshot_to_mei if kind == "snapshot" else events_to_mei
    with pytest.raises(ValueError, match="2 row.*0–9.*encodable frequency range.*Filter by freq"):
        build(table)


def test_glide_to_zero_uses_only_start_frequency(make_mei):
    table = events([(0, 1, 69)])
    table.freq_stop = 0
    _, root = make_mei(table, kind="events")
    assert root.xpath("//m:note/@pname", namespaces=NS) == ["a"]


@pytest.mark.parametrize("kind, options", [
    ("snapshot", {"labels": True}),
    ("snapshot", {"layout": "sequence", "labels": True}),
    ("events", {"time_labels": True, "labels": True}),
    ("snapshot", {"resolution": "eighth", "naturals": True}),
    ("snapshot", {"resolution": "eighth", "naturals": True, "spelling": "flat"}),
    ("snapshot", {"clef": "F", "layout": "sequence"}),
])
def test_render_without_c_level_warnings(make_mei, capfd, kind, options):
    table = snapshot([48, 59.5, 60, 60.5, 60.5, 69.12]) if kind == "snapshot" else None
    mei, root = make_mei(table, kind=kind, **options)
    capfd.readouterr()
    svg = render_svg(mei)
    assert etree.fromstring(svg.encode()).tag == "{http://www.w3.org/2000/svg}svg"
    assert svg.count('class="notehead"') == len(root.xpath("//m:note", namespaces=NS))
    pages = render_svg_pages(mei)
    assert pages and all(etree.fromstring(s.encode()).tag.endswith("}svg") for s in pages)
    captured = capfd.readouterr()
    assert "[Warning]" not in captured.out + captured.err
    assert "[Error]" not in captured.out + captured.err


def test_verovio_63_diagnostics_require_fd_capture(capfd):
    import verovio

    toolkit = verovio.toolkit()
    if not toolkit.getVersion().startswith("6.3."):
        pytest.skip("This diagnostic contract describes the Verovio 6.3 binding")
    assert not hasattr(verovio, "enableLogToBuffer")
    assert not hasattr(toolkit, "enableLogToBuffer")
    capfd.readouterr()
    # Trigger a diagnostic without constructing invalid MEI. Unlike successful
    # notation renders, this deliberately requests an unavailable font.
    toolkit.setOptions({"font": "AudioSpyltMissingFont"})
    captured = capfd.readouterr()
    assert "[Warning]" in captured.err
    assert captured.out == ""
    assert toolkit.getLog() == ""


def test_multiple_pages_and_option_override(make_mei, capfd):
    mei, _ = make_mei(snapshot(np.tile([60, 62, 64, 65, 67, 69, 71, 72], 10)), layout="sequence")
    options = {"pageHeight": 700, "pageWidth": 1000, "scale": 60, "adjustPageHeight": False}
    before = options.copy()
    pages = render_svg_pages(mei, options=options)
    assert len(pages) > 1
    assert options == before
    assert "<svg" in render_svg(mei, page=len(pages), options=options)
    with pytest.raises(ValueError, match="page"):
        render_svg(mei, page=len(pages) + 1, options=options)
    captured = capfd.readouterr()
    assert "[Warning]" not in captured.out + captured.err
    assert "[Error]" not in captured.out + captured.err


def test_sequence_labels_render_at_one_height(make_mei, capfd):
    # G#6 with a sharp: without a shared @vgrp Verovio lifts '-19' (wide enough to
    # reach the sharp) above '-3', so the labels wobble (y 275 vs 434 in Verovio 6.3).
    mei, _ = make_mei(snapshot([91.81, 91.97, 91.81, 91.97]), layout="sequence", labels=True)
    svg = ET.fromstring(render_svg(mei))
    texts = svg.findall(".//{http://www.w3.org/2000/svg}g[@class='dir cents']/{http://www.w3.org/2000/svg}text")
    assert ["".join(t.itertext()).strip() for t in texts] == ["-19", "-3", "-19", "-3"]
    assert len({t.get("y") for t in texts}) == 1
    captured = capfd.readouterr()
    assert "[Warning]" not in captured.out + captured.err


def test_default_render_has_no_title_header_and_fits_the_music(make_mei, capfd):
    mei, _ = make_mei(title="Spectral chord title")
    svg = render_svg(mei)
    assert "Spectral chord title" not in svg
    width = float(etree.fromstring(svg.encode()).get("width").removesuffix("px"))
    assert width < 1000
    assert "Spectral chord title" in render_svg(mei, options={"header": "auto"})
    captured = capfd.readouterr()
    assert "[Warning]" not in captured.out + captured.err


@pytest.mark.parametrize("page", [0, -1, 2, 1.5, True, np.bool_(True), "1"])
def test_bad_page(make_mei, page):
    mei, _ = make_mei()
    with pytest.raises(ValueError, match="page"):
        render_svg(mei, page=page)


@pytest.mark.parametrize("render", [render_svg, render_svg_pages])
def test_invalid_mei(render, capfd):
    # Deliberately malformed input is the sole document not schema-validated.
    with pytest.raises(ValueError, match="could not load"):
        render("<mei>")
    capfd.readouterr()  # Verovio itself may emit errors for intentionally bad XML.


@pytest.mark.parametrize("render", [render_svg, render_svg_pages])
def test_verovio_missing(make_mei, monkeypatch, render):
    mei, _ = make_mei()
    monkeypatch.setitem(sys.modules, "verovio", None)
    with pytest.raises(ImportError, match=r"audiospylt\[notation\]"):
        render(mei)


def test_core_has_no_display_imports_and_verovio_is_lazy():
    # Static rule here; test_notation_modules_load_no_display_or_verovio checks sys.modules.
    directory = Path(__file__).parent.parent / "audiospylt" / "notation"
    forbidden = {"IPython", "ipywidgets", "matplotlib", "plotly"}
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [(node.module or "").split(".")[0]]
            else:
                continue
            assert not forbidden.intersection(modules), path
            if "verovio" in modules:
                ancestor = parents.get(node)
                while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ancestor = parents.get(ancestor)
                assert ancestor is not None, path
    init = ast.parse((directory / "__init__.py").read_text())
    assert len(init.body) == 1 and isinstance(init.body[0], ast.Expr)
    assert isinstance(init.body[0].value, ast.Constant) and isinstance(init.body[0].value.value, str)


def test_notation_modules_load_no_display_or_verovio():
    root = Path(__file__).parent.parent
    script = (
        "import sys\n"
        "import audiospylt.notation.mei, audiospylt.notation.pitch, audiospylt.notation.render\n"
        "tops = {name.partition('.')[0] for name in sys.modules}\n"
        "assert not {'IPython', 'ipywidgets', 'matplotlib', 'plotly', 'verovio'} & tops, tops\n"
    )
    modules = {path.stem for path in (root / "audiospylt" / "notation").glob("*.py")} - {"__init__"}
    assert modules == {"mei", "pitch", "render"}
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(root)})
    assert result.returncode == 0, result.stderr
