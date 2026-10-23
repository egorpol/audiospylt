"""Deterministic MEI 5.1 notation of v1 snapshots and temporal chords."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np
import pandas as pd

from audiospylt.model.schema import validate
from audiospylt.notation.pitch import (
    _decimals, _positive_scalar, cents_label, pitch_table,
)

__all__ = ["snapshot_to_mei", "events_to_mei"]


_NS = "http://www.music-encoding.org/ns/mei"
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
# The viz light colorway without its orange and red, which nb.show_mei uses
# for hover. Copied, not imported: viz needs plotly.
_COLORS = ("#2a78d6", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#008300")
_HEX_COLOR = re.compile(r"#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})")
_DAY_US = 86_400_000_000


def _element(parent, name, **attrs):
    return ET.SubElement(parent, name, attrs)


def _label(measure, staff_no, note_id, place, text, kind, vgrp=None):
    # A bare <dir> starts at its note's left edge; <rend halign> centres it on the head.
    # Verovio copies @type into the SVG class ("dir cents"), which nb.show_mei uses for hover.
    direction = _element(measure, "dir", type=kind, staff=str(staff_no), startid=f"#{note_id}",
                         place=place, **({"vgrp": str(vgrp)} if vgrp is not None else {}))
    _element(direction, "rend", halign="center").text = text


def _prepare(table, kind, resolution, spelling, labels, decimals, clef, a4):
    table = validate(table, kind)
    if table.empty:
        raise ValueError(f"{kind} must not be empty")
    column = "freq" if kind == "snapshot" else "freq_start"
    zeros = table[column].eq(0)
    if zeros.any():
        raise ValueError(f"{int(zeros.sum())} row(s) at 0 Hz are unnotatable; filter them first")
    a4 = _positive_scalar(a4, "a4")
    pitches = pitch_table(table[column], resolution=resolution, spelling=spelling, a4=a4)
    decimals = _decimals(decimals)
    if not isinstance(clef, str) or clef not in ("G", "F", "GF"):
        raise ValueError("clef must be one of ('G', 'F', 'GF')")
    if labels is not None and not isinstance(labels, (bool, np.bool_)):
        raise ValueError("labels must be None, True, or False")
    if resolution == "eighth" and labels is not None and not labels:
        raise ValueError("resolution='eighth' requires cents labels; labels=False is invalid")
    labels = resolution == "eighth" if labels is None else labels
    outside = ~pitches.octave.between(0, 9)
    if outside.any():
        grid = 1.0 if resolution == "semitone" else 0.5
        # Include the accidental's octave: C0 quarter-flat is encodable,
        # whereas B9 quarter-sharp spelled as C10 quarter-flat is not.
        low = 11.5 if spelling == "flat" and grid == 0.5 else 12.0
        high = 131.5 if spelling == "sharp" and grid == 0.5 else 131.0
        low_hz = a4 * 2 ** ((low - grid / 2 - 69) / 12)
        high_hz = a4 * 2 ** ((high + grid / 2 - 69) / 12)
        raise ValueError(
            f"{int(outside.sum())} row(s) have written octaves outside MEI's 0–9; "
            f"encodable frequency range for {spelling}/{resolution}, a4={a4:g}: "
            f"[{low_hz:.12g}, {high_hz:.12g}) Hz (half-up boundaries). Filter by freq first."
        )
    return table, pitches, labels, decimals


def _document(clef, title):
    # A document-local default namespace avoids register_namespace's global
    # registry, which would also change serialization of the legacy 0.6 MEI.
    root = ET.Element("mei", {"xmlns": _NS, "meiversion": "5.1"})
    file_desc = _element(_element(root, "meiHead"), "fileDesc")
    _element(_element(file_desc, "titleStmt"), "title").text = title
    _element(file_desc, "pubStmt")
    score = _element(_element(_element(_element(root, "music"), "body"), "mdiv"), "score")
    group = _element(_element(score, "scoreDef"), "staffGrp",
                     **({"symbol": "brace"} if clef == "GF" else {}))
    staves = list(enumerate(clef, 1))
    for number, shape in staves:
        _element(group, "staffDef", n=str(number), lines="5",
                 **{"clef.shape": shape, "clef.line": "2" if shape == "G" else "4"})
    return root, _element(score, "section"), staves


def _height_groups(pitches, staves):
    # Verovio lifts each label clear of its own note (a wide label over a sharp
    # sits higher than a narrow one), so a sequence's labels wobble. Labels with
    # one @vgrp share the outermost height; a run ends at a staff change or
    # where its notes would span more than a third.
    groups, number, low, high, upper = [], 0, None, None, None
    for row in pitches.itertuples():
        step = row.octave * 7 + "cdefgab".index(row.step)
        on_upper = len(staves) == 2 and row.written_midi >= 60
        if low is None or on_upper != upper or max(high, step) - min(low, step) > 2:
            number, low, high, upper = number + 1, step, step, on_upper
        low, high = min(low, step), max(high, step)
        groups.append(number)
    return groups


def _measure(section, pitches, staves, next_id, *, chord=True, labels=False,
             decimals=0, naturals=False, onset_text=None, vgrp=None, when=None, color=None):
    number = len(section) + 1
    measure = _element(section, "measure", n=str(number), right="invis",
                       **{_XML_ID: f"m{number}"}, **({"when": when} if when else {}))
    ordered = pitches.sort_values("written_midi", kind="stable")
    parts = []
    for staff_no, clef in staves:
        rows = ordered
        if len(staves) == 2:
            rows = ordered[ordered.written_midi.ge(60) if clef == "G"
                           else ordered.written_midi.lt(60)]
        # Verovio draws at most two heads per step in a chord (either side of the
        # stem) and puts a third exactly on top of one, so a step's third and
        # fourth notes continue the chord in a second column, and so on.
        columns = rows.groupby(["step", "octave"]).cumcount().to_numpy() // 2
        parts.append((staff_no, clef, rows, columns))
    width = max((int(columns.max()) + 1 for *_, columns in parts if columns.size), default=1)
    top_anchor = None
    for staff_no, clef, rows, columns in parts:
        layer = _element(_element(measure, "staff", n=str(staff_no)), "layer", n="1")
        # Ids follow document order: column by column, each low to high.
        ids = np.empty(len(rows), dtype=object)
        ids[np.argsort(columns, kind="stable")] = [f"n{next_id + i}" for i in range(len(rows))]
        next_id += len(rows)
        altered = {(r.step, r.octave) for r in rows.itertuples() if r.accid}
        for column in range(width):
            at = np.flatnonzero(columns == column)
            if not at.size:
                _element(layer, "space", dur="4")
                continue
            parent = (_element(layer, "chord", dur="4", **{"stem.visible": "false"},
                               **({"color": color} if color else {}))
                      if chord else layer)
            texts = []
            for note_id, row in zip(ids[at], rows.iloc[at].itertuples()):
                attrs = {_XML_ID: note_id, "pname": row.step, "oct": str(row.octave)}
                if not chord:
                    attrs.update({"dur": "4", "stem.visible": "false"})
                accid = row.accid or ("n" if naturals or (row.step, row.octave) in altered else "")
                if accid:
                    attrs["accid"] = accid
                _element(parent, "note", **attrs)
                if labels:
                    texts.append((note_id, cents_label(row.residual, decimals)))
            # Verovio stacks dirs outward from the staff in document order, so labels
            # below the F staff go high to low to keep each column in the chord's order.
            for note_id, text in reversed(texts) if clef == "F" else texts:
                _label(measure, staff_no, note_id, "above" if clef == "G" else "below", text,
                       "cents", vgrp)
        # The first nonempty staff is uppermost; its last row is highest.
        if top_anchor is None and len(rows):
            top_anchor = (staff_no, ids[-1])
    if onset_text is not None and top_anchor is not None:
        staff_no, note_id = top_anchor
        _label(measure, staff_no, note_id, "above", onset_text, "onset")
    return next_id


def _iso_time(seconds):
    # MEI's ISO time (xsd:time) in whole microseconds, trailing zeros dropped.
    micro = round(seconds * 1e6)
    if micro >= _DAY_US:
        raise ValueError(f"MEI time values stop below 24 h; got {float(seconds)!r} s")
    clock, fraction = divmod(micro, 1_000_000)
    text = f"{clock // 3600:02d}:{clock // 60 % 60:02d}:{clock % 60:02d}"
    return text + f".{fraction:06d}".rstrip("0") if fraction else text


def _recording(root, end):
    # <performance> precedes <body>; @betype makes every <when absolute> ISO time.
    music = root.find("music")
    performance = ET.Element("performance")
    music.insert(0, performance)
    return _element(performance, "recording", begin="00:00:00", end=_iso_time(end),
                    betype="time")


def _palette(colors):
    if isinstance(colors, (bool, np.bool_)):
        return _COLORS if colors else None
    if (isinstance(colors, str) or not isinstance(colors, Sequence) or not colors
            or not all(isinstance(c, str) and _HEX_COLOR.fullmatch(c) for c in colors)):
        raise ValueError("colors must be True, False, or a nonempty sequence of "
                         f"'#rrggbb' / '#rrggbbaa' strings; got {colors!r}")
    return tuple(colors)


def _serialize(root):
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def snapshot_to_mei(
    snapshot: pd.DataFrame, *, layout: str = "chord", resolution: str = "quarter",
    spelling: str = "sharp", labels: bool | None = None, decimals: int = 0,
    naturals: bool = False, clef: str = "GF", a4: float = 440.0,
    title: str = "AudioSpylt notation",
) -> str:
    """Turn a v1 snapshot into a complete MEI 5.1 score.

    Parameters
    ----------
    snapshot : pandas.DataFrame
        Valid v1 snapshot. Amplitude is not notated; filter 0 Hz rows first.
    layout : {'chord', 'sequence'}, default 'chord'
        One chord per staff in one measure, ordered by written pitch (stable
        ties), or one note per measure in input row order. Verovio draws at
        most two noteheads per staff step (line or space) in a chord, so a
        staff with three or more notes on one step continues the chord in
        further columns of the same measure, two notes per step each (low to
        high); the other staff is padded with spaces. Chords read well
        when their notes lie a third or more apart; in denser snapshots (many
        partials within a semitone or two) Verovio puts noteheads on both
        sides of the stem and spreads accidentals over many columns. Use
        ``layout='sequence'`` for those (sort by ``freq`` first for a scale).
        Keeping fewer partials (``snapshot.nlargest(n, 'amp')``) helps only
        when the strongest ones are that far apart; in speech or noise
        spectra they often cluster.
    resolution, spelling, a4
        See pitch_table. Eighth uses the quarter-tone grid with mandatory labels.
    labels : bool or None, default None
        Cents remainder relative to each written note, centred on its notehead
        (MEI ``<dir type="cents">`` with ``<rend halign="center">``). In a chord the labels
        of each staff column stack, above the G staff or below the F staff,
        read from the top in the chord's order (high to low). In a
        sequence, consecutive notes on one staff spanning at most a third
        share a label height (MEI ``@vgrp``), so labels do not move up and
        down with each note's accidental. None enables labels only for
        eighth resolution; False is invalid in eighth mode.
    decimals : int, default 0
        Decimal places in cents labels.
    naturals : bool, default False
        Show all natural signs. Same-step/octave naturals sharing a staff's
        chord with an altered note always show a sign, regardless of this flag.
    clef : {'G', 'F', 'GF'}, default 'GF'
        Grand staff splits at written MIDI 60, inclusive on the upper G staff.
    title : str, default 'AudioSpylt notation'
        Document title; XML special characters are escaped.

    Returns
    -------
    str
        Deterministic MEI 5.1 XML, with note ids n1, n2, ... in document order.
        No files, printing, or display. Exact frequency is represented only by
        cents labels; no pnum or arrow accidentals are emitted.

    Raises
    ------
    audiospylt.model.schema.SchemaError
        The table violates the v1 snapshot schema.
    ValueError
        Empty/unnotatable input, written octave outside 0–9, or invalid options.
    """
    _, pitches, labels, decimals = _prepare(
        snapshot, "snapshot", resolution, spelling, labels, decimals, clef, a4)
    if not isinstance(layout, str) or layout not in ("chord", "sequence"):
        raise ValueError("layout must be one of ('chord', 'sequence')")
    root, section, staves = _document(clef, title)
    next_id = 1
    groups = [pitches] if layout == "chord" else [pitches.iloc[[i]] for i in range(len(pitches))]
    heights = (_height_groups(pitches, staves) if layout == "sequence" and labels
               else [None] * len(groups))
    for group, vgrp in zip(groups, heights):
        next_id = _measure(section, group, staves, next_id, chord=layout == "chord",
                           labels=labels, decimals=decimals, naturals=naturals, vgrp=vgrp)
    return _serialize(root)


def events_to_mei(
    events: pd.DataFrame, *, measure_seconds: float = 0.5, time_labels: bool = False,
    time_format: str = "{:.3f} s", resolution: str = "quarter", spelling: str = "sharp",
    labels: bool | None = None, decimals: int = 0, naturals: bool = False,
    clef: str = "GF", a4: float = 440.0, title: str = "AudioSpylt notation",
    colors: bool | Sequence[str] = False,
) -> str:
    """Notate event onsets as temporal chords; glides are not drawn.

    Parameters
    ----------
    events : pandas.DataFrame
        Valid v1 events. Only freq_start is notated; freq_stop and amplitudes
        do not affect written pitch. Filter rows starting at 0 Hz first.
    measure_seconds : float, default 0.5
        Finite positive measure span. Each onset group occupies
        max(1, round(span / measure_seconds)) measures: a chord followed by
        empty padding. Span is the gap to the next onset, or for the final
        group, its latest time_stop minus its onset. Overlaps do not extend
        earlier groups. round uses Python's nearest-even tie rule.
    time_labels : bool, default False
        Label each onset above the highest note of its uppermost occupied staff
        (MEI ``<dir type="onset">``).
    time_format : str, default '{:.3f} s'
        Format string applied to the onset in seconds.
    resolution, spelling, labels, decimals, naturals, clef, a4, title
        See snapshot_to_mei.
    colors : bool or sequence of str, default False
        Colour each onset's chords (MEI ``chord@color``), cycling through a
        palette: True uses six colours of the ``viz`` light theme (not the
        orange of the ``nb.show_mei`` hover), or pass ``'#rrggbb'`` /
        ``'#rrggbbaa'`` strings. The timeline drawn by render_svg uses the
        same colour for the onset's connector and dot.

    Returns
    -------
    str
        Deterministic MEI 5.1 XML; equal time_start rows share a chord (in
        columns as in snapshot_to_mei), onsets are sorted, and every measure
        contains every configured staff. Measures have ids m1, m2, ... The
        real times are encoded too: ``<performance><recording>`` (begin 0,
        end = the latest time_stop, ``betype="time"``) holds one
        ``<when xml:id="w1" absolute="HH:MM:SS.ffffff">`` per onset (whole
        microseconds), and the measure starting at that onset points at it
        with ``@when``. render_svg draws these as a time ruler.

    Raises
    ------
    audiospylt.model.schema.SchemaError
        The table violates the v1 events schema.
    ValueError
        Empty/unnotatable input, written octave outside 0–9, times of 24 h
        or more (MEI's time format), or invalid options.
    """
    events, pitches, labels, decimals = _prepare(
        events, "events", resolution, spelling, labels, decimals, clef, a4)
    measure_seconds = _positive_scalar(measure_seconds, "measure_seconds")
    palette = _palette(colors)
    root, section, staves = _document(clef, title)
    recording = _recording(root, events.time_stop.max())
    groups = list(events.groupby("time_start", sort=True))
    next_id = 1
    for i, (onset, group) in enumerate(groups):
        end = groups[i + 1][0] if i + 1 < len(groups) else group.time_stop.max()
        measures = max(1, round((end - onset) / measure_seconds))
        _element(recording, "when", **{_XML_ID: f"w{i + 1}"}, absolute=_iso_time(onset))
        next_id = _measure(section, pitches.loc[group.index], staves, next_id,
                           labels=labels, decimals=decimals, naturals=naturals,
                           onset_text=time_format.format(onset) if time_labels else None,
                           when=f"#w{i + 1}",
                           color=palette[i % len(palette)] if palette else None)
        for _ in range(measures - 1):
            _measure(section, pitches.iloc[:0], staves, next_id)
    return _serialize(root)
