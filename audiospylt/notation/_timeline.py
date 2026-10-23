"""Real-time ruler under each system of a Verovio rendering of events MEI.

The engraving is left as Verovio spaced it. Below each system a ruler shows
the system's time span at true scale, with a dot at each onset and a
connector from the onset's chord down to the dot. render.py makes room with
spacingSystem and asks for bounding boxes, which give the exact extent of
each system's content and each notehead; they are removed again here.
"""

from __future__ import annotations

import math
import re
from html import escape
import xml.etree.ElementTree as ET
from dataclasses import dataclass

_MEI = "{http://www.music-encoding.org/ns/mei}"
_SVG = "{http://www.w3.org/2000/svg}"
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
_ISO_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)")
_STAFF_LINE = re.compile(r"M\s*(-?[\d.]+)[\s,]+(-?[\d.]+)\s*L\s*(-?[\d.]+)[\s,]+(-?[\d.]+)")
_TRANSLATE = re.compile(r"translate\(\s*(-?[\d.]+)[\s,]+(-?[\d.]+)\s*\)")
_BOXES = re.compile(r'<g id="bbox-[^"]*"[^>]*/>|<g id="bbox-[^"]*"[^>]*>\s*<rect[^>]*/>\s*</g>')
_SVG_TAG = re.compile(r"<svg\b[^>]*>")
_VIEW_BOX = re.compile(r'viewBox="([^"]*)"')

# Strip layout in staff spaces below a system's lowest content: gap, connector
# (down, slanted, down), ruler, scale ticks and labels. Verovio's own CSS sets
# stroke on path/polyline/polygon/rect/ellipse, so the strip uses <line> and
# <circle>, whose attributes it leaves alone.
_GAP, _DROP, _SLANT, _RISE, _TICK, _FONT = 1.0, 0.5, 2.5, 0.5, 0.5, 1.8
_RULER = _DROP + _SLANT + _RISE
STRIP_SPACES = 8  # whole staff spaces reserved under every system
_CONNECTOR = "#898781"  # viz muted grey


@dataclass(frozen=True)
class Timeline:
    """Onset seconds and <when> id per MEI measure id, in document order; end time."""

    onsets: dict[str, float]
    whens: dict[str, str]
    end: float | None


def _seconds(text):
    match = _ISO_TIME.fullmatch(text.strip())
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_timeline(mei: str) -> Timeline | None:
    """Measures whose @when points at a <when> in ISO time; None if there are none."""
    try:
        root = ET.fromstring(mei)
    except ET.ParseError:
        return None
    whens: dict[str, float] = {}
    end = None
    for recording in root.iter(f"{_MEI}recording"):
        betype = recording.get("betype")
        if betype == "time":
            end = _seconds(recording.get("end", ""))
        for when in recording.iter(f"{_MEI}when"):
            ident, absolute = when.get(_XML_ID), when.get("absolute")
            if ident and absolute is not None and when.get("abstype", betype) == "time":
                seconds = _seconds(absolute)
                if seconds is not None:
                    whens[ident] = seconds
    onsets: dict[str, float] = {}
    refs: dict[str, str] = {}
    for measure in root.iter(f"{_MEI}measure"):
        ident, target = measure.get(_XML_ID), (measure.get("when") or "").strip()
        if ident and target.startswith("#") and target[1:] in whens:
            onsets[ident] = whens[target[1:]]
            refs[ident] = target
    return Timeline(onsets, refs, end) if onsets else None


def _classes(element):
    return element.get("class", "").split()


def _groups(element, name):
    # Verovio's bounding-box groups repeat their element's class ("note bounding-box").
    return [g for g in element.iter(f"{_SVG}g")
            if name in _classes(g) and (name == "bounding-box" or "bounding-box" not in _classes(g))]


def _rect(group):
    rect = group.find(f"{_SVG}rect")
    if rect is None:
        return None
    x, y = float(rect.get("x")), float(rect.get("y"))
    return x, y, x + float(rect.get("width")), y + float(rect.get("height"))


def _staff_lines(system):
    for staff in _groups(system, "staff"):
        for path in staff.findall(f"{_SVG}path"):
            match = _STAFF_LINE.match(path.get("d", ""))
            if match and match.group(2) == match.group(4):
                yield tuple(float(v) for v in match.groups())


def _anchor(measure):
    # Centre of the heads of each staff's first chord: the notehead itself, or
    # the stem position when a chord puts heads on both sides of the stem.
    spans = []
    for layer in _groups(measure, "layer"):
        # The first chord or note in document order, also inside a beam or tuplet.
        first = next((g for g in layer.iter(f"{_SVG}g") if {"chord", "note"} & set(_classes(g))
                      and "bounding-box" not in _classes(g)), None)
        if first is None:
            continue
        notes = [first] if "note" in _classes(first) else _groups(first, "note")
        for note in notes:
            box = next((_rect(g) for g in note if "bounding-box" in _classes(g)), None)
            if box is not None:
                spans.append(box)
    if not spans:
        return None
    return (min(b[0] for b in spans) + max(b[2] for b in spans)) / 2


def _colour(measure):
    return next((c.get("color") for c in _groups(measure, "chord") if c.get("color")), None)


def _step(t0, t1, width, font):
    # The smallest 1/2/2.5/5 step whose labels (the widest with " s") fit with
    # a gap of one label height between them.
    for exponent in range(-6, 7):
        for mantissa in (1, 2, 2.5, 5):
            step = mantissa * 10.0 ** exponent
            decimals = max(0, -exponent + (mantissa == 2.5))
            widest = len(f"{t1:.{decimals}f}") + 2
            if (t1 - t0) / step * (0.55 * font * widest + font) <= width:
                return step, decimals
    return None, 0


def _number(value):
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _line(x1, y1, x2, y2, css_class, stroke, width):
    return (f'<line class="{css_class}" x1="{_number(x1)}" y1="{_number(y1)}" '
            f'x2="{_number(x2)}" y2="{_number(y2)}" stroke="{escape(stroke)}" '
            f'stroke-width="{_number(width)}" stroke-linecap="round" />')


def _ruler(x0, x1, y, t0, t1, sp, step, decimals):
    font = _FONT * sp
    parts = [_line(x0, y, x1, y, "asp-axis", "currentColor", 0.1 * sp)]
    ticks = []
    if t1 > t0 and step is not None:
        first = math.ceil(t0 / step - 1e-9)
        last = math.floor(t1 / step + 1e-9)
        ticks = [(k * step, f"{k * step:.{decimals}f}") for k in range(first, last + 1)]
    if not ticks:
        ticks = [(t0, f"{t0:.3f}".rstrip("0").rstrip("."))]
    scale = (x1 - x0) / (t1 - t0) if t1 > t0 else 0.0
    for index, (seconds, label) in enumerate(ticks):
        x = x0 + (seconds - t0) * scale
        unit = " s" if index == len(ticks) - 1 else ""
        parts.append(_line(x, y, x, y + _TICK * sp, "asp-tick", "currentColor", 0.07 * sp))
        parts.append(f'<text x="{_number(x)}" y="{_number(y + _TICK * sp + 0.95 * font)}" '
                     f'font-size="{_number(font)}px" text-anchor="middle" '
                     f'fill="currentColor">{label}{unit}</text>')
    return '<g class="asp-ruler">' + "".join(parts) + "</g>"


def _onset(measure_id, when, xa, xt, top, sp, colour):
    y_drop, y_slant, y_ruler = top + _DROP * sp, top + (_DROP + _SLANT) * sp, top + _RULER * sp
    points = [(xa, top, xa, y_drop), (xa, y_drop, xt, y_slant), (xt, y_slant, xt, y_ruler)]
    # Transparent copies make the thin connector and the dot easy to hover.
    hits = [_line(*p, "asp-hit", "transparent", 0.8 * sp) for p in points]
    lines = [_line(*p, "asp-connector", colour or _CONNECTOR, 0.12 * sp) for p in points]
    # Ids and colours come from the MEI document: escape them as attribute values.
    return (f'<g class="asp-when" data-when="{escape(when)}" data-measure="{escape(measure_id)}">'
            + "".join(hits)
            + f'<circle class="asp-hit" cx="{_number(xt)}" cy="{_number(y_ruler)}" '
            f'r="{_number(0.6 * sp)}" fill="transparent" />'
            + "".join(lines)
            + f'<circle class="asp-dot" cx="{_number(xt)}" cy="{_number(y_ruler)}" '
            f'r="{_number(0.35 * sp)}" fill="{escape(colour or "currentColor")}" /></g>')


def _view_box(tag):
    match = _VIEW_BOX.search(tag)
    if match is None:
        raise ValueError("the SVG has no Verovio definition-scale viewBox")
    return [float(v) for v in match.group(1).split()]


def _resize(svg, extra):
    # Grow the outer <svg> (px height or viewBox) and the inner definition-scale
    # viewBox by `extra` viewBox units; Verovio cropped the page to the music.
    tags = _SVG_TAG.findall(svg)
    outer, inner = tags[0], tags[1]
    box = _view_box(inner)
    ratio = (box[3] + extra) / box[3]

    def grow(match):
        number = float(match.group(2)) * ratio
        return f'{match.group(1)}"{_number(number)}{match.group(3)}"'

    new_outer = re.sub(r'(\sheight=)"([\d.]+)(px|)"', grow, outer, count=1)
    view = _VIEW_BOX.search(new_outer)
    if view:
        values = [float(v) for v in view.group(1).split()]
        values[3] *= ratio
        new_outer = new_outer.replace(view.group(0),
                                      f'viewBox="{" ".join(_number(v) for v in values)}"')
    box[3] += extra
    new_inner = _VIEW_BOX.sub(f'viewBox="{" ".join(_number(v) for v in box)}"', inner, count=1)
    return svg.replace(outer, new_outer, 1).replace(inner, new_inner, 1)


def draw(svg: str, timeline: Timeline, *, keep_boxes: bool = False) -> str:
    """Add the rulers to one rendered page (bounding boxes required)."""
    root = ET.fromstring(svg)
    margin = next(iter(_groups(root, "page-margin")), None)
    if margin is None:
        raise ValueError("the SVG has no Verovio page-margin group")
    offset = _TRANSLATE.search(margin.get("transform", ""))
    top_margin = float(offset.group(2)) if offset else 0.0
    order = list(timeline.onsets)
    position = {measure_id: index for index, measure_id in enumerate(order)}
    systems = []
    for system in _groups(margin, "system"):
        lines = list(_staff_lines(system))
        boxes = [box for g in _groups(system, "bounding-box") if (box := _rect(g))]
        if not lines or not boxes:
            continue
        ys = sorted({line[1] for line in lines})
        sp = min(b - a for a, b in zip(ys, ys[1:])) if len(ys) > 1 else 180.0
        right = max(line[2] for line in lines)
        bottom = max(box[3] for box in boxes)
        onsets = []
        for measure in _groups(system, "measure"):
            measure_id = measure.get("id")
            if measure_id in timeline.onsets:
                x = _anchor(measure)
                if x is not None:
                    onsets.append((measure_id, x, _colour(measure)))
        if not onsets:
            continue
        index = position[onsets[-1][0]] + 1
        t0 = timeline.onsets[onsets[0][0]]
        if index < len(order):
            t1 = timeline.onsets[order[index]]
        else:
            t1 = max(timeline.end if timeline.end is not None else t0,
                     timeline.onsets[onsets[-1][0]])
        x0 = onsets[0][1]
        systems.append((x0, max(right, x0), bottom + _GAP * sp, t0, t1, sp, onsets))
    # One tick step per page: the coarsest that any of its systems needs.
    steps = [_step(t0, t1, x1 - x0, _FONT * sp)
             for x0, x1, _, t0, t1, sp, _ in systems if t1 > t0]
    step, decimals = max((s for s in steps if s[0] is not None), default=(None, 0))
    marks, lowest_strip = [], -math.inf
    for x0, x1, top, t0, t1, sp, onsets in systems:
        scale = (x1 - x0) / (t1 - t0) if t1 > t0 else 0.0
        marks.append(_ruler(x0, x1, top + _RULER * sp, t0, t1, sp, step, decimals))
        for measure_id, xa, colour in onsets:
            xt = x0 + (timeline.onsets[measure_id] - t0) * scale
            marks.append(_onset(measure_id, timeline.whens[measure_id], xa, xt, top, sp, colour))
        lowest_strip = max(lowest_strip, top + (_RULER + _TICK + 1.25 * _FONT) * sp)
    if not keep_boxes:
        svg = _BOXES.sub("", svg)
    if not marks:
        return svg
    close_inner = svg.rfind("</svg>", 0, svg.rfind("</svg>"))
    close_margin = svg.rfind("</g>", 0, close_inner)
    svg = (svg[:close_margin] + '<g class="asp-timeline">' + "".join(marks) + "</g>"
           + svg[close_margin:])
    # Keep Verovio's bottom margin (equal to its top margin by default).
    height = _view_box(_SVG_TAG.findall(svg)[1])[3]
    needed = top_margin + lowest_strip + top_margin
    return _resize(svg, needed - height) if needed > height else svg
