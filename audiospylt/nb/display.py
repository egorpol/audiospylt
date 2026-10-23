"""Notebook display helpers; IPython is imported only inside callables."""

from __future__ import annotations

import re
import sys
import uuid
from typing import TYPE_CHECKING

import numpy as np

from audiospylt._optional import import_optional
from audiospylt.io.audio import _audio_array, _sample_rate

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "show",
    "show_svg",
    "show_mei",
    "play",
]

_HEX = re.compile(r"^#[0-9A-Fa-f]{3,8}$")
_RGB = re.compile(r"^rgba?\([^)]*\)$", re.IGNORECASE)
_CSS_NAME = re.compile(r"^[A-Za-z]+$")
_XML_DECL = re.compile(r"^\s*<\?xml\b[^>]*\?>", re.IGNORECASE)
_DOCTYPE = re.compile(r"^\s*<!DOCTYPE\b[^>]*>", re.IGNORECASE | re.DOTALL)
_SVG_GROUP = re.compile(r"<g\b[^>]*>")
_SVG_ATTR = re.compile(r'([\w:.-]+)="([^"]*)"')


def _ipython_display():
    import_optional("IPython", extra="notebook", feature="notebook display")
    from IPython.display import display
    return display


def _is_svg(text: str) -> bool:
    trimmed = text.lstrip()
    if trimmed.lower().startswith("<?xml"):
        end = trimmed.find("?>")
        if end != -1:
            trimmed = trimmed[end + 2:].lstrip()
    return trimmed[:4].lower() == "<svg"


def _contains_mei(text: str) -> bool:
    return "<mei" in text.lower()


def _validate_colour(colour: str, name: str = "background") -> str:
    if isinstance(colour, str):
        value = colour.strip()
        if _HEX.fullmatch(value) or _RGB.fullmatch(value) or _CSS_NAME.fullmatch(value):
            return value
    raise ValueError(
        f"{name} must be a #hex, CSS colour name, or rgb()/rgba(); got {colour!r}"
    )


def _svg_div(svg: str, background: str | None, *, scope: str | None = None,
             css: str = "") -> str:
    body = svg.lstrip()
    body = _XML_DECL.sub("", body, count=1).lstrip()
    body = _DOCTYPE.sub("", body, count=1).lstrip()
    box = ("display:inline-block" if background is None
           else f"background:{background};display:inline-block;padding:8px")
    ident = f' id="{scope}"' if scope is not None else ""
    return f'<div{ident} style="{box}">{css}{body}</div>'


def _hover_notes(svg: str) -> list[str]:
    # Verovio SVG: <g id="n5" class="note">, and with svgAdditionalAttribute
    # dir@startid, <g class="dir cents" data-startid="#n5">. Onset labels
    # (class "dir onset") belong to the whole event, not to one note.
    notes, targets = set(), []
    for tag in _SVG_GROUP.findall(svg):
        attrs = dict(_SVG_ATTR.findall(tag))
        classes = attrs.get("class", "").split()
        if "note" in classes and "id" in attrs:
            notes.add(attrs["id"])
        elif "dir" in classes and "onset" not in classes:
            start = attrs.get("data-startid", "")
            if start.startswith("#") and start[1:] not in targets:
                targets.append(start[1:])
    return [note for note in targets if note in notes]


def _hover_measures(svg: str) -> list[str]:
    # notation.render's timeline: <g class="asp-when" data-measure="m3"> per
    # onset, next to Verovio's <g id="m3" class="measure">.
    measures: set[str] = set()
    targets: list[str] = []
    for tag in _SVG_GROUP.findall(svg):
        attrs = dict(_SVG_ATTR.findall(tag))
        classes = attrs.get("class", "").split()
        if "measure" in classes and "id" in attrs:
            measures.add(attrs["id"])
        elif "asp-when" in classes and attrs.get("data-measure") not in (None, *targets):
            targets.append(attrs["data-measure"])
    return [measure for measure in targets if measure in measures]


def _hover_css(svg: str, scope: str, colour: str) -> str:
    # One selector per pair: hovering either member colours both. Scoped to
    # the output's div because notebooks repeat ids n1, n2, … across scores.
    fills, strokes = [], []
    for note in _hover_notes(svg):
        pair = f'g.note[id="{note}"], g.dir:not(.onset)[data-startid="#{note}"]'
        fills.append(f"#{scope}:has(:is({pair}):hover) :is({pair})")
    # Any note of an onset, its onset label, or its connector/dot colours the
    # connector, dot and label; the connector/dot or label also colours every
    # note of the onset and their cents labels.
    for measure in _hover_measures(svg):
        notes, when = f'g.measure[id="{measure}"]', f'g.asp-when[data-measure="{measure}"]'
        time = f":is({when}, {notes} g.dir.onset)"
        anything = f":is({notes} g.note, {when}, {notes} g.dir.onset)"
        strokes.append(f"#{scope}:has({anything}:hover) {when} .asp-connector")
        fills.append(f"#{scope}:has({anything}:hover) :is({when} .asp-dot, {notes} g.dir.onset)")
        fills.append(f"#{scope}:has({time}:hover) :is({notes} g.note, {notes} g.dir.cents)")
    blocks = []
    if strokes:
        blocks.append(",\n".join(strokes) + f" {{ stroke: {colour}; }}")
    if fills:
        blocks.append(",\n".join(fills) + f" {{ fill: {colour}; }}")
    return "<style>" + "\n".join(blocks) + "</style>" if blocks else ""


def show(obj: object, *, renderer: str | None = None) -> None:
    """Display a Plotly figure, SVG string, or arbitrary notebook object.

    Parameters
    ----------
    obj : object
        A Plotly ``BaseFigure`` (calls ``show``), an SVG string (``show_svg``),
        or anything else passed to ``IPython.display.display``. A string
        containing ``<mei`` raises, pointing at ``show_mei``.
    renderer : str or None
        Optional Plotly renderer name.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        IPython is missing (for anything but a Plotly figure).
    ValueError
        ``obj`` looks like MEI.
    """
    # A figure exists only once plotly is loaded, so no import is needed to test for one.
    plotly_base = sys.modules.get("plotly.basedatatypes")
    if plotly_base is not None and isinstance(obj, plotly_base.BaseFigure):
        if renderer is None:
            obj.show()
        else:
            obj.show(renderer=renderer)
        return None
    if isinstance(obj, str):
        if _contains_mei(obj):
            raise ValueError("MEI strings must be shown with show_mei, not show")
        if _is_svg(obj):
            return show_svg(obj)
    display = _ipython_display()
    display(obj)
    return None


def show_svg(svg: str, *, background: str | None = "#ffffff") -> None:
    """Display an SVG string, optionally on a solid background.

    With a background (default white), strips a leading XML declaration or
    doctype and wraps the SVG in an HTML div so it stays legible in dark
    notebook themes (0.6 ``notebook_white_background``). Files are unchanged.

    Parameters
    ----------
    svg : str
        SVG markup.
    background : str or None, default '#ffffff'
        ``None`` displays raw SVG; otherwise a ``#hex`` (3–8 digits), CSS
        colour name, or ``rgb(...)`` / ``rgba(...)``.

    Returns
    -------
    None
    """
    display = _ipython_display()
    if background is None:
        from IPython.display import SVG
        display(SVG(svg))
        return None
    background = _validate_colour(background)
    from IPython.display import HTML
    display(HTML(_svg_div(svg, background)))
    return None


def show_mei(
    mei: str,
    *,
    page: int | None = None,
    options: dict | None = None,
    background: str | None = "#ffffff",
    highlight: str | None = "#eb6834",
    timeline: bool | None = None,
) -> None:
    """Render MEI to SVG page(s) and display each page.

    With ``highlight``, hovering a notehead colours it and its cents label,
    and hovering a label colours it and its note: every MEI ``<dir>`` with a
    ``@startid`` pairs with that note, except onset labels
    (``<dir type="onset">``). With a time ruler (events_to_mei output, see
    ``notation.render.render_svg``), hovering a note or the onset label
    also colours the onset's connector and dot, and hovering the connector,
    dot or onset label colours all the onset's notes and labels. This is CSS
    only (``:has()``, no script), scoped to each output; viewers that strip
    ``<style>`` from outputs show the plain score. Pages without anything to
    pair are shown with ``show_svg``.

    Parameters
    ----------
    mei : str
        MEI document.
    page : int or None
        One-based page; ``None`` renders every page.
    options : dict or None
        Verovio options for ``notation.render``. With ``highlight``,
        ``"dir@startid"`` is added to ``svgAdditionalAttribute`` (the
        caller's dictionary is not modified).
    background : str or None
        As in ``show_svg``. ``None`` with hover pairs keeps the HTML wrapper,
        without a background or padding.
    highlight : str or None, default '#eb6834'
        Hover colour: a ``#hex`` (3–8 digits), CSS colour name, or
        ``rgb(...)`` / ``rgba(...)``. ``None`` turns hover off.
    timeline : bool or None, default None
        Time ruler under each system, as in ``notation.render.render_svg``:
        None draws it when the MEI has onset times, True requires them,
        False never draws it.

    Returns
    -------
    None
    """
    from audiospylt.notation.render import render_svg, render_svg_pages

    if background is not None:
        background = _validate_colour(background)
    if highlight is not None:
        highlight = _validate_colour(highlight, "highlight")
        options = dict(options) if options is not None else {}
        extra = options.get("svgAdditionalAttribute", [])
        extra = [extra] if isinstance(extra, str) else list(extra)
        if "dir@startid" not in extra:
            extra.append("dir@startid")
        options["svgAdditionalAttribute"] = extra
    if page is None:
        pages = render_svg_pages(mei, options=options, timeline=timeline)
    else:
        pages = [render_svg(mei, page=page, options=options, timeline=timeline)]
    for svg in pages:
        scope = f"asp-mei-{uuid.uuid4().hex[:12]}"
        css = "" if highlight is None else _hover_css(svg, scope, highlight)
        if not css:
            show_svg(svg, background=background)
            continue
        display = _ipython_display()
        from IPython.display import HTML
        display(HTML(_svg_div(svg, background, scope=scope, css=css)))
    return None


def play(audio: ArrayLike, sr: int, *, normalize: bool = False) -> None:
    """Play audio in the notebook via ``IPython.display.Audio``.

    Channel-first 2-D arrays are passed through. With ``normalize=False`` and
    peak above full scale, raises instead of clipping (same policy as
    ``write_audio``).

    Parameters
    ----------
    audio : array-like
        1-D ``(samples,)`` or 2-D ``(channels, samples)``.
    sr : int
        Positive sample rate.
    normalize : bool, default False
        Forwarded to ``IPython.display.Audio``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Invalid audio, or peak > 1 with ``normalize=False``.
    ImportError
        IPython is missing.
    """
    array = _audio_array(audio)
    sr = _sample_rate(sr)
    if array.size == 0:
        raise ValueError("audio must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError("audio must contain only finite samples")
    peak = float(np.max(np.abs(array)))
    if not normalize and peak > 1.0:
        raise ValueError(
            f"audio peaks at {20 * np.log10(peak):+.2f} dB re full scale "
            f"({peak:.6g}), above full scale; pass normalize=True to play "
            f"without clipping"
        )
    display = _ipython_display()
    from IPython.display import Audio
    display(Audio(data=array, rate=sr, normalize=normalize))
    return None
