"""Render MEI to SVG strings without file or display side effects.

Verovio 6.3's Python binding writes warnings to C-level stderr (fd 2):
toolkit.getLog() stays empty and enableLogToBuffer is unavailable. The library
does not capture process file descriptors; callers/tests can use fd capture.
"""

from __future__ import annotations

import json
from numbers import Integral

from audiospylt.notation import _timeline

__all__ = ["render_svg", "render_svg_pages"]


# No page header (it would print the meiHead title); crop the page to the music.
_DEFAULT_OPTIONS = {"adjustPageHeight": True, "adjustPageWidth": True, "header": "none",
                    "pageWidth": 3000, "scale": 50}


def _timeline_for(mei, timeline, options):
    if timeline is not None and not isinstance(timeline, bool):
        raise ValueError(f"timeline must be None, True, or False; got {timeline!r}")
    marks = None if timeline is False else _timeline.read_timeline(mei)
    if timeline and marks is None:
        raise ValueError("timeline=True needs real onset times in the MEI (measure@when "
                         "pointing at <when absolute> in ISO time), as events_to_mei writes")
    if marks is not None and (options or {}).get("svgHtml5"):
        raise ValueError("the timeline cannot be drawn with svgHtml5; pass timeline=False")
    return marks


def _load(mei, options, timeline=None):
    try:
        import verovio
    except ImportError as error:
        raise ImportError("SVG rendering requires verovio; install audiospylt[notation]") from error
    toolkit = verovio.toolkit()
    toolkit.setOptions({**_DEFAULT_OPTIONS, **(options if options is not None else {})})
    if timeline is not None:
        # Room for the ruler under every system, and exact extents to place it.
        current = toolkit.getOptions()  # a dict in Verovio 6; JSON text in older bindings
        if isinstance(current, str):
            current = json.loads(current)
        spacing = int(current["spacingSystem"])
        toolkit.setOptions({"spacingSystem": spacing + 2 * _timeline.STRIP_SPACES,
                            "svgBoundingBoxes": True})
    if not toolkit.loadData(mei):
        raise ValueError("Verovio could not load the MEI document")
    return toolkit


def _render(toolkit, page, timeline, options):
    svg = toolkit.renderToSVG(page)
    if timeline is None:
        return svg
    return _timeline.draw(svg, timeline,
                          keep_boxes=bool((options or {}).get("svgBoundingBoxes")))


def render_svg(mei: str, *, page: int = 1, options: dict | None = None,
               timeline: bool | None = None) -> str:
    """Render one page of MEI with a lazily imported Verovio binding.

    Parameters
    ----------
    mei : str
        MEI XML document.
    page : int, default 1
        One-based page number (booleans rejected).
    options : dict or None, default None
        Verovio options overriding adjustPageHeight=True, adjustPageWidth=True,
        header='none' (no title header), pageWidth=3000, scale=50. The caller's
        dictionary is not modified.
    timeline : bool or None, default None
        Draw a time ruler under each system, as for events_to_mei output: the
        system's time span at true scale (from its first onset to the next
        system's first onset, or the recording end), a dot at each onset and a
        connector from the onset's chord to its dot, in the chord's colour if
        it has one. The engraving keeps Verovio's spacing; ``spacingSystem``
        grows by 16 MEI units to make room. Onsets are measures whose
        ``@when`` points at a ``<when absolute>`` in ISO time (a ``recording``
        with ``betype="time"`` or ``abstype="time"``). None draws the ruler
        when the MEI has such onsets, True requires them, False never draws.
        The ruler's SVG groups carry classes ``asp-ruler`` and ``asp-when``
        (``data-measure``, ``data-when``), which ``nb.show_mei`` uses for hover.

    Returns
    -------
    str
        SVG XML, without writing files, printing, or displaying it.

    Raises
    ------
    ImportError
        Verovio is missing; install audiospylt[notation].
    ValueError
        MEI cannot be loaded, page is outside 1 through the page count,
        ``timeline=True`` without onset times, or a timeline with the
        ``svgHtml5`` option.
    """
    marks = _timeline_for(mei, timeline, options)
    toolkit = _load(mei, options, marks)
    count = toolkit.getPageCount()
    if isinstance(page, bool) or not isinstance(page, Integral) or not 1 <= page <= count:
        raise ValueError(f"page must be an integer in 1…{count}; got {page!r}")
    return _render(toolkit, int(page), marks, options)


def render_svg_pages(mei: str, *, options: dict | None = None,
                     timeline: bool | None = None) -> list[str]:
    """Render all MEI pages with a lazily imported Verovio binding.

    Parameters
    ----------
    mei : str
        MEI XML document.
    options : dict or None, default None
        Verovio options merged over the spectral-chord defaults of render_svg.
    timeline : bool or None, default None
        See render_svg; a system's span runs to the next onset even when that
        is on the next page.

    Returns
    -------
    list[str]
        SVG pages in order, with no file writes, printing, or display.

    Raises
    ------
    ImportError
        Verovio is missing; install audiospylt[notation].
    ValueError
        MEI cannot be loaded, or the timeline cannot be drawn (see render_svg).
    """
    marks = _timeline_for(mei, timeline, options)
    toolkit = _load(mei, options, marks)
    return [_render(toolkit, page, marks, options)
            for page in range(1, toolkit.getPageCount() + 1)]
