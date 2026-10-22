"""Render MEI to SVG strings without file or display side effects.

Verovio 6.3's Python binding writes warnings to C-level stderr (fd 2):
toolkit.getLog() stays empty and enableLogToBuffer is unavailable. The library
does not capture process file descriptors; callers/tests can use fd capture.
"""

from __future__ import annotations

from numbers import Integral

__all__ = ["render_svg", "render_svg_pages"]


# No page header (it would print the meiHead title); crop the page to the music.
_DEFAULT_OPTIONS = {"adjustPageHeight": True, "adjustPageWidth": True, "header": "none",
                    "pageWidth": 3000, "scale": 50}


def _load(mei, options):
    try:
        import verovio
    except ImportError as error:
        raise ImportError("SVG rendering requires verovio; install audiospylt[notation]") from error
    toolkit = verovio.toolkit()
    toolkit.setOptions({**_DEFAULT_OPTIONS, **(options if options is not None else {})})
    if not toolkit.loadData(mei):
        raise ValueError("Verovio could not load the MEI document")
    return toolkit


def render_svg(mei: str, *, page: int = 1, options: dict | None = None) -> str:
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

    Returns
    -------
    str
        SVG XML, without writing files, printing, or displaying it.

    Raises
    ------
    ImportError
        Verovio is missing; install audiospylt[notation].
    ValueError
        MEI cannot be loaded, or page is outside 1 through the page count.
    """
    toolkit = _load(mei, options)
    count = toolkit.getPageCount()
    if isinstance(page, bool) or not isinstance(page, Integral) or not 1 <= page <= count:
        raise ValueError(f"page must be an integer in 1…{count}; got {page!r}")
    return toolkit.renderToSVG(int(page))


def render_svg_pages(mei: str, *, options: dict | None = None) -> list[str]:
    """Render all MEI pages with a lazily imported Verovio binding.

    Parameters
    ----------
    mei : str
        MEI XML document.
    options : dict or None, default None
        Verovio options merged over the spectral-chord defaults of render_svg.

    Returns
    -------
    list[str]
        SVG pages in order, with no file writes, printing, or display.

    Raises
    ------
    ImportError
        Verovio is missing; install audiospylt[notation].
    ValueError
        MEI cannot be loaded.
    """
    toolkit = _load(mei, options)
    return [toolkit.renderToSVG(page) for page in range(1, toolkit.getPageCount() + 1)]
