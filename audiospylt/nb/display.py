"""Notebook display helpers; IPython is imported only inside callables."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

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


def _ipython_display():
    try:
        from IPython.display import display
    except ImportError as error:
        raise ImportError(
            "notebook display requires IPython; install audiospylt[notebook]"
        ) from error
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


def _validate_background(background: str) -> str:
    if not isinstance(background, str):
        raise ValueError(
            f"background must be a #hex, CSS colour name, or rgb()/rgba(); "
            f"got {background!r}"
        )
    value = background.strip()
    if _HEX.fullmatch(value) or _RGB.fullmatch(value) or _CSS_NAME.fullmatch(value):
        return value
    raise ValueError(
        f"background must be a #hex, CSS colour name, or rgb()/rgba(); "
        f"got {background!r}"
    )


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
        IPython (or Plotly, for figures) is missing.
    ValueError
        ``obj`` looks like MEI.
    """
    try:
        from plotly.basedatatypes import BaseFigure
    except ImportError:
        BaseFigure = ()

    if BaseFigure and isinstance(obj, BaseFigure):
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
    background = _validate_background(background)
    body = svg.lstrip()
    body = _XML_DECL.sub("", body, count=1).lstrip()
    body = _DOCTYPE.sub("", body, count=1).lstrip()
    from IPython.display import HTML
    display(HTML(
        f'<div style="background:{background};display:inline-block;'
        f'padding:8px">{body}</div>'
    ))
    return None


def show_mei(
    mei: str,
    *,
    page: int | None = None,
    options: dict | None = None,
    background: str | None = "#ffffff",
) -> None:
    """Render MEI to SVG page(s) and display each with ``show_svg``.

    Parameters
    ----------
    mei : str
        MEI document.
    page : int or None
        One-based page; ``None`` renders every page.
    options : dict or None
        Verovio options for ``notation.render``.
    background : str or None
        Passed to ``show_svg``.

    Returns
    -------
    None
    """
    from audiospylt.notation.render import render_svg, render_svg_pages

    if page is None:
        pages = render_svg_pages(mei, options=options)
    else:
        pages = [render_svg(mei, page=page, options=options)]
    for svg in pages:
        show_svg(svg, background=background)
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
    try:
        from IPython.display import Audio, display
    except ImportError as error:
        raise ImportError(
            "notebook display requires IPython; install audiospylt[notebook]"
        ) from error
    display(Audio(data=array, rate=sr, normalize=normalize))
    return None
