"""
Small helpers for plotting that make notebooks/scripts more robust.

The most common failure mode we see in Jupyter is Plotly raising:
  ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed

This module provides a `show_plotly(fig)` wrapper that:
- tries normal `fig.show()` first (inline in notebooks)
- if that fails due to missing notebook render deps, falls back to a non-inline renderer
  (default: "browser") or simply returns without raising.
"""

from __future__ import annotations

from typing import Any, Optional


def show_plotly(
    fig: Any,
    *,
    renderer: Optional[str] = None,
    fallback_renderer: str = "browser",
    warn: bool = True,
) -> Any:
    """
    Safely show a Plotly figure.

    - renderer: passed to fig.show(renderer=...)
    - fallback_renderer: used when inline rendering fails (e.g. nbformat missing)
    - warn: print a short message when falling back
    """
    try:
        if renderer is None:
            return fig.show()
        return fig.show(renderer=renderer)
    except Exception as e:
        msg = str(e)
        needs_nbformat = (
            "Mime type rendering requires nbformat" in msg
            or "requires nbformat>=4.2.0" in msg
            or "nbformat" in msg
        )
        if not needs_nbformat:
            raise

        if warn:
            print(
                "Plotly couldn't render inline (missing/old `nbformat`). "
                "Install `nbformat>=4.2.0` (and `ipython`) for notebook inline plots, "
                f"or we'll fall back to renderer='{fallback_renderer}'."
            )

        try:
            return fig.show(renderer=fallback_renderer)
        except Exception:
            # Last resort: don't crash user code; just return the figure.
            return fig


