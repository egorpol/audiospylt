"""Notebook helpers for AudioSpylt v1; display needs ``audiospylt[notebook]``.

IPython is imported only when a helper is called.
"""

from .display import play, show, show_mei, show_svg

__all__ = ["show", "show_svg", "show_mei", "play"]
