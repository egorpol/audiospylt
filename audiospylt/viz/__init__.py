"""Plotly figures for AudioSpylt v1; needs ``audiospylt[plot]``.

Every function returns a ``plotly.graph_objects.Figure`` and displays nothing.
"""

from .morph import plot_morph_map
from .signal import plot_spectrogram, plot_spectrum, plot_waveform
from .tables import plot_events, plot_events_3d, plot_snapshot

__all__ = [
    "plot_waveform",
    "plot_spectrum",
    "plot_spectrogram",
    "plot_snapshot",
    "plot_events",
    "plot_events_3d",
    "plot_morph_map",
]
