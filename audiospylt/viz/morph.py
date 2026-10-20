"""Plotly figure for a ``transform.morph.pair`` pairing table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audiospylt.analysis.spectrum import amplitude_to_db
from audiospylt.viz._common import (
    _freq_scale,
    _theme_name,
    apply_layout,
    colorway_colour,
    go,
    marker_style,
    pitch_hover_label,
    theme_colors,
)

_REQUIRED = (
    "src", "dst", "freq_start", "freq_stop",
    "amp_source", "amp_target", "mass",
    "primary_start", "primary_stop", "amp_start", "amp_stop",
)
_MASS_EQUAL_TOL = 1e-12
# Visual style and legend rank (lower legendrank appears first in the legend).
_CATEGORY_STYLE = {
    "kept": ("solid", 0, 1),
    "birth": ("dash", 1, 2),
    "death": ("dot", 2, 3),
    "silent": ("dashdot", None, 4),
}
# Draw order: silent under death under birth under kept (kept on top).
_DRAW_ORDER = ("silent", "death", "birth", "kept")


def _category(row) -> str:
    start = bool(row.primary_start)
    stop = bool(row.primary_stop)
    if start and stop:
        return "kept"
    if stop and not start:
        return "birth"
    if start and not stop:
        return "death"
    return "silent"


def _default_title(attrs: dict) -> str | None:
    morph = attrs.get("morph")
    if not isinstance(morph, dict):
        return None
    method = morph.get("method", "morph")
    parts = [f"morph map: {method}"]
    parameter = morph.get("parameter")
    weight = morph.get("weight")
    if parameter is not None:
        parts.append(f"parameter={parameter:g}")
    if weight is not None:
        parts.append(f"weight={weight}")
    if len(parts) == 1:
        return parts[0]
    return parts[0] + " (" + ", ".join(parts[1:]) + ")"


def plot_morph_map(pairing, *, freq_scale="log", theme="light", title=None):
    """Return a 2-D morph map of a ``pair`` result (source x=0, target x=1).

    Colour and dash both encode the D13 category (kept / birth / death /
    silent). Never shows, prints, or writes files; inputs are unchanged.

    Parameters
    ----------
    pairing : pandas.DataFrame
        Output of ``transform.morph.pair``. Missing required columns raise
        ValueError listing them. ``attrs['morph']`` is optional.
    freq_scale : {'linear', 'log'}, default 'log'
    theme : {'light', 'dark'}
    title : str or None
        Default from ``attrs['morph']`` when present.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        Missing columns or invalid theme / freq_scale.
    """
    if not isinstance(pairing, pd.DataFrame):
        raise TypeError(f"pairing must be a DataFrame; got {type(pairing).__name__}")
    missing = [name for name in _REQUIRED if name not in pairing.columns]
    if missing:
        raise ValueError(f"pairing is missing columns: {missing}")
    theme = _theme_name(theme)
    freq_scale = _freq_scale(freq_scale)
    if title is None:
        title = _default_title(getattr(pairing, "attrs", {}))

    masses = pairing["mass"].to_numpy(dtype=np.float64)
    max_mass = float(masses.max()) if len(masses) else 1.0
    equal_mass = (
        len(masses) == 0
        or float(np.max(np.abs(masses - masses[0]))) <= _MASS_EQUAL_TOL * max(1.0, abs(masses[0]))
    )

    categories: dict[str, list[int]] = {name: [] for name in _DRAW_ORDER}
    for index, row in pairing.iterrows():
        categories[_category(row)].append(index)

    fig = go.Figure()
    present = []
    for name in _DRAW_ORDER:
        rows = categories[name]
        if not rows:
            continue
        present.append(name)
        dash, slot, legendrank = _CATEGORY_STYLE[name]
        if name == "silent":
            colour = theme_colors(theme)["muted"]
            legend = f"silent ({len(rows)}), dropped by morph"
        else:
            colour = colorway_colour(theme, slot)
            legend = f"{name} ({len(rows)})"
        first = True
        for row_index in rows:
            row = pairing.loc[row_index]
            f0 = float(row.freq_start)
            f1 = float(row.freq_stop)
            if freq_scale == "log" and (f0 <= 0 or f1 <= 0):
                continue
            mass = float(row.mass)
            if name == "silent":
                width = 1.0
            elif equal_mass:
                width = 2.0
            else:
                width = 1.0 + 5.0 * mass / max_mass
            hover = (
                f"row: {row_index}<br>"
                f"{int(row.src)} → {int(row.dst)}<br>"
                f"mass: {mass:.4g}<br>"
                f"amp: {float(row.amp_start):.4g} → {float(row.amp_stop):.4g}"
                f"<extra></extra>"
            )
            fig.add_trace(go.Scatter(
                x=[0.0, 1.0], y=[f0, f1], mode="lines",
                name=legend, legendgroup=name, showlegend=first,
                legendrank=legendrank,
                line=dict(width=width, color=colour, dash=dash),
                hovertemplate=hover,
            ))
            first = False

    # Partials: one marker per distinct src at x=0 and dst at x=1
    src_amp = {}
    dst_amp = {}
    src_freq = {}
    dst_freq = {}
    for _, row in pairing.iterrows():
        src, dst = int(row.src), int(row.dst)
        src_amp[src] = max(src_amp.get(src, 0.0), float(row.amp_source))
        dst_amp[dst] = max(dst_amp.get(dst, 0.0), float(row.amp_target))
        src_freq[src] = float(row.freq_start)
        dst_freq[dst] = float(row.freq_stop)
    all_amps = list(src_amp.values()) + list(dst_amp.values())
    max_amp = max(all_amps) if all_amps else 1.0
    if max_amp <= 0:
        max_amp = 1.0
    ink = theme_colors(theme)["ink"]

    def _add_partials(side: str, indices, freqs, amps, x: float):
        xs, ys, sizes, texts = [], [], [], []
        for index in indices:
            freq = freqs[index]
            if freq_scale == "log" and freq <= 0:
                continue
            amp = amps[index]
            size = 6.0 + 12.0 * np.sqrt(amp / max_amp)
            xs.append(x)
            ys.append(freq)
            sizes.append(size)
            db = float(amplitude_to_db(amp))
            pitch = pitch_hover_label(freq)
            text = (
                f"partial: {index}<br>"
                f"frequency: {freq:.2f} Hz<br>"
                + (f"pitch: {pitch}<br>" if pitch else "")
                + f"amplitude: {db:.1f} dB"
            )
            texts.append(text)
        if not xs:
            return
        style = marker_style(ink, theme, size=8)
        style["size"] = sizes
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=side, showlegend=False,
            marker=style,
            hovertemplate="%{text}<extra></extra>",
            text=texts,
        ))

    _add_partials("source", sorted(src_amp), src_freq, src_amp, 0.0)
    _add_partials("target", sorted(dst_amp), dst_freq, dst_amp, 1.0)

    n_legend = len(present)
    apply_layout(fig, theme=theme, title=title, n_series=max(n_legend, 1))
    fig.update_xaxes(
        title_text="",
        tickmode="array", tickvals=[0, 1], ticktext=["source", "target"],
        range=[-0.05, 1.05],
    )
    fig.update_yaxes(title_text="frequency (Hz)", type=freq_scale)
    return fig
