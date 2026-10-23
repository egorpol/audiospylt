"""Shared Plotly theme, validation, and display-only helpers for viz."""

from __future__ import annotations

from numbers import Integral
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_number

from audiospylt._optional import import_optional
from audiospylt.analysis.spectrum import amplitude_to_db
from audiospylt.model.schema import TABLE_CONTRACTS, infer_kind, validate
from audiospylt.notation.pitch import cents_label, pitch_table

import_optional("plotly", extra="plot", feature="plotting")
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
_SEQUENTIAL = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
_THEMES = {
    "light": {
        "paper": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "colorway": (
            "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#6250d6", "#e34948",
        ),
        "sequential": list(_SEQUENTIAL),
    },
    "dark": {
        "paper": "#1a1a19",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "colorway": (
            "#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767",
        ),
        "sequential": list(reversed(_SEQUENTIAL)),
    },
}
_TEMPLATES: dict[str, go.layout.Template] = {}


def _require_plotly() -> None:
    """No-op that documents the module-level plotly import contract."""


def _finite_scalar(value: float, name: str) -> float:
    try:
        if (np.ndim(value) != 0 or not is_number(value)
                or isinstance(value, (bool, np.bool_, complex, np.complexfloating))):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite; got {value!r}") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _optional_max_points(value, name: str = "max_points"):
    if value is None:
        return None
    return _positive_int(value, name, 2)


def _theme_name(theme: str) -> str:
    if not isinstance(theme, str) or theme not in _THEMES:
        raise ValueError(f"theme must be 'light' or 'dark'; got {theme!r}")
    return theme


def _freq_scale(freq_scale: str) -> str:
    if not isinstance(freq_scale, str) or freq_scale not in ("linear", "log"):
        raise ValueError(f"freq_scale must be 'linear' or 'log'; got {freq_scale!r}")
    return freq_scale


def theme_colors(theme: str = "light") -> dict:
    """Return the validated reference palette for ``theme``."""
    return _THEMES[_theme_name(theme)]


def sequential_colorscale(theme: str = "light") -> list[list]:
    """Return the theme sequential colorscale as Plotly ``[[pos, color], ...]``."""
    colours = theme_colors(theme)["sequential"]
    n = len(colours)
    if n == 1:
        return [[0.0, colours[0]], [1.0, colours[0]]]
    return [[i / (n - 1), colour] for i, colour in enumerate(colours)]


def get_template(theme: str = "light") -> go.layout.Template:
    """Return a ``plotly_white``-based layout template for ``theme``."""
    theme = _theme_name(theme)
    cached = _TEMPLATES.get(theme)
    if cached is not None:
        return cached
    colours = _THEMES[theme]
    template = go.layout.Template(pio.templates["plotly_white"])
    layout = template.layout
    layout.paper_bgcolor = colours["paper"]
    layout.plot_bgcolor = colours["paper"]
    layout.font.family = _FONT
    layout.font.color = colours["ink"]
    layout.title.font.color = colours["ink"]
    layout.legend.font.color = colours["ink"]
    layout.colorway = list(colours["colorway"])
    for axis_name in ("xaxis", "yaxis", "xaxis2", "yaxis2", "scene"):
        axis = getattr(layout, axis_name, None)
        if axis is None:
            continue
        if axis_name == "scene":
            for sub in ("xaxis", "yaxis", "zaxis"):
                sub_axis = getattr(axis, sub)
                _style_axis(sub_axis, colours)
            continue
        _style_axis(axis, colours)
    _TEMPLATES[theme] = template
    return template


def _style_axis(axis, colours) -> None:
    axis.color = colours["secondary"]
    axis.title.font.color = colours["secondary"]
    axis.tickfont.color = colours["secondary"]
    axis.gridcolor = colours["grid"]
    axis.gridwidth = 1
    # scene axes in Plotly have no griddash property
    if hasattr(axis, "griddash"):
        axis.griddash = "solid"
    axis.linecolor = colours["axis"]
    axis.zerolinecolor = colours["axis"]
    axis.zerolinewidth = 1


def colorway_colour(theme: str, index: int) -> str:
    """Return categorical slot ``index`` (0-based) from the theme colorway."""
    colourway = theme_colors(theme)["colorway"]
    if index < 0 or index >= len(colourway):
        raise ValueError(
            f"at most {len(colourway)} series per figure; facet instead"
        )
    return colourway[index]


def check_series_count(count: int, *, what: str = "series") -> None:
    if count > 8:
        raise ValueError(
            f"at most 8 {what} per figure (palette has 8 slots); facet instead"
        )
    if count < 1:
        raise ValueError(f"need at least one {what}")


def _table(table: pd.DataFrame, kind: str | None) -> tuple[pd.DataFrame, str]:
    if kind is None:
        hint = getattr(table, "attrs", {}).get("audiospylt_table")
        kind = hint if isinstance(hint, str) and hint in TABLE_CONTRACTS else infer_kind(table)
    return validate(table, kind), kind


def _labels(labels, count: int, *, prefix: str = "table") -> list[str]:
    if labels is None:
        return [f"{prefix} {i}" for i in range(1, count + 1)]
    if len(labels) != count:
        raise ValueError(
            f"labels length ({len(labels)}) must match number of tables ({count})"
        )
    return [str(label) for label in labels]


def amplitude_axis_label(*, db: bool) -> str:
    if db:
        return "amplitude (dB re full scale)"
    return "amplitude (re full scale)"


def to_plot_amplitude(amp, *, db: bool, floor_db: float):
    floor_db = _finite_scalar(floor_db, "floor_db")
    amp = np.asarray(amp, dtype=np.float64)
    if db:
        return amplitude_to_db(amp, floor_db=floor_db)
    return amp


def format_amplitude(value: float, *, db: bool) -> str:
    if db:
        return f"{float(value):.1f}"
    return f"{float(value):.4g}"


def pitch_hover_label(freq: float) -> str:
    """Return e.g. ``A4 +13 c``, or ``''`` when ``freq <= 0``."""
    if not np.isfinite(freq) or freq <= 0:
        return ""
    row = pitch_table(freq, resolution="semitone").iloc[0]
    accidental = "♯" if row.accid == "s" else ""
    return f"{str(row.step).upper()}{accidental}{int(row.octave)} {cents_label(row.residual)} c"


def hover_amp_freq(
    freq: float, amp: float, *, db: bool, extra: str = "",
) -> str:
    pitch = pitch_hover_label(freq)
    parts = [
        f"frequency: {freq:.2f} Hz",
        f"amplitude: {format_amplitude(amp, db=db)}"
        + (" dB" if db else ""),
    ]
    if pitch:
        parts.append(f"pitch: {pitch}")
    if extra:
        parts.append(extra)
    return "<br>".join(parts) + "<extra></extra>"


def apply_layout(
    fig: go.Figure, *, theme: str, title: str | None, n_series: int,
    **kwargs,
) -> go.Figure:
    fig.update_layout(
        template=get_template(theme),
        title=None if title is None else dict(text=title, font=dict(color=theme_colors(theme)["ink"])),
        showlegend=n_series >= 2,
        legend=dict(font=dict(color=theme_colors(theme)["ink"])),
        **kwargs,
    )
    return fig


def marker_style(colour: str, theme: str, *, size: float = 8.0) -> dict:
    return dict(
        size=size,
        color=colour,
        line=dict(width=2, color=theme_colors(theme)["paper"]),
    )


def filter_positive_freq(freq, *arrays, scale: str, warn_rows: bool = False):
    """Drop non-positive frequencies on a log axis; optionally warn for tables."""
    freq = np.asarray(freq, dtype=np.float64)
    if scale != "log":
        return (freq, *arrays)
    mask = freq > 0
    dropped = int((~mask).sum())
    if dropped and warn_rows:
        warnings.warn(
            f"{dropped} row(s) with frequency <= 0 left out on a log axis",
            UserWarning,
            stacklevel=3,
        )
    out = [freq[mask]]
    for array in arrays:
        array = np.asarray(array)
        out.append(array[mask] if array.shape[0] == mask.shape[0] else array)
    return tuple(out)


def check_fmin_fmax(fmin: float, fmax: float | None) -> tuple[float, float | None]:
    fmin = _finite_scalar(fmin, "fmin")
    if fmin < 0:
        raise ValueError(f"fmin must be >= 0; got {fmin!r}")
    if fmax is not None:
        fmax = _finite_scalar(fmax, "fmax")
        if fmax <= fmin:
            raise ValueError(f"fmax must be > fmin; got fmin={fmin!r}, fmax={fmax!r}")
    return fmin, fmax


def axis_range(lo: float, hi: float, scale: str) -> list[float]:
    """Return a Plotly axis ``range``; log axes use log10 units."""
    if scale == "log":
        return [float(np.log10(lo)), float(np.log10(hi))]
    return [float(lo), float(hi)]


def freq_mask(freq, fmin: float, fmax: float | None) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    mask = freq >= fmin
    if fmax is not None:
        mask &= freq <= fmax
    return mask


def _minmax(x, y, max_points: int | None):
    """Display-only min/max decimation.

    When ``len(y) <= max_points`` (or ``max_points`` is None), returns ``x`` and
    ``y`` unchanged. Otherwise splits the samples into ``max_points // 2``
    contiguous, nearly equal chunks and keeps each chunk's minimum and maximum
    sample in their original order. Output length is at most ``max_points``;
    the global min and max survive; ``x`` stays increasing.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)
    if max_points is None or n <= max_points:
        return x, y
    n_buckets = max_points // 2
    edges = np.linspace(0, n, n_buckets + 1, dtype=np.int64)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_buckets):
        start, stop = int(edges[i]), int(edges[i + 1])
        if start >= stop:
            continue
        chunk_y = y[start:stop]
        chunk_x = x[start:stop]
        i_min = int(np.argmin(chunk_y))
        i_max = int(np.argmax(chunk_y))
        order = (i_min,) if i_min == i_max else (
            (i_min, i_max) if i_min < i_max else (i_max, i_min)
        )
        for index in order:
            xs.append(float(chunk_x[index]))
            ys.append(float(chunk_y[index]))
    return np.asarray(xs, dtype=x.dtype), np.asarray(ys, dtype=y.dtype)


def baseline_amplitude(*, db: bool, floor_db: float) -> float:
    return float(floor_db) if db else 0.0
