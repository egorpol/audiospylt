"""Plotly figures for snapshot, events, and tracks tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from audiospylt.analysis.spectrum import amplitude_to_db
from audiospylt.viz._common import (
    _finite_scalar,
    _freq_scale,
    _labels,
    _modes,
    _table,
    _theme_name,
    amplitude_axis_label,
    apply_layout,
    baseline_amplitude,
    check_fmin_fmax,
    check_series_count,
    colorway_colour,
    filter_positive_freq,
    go,
    hover_amp_freq,
    marker_style,
    to_plot_amplitude,
)

__all__ = ["plot_snapshot", "plot_events", "plot_events_3d"]


_MARKER_SIZE = 4


def _kind_error(kind: str, function: str, can: str) -> ValueError:
    if kind == "snapshot":
        return ValueError(
            f"{function} cannot draw kind 'snapshot' where time is needed; "
            f"use model.convert.snapshot_to_events"
        )
    return ValueError(f"{function} cannot draw kind {kind!r}; use {can}")


def _snapshot_stem_xy(freqs, amps, baseline):
    """None-separated stem coordinates from baseline to each amplitude."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    for freq, amp in zip(freqs, amps):
        xs.extend([float(freq), float(freq), None])
        ys.extend([baseline, float(amp), None])
    return xs, ys


def plot_snapshot(
    *snapshots: pd.DataFrame,
    labels: Sequence[str] | None = None,
    db: bool = True,
    floor_db: float = -120.0,
    fmin: float = 0.0,
    fmax: float | None = None,
    freq_scale: str = "linear",
    theme: str = "light",
    title: str | None = None,
) -> go.Figure:
    """Return a Plotly stem figure of one or more snapshot tables.

    Beyond three overlaid snapshots colours stop being reliably distinguishable
    (palette all-pairs cap). Never shows, prints, or writes files.

    Parameters
    ----------
    *snapshots : DataFrame
        Snapshot tables. An empty snapshot draws nothing but keeps its legend
        entry.
    labels : sequence or None
        Default ``table 1``, ``table 2``, …
    db, floor_db, fmin, fmax, freq_scale, theme, title

    Returns
    -------
    plotly.graph_objects.Figure

    Notes
    -----
    Each snapshot is one None-separated 2 px line trace plus one marker trace
    sharing a legendgroup and one legend entry.

    Raises
    ------
    ValueError
        No snapshots, wrong kind, label length, or invalid arguments.
    """
    if not snapshots:
        raise ValueError("plot_snapshot requires at least one snapshot")
    theme = _theme_name(theme)
    freq_scale = _freq_scale(freq_scale)
    fmin, fmax = check_fmin_fmax(fmin, fmax)
    check_series_count(len(snapshots), what="tables")
    names = _labels(labels, len(snapshots))
    baseline = baseline_amplitude(db=db, floor_db=floor_db)
    fig = go.Figure()
    for index, (snap, name) in enumerate(zip(snapshots, names)):
        table, kind = _table(snap, None)
        if kind != "snapshot":
            raise _kind_error(kind, "plot_snapshot", "plot_events")
        colour = colorway_colour(theme, index)
        freqs = table["freq"].to_numpy(dtype=np.float64)
        amps = table["amp"].to_numpy(dtype=np.float64)
        mask = np.ones(len(table), dtype=bool)
        if fmax is not None:
            mask &= (freqs >= fmin) & (freqs <= fmax)
        else:
            mask &= freqs >= fmin
        freqs, amps = freqs[mask], amps[mask]
        if freq_scale == "log":
            freqs, amps = filter_positive_freq(
                freqs, amps, scale="log", warn_rows=True,
            )
        plot_amp = to_plot_amplitude(amps, db=db, floor_db=floor_db)
        xs, ys = _snapshot_stem_xy(freqs, plot_amp, baseline)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=name, legendgroup=name, showlegend=True,
            line=dict(width=2, color=colour),
            hoverinfo="skip",
        ))
        texts = [hover_amp_freq(f, a, db=db) for f, a in zip(freqs, plot_amp)]
        fig.add_trace(go.Scatter(
            x=freqs, y=np.asarray(plot_amp, dtype=np.float64),
            mode="markers",
            name=name, legendgroup=name, showlegend=False,
            marker=marker_style(colour, theme),
            hovertemplate="%{text}<extra></extra>",
            text=texts,
        ))
    apply_layout(fig, theme=theme, title=title, n_series=len(snapshots))
    fig.update_xaxes(title_text="frequency (Hz)", type=freq_scale)
    fig.update_yaxes(title_text=amplitude_axis_label(db=db))
    return fig


def _segment_level_db(amp_a, amp_b, floor_db):
    mean = 0.5 * (float(amp_a) + float(amp_b))
    return float(amplitude_to_db(mean, floor_db=floor_db))


def _opacity_for_bin(k: int) -> float:
    return 0.15 + 0.85 * k / 7


def _events_segments(table: pd.DataFrame):
    """Yield (x0, y0, a0, x1, y1, a1) for each events row."""
    for row in table.itertuples(index=False):
        yield (
            float(row.time_start), float(row.freq_start), float(row.amp_start),
            float(row.time_stop), float(row.freq_stop), float(row.amp_stop),
        )


def _tracks_polylines(table: pd.DataFrame):
    """Yield (times, freqs, amps) per track_id, sorted by time."""
    for _, group in table.groupby("track_id", sort=False):
        ordered = group.sort_values("time", kind="stable")
        yield (
            ordered["time"].to_numpy(dtype=np.float64),
            ordered["freq"].to_numpy(dtype=np.float64),
            ordered["amp"].to_numpy(dtype=np.float64),
        )


def _vertex_hovers(times, freqs, amps_plot, *, db: bool) -> list[str]:
    """Hover text per vertex: time, frequency, plotted amplitude and pitch."""
    from audiospylt.viz._common import format_amplitude, pitch_hover_labels

    unit = " dB" if db else ""
    texts = []
    for time, freq, amp_plot, pitch in zip(times, freqs, amps_plot, pitch_hover_labels(freqs)):
        parts = [
            f"time: {float(time):.4g} s",
            f"frequency: {float(freq):.2f} Hz",
            f"amplitude: {format_amplitude(amp_plot, db=db)}{unit}",
        ]
        if pitch:
            parts.append(f"pitch: {pitch}")
        texts.append("<br>".join(parts) + "<extra></extra>")
    return texts


def _none_separated(paths, texts):
    """x, y (and z) lists with None between paths; texts are per vertex, in order."""
    columns: list[list] = [[] for _ in range(len(paths[0]) if paths else 0)]
    out_texts: list[str | None] = []
    k = 0
    for path in paths:
        n = len(path[0])
        for column, values in zip(columns, path):
            column.extend(np.asarray(values, dtype=np.float64).tolist())
            column.append(None)
        out_texts.extend(texts[k:k + n])
        out_texts.append(None)
        k += n
    return columns, out_texts


# A row ramps time, frequency and linear amplitude linearly, as ``synthesize``
# renders it. On a log frequency axis or a dB amplitude axis that ramp is a
# curve, so a straight line between the row's ends would misdraw it; vertices
# at equal steps of cents and of dB draw the curve instead.
_DRAW_CENTS = 100.0
_DRAW_DB = 3.0
_DRAW_MAX_STEPS = 256


def _ramp_fractions(v0: float, v1: float, lo: float, hi: float, step_ratio: float):
    """Fractions in (0, 1) where a linear ramp v0 -> v1 crosses a geometric grid on [lo, hi]."""
    steps = min(_DRAW_MAX_STEPS, int(np.ceil(np.log(hi / lo) / np.log(step_ratio) - 1e-9)))
    if steps < 2:
        return None
    targets = np.geomspace(lo, hi, steps + 1)[1:-1]
    return (targets - v0) / (v1 - v0)


def _path_fractions(f0, a0, f1, a1, *, freq_log: bool, db: bool, floor_amp: float):
    """Sorted fractions in [0, 1] at which to draw one row, or None for its two ends."""
    parts = []
    if freq_log and f0 != f1 and f0 > 0 and f1 > 0:
        lo, hi = sorted((f0, f1))
        part = _ramp_fractions(f0, f1, lo, hi, 2.0 ** (_DRAW_CENTS / 1200.0))
        if part is not None:
            parts.append(part)
    if db and a0 != a1:
        lo, hi = max(min(a0, a1), floor_amp), max(a0, a1)
        if hi > lo:
            part = _ramp_fractions(a0, a1, lo, hi, 10.0 ** (_DRAW_DB / 20.0))
            if part is not None:
                parts.append(part)
            if min(a0, a1) < floor_amp:
                # Where the ramp meets the floor; it runs along the floor after.
                parts.append(np.array([(floor_amp - a0) / (a1 - a0)]))
    if not parts:
        return None
    return np.unique(np.clip(np.concatenate([[0.0, 1.0], *parts]), 0.0, 1.0))


def _drawn_path(times, freqs, amps, *, freq_log: bool, db: bool, floor_amp: float = 0.0):
    """Vertices drawing a polyline of linear rows on the plot's axes.

    Returns the inputs unchanged when no segment bends on these axes (linear
    frequency and amplitude axes never bend one).
    """
    times = np.asarray(times, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    if len(times) < 2 or not (freq_log or db):
        return times, freqs, amps
    out_t, out_f, out_a = [times[:1]], [freqs[:1]], [amps[:1]]
    bent = False
    for i in range(len(times) - 1):
        u = _path_fractions(
            freqs[i], amps[i], freqs[i + 1], amps[i + 1],
            freq_log=freq_log, db=db, floor_amp=floor_amp,
        )
        if u is None:
            out_t.append(times[i + 1:i + 2])
            out_f.append(freqs[i + 1:i + 2])
            out_a.append(amps[i + 1:i + 2])
            continue
        bent = True
        u = u[1:]
        out_t.append(times[i] + (times[i + 1] - times[i]) * u)
        out_f.append(freqs[i] + (freqs[i + 1] - freqs[i]) * u)
        out_a.append(amps[i] + (amps[i + 1] - amps[i]) * u)
        # Exact end values, not t0 + (t1 - t0) * 1.0.
        out_t[-1][-1], out_f[-1][-1], out_a[-1][-1] = times[i + 1], freqs[i + 1], amps[i + 1]
    if not bent:
        return times, freqs, amps
    return np.concatenate(out_t), np.concatenate(out_f), np.concatenate(out_a)


def _segment_lines(segments, *, db: bool, floor_db: float, freq_log: bool):
    """None-separated x, y and hover text for segment tuples."""
    paths = [
        _drawn_path((t0, t1), (f0, f1), (a0, a1), freq_log=freq_log, db=False)
        for t0, f0, a0, t1, f1, a1 in segments
    ]
    if not paths:
        return [], [], []
    times, freqs, amps = (np.concatenate(column) for column in zip(*paths))
    texts = _vertex_hovers(times, freqs, to_plot_amplitude(amps, db=db, floor_db=floor_db), db=db)
    (xs, ys), out_texts = _none_separated([path[:2] for path in paths], texts)
    return xs, ys, out_texts


def _bin_index(level: float, l_max: float, range_db: float) -> int:
    low = l_max - range_db
    if level < low:
        return 0
    return min(7, max(0, int((level - low) / range_db * 8)))


def _opacity_segment_traces(
    segments, *, colour, label, legendgroup, db, floor_db, range_db, l_max,
    freq_log=False,
):
    """One None-separated trace per non-empty opacity bin for segment lists."""
    levels = [_segment_level_db(a0, a1, floor_db) for _, _, a0, _, _, a1 in segments]
    if l_max is None:
        l_max = max(levels) if levels else 0.0
    bins: list[list] = [[] for _ in range(8)]
    for seg, level in zip(segments, levels):
        bins[_bin_index(level, l_max, range_db)].append(seg)
    traces = []
    first_legend = True
    for k, bucket in enumerate(bins):
        if not bucket:
            continue
        xs, ys, texts = _segment_lines(bucket, db=db, floor_db=floor_db, freq_log=freq_log)
        traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=label, legendgroup=legendgroup, showlegend=first_legend,
            line=dict(width=2, color=colour),
            opacity=_opacity_for_bin(k),
            hovertemplate="%{text}", text=texts,
        ))
        first_legend = False
    if first_legend:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=label, legendgroup=legendgroup, showlegend=True,
            line=dict(width=2, color=colour), opacity=1.0,
        ))
    return traces


def _events_line_traces(
    segments, *, colour, label, legendgroup, amp, db, floor_db, range_db, l_max,
    freq_log=False,
):
    if amp is None:
        xs, ys, texts = _segment_lines(segments, db=db, floor_db=floor_db, freq_log=freq_log)
        return [go.Scatter(
            x=xs, y=ys, mode="lines",
            name=label, legendgroup=legendgroup, showlegend=True,
            line=dict(width=2, color=colour),
            opacity=1.0,
            hovertemplate="%{text}",
            text=texts,
        )]
    return _opacity_segment_traces(
        segments, colour=colour, label=label, legendgroup=legendgroup,
        db=db, floor_db=floor_db, range_db=range_db, l_max=l_max, freq_log=freq_log,
    )


def _tracks_line_traces(
    polylines, *, colour, label, legendgroup, amp, db, floor_db, range_db, l_max,
    freq_log=False,
):
    if amp is None:
        paths = [
            _drawn_path(times, freqs, amps, freq_log=freq_log, db=False)
            for times, freqs, amps in polylines
        ]
        xs: list[float | None] = []
        ys: list[float | None] = []
        texts: list[str | None] = []
        if paths:
            times, freqs, amps = (np.concatenate(column) for column in zip(*paths))
            hover = _vertex_hovers(
                times, freqs, to_plot_amplitude(amps, db=db, floor_db=floor_db), db=db,
            )
            (xs, ys), texts = _none_separated([path[:2] for path in paths], hover)
        return [go.Scatter(
            x=xs, y=ys, mode="lines",
            name=label, legendgroup=legendgroup, showlegend=True,
            line=dict(width=2, color=colour), opacity=1.0,
            hovertemplate="%{text}", text=texts,
        )]

    segments = []
    for times, freqs, amps in polylines:
        for i in range(len(times) - 1):
            segments.append((
                float(times[i]), float(freqs[i]), float(amps[i]),
                float(times[i + 1]), float(freqs[i + 1]), float(amps[i + 1]),
            ))
    return _opacity_segment_traces(
        segments, colour=colour, label=label, legendgroup=legendgroup,
        db=db, floor_db=floor_db, range_db=range_db, l_max=l_max, freq_log=freq_log,
    )


def _event_points(segments):
    """Row end points, each once: consecutive rows of one track share one."""
    ends = (point for t0, f0, a0, t1, f1, a1 in segments
            for point in ((t0, f0, a0), (t1, f1, a1)))
    return list(dict.fromkeys(ends))


def _track_points(polylines, *, every: bool):
    """(time, freq, amp) of every vertex, or of one-point tracks only."""
    return [
        (float(t), float(f), float(a))
        for times, freqs, amps in polylines
        if every or len(times) == 1
        for t, f, a in zip(times, freqs, amps)
    ]


def _point_level_db(amp, floor_db):
    return float(amplitude_to_db(float(amp), floor_db=floor_db))


def _marker_traces(
    points, *, colour, label, legendgroup, amp, db, floor_db, range_db, l_max,
    showlegend,
):
    """Dot traces: one per non-empty opacity bin, or one opaque trace."""
    if amp is None:
        buckets = [(1.0, points)]
    else:
        levels = [_point_level_db(a, floor_db) for _, _, a in points]
        if l_max is None:
            l_max = max(levels) if levels else 0.0
        bins: list[list] = [[] for _ in range(8)]
        for point, level in zip(points, levels):
            bins[_bin_index(level, l_max, range_db)].append(point)
        buckets = [(_opacity_for_bin(k), bucket) for k, bucket in enumerate(bins)]
    traces: list = []
    for opacity, bucket in buckets:
        if not bucket:
            continue
        times, freqs, amps = (list(column) for column in zip(*bucket))
        plot_amp = to_plot_amplitude(amps, db=db, floor_db=floor_db)
        traces.append(go.Scatter(
            x=times, y=freqs, mode="markers",
            name=label, legendgroup=legendgroup, showlegend=showlegend and not traces,
            marker=dict(size=_MARKER_SIZE, color=colour), opacity=opacity,
            hovertemplate="%{text}",
            text=_vertex_hovers(times, freqs, plot_amp, db=db),
        ))
    if showlegend and not traces:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=label, legendgroup=legendgroup, showlegend=True,
            marker=dict(size=_MARKER_SIZE, color=colour), opacity=1.0,
        ))
    return traces


def _drawn_levels(frame, kind, mode, floor_db):
    """dB levels of what ``mode`` draws: segments for lines, points for dots."""
    if kind == "events":
        segments = list(_events_segments(frame))
        points = _event_points(segments) if "markers" in mode else []
    else:
        polylines = list(_tracks_polylines(frame))
        segments = [
            (None, None, amps[i], None, None, amps[i + 1])
            for _, _, amps in polylines for i in range(len(amps) - 1)
        ]
        points = _track_points(polylines, every="markers" in mode)
    levels = [_point_level_db(a, floor_db) for _, _, a in points]
    if "lines" in mode:
        levels += [_segment_level_db(a0, a1, floor_db) for _, _, a0, _, _, a1 in segments]
    return levels


def _line_traces_for_table(
    table, kind, *, colour, label, theme, amp, db, floor_db, range_db,
    freq_scale, legendgroup, l_max, mode="lines",
):
    """Build Scatter line and dot traces for one events or tracks table.

    Dots mark every point when ``mode`` has markers; in 'lines' mode they mark
    only one-point tracks, which a line cannot show.
    """
    if amp not in ("opacity", None):
        raise ValueError(f"amp must be 'opacity' or None; got {amp!r}")
    style = dict(
        colour=colour, label=label, legendgroup=legendgroup, amp=amp, db=db,
        floor_db=floor_db, range_db=range_db, l_max=l_max,
    )

    if kind == "events":
        segments = list(_events_segments(table))
        if freq_scale == "log":
            kept = []
            dropped = 0
            for seg in segments:
                if seg[1] <= 0 or seg[4] <= 0:
                    dropped += 1
                else:
                    kept.append(seg)
            if dropped:
                import warnings
                warnings.warn(
                    f"{dropped} row(s) with frequency <= 0 left out on a log axis",
                    UserWarning,
                    stacklevel=3,
                )
            segments = kept
        traces = (_events_line_traces(segments, freq_log=freq_scale == "log", **style)
                  if "lines" in mode else [])
        points = _event_points(segments) if "markers" in mode else []
    else:
        polylines = list(_tracks_polylines(table))
        if freq_scale == "log":
            filtered = []
            dropped = 0
            for times, freqs, amps in polylines:
                keep = freqs > 0
                dropped += int((~keep).sum())
                if keep.any():
                    filtered.append((times[keep], freqs[keep], amps[keep]))
            if dropped:
                import warnings
                warnings.warn(
                    f"{dropped} row(s) with frequency <= 0 left out on a log axis",
                    UserWarning,
                    stacklevel=3,
                )
            polylines = filtered
        traces = (_tracks_line_traces(polylines, freq_log=freq_scale == "log", **style)
                  if "lines" in mode else [])
        points = _track_points(polylines, every="markers" in mode)

    if points or not traces:
        traces += _marker_traces(points, showlegend=not traces, **style)
    return traces


def plot_events(
    *tables: pd.DataFrame,
    labels: Sequence[str] | None = None,
    mode: str | Sequence[str] = "lines",
    amp: str | None = "opacity",
    db: bool = True,
    floor_db: float = -120.0,
    range_db: float = 80.0,
    freq_scale: str = "linear",
    theme: str = "light",
    title: str | None = None,
) -> go.Figure:
    """Return a Plotly figure of events and/or tracks tables.

    With ``amp='opacity'`` (SPEAR convention), segment level is the dB of the
    mean of its two end amplitudes and a dot's level the dB of its amplitude;
    eight equal bins span ``[L_max - range_db, L_max]``, where ``L_max`` is the
    loudest level drawn. Never shows, prints, or writes files.

    Parameters
    ----------
    *tables : DataFrame
        Events and tracks may be mixed.
    labels : sequence or None
    mode : {'lines', 'markers', 'lines+markers'} or sequence, default 'lines'
        One mode for all tables, or one per table. Markers draw each point once
        as a 4 px dot (events: each row end, shared ends once). In 'lines'
        mode, one-point tracks, which a line cannot show, are drawn as dots.
    amp : {'opacity', None}
    db, floor_db, range_db, theme, title
    freq_scale : {'linear', 'log'}, default 'linear'
        On a log axis each row is drawn along its linear-Hz path (a curve),
        as ``synthesize`` renders it; rows with a frequency <= 0 are left out
        with a warning.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not tables:
        raise ValueError("plot_events requires at least one table")
    theme = _theme_name(theme)
    freq_scale = _freq_scale(freq_scale)
    range_db = _finite_scalar(range_db, "range_db")
    if range_db <= 0:
        raise ValueError(f"range_db must be > 0; got {range_db!r}")
    if amp not in ("opacity", None):
        raise ValueError(f"amp must be 'opacity' or None; got {amp!r}")
    check_series_count(len(tables), what="tables")
    names = _labels(labels, len(tables))
    modes = _modes(mode, len(tables))

    resolved = []
    for table in tables:
        frame, kind = _table(table, None)
        if kind not in ("events", "tracks"):
            raise _kind_error(kind, "plot_events", "plot_snapshot")
        resolved.append((frame, kind))

    l_max = None
    if amp == "opacity":
        levels = []
        for (frame, kind), table_mode in zip(resolved, modes):
            levels += _drawn_levels(frame, kind, table_mode, floor_db)
        l_max = max(levels) if levels else 0.0

    fig = go.Figure()
    for index, ((frame, kind), name, table_mode) in enumerate(zip(resolved, names, modes)):
        colour = colorway_colour(theme, index)
        for trace in _line_traces_for_table(
            frame, kind, colour=colour, label=name, theme=theme,
            amp=amp, db=db, floor_db=floor_db, range_db=range_db,
            freq_scale=freq_scale, legendgroup=name, l_max=l_max, mode=table_mode,
        ):
            fig.add_trace(trace)
    apply_layout(fig, theme=theme, title=title, n_series=len(tables))
    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text="frequency (Hz)", type=freq_scale)
    return fig


_HEIGHT_3D = 520
_ASPECT_3D = {"x": 1.6, "y": 1.0, "z": 0.6}
_CAMERA_3D = {"eye": {"x": -1.2, "y": -1.75, "z": 0.75}, "center": {"x": -0.16, "y": 0.11, "z": -0.12}}
_SHADOW_OPACITY = 0.25


def _nice_ticks(lo: float, hi: float, count: int = 6) -> list[float]:
    """Round tick values (1, 2 or 5 times a power of ten apart) inside [lo, hi]."""
    raw = (hi - lo) / count
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1.0, 2.0, 5.0, 10.0) if m * magnitude >= raw)
    # Integer multiples of the step, so a tick at zero is exactly 0.
    first, last = int(np.ceil(lo / step - 1e-9)), int(np.floor(hi / step + 1e-9))
    return [float(f"{k * step:.12g}") for k in range(first, last + 1)]


def _hz_label(value: float) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _log_freq_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 (or 1-9 on a narrow range) ticks per decade inside [lo, hi] Hz."""
    for mantissas in ((1, 2, 5), tuple(range(1, 10))):
        ticks = [
            float(m * 10.0 ** k)
            for k in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
            for m in mantissas
            if lo <= m * 10.0 ** k <= hi
        ]
        if len(ticks) >= 3:
            return ticks
    return ticks


def _paths_3d(frame, kind, *, freq_log):
    """(times, freqs, amps) per events row or track; frequencies <= 0 left out on a log axis."""
    dropped = 0
    if kind == "events":
        paths = [
            (np.array([t0, t1]), np.array([f0, f1]), np.array([a0, a1]))
            for t0, f0, a0, t1, f1, a1 in _events_segments(frame)
        ]
        if freq_log:
            kept = [path for path in paths if (path[1] > 0).all()]
            dropped = len(paths) - len(kept)
            paths = kept
    else:
        paths = list(_tracks_polylines(frame))
        if freq_log:
            dropped = sum(int((path[1] <= 0).sum()) for path in paths)
            paths = [
                (times[freqs > 0], freqs[freqs > 0], amps[freqs > 0])
                for times, freqs, amps in paths if (freqs > 0).any()
            ]
    if freq_log and dropped:
        import warnings
        warnings.warn(
            f"{dropped} row(s) with frequency <= 0 left out on a log axis",
            UserWarning,
            stacklevel=3,
        )
    return paths


def _scene_axes_3d(times, freqs, *, z_range, freq_log, db):
    """Scene axis dicts: tight time and frequency ranges, z ticks above the floor."""
    zaxis: dict[str, object] = {"title": amplitude_axis_label(db=db), "range": list(z_range)}
    z_lo, z_hi = z_range
    # The floor value is left unlabelled so that it never sits on another axis's end label.
    zaxis["tickvals"] = [v for v in _nice_ticks(z_lo, z_hi) if v > z_lo + 0.04 * (z_hi - z_lo)]
    xaxis: dict[str, object] = {"title": "time (s)"}
    yaxis: dict[str, object] = {"title": "frequency (Hz)", "type": "log" if freq_log else "linear"}
    if not len(times):
        return xaxis, yaxis, zaxis
    t_lo, t_hi = float(np.min(times)), float(np.max(times))
    if t_hi > t_lo:
        xaxis["range"] = [t_lo, t_hi]
    f_lo, f_hi = float(np.min(freqs)), float(np.max(freqs))
    if freq_log:
        lo, hi = np.log10(f_lo), np.log10(f_hi)
        pad = 0.03 * (hi - lo) if hi > lo else 0.05
        yaxis["range"] = [float(lo - pad), float(hi + pad)]
        ticks = _log_freq_ticks(10.0 ** (lo - pad), 10.0 ** (hi + pad))
        if ticks:
            yaxis["tickvals"] = ticks
            yaxis["ticktext"] = [_hz_label(v) for v in ticks]
    else:
        pad = 0.03 * (f_hi - f_lo) if f_hi > f_lo else max(0.05 * f_hi, 1.0)
        # Start above 0 Hz when the data do, so no "0" tick joins time's at the corner.
        lo = f_lo - pad if f_lo - pad > 0 else 0.5 * f_lo
        yaxis["range"] = [float(lo), float(f_hi + pad)]
    return xaxis, yaxis, zaxis


def plot_events_3d(
    *tables: pd.DataFrame,
    labels: Sequence[str] | None = None,
    db: bool = True,
    floor_db: float = -120.0,
    range_db: float = 80.0,
    freq_scale: str = "linear",
    shadow: bool = True,
    theme: str = "light",
    title: str | None = None,
) -> go.Figure:
    """Return a 3-D Plotly figure of events/tracks (time, frequency, amplitude).

    Time runs left to right, frequency into depth, amplitude up. Each events
    row is drawn along the path ``synthesize`` renders: time, frequency and
    linear amplitude ramp linearly, so on a dB or log axis the row is a curve.
    The figure is 520 px high; change it with ``fig.update_layout(height=...)``.
    Never shows, prints, or writes files; inputs are unchanged.

    Parameters
    ----------
    *tables : DataFrame
        Events and tracks may be mixed; one line trace per table. One-point
        tracks, which a line cannot show, are drawn as dots.
    labels : sequence or None
        Default ``table 1``, ``table 2``, …
    db : bool, default True
        Amplitude in dB re full scale. The axis spans ``range_db`` below the
        loudest level drawn (not below ``floor_db``); quieter values lie on its
        floor. ``db=False`` plots linear amplitude from 0.
    floor_db : float, default -120.0
    range_db : float, default 80.0
        Used with ``db=True`` only; must be > 0.
    freq_scale : {'linear', 'log'}, default 'linear'
        On a log axis, rows with a frequency <= 0 are left out with a warning.
    shadow : bool, default True
        Draw a faint copy of each line on the floor of the box: the
        time-frequency view of ``plot_events``.
    theme : {'light', 'dark'}, default 'light'
    title : str or None

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        No tables, a snapshot table, label length, or invalid arguments.
    """
    if not tables:
        raise ValueError("plot_events_3d requires at least one table")
    theme = _theme_name(theme)
    freq_log = _freq_scale(freq_scale) == "log"
    floor_db = _finite_scalar(floor_db, "floor_db")
    range_db = _finite_scalar(range_db, "range_db")
    if range_db <= 0:
        raise ValueError(f"range_db must be > 0; got {range_db!r}")
    check_series_count(len(tables), what="tables")
    names = _labels(labels, len(tables))
    resolved = []
    for table in tables:
        frame, kind = _table(table, None)
        if kind not in ("events", "tracks"):
            raise _kind_error(kind, "plot_events_3d", "plot_snapshot")
        resolved.append(_paths_3d(frame, kind, freq_log=freq_log))

    amps = [path[2] for paths in resolved for path in paths]
    a_max = float(max((np.max(a) for a in amps), default=0.0))
    if db:
        l_max = float(amplitude_to_db(a_max, floor_db=floor_db))
        z_floor = max(floor_db, l_max - range_db)
        # A silent or empty table still gets a range_db-tall axis.
        z_range = (z_floor, (l_max if l_max > z_floor else z_floor + range_db) + 0.03 * range_db)
        floor_amp = 10.0 ** (z_floor / 20.0)
    else:
        z_floor = 0.0
        z_range = (0.0, 1.03 * a_max if a_max > 0 else 1.0)
        floor_amp = 0.0

    fig = go.Figure()
    all_t: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    for index, (paths, name) in enumerate(zip(resolved, names)):
        colour = colorway_colour(theme, index)
        drawn = []
        dots: list[tuple[float, float, float]] = []
        for times, freqs, amps_ in paths:
            all_t.append(times)
            all_f.append(freqs)
            if len(times) == 1:
                level = float(to_plot_amplitude(amps_[0], db=db, floor_db=floor_db))
                dots.append((float(times[0]), float(freqs[0]), max(level, z_floor)))
                continue
            times, freqs, amps_ = _drawn_path(
                times, freqs, amps_, freq_log=freq_log, db=db, floor_amp=floor_amp,
            )
            levels = np.maximum(to_plot_amplitude(amps_, db=db, floor_db=floor_db), z_floor)
            drawn.append((times, freqs, levels))
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        texts: list[str | None] = []
        if drawn:
            hover = _vertex_hovers(*(np.concatenate(column) for column in zip(*drawn)), db=db)
            (xs, ys, zs), texts = _none_separated(drawn, hover)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            name=name, legendgroup=name, showlegend=True,
            line=dict(width=3, color=colour),
            hovertemplate="%{text}", text=texts,
        ))
        if dots:
            dt, df, dz = (list(column) for column in zip(*dots))
            fig.add_trace(go.Scatter3d(
                x=dt, y=df, z=dz, mode="markers",
                name=name, legendgroup=name, showlegend=False,
                marker=dict(size=_MARKER_SIZE, color=colour),
                hovertemplate="%{text}",
                text=_vertex_hovers(dt, df, dz, db=db),
            ))
        if shadow and len(xs):
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=[None if z is None else z_range[0] for z in zs], mode="lines",
                name=name, legendgroup=name, showlegend=False, hoverinfo="skip",
                line=dict(width=2, color=colour), opacity=_SHADOW_OPACITY,
            ))
    apply_layout(fig, theme=theme, title=title, n_series=len(tables))
    times_all = np.concatenate(all_t) if all_t else np.empty(0)
    freqs_all = np.concatenate(all_f) if all_f else np.empty(0)
    xaxis, yaxis, zaxis = _scene_axes_3d(
        times_all, freqs_all, z_range=z_range, freq_log=freq_log, db=db,
    )
    fig.update_layout(
        height=_HEIGHT_3D,
        margin=dict(l=0, r=0, b=0, t=10 if title is None else 50),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top", tracegroupgap=0),
        scene=dict(
            xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
            aspectmode="manual", aspectratio=dict(_ASPECT_3D), camera=_CAMERA_3D,
        ),
    )
    return fig
