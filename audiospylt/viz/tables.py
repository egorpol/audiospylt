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


def _vertex_hover(time, freq, amp_plot, *, db: bool) -> str:
    pitch = ""
    from audiospylt.viz._common import pitch_hover_label, format_amplitude
    pitch = pitch_hover_label(freq)
    parts = [
        f"time: {time:.4g} s",
        f"frequency: {freq:.2f} Hz",
        f"amplitude: {format_amplitude(amp_plot, db=db)}" + (" dB" if db else ""),
    ]
    if pitch:
        parts.append(f"pitch: {pitch}")
    return "<br>".join(parts) + "<extra></extra>"


def _bin_index(level: float, l_max: float, range_db: float) -> int:
    low = l_max - range_db
    if level < low:
        return 0
    return min(7, max(0, int((level - low) / range_db * 8)))


def _opacity_segment_traces(
    segments, *, colour, label, legendgroup, db, floor_db, range_db, l_max,
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
        xs, ys, texts = [], [], []
        for t0, f0, a0, t1, f1, a1 in bucket:
            p0 = float(to_plot_amplitude(a0, db=db, floor_db=floor_db))
            p1 = float(to_plot_amplitude(a1, db=db, floor_db=floor_db))
            xs.extend([t0, t1, None])
            ys.extend([f0, f1, None])
            texts.extend([
                _vertex_hover(t0, f0, p0, db=db),
                _vertex_hover(t1, f1, p1, db=db),
                None,
            ])
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
):
    if amp is None:
        xs: list[float | None] = []
        ys: list[float | None] = []
        texts: list[str | None] = []
        for t0, f0, a0, t1, f1, a1 in segments:
            p0 = float(to_plot_amplitude(a0, db=db, floor_db=floor_db))
            p1 = float(to_plot_amplitude(a1, db=db, floor_db=floor_db))
            xs.extend([t0, t1, None])
            ys.extend([f0, f1, None])
            texts.extend([
                _vertex_hover(t0, f0, p0, db=db),
                _vertex_hover(t1, f1, p1, db=db),
                None,
            ])
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
        db=db, floor_db=floor_db, range_db=range_db, l_max=l_max,
    )


def _tracks_line_traces(
    polylines, *, colour, label, legendgroup, amp, db, floor_db, range_db, l_max,
):
    if amp is None:
        xs: list[float | None] = []
        ys: list[float | None] = []
        texts: list[str | None] = []
        for times, freqs, amps in polylines:
            plot_amp = to_plot_amplitude(amps, db=db, floor_db=floor_db)
            for i in range(len(times)):
                xs.append(float(times[i]))
                ys.append(float(freqs[i]))
                texts.append(_vertex_hover(
                    float(times[i]), float(freqs[i]), float(plot_amp[i]), db=db,
                ))
            xs.append(None)
            ys.append(None)
            texts.append(None)
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
        db=db, floor_db=floor_db, range_db=range_db, l_max=l_max,
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
            text=[_vertex_hover(t, f, float(p), db=db)
                  for t, f, p in zip(times, freqs, plot_amp)],
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
        traces = _events_line_traces(segments, **style) if "lines" in mode else []
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
        traces = _tracks_line_traces(polylines, **style) if "lines" in mode else []
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
    db, floor_db, range_db, freq_scale, theme, title

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


def plot_events_3d(
    *tables: pd.DataFrame,
    labels: Sequence[str] | None = None,
    db: bool = False,
    floor_db: float = -120.0,
    theme: str = "light",
    title: str | None = None,
) -> go.Figure:
    """Return a 3-D Plotly figure of events/tracks (time, frequency, amplitude).

    Never shows, prints, or writes files; inputs are unchanged.

    Parameters
    ----------
    *tables : DataFrame
        Events and tracks may be mixed; one 3-px line trace per table.
    labels : sequence or None
        Default ``table 1``, ``table 2``, …
    db : bool, default False
        Plot amplitude in dB (floored at ``floor_db``); linear amplitude uses
        a z axis starting at 0.
    floor_db : float, default -120.0
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
    check_series_count(len(tables), what="tables")
    names = _labels(labels, len(tables))
    fig = go.Figure()
    z_min = 0.0 if not db else None
    for index, (table, name) in enumerate(zip(tables, names)):
        frame, kind = _table(table, None)
        if kind not in ("events", "tracks"):
            raise _kind_error(kind, "plot_events_3d", "plot_snapshot")
        colour = colorway_colour(theme, index)
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        if kind == "events":
            for t0, f0, a0, t1, f1, a1 in _events_segments(frame):
                p0 = float(to_plot_amplitude(a0, db=db, floor_db=floor_db))
                p1 = float(to_plot_amplitude(a1, db=db, floor_db=floor_db))
                xs.extend([t0, t1, None])
                ys.extend([f0, f1, None])
                zs.extend([p0, p1, None])
        else:
            for times, freqs, amps in _tracks_polylines(frame):
                plot_amp = to_plot_amplitude(amps, db=db, floor_db=floor_db)
                for i in range(len(times)):
                    xs.append(float(times[i]))
                    ys.append(float(freqs[i]))
                    zs.append(float(plot_amp[i]))
                xs.append(None)
                ys.append(None)
                zs.append(None)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            name=name,
            line=dict(width=3, color=colour),
        ))
    apply_layout(fig, theme=theme, title=title, n_series=len(tables))
    scene: dict[str, object] = dict(
        xaxis_title="time (s)",
        yaxis_title="frequency (Hz)",
        zaxis_title=amplitude_axis_label(db=db),
    )
    if z_min is not None:
        scene["zaxis"] = dict(title=amplitude_axis_label(db=db), range=[0, None])
    fig.update_layout(scene=scene)
    return fig
