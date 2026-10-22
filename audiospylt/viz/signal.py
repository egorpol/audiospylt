"""Plotly figures for waveforms, spectra, and spectrograms."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from audiospylt._optional import import_optional
from audiospylt.analysis.spectrogram import MelSpectrogram, Spectrogram
from audiospylt.analysis.spectrum import Spectrum
from audiospylt.io.audio import _audio_array, _sample_rate
from audiospylt.viz._common import (
    _finite_scalar,
    _freq_scale,
    _labels,
    _minmax,
    _optional_max_points,
    _positive_int,
    _table,
    _theme_name,
    amplitude_axis_label,
    apply_layout,
    axis_range,
    check_fmin_fmax,
    check_series_count,
    colorway_colour,
    freq_mask,
    go,
    hover_amp_freq,
    marker_style,
    sequential_colorscale,
    theme_colors,
    to_plot_amplitude,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["plot_waveform", "plot_spectrum", "plot_spectrogram"]

_MEL_TICK_HZ = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)


def _validate_audio(audio, sr):
    array = _audio_array(audio)
    sr = _sample_rate(sr)
    if array.size == 0:
        raise ValueError("audio must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError("audio must contain only finite samples")
    return array, sr


def _channel_labels(labels, n_channels: int) -> list[str]:
    if labels is None:
        return [f"ch {i}" for i in range(1, n_channels + 1)]
    if len(labels) != n_channels:
        raise ValueError(
            f"channel_labels length ({len(labels)}) must match "
            f"number of channels ({n_channels})"
        )
    return [str(label) for label in labels]


def plot_waveform(
    audio: ArrayLike,
    sr: int,
    *,
    time_offset: float = 0.0,
    channel_labels: Sequence[str] | None = None,
    max_points: int | None = 50_000,
    gl: bool = False,
    theme: str = "light",
    title: str | None = None,
) -> go.Figure:
    """Return a Plotly figure of a mono or multichannel waveform.

    Dense traces are display-decimated with ``_minmax`` when longer than
    ``max_points``. Never shows, prints, or writes files; inputs are unchanged.

    Parameters
    ----------
    audio : array-like
        1-D ``(samples,)`` or 2-D ``(channels, samples)`` as ``io.audio`` reads.
    sr : int
        Positive sample rate in Hz.
    time_offset : float, default 0
        Finite seconds added to ``n / sr``.
    channel_labels : sequence of str or None
        Legend labels for multichannel audio; default ``ch 1``, ``ch 2``, …
    max_points : int or None, default 50000
        Display-only min/max budget per channel (``>= 2`` or None).
    gl : bool, default False
        Use ``Scattergl``. Browsers cap WebGL contexts at roughly 8–16 per page,
        so this is opt-in.
    theme : {'light', 'dark'}, default 'light'
    title : str or None

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        Invalid audio, rate, labels, theme, or max_points.
    ImportError
        Plotly is missing; install audiospylt[plot].
    """
    theme = _theme_name(theme)
    array, sr = _validate_audio(audio, sr)
    time_offset = _finite_scalar(time_offset, "time_offset")
    max_points = _optional_max_points(max_points)
    if array.ndim == 1:
        channels = array[np.newaxis, :]
        mono = True
    else:
        channels = array
        mono = False
    n_channels = channels.shape[0]
    check_series_count(n_channels, what="channels")
    labels = _channel_labels(channel_labels, n_channels)
    scatter = go.Scattergl if gl else go.Scatter
    fig = go.Figure()
    n = channels.shape[1]
    x_full = time_offset + np.arange(n, dtype=np.float64) / sr
    for index in range(n_channels):
        x, y = _minmax(x_full, channels[index], max_points)
        fig.add_trace(scatter(
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float32),
            mode="lines",
            name=labels[index],
            line=dict(width=1, color=colorway_colour(theme, index)),
            hovertemplate="time: %{x:.4g} s<br>amplitude: %{y:.4g}<extra></extra>",
            showlegend=not mono,
        ))
    apply_layout(fig, theme=theme, title=title, n_series=1 if mono else n_channels)
    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text="amplitude (re full scale)")
    return fig


def plot_spectrum(
    spec: Spectrum,
    peaks: pd.DataFrame | None = None,
    *,
    db: bool = True,
    floor_db: float = -120.0,
    fmin: float = 0.0,
    fmax: float | None = None,
    freq_scale: str = "linear",
    max_points: int | None = 50_000,
    gl: bool = False,
    theme: str = "light",
    title: str | None = None,
) -> go.Figure:
    """Return a Plotly figure of a ``Spectrum``, optionally with peak markers.

    The spectrum line is display-decimated with ``_minmax`` after cropping.
    Never shows, prints, or writes files; inputs are unchanged.

    Parameters
    ----------
    spec : Spectrum
    peaks : DataFrame or None
        Optional snapshot of peak partials (markers in colour slot 2).
    db, floor_db : amplitude display
    fmin, fmax : float, float or None
        Frequency crop in Hz; ``fmax=None`` means ``spec.sr / 2``.
    freq_scale : {'linear', 'log'}
    max_points : int or None
        Display-only min/max budget for the spectrum line.
    gl : bool
        Opt-in ``Scattergl`` (WebGL context limits apply).
    theme, title

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    TypeError
        ``spec`` is not a Spectrum.
    ValueError, SchemaError
        Invalid arguments or peaks table.
    """
    if not isinstance(spec, Spectrum):
        raise TypeError(f"spec must be a Spectrum; got {type(spec).__name__}")
    theme = _theme_name(theme)
    freq_scale = _freq_scale(freq_scale)
    max_points = _optional_max_points(max_points)
    fmin, fmax = check_fmin_fmax(fmin, fmax)
    if fmax is None:
        fmax = float(spec.sr / 2)
    elif fmax <= fmin:
        raise ValueError(f"fmax must be > fmin; got fmin={fmin!r}, fmax={fmax!r}")
    mask = freq_mask(spec.freqs, fmin, fmax)
    freqs = np.asarray(spec.freqs[mask], dtype=np.float64)
    amps = np.asarray(spec.amp[mask], dtype=np.float64)
    if freq_scale == "log":
        keep = freqs > 0
        freqs, amps = freqs[keep], amps[keep]
    y = np.asarray(to_plot_amplitude(amps, db=db, floor_db=floor_db), dtype=np.float64)
    x, y = _minmax(freqs, y, max_points)
    scatter = go.Scattergl if gl else go.Scatter
    fig = go.Figure()
    n_series = 1
    fig.add_trace(scatter(
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float32),
        mode="lines",
        name="spectrum",
        line=dict(width=1, color=colorway_colour(theme, 0)),
        hovertemplate=(
            "frequency: %{x:.2f} Hz<br>amplitude: %{y:"
            + (".1f" if db else ".4g")
            + "}" + (" dB" if db else "") + "<extra></extra>"
        ),
        showlegend=peaks is not None,
    ))
    if peaks is not None:
        table, kind = _table(peaks, None)
        if kind != "snapshot":
            raise ValueError(
                f"peaks must be a snapshot; got kind {kind!r}"
            )
        n_series = 2
        pf = table["freq"].to_numpy(dtype=np.float64)
        pa = table["amp"].to_numpy(dtype=np.float64)
        pm = (pf >= fmin) & (pf <= fmax)
        if freq_scale == "log":
            dropped = int(((pf <= 0) & pm).sum())
            pm &= pf > 0
            if dropped:
                import warnings
                warnings.warn(
                    f"{dropped} row(s) with frequency <= 0 left out on a log axis",
                    UserWarning,
                    stacklevel=2,
                )
        pf, pa = pf[pm], pa[pm]
        py = to_plot_amplitude(pa, db=db, floor_db=floor_db)
        texts = [hover_amp_freq(f, a, db=db) for f, a in zip(pf, py)]
        fig.add_trace(go.Scatter(
            x=pf,
            y=np.asarray(py, dtype=np.float64),
            mode="markers",
            name=f"peaks ({len(pf)})",
            marker=marker_style(colorway_colour(theme, 1), theme),
            hovertemplate="%{text}<extra></extra>",
            text=texts,
            showlegend=True,
        ))
    x0: float | None = fmin
    if freq_scale == "log":
        positive = freqs[freqs > 0] if len(freqs) else np.array([])
        if len(positive):
            x0 = float(max(fmin, positive[0])) if fmin > 0 else float(positive[0])
        else:
            x0 = None
    apply_layout(fig, theme=theme, title=title, n_series=n_series)
    xaxis_kwargs: dict[str, object] = dict(title_text="frequency (Hz)", type=freq_scale)
    if x0 is not None:
        xaxis_kwargs["range"] = axis_range(x0, fmax, freq_scale)
    fig.update_xaxes(**xaxis_kwargs)
    fig.update_yaxes(title_text=amplitude_axis_label(db=db))
    return fig


def _pool_frames(times, amp, max_frames):
    """Display-only max-pool of consecutive frames; x is the group mean time."""
    n_frames = amp.shape[1]
    if max_frames is None or n_frames <= max_frames:
        return times, amp
    group = int(ceil(n_frames / max_frames))
    n_out = int(ceil(n_frames / group))
    out_amp = np.empty((amp.shape[0], n_out), dtype=np.float64)
    out_times = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        start = i * group
        stop = min(start + group, n_frames)
        out_amp[:, i] = amp[:, start:stop].max(axis=1)
        out_times[i] = times[start:stop].mean()
    return out_times, out_amp


def _mel_ticks(y_min: float, y_max: float):
    """Tick values in Slaney mel, thinned so neighbours are ≥ 5% of the span."""
    librosa = import_optional("librosa", extra="librosa", feature="mel spectrogram support")

    candidates = []
    for hz in _MEL_TICK_HZ:
        mel = float(librosa.hz_to_mel(hz, htk=False))
        if y_min <= mel <= y_max:
            text = f"{hz // 1000}k" if hz >= 1000 else str(hz)
            candidates.append((mel, text))
    if not candidates:
        return [], []
    span = y_max - y_min
    min_gap = 0.05 * span if span > 0 else 0.0
    # Greedy from the top: keep a tick only if ≥ 5% of the range below the last kept.
    kept: list[tuple[float, str]] = []
    for mel, text in reversed(candidates):
        if not kept or (kept[-1][0] - mel) >= min_gap:
            kept.append((mel, text))
    kept.reverse()
    return [m for m, _ in kept], [t for _, t in kept]


def _hz_to_mel_coords(y):
    """Convert Hz coordinates to Slaney mel, preserving None separators."""
    librosa = import_optional("librosa", extra="librosa", feature="mel spectrogram support")

    out: list[float | None] = []
    for value in y:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            out.append(None)
        else:
            out.append(float(librosa.hz_to_mel(float(value), htk=False)))
    return out


def plot_spectrogram(
    spec: Spectrogram | MelSpectrogram,
    *,
    db: bool = True,
    floor_db: float = -120.0,
    range_db: float = 80.0,
    fmin: float = 0.0,
    fmax: float | None = None,
    freq_scale: str = "linear",
    max_frames: int | None = 2000,
    overlay: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
    overlay_labels: Sequence[str] | None = None,
    audio: ArrayLike | None = None,
    colorscale: str | list[list] | None = None,
    theme: str = "light",
    title: str | None = None,
) -> go.Figure:
    """Return a Plotly heatmap of a spectrogram or mel spectrogram.

    Frame pooling with ``max_frames`` is display-only. Never shows, prints, or
    writes files; inputs are unchanged. Plotting a ``MelSpectrogram`` needs
    ``audiospylt[plot]`` and ``audiospylt[librosa]``.

    Parameters
    ----------
    spec : Spectrogram or MelSpectrogram
    db, floor_db, range_db : colour mapping
        In dB mode, ``zmax`` is the max of shown z and ``zmin = zmax - range_db``.
    fmin, fmax : frequency crop (Hz); mel axis uses the same Hz bounds.
    freq_scale : {'linear', 'log'}
        Log is invalid for ``MelSpectrogram``.
    max_frames : int or None
        Display-only max-pool budget (``>= 1`` or None).
    overlay : table, list of tables, or None
        Events or tracks (snapshot raises, naming ``snapshot_to_events``).
    overlay_labels : sequence or None
    audio : array-like or None
        Optional waveform under the heatmap (shared x, heights 0.75 / 0.25).
    colorscale : Plotly colorscale or None
        None uses the theme sequential scale.
    theme, title

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        ``spec`` is a ``MelSpectrogram`` and librosa is not installed.
    """
    if not isinstance(spec, (Spectrogram, MelSpectrogram)):
        raise TypeError(
            f"spec must be a Spectrogram or MelSpectrogram; got {type(spec).__name__}"
        )
    is_mel = isinstance(spec, MelSpectrogram)
    theme = _theme_name(theme)
    freq_scale = _freq_scale(freq_scale)
    if is_mel and freq_scale == "log":
        raise ValueError("freq_scale='log' is not supported for MelSpectrogram")
    range_db = _finite_scalar(range_db, "range_db")
    if range_db <= 0:
        raise ValueError(f"range_db must be > 0; got {range_db!r}")
    fmin, fmax = check_fmin_fmax(fmin, fmax)
    if fmax is None:
        fmax = float(spec.freqs.max() if len(spec.freqs) and is_mel else spec.sr / 2)
    if max_frames is not None:
        max_frames = _positive_int(max_frames, "max_frames", 1)

    freqs = np.asarray(spec.freqs, dtype=np.float64)
    times = np.asarray(spec.times, dtype=np.float64)
    amp = np.asarray(spec.amp, dtype=np.float64)
    mask = freq_mask(freqs, fmin, fmax)
    if freq_scale == "log":
        mask &= freqs > 0
    freqs = freqs[mask]
    amp = amp[mask, :]
    times, amp = _pool_frames(times, amp, max_frames)

    if is_mel:
        librosa = import_optional("librosa", extra="librosa", feature="mel spectrogram support")
        y_vals = np.asarray(librosa.hz_to_mel(freqs, htk=False), dtype=np.float64)
    else:
        y_vals = freqs

    z = np.asarray(to_plot_amplitude(amp, db=db, floor_db=floor_db), dtype=np.float32)
    if db:
        zmax = float(np.max(z)) if z.size else float(floor_db)
        zmin = zmax - range_db
        colorbar_title = "dB re FS"
    else:
        zmin, zmax = 0.0, None
        colorbar_title = "amplitude (re full scale)"

    overlays = []
    if overlay is not None:
        items = overlay if isinstance(overlay, (list, tuple)) else [overlay]
        for item in items:
            table, kind = _table(item, None)
            if kind == "snapshot":
                raise ValueError(
                    "overlay kind 'snapshot' needs time; use "
                    "model.convert.snapshot_to_events"
                )
            if kind not in ("events", "tracks"):
                raise ValueError(
                    f"plot_spectrogram overlay cannot draw kind {kind!r}; "
                    f"use plot_events"
                )
            overlays.append((table, kind))
    labels = _labels(overlay_labels, len(overlays)) if overlays else []

    with_audio = audio is not None
    wave = None
    wave_sr = None
    n_channels = 0
    mono_audio = True
    if with_audio:
        wave, wave_sr = _validate_audio(audio, spec.sr)
        if wave_sr != spec.sr:
            raise ValueError(
                f"audio sample rate must equal spec.sr ({spec.sr}); got {wave_sr}"
            )
        channels = wave[np.newaxis, :] if wave.ndim == 1 else wave
        n_channels = channels.shape[0]
        mono_audio = n_channels == 1
    else:
        channels = None

    # Heatmap uses slot 1; overlays slots 2…; multichannel audio after overlays.
    n_colour_slots = 1 + len(overlays) + (0 if mono_audio or not with_audio else n_channels)
    if n_colour_slots > 8:
        raise ValueError(
            "at most 8 series per figure (palette has 8 slots); facet instead"
        )

    scale = sequential_colorscale(theme) if colorscale is None else colorscale
    amp_fmt = ":.1f" if db else ":.4g"
    amp_unit = " dB" if db else ""
    colorbar: dict[str, object] = dict(title=colorbar_title)
    if with_audio:
        colorbar.update(y=1, yanchor="top", len=0.75, lenmode="fraction")
    if is_mel:
        # Band Hz as float32 customdata (n_mels × frames); mel y is not Hz.
        custom = np.broadcast_to(
            freqs.astype(np.float32)[:, np.newaxis], z.shape,
        ).copy()
        hover = (
            f"time: %{{x:.4g}} s<br>frequency: %{{customdata:.2f}} Hz<br>"
            f"amplitude: %{{z{amp_fmt}}}{amp_unit}<extra></extra>"
        )
        heatmap_kwargs = dict(
            x=times, y=y_vals, z=z, colorscale=scale, zmin=zmin,
            colorbar=colorbar, hovertemplate=hover, customdata=custom,
            showscale=True,
        )
    else:
        hover = (
            f"time: %{{x:.4g}} s<br>frequency: %{{y:.2f}} Hz<br>"
            f"amplitude: %{{z{amp_fmt}}}{amp_unit}<extra></extra>"
        )
        heatmap_kwargs = dict(
            x=times, y=y_vals, z=z, colorscale=scale, zmin=zmin,
            colorbar=colorbar, hovertemplate=hover, showscale=True,
        )
    if zmax is not None:
        heatmap_kwargs["zmax"] = zmax
    if with_audio:
        heatmap_kwargs.update(xaxis="x", yaxis="y")

    fig = go.Figure()
    fig.add_trace(go.Heatmap(**heatmap_kwargs))

    if overlays:
        from audiospylt.viz import tables as tables_mod

        for index, ((table, kind), label) in enumerate(zip(overlays, labels)):
            colour = colorway_colour(theme, index + 1)
            for trace in tables_mod._line_traces_for_table(
                table, kind, colour=colour, label=label, theme=theme,
                amp=None, db=db, floor_db=floor_db, range_db=range_db,
                freq_scale="linear" if is_mel else freq_scale,
                legendgroup=label, l_max=None,
            ):
                if is_mel:
                    trace.y = _hz_to_mel_coords(trace.y)
                if with_audio:
                    trace.update(xaxis="x", yaxis="y")
                fig.add_trace(trace)

    if with_audio:
        x_full = np.arange(channels.shape[1], dtype=np.float64) / wave_sr
        secondary = theme_colors(theme)["secondary"]
        for index in range(n_channels):
            x, y = _minmax(x_full, channels[index], 50_000)
            if mono_audio:
                colour = secondary
                show_leg = False
            else:
                colour = colorway_colour(theme, 1 + len(overlays) + index)
                show_leg = True
            # x starts at 0 here (no time_offset), so float32 resolves well below
            # the point spacing and halves the payload.
            fig.add_trace(go.Scatter(
                x=np.asarray(x, dtype=np.float32),
                y=np.asarray(y, dtype=np.float32),
                mode="lines", name=f"ch {index + 1}",
                line=dict(width=1, color=colour),
                xaxis="x", yaxis="y2",
                hovertemplate="time: %{x:.4g} s<br>amplitude: %{y:.4g}<extra></extra>",
                showlegend=show_leg,
            ))

    legend_entries = len(overlays) + (0 if mono_audio or not with_audio else n_channels)
    show_legend = legend_entries >= 1
    apply_layout(fig, theme=theme, title=title, n_series=2 if show_legend else 1)
    if show_legend:
        legend = dict(
            orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
            font=dict(color=theme_colors(theme)["ink"]),
        )
        layout_update = dict(legend=legend, showlegend=True)
        if title is not None:
            # Plotly 7 clamps paper title.y to [0, 1]; use container coords so the
            # title sits above the horizontal legend in the top margin.
            layout_update["title"] = dict(
                text=title,
                font=dict(color=theme_colors(theme)["ink"]),
                y=1.0, yref="container", yanchor="top",
            )
            layout_update["margin"] = dict(t=80)
        fig.update_layout(**layout_update)

    y_title = "frequency (mel)" if is_mel else "frequency (Hz)"
    yaxis_kwargs: dict = {"title_text": y_title}
    if len(y_vals):
        # First/last shown bin (cropped freqs stay ascending; mel centres too).
        lo, hi = float(y_vals[0]), float(y_vals[-1])
        if not is_mel and freq_scale == "log":
            yaxis_kwargs["range"] = axis_range(lo, hi, "log")
        else:
            yaxis_kwargs["range"] = [lo, hi]
    if is_mel and len(y_vals):
        tickvals, ticktext = _mel_ticks(float(y_vals.min()), float(y_vals.max()))
        yaxis_kwargs.update(
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
        )
    elif not is_mel:
        yaxis_kwargs["type"] = freq_scale

    if with_audio:
        fig.update_layout(
            xaxis=dict(title_text="time (s)", domain=[0, 1], anchor="y2"),
            yaxis=dict(domain=[0.25, 1.0], anchor="x", **{
                k: v for k, v in yaxis_kwargs.items() if v is not None
            }),
            yaxis2=dict(
                domain=[0.0, 0.25], anchor="x",
                title_text="amplitude (re full scale)",
            ),
        )
    else:
        fig.update_xaxes(title_text="time (s)")
        fig.update_yaxes(**{k: v for k, v in yaxis_kwargs.items() if v is not None})
    return fig
