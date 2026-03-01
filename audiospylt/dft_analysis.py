import numpy as np
import pandas as pd
from scipy.signal import find_peaks, get_window
import plotly.graph_objects as go
import os
from urllib.parse import urlparse

try:
    # Optional (works in notebooks). Falls back gracefully in non-IPython contexts.
    from IPython.display import display as ipy_display
except Exception:  # pragma: no cover
    ipy_display = None


def _basename_for_title(name: str) -> str:
    """
    Return a "basename-like" string for plot titles.

    - For local paths: `os.path.basename(...)`
    - For URLs: basename of the URL path (ignores query/fragment)
    """
    s = str(name).strip().rstrip("/\\")
    if not s:
        return s
    try:
        p = urlparse(s)
        if p.scheme and p.netloc:
            base = os.path.basename((p.path or "").rstrip("/"))
            return base or s
    except Exception:
        pass
    base = os.path.basename(s)
    return base or s


def _format_title_name(name: str, mode: str) -> str:
    mode_norm = str(mode).lower().strip().replace("-", "_")
    if mode_norm == "basename":
        return _basename_for_title(name)
    if mode_norm == "full":
        return str(name)
    raise ValueError(f"plot_title_mode must be 'full' or 'basename'; got {mode!r}")


def _show_plotly(fig):
    try:  # pragma: no cover
        from .plotting import show_plotly
    except Exception:  # pragma: no cover
        from audiospylt.plotting import show_plotly
    return show_plotly(fig)

def apply_window(signal, window_type):
    window = get_window(window_type, len(signal))
    return signal * window


def _compute_window(window_type: str | None, n: int) -> np.ndarray:
    if not window_type:
        return np.ones(int(n), dtype=float)
    return np.asarray(get_window(window_type, int(n)), dtype=float)

def compute_fft(signal, sr):
    fft = np.fft.rfft(signal) / len(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    spec = np.abs(fft)
    return freqs, spec

def _hz_to_mel(hz):
    """HTK-ish mel conversion; avoids extra deps here."""
    hz = np.asarray(hz, dtype=float)
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mixed_warp(freqs_hz, mix, fmax_hz, log_floor_hz):
    """
    Warp frequency coordinates to continuously blend linear->log spacing.

    mix=0   => y = f (linear)
    mix=1   => y = log-scaled coordinate normalized to [0, fmax_hz]
    """
    mix = float(mix)
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"freq_axis_mix must be in [0,1]; got {mix}")
    log_floor_hz = float(log_floor_hz)
    if log_floor_hz <= 0:
        raise ValueError(f"mixed_log_floor_hz must be > 0; got {log_floor_hz}")

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if freqs_hz.ndim != 1:
        raise ValueError("freqs_hz must be a 1D array")
    fmax_hz = float(fmax_hz)
    if fmax_hz <= 0:
        return freqs_hz.copy()

    log0 = np.log10(log_floor_hz)
    logf = np.log10(freqs_hz + log_floor_hz)
    log_max = np.log10(fmax_hz + log_floor_hz)
    denom = max(1e-12, (log_max - log0))
    log_scaled = (logf - log0) / denom * fmax_hz
    return (1.0 - mix) * freqs_hz + mix * log_scaled

def _mixed_warp_values(values, mix, vmax, log_floor, mix_param_name="mix", floor_param_name="log_floor"):
    """
    Generic warp for non-negative values to blend linear->log-like spacing.

    mix=0 => identity
    mix=1 => log-scaled and normalized to [0, vmax]
    """
    mix = float(mix)
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"{mix_param_name} must be in [0,1]; got {mix}")
    log_floor = float(log_floor)
    if log_floor <= 0:
        raise ValueError(f"{floor_param_name} must be > 0; got {log_floor}")

    values = np.asarray(values, dtype=float)
    vmax = float(vmax)
    if vmax <= 0:
        return values.copy()

    log0 = np.log10(log_floor)
    logv = np.log10(values + log_floor)
    log_max = np.log10(vmax + log_floor)
    denom = max(1e-12, (log_max - log0))
    log_scaled = (logv - log0) / denom * vmax
    return (1.0 - mix) * values + mix * log_scaled

def _default_tick_freqs_hz(fmax_hz):
    fmax_hz = float(fmax_hz)
    if fmax_hz <= 0:
        return np.array([0.0])
    if fmax_hz <= 200:
        return np.linspace(0.0, fmax_hz, num=6)
    start = 20.0
    if fmax_hz < start:
        start = max(1.0, fmax_hz / 10.0)
    return np.concatenate(([0.0], np.geomspace(start, fmax_hz, num=6)))

def _format_hz(v):
    v = float(v)
    if v >= 1000:
        return f"{v/1000:.1f} kHz"
    return f"{v:.0f} Hz"


def _partial_color_pairs(n_groups: int):
    """
    Return (line_colors, fill_colors) for partial overlays.
    """
    n = max(0, int(n_groups))
    palette_rgb = [
        (0, 180, 0),
        (255, 127, 14),
        (31, 119, 180),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    line_colors = []
    fill_colors = []
    for i in range(n):
        r, g, b = palette_rgb[i % len(palette_rgb)]
        line_colors.append(f"rgba({r}, {g}, {b}, 0.75)")
        fill_colors.append(f"rgba({r}, {g}, {b}, 0.12)")
    return line_colors, fill_colors


def _expand_color_spec(color_spec, n: int, fallback: list[str]) -> list[str]:
    """
    Expand a scalar/list color spec to n items.
    """
    n = max(0, int(n))
    if n == 0:
        return []
    if isinstance(color_spec, (list, tuple)):
        seq = [str(c) for c in color_spec if str(c).strip()]
        if seq:
            return [seq[i % len(seq)] for i in range(n)]
    if isinstance(color_spec, str) and color_spec.strip():
        return [color_spec] * n
    if fallback:
        return [fallback[i % len(fallback)] for i in range(n)]
    return ["rgba(0, 180, 0, 0.75)"] * n

def filter_peaks(
    spec,
    freqs,
    thresh_amp_low,
    thresh_amp_high,
    thresh_freq_low,
    thresh_freq_high,
    prominence=None,
    width=None,
    prominence_rel=None,
    width_hz=None,
    distance_hz=None,
):
    """
    Peak filtering helper.

    - prominence: raw scipy units (same as spec amplitude); kept for backwards compatibility
    - prominence_rel: [0,1] prominence as a fraction of max(spec) (more intuitive)
    - width: raw scipy bins
    - width_hz: peak width in Hz (converted to bins using FFT resolution)
    - distance_hz: minimum distance between peaks in Hz (converted to bins)
    """
    if len(freqs) < 2:
        return []

    freq_resolution = float(freqs[1] - freqs[0])
    if freq_resolution <= 0:
        return []

    # Derive scipy parameters from the more intuitive variants when provided.
    prom_val = prominence
    if prominence_rel is not None:
        pr = float(prominence_rel)
        if not (0.0 <= pr <= 1.0):
            raise ValueError(f"prominence_rel must be in [0,1]; got {prominence_rel}")
        prom_val = pr * float(np.max(spec))

    width_val = width
    if width_hz is not None:
        width_val = max(1.0, float(width_hz) / freq_resolution)

    distance_val = None
    if distance_hz is not None:
        distance_val = max(1, int(round(float(distance_hz) / freq_resolution)))

    find_peaks_kwargs = {
        "height": (thresh_amp_low, thresh_amp_high),
        "prominence": prom_val,
        "width": width_val,
    }
    if distance_val is not None:
        find_peaks_kwargs["distance"] = distance_val

    peaks, _ = find_peaks(spec, **find_peaks_kwargs)
    peaks = [
        peak
        for peak in peaks
        if thresh_freq_low <= freqs[peak]
        and (thresh_freq_high is None or freqs[peak] <= thresh_freq_high)
    ]
    return peaks

def plot_spectrum(
    freqs_hz,
    spec,
    peaks,
    thresh_amp_low,
    thresh_amp_high,
    thresh_freq_low,
    thresh_freq_high,
    freq_axis_mode="linear",
    freq_axis_mix=0.5,
    mixed_log_floor_hz=1.0,
    amp_axis_mode="linear",
    amp_axis_mix=0.5,
    amp_log_floor=1e-12,
    auto_plot_range=False,
    freq_plot_pad_hz=None,
    freq_plot_pad_frac=0.05,
    amp_plot_pad=None,
    amp_plot_pad_frac=0.10,
    amp_plot_pad_ratio=0.15,
    *,
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict | None = None,
    title: str | None = None,
    plot_partials: bool = False,
    partials_hz: list[float] | None = None,
    partials_by_f0: dict[float, list[float]] | None = None,
    partial_bandwidth_hz: float = 20.0,
    f0_hz: float | list[float] | None = None,
    spectrum_color: str | None = None,
    peaks_color: str | None = None,
    threshold_color: str = "Red",
    partials_color: str | list[str] = "rgba(0, 180, 0, 0.65)",
    partials_band_fill: str | list[str] = "rgba(0, 180, 0, 0.10)",
):
    max_amp = np.max(spec)
    fig = go.Figure()

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    fmax_hz = float(freqs_hz[-1]) if len(freqs_hz) else 0.0

    # Build frequency coordinates for plotting.
    x_axis_type = "linear"
    x_title = "Frequency (Hz)"
    x_vals = freqs_hz
    peak_x_vals = freqs_hz[peaks] if len(peaks) else np.array([])
    tickvals = None
    ticktext = None

    if freq_axis_mode == "log":
        # Use plotly's log axis; drop 0 Hz if present.
        if len(freqs_hz) and freqs_hz[0] == 0:
            x_vals = freqs_hz[1:]
            spec = spec[1:]
            # Remap peaks to the sliced array if needed.
            peaks = [p - 1 for p in peaks if p > 0]
            peak_x_vals = x_vals[peaks] if len(peaks) else np.array([])
        x_axis_type = "log"
    elif freq_axis_mode == "mel":
        x_vals = _hz_to_mel(freqs_hz)
        peak_x_vals = x_vals[peaks] if len(peaks) else np.array([])
        x_title = "Frequency (mel)"
        tick_freqs = _default_tick_freqs_hz(fmax_hz)
        tickvals = _hz_to_mel(tick_freqs)
        ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode == "mixed":
        x_vals = _mixed_warp(freqs_hz, freq_axis_mix, fmax_hz=fmax_hz, log_floor_hz=mixed_log_floor_hz)
        peak_x_vals = x_vals[peaks] if len(peaks) else np.array([])
        x_title = "Frequency"
        tick_freqs = _default_tick_freqs_hz(fmax_hz)
        tickvals = _mixed_warp(tick_freqs, freq_axis_mix, fmax_hz=fmax_hz, log_floor_hz=mixed_log_floor_hz)
        ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode != "linear":
        raise ValueError(f"freq_axis_mode must be linear|log|mel|mixed; got {freq_axis_mode}")

    spectrum_trace = go.Scatter(x=x_vals, y=spec, mode='lines', name='Spectrum')
    if spectrum_color is not None:
        spectrum_trace.update(line=dict(color=spectrum_color))
    fig.add_trace(spectrum_trace)
    if len(peaks):
        peaks_trace = go.Scatter(x=peak_x_vals, y=spec[peaks], mode='markers', name='Peaks')
        if peaks_color is not None:
            peaks_trace.update(marker=dict(color=peaks_color))
        fig.add_trace(peaks_trace)

    # Amplitude-axis scaling (plot only; peak detection stays linear).
    y_axis_type = "linear"
    spec_for_plot = spec
    peaks_y_for_plot = spec[peaks] if len(peaks) else np.array([])
    thresh_amp_low_for_plot = thresh_amp_low
    thresh_amp_high_for_plot = thresh_amp_high
    y_tickvals = None
    y_ticktext = None

    if amp_axis_mode == "log":
        y_axis_type = "log"
        floor = float(amp_log_floor)
        if floor <= 0:
            raise ValueError(f"amp_log_floor must be > 0; got {amp_log_floor}")
        # Plotly log axis cannot display non-positive values.
        if thresh_amp_low_for_plot <= 0 or thresh_amp_high_for_plot <= 0:
            raise ValueError("For amp_axis_mode='log', thresh_amp_low/high must be > 0.")
        spec_for_plot = np.maximum(spec_for_plot, floor)
        peaks_y_for_plot = np.maximum(peaks_y_for_plot, floor) if len(peaks) else peaks_y_for_plot
    elif amp_axis_mode == "mixed":
        y_axis_type = "linear"
        vmax = float(np.max(spec)) if len(spec) else 0.0
        spec_for_plot = _mixed_warp_values(
            spec_for_plot,
            mix=amp_axis_mix,
            vmax=vmax,
            log_floor=amp_log_floor,
            mix_param_name="amp_axis_mix",
            floor_param_name="amp_log_floor",
        )
        peaks_y_for_plot = spec_for_plot[peaks] if len(peaks) else peaks_y_for_plot
        thresh_amp_low_for_plot = float(_mixed_warp_values(
            [thresh_amp_low_for_plot],
            mix=amp_axis_mix,
            vmax=vmax,
            log_floor=amp_log_floor,
            mix_param_name="amp_axis_mix",
            floor_param_name="amp_log_floor",
        )[0])
        thresh_amp_high_for_plot = float(_mixed_warp_values(
            [thresh_amp_high_for_plot],
            mix=amp_axis_mix,
            vmax=vmax,
            log_floor=amp_log_floor,
            mix_param_name="amp_axis_mix",
            floor_param_name="amp_log_floor",
        )[0])

        # Critical: in mixed mode, the plotted y-values live in a warped coordinate space.
        # If we leave default numeric ticks, users will compare threshold lines (labeled in raw
        # amplitude) to warped y-axis numbers and it will look "misaligned".
        #
        # Instead, we label the y-axis with *raw amplitude* values at selected tick positions,
        # mapped into the warped coordinate space.
        def _format_amp(v: float) -> str:
            v = float(v)
            if v == 0:
                return "0"
            if abs(v) < 1e-3 or abs(v) >= 1e3:
                return f"{v:.2e}"
            return f"{v:.4g}"

        try:
            floor = float(amp_log_floor)
        except Exception:
            floor = 1e-12
        floor = max(1e-300, floor)

        raw_max = float(np.max(spec)) if len(spec) else 0.0
        raw_max = max(raw_max, float(thresh_amp_high), float(thresh_amp_low), floor)

        # Pick a small, stable set of meaningful raw ticks.
        # Include 0 + thresholds + a few log-ish values up to max.
        tick_raw = [0.0, float(thresh_amp_low), float(thresh_amp_high)]
        if raw_max > 0:
            start = max(floor, raw_max / 1e6)
            tick_raw += list(np.geomspace(start, raw_max, num=5))
        # Deduplicate and keep finite, sorted.
        tick_raw = sorted({float(v) for v in tick_raw if np.isfinite(v) and float(v) >= 0.0})

        y_tickvals = _mixed_warp_values(
            tick_raw,
            mix=amp_axis_mix,
            vmax=raw_max,
            log_floor=floor,
            mix_param_name="amp_axis_mix",
            floor_param_name="amp_log_floor",
        )
        y_ticktext = [_format_amp(v) for v in tick_raw]
    elif amp_axis_mode != "linear":
        raise ValueError(f"amp_axis_mode must be linear|log|mixed; got {amp_axis_mode}")

    # Update traces with possibly-transformed amplitude values.
    fig.data[0].y = spec_for_plot
    if len(peaks) and len(fig.data) > 1:
        fig.data[1].y = peaks_y_for_plot

    # Threshold "lines" as traces (so they appear in the legend).
    # We plot them in the *axis coordinate space* (Hz, mel, or mixed warp), and in the
    # amplitude plot space (linear/log/mixed), so they always align with the plotted spectrum.
    x_min = float(np.min(x_vals)) if len(x_vals) else 0.0
    x_max = float(np.max(x_vals)) if len(x_vals) else 0.0
    y_max = float(np.max(spec_for_plot)) if len(spec_for_plot) else 1.0
    y_max = max(y_max, float(thresh_amp_high_for_plot)) * 1.05 if y_max > 0 else 1.0
    if y_axis_type == "log":
        # Log axis can't show <=0.
        y_min = float(np.min(spec_for_plot[spec_for_plot > 0])) if np.any(spec_for_plot > 0) else float(amp_log_floor)
        y_min = max(float(amp_log_floor), y_min)
    else:
        y_min = 0.0

    thr_line_style = dict(color=threshold_color, width=2, dash="dash")
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[thresh_amp_low_for_plot, thresh_amp_low_for_plot],
            mode="lines",
            name=f"Amp low ({thresh_amp_low:g})",
            line=thr_line_style,
            hoverinfo="skip",
            legendgroup="thresholds",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[thresh_amp_high_for_plot, thresh_amp_high_for_plot],
            mode="lines",
            name=f"Amp high ({thresh_amp_high:g})",
            line=thr_line_style,
            hoverinfo="skip",
            legendgroup="thresholds",
        )
    )

    # Frequency thresholds: map Hz -> axis coordinate space for mel/mixed; keep Hz for linear/log.
    def _map_freq_for_axis(f_hz):
        if freq_axis_mode == "mel":
            return float(_hz_to_mel([f_hz])[0])
        if freq_axis_mode == "mixed":
            return float(
                _mixed_warp(
                    [f_hz],
                    freq_axis_mix,
                    fmax_hz=fmax_hz,
                    log_floor_hz=mixed_log_floor_hz,
                )[0]
            )
        return float(f_hz)

    def _maybe_add_freq_line(f_hz: float, label: str):
        f_hz = float(f_hz)
        if x_axis_type == "log" and f_hz <= 0:
            # Can't draw 0 Hz on a log axis.
            return
        x_thr = _map_freq_for_axis(f_hz)
        fig.add_trace(
            go.Scatter(
                x=[x_thr, x_thr],
                y=[y_min, y_max],
                mode="lines",
                name=f"{label} ({_format_hz(f_hz)})",
                line=thr_line_style,
                hoverinfo="skip",
                legendgroup="thresholds",
            )
        )

    _maybe_add_freq_line(thresh_freq_low, "Freq low")
    if thresh_freq_high is not None:
        _maybe_add_freq_line(thresh_freq_high, "Freq high")

    # Optional: overlay partial grid (k*f0) and +/- bandwidth bands.
    if plot_partials and (partials_hz or partials_by_f0):
        bw = float(partial_bandwidth_hz)
        if bw < 0:
            raise ValueError(f"partial_bandwidth_hz must be >= 0; got {partial_bandwidth_hz}")

        grouped_partials = []
        if partials_by_f0:
            for f0_i, partials_i in partials_by_f0.items():
                f0_val = float(f0_i)
                if f0_val <= 0:
                    continue
                pts = [float(p) for p in partials_i if float(p) > 0]
                if pts:
                    grouped_partials.append((f0_val, pts))
        elif partials_hz:
            grouped_partials.append((None, [float(p) for p in partials_hz if float(p) > 0]))

        if grouped_partials:
            n_groups = len(grouped_partials)
            palette_line, palette_fill = _partial_color_pairs(n_groups)

            # Keep figures responsive when f0 is very low and multiple fundamentals are used.
            max_partials_total = 200
            max_partials_per_group = max(1, max_partials_total // n_groups)

            if n_groups > 1:
                line_colors = _expand_color_spec(partials_color, n_groups, palette_line)
                if isinstance(partials_color, str):
                    line_colors = palette_line

                fill_colors = _expand_color_spec(partials_band_fill, n_groups, palette_fill)
                if isinstance(partials_band_fill, str):
                    fill_colors = palette_fill
            else:
                line_colors = _expand_color_spec(partials_color, 1, palette_line)
                fill_colors = _expand_color_spec(partials_band_fill, 1, palette_fill)

            for gi, (f0_i, partials_i) in enumerate(grouped_partials):
                partials_use = list(partials_i)[:max_partials_per_group]
                if not partials_use:
                    continue

                band_label = (
                    f"Partials (f0={_format_hz(f0_i)}) ±{bw:g} Hz"
                    if f0_i is not None
                    else f"Partials ±{bw:g} Hz"
                )
                center_line = dict(color=line_colors[gi], width=1, dash="dot")
                band_fill = fill_colors[gi]
                legend_group = f"partials_{gi}"
                band_x = []
                band_y = []
                center_x = []
                center_y = []

                for f_hz in partials_use:
                    if x_axis_type == "log" and f_hz <= 0:
                        continue
                    x0 = _map_freq_for_axis(max(0.0, f_hz - bw))
                    x1 = _map_freq_for_axis(f_hz + bw)
                    x_center = _map_freq_for_axis(f_hz)

                    # Build one multi-polygon trace so it can be legend-group toggled with center lines.
                    band_x.extend([x0, x1, x1, x0, x0, None])
                    band_y.extend([y_min, y_min, y_max, y_max, y_min, None])

                    # Build one multi-segment trace for all dotted center lines in this group.
                    center_x.extend([x_center, x_center, None])
                    center_y.extend([y_min, y_max, None])

                if not center_x:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=band_x,
                        y=band_y,
                        mode="lines",
                        fill="toself",
                        fillcolor=band_fill,
                        line=dict(width=0, color=band_fill),
                        hoverinfo="skip",
                        legendgroup=legend_group,
                        showlegend=False,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=center_x,
                        y=center_y,
                        mode="lines",
                        name=band_label,
                        line=center_line,
                        hoverinfo="skip",
                        legendgroup=legend_group,
                        showlegend=True,
                    )
                )
    
    fig.update_layout(
        title=('Spectrum and Peaks' if title is None else title),
        xaxis_title=x_title,
        yaxis_title='Amplitude',
        autosize=False,
        width=(900 if plot_width is None else int(plot_width)),
        height=(600 if plot_height is None else int(plot_height)),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),
    )

    fig.update_xaxes(type=x_axis_type)
    if tickvals is not None and ticktext is not None:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    fig.update_yaxes(type=y_axis_type)
    if y_tickvals is not None and y_ticktext is not None:
        fig.update_yaxes(tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext)

    # Auto-zoom plotting ranges based on thresholds (+ padding). Plot-only.
    if auto_plot_range:
        # Frequency range in Hz.
        fmin_hz = float(thresh_freq_low)
        fmax_hz_thr = float(thresh_freq_high) if thresh_freq_high is not None else fmax_hz
        fmin_hz = max(0.0, min(fmin_hz, fmax_hz))
        fmax_hz_thr = max(fmin_hz, min(fmax_hz_thr, fmax_hz))
        fspan = max(1e-12, fmax_hz_thr - fmin_hz)
        fpad = float(freq_plot_pad_hz) if freq_plot_pad_hz is not None else float(freq_plot_pad_frac) * fspan
        fmin_hz_p = max(0.0, fmin_hz - fpad)
        fmax_hz_p = min(fmax_hz, fmax_hz_thr + fpad)

        # Amplitude range (use plot-space thresholds, so mixed/log behave intuitively).
        a_low = float(thresh_amp_low_for_plot)
        a_high = float(thresh_amp_high_for_plot)
        a_low = max(0.0, min(a_low, a_high))
        a_high = max(a_low, a_high)
        aspan = max(1e-12, a_high - a_low)
        apad = float(amp_plot_pad) if amp_plot_pad is not None else float(amp_plot_pad_frac) * aspan
        a_low_p = max(0.0, a_low - apad)
        a_high_p = a_high + apad

        # Apply x-axis range.
        if x_axis_type == "log":
            # Plotly expects log10 values for axis range when type='log'.
            positive_freqs = freqs_hz[freqs_hz > 0]
            min_pos = float(np.min(positive_freqs)) if len(positive_freqs) else 1e-6
            fmin_use = max(min_pos, fmin_hz_p)
            fmax_use = max(fmin_use * (1.0 + 1e-6), fmax_hz_p)
            fig.update_xaxes(range=[np.log10(fmin_use), np.log10(fmax_use)])
        else:
            # Map Hz threshold range to the axis coordinate space.
            def _map_freq_for_axis_range(f_hz):
                if freq_axis_mode == "mel":
                    return float(_hz_to_mel([f_hz])[0])
                if freq_axis_mode == "mixed":
                    return float(_mixed_warp([f_hz], freq_axis_mix, fmax_hz=fmax_hz, log_floor_hz=mixed_log_floor_hz)[0])
                return float(f_hz)

            fig.update_xaxes(range=[_map_freq_for_axis_range(fmin_hz_p), _map_freq_for_axis_range(fmax_hz_p)])

        # Apply y-axis range.
        if y_axis_type == "log":
            # Use multiplicative padding in log mode (more sensible than additive).
            floor = float(amp_log_floor)
            low_raw = max(floor, float(thresh_amp_low))
            high_raw = max(low_raw * (1.0 + 1e-9), float(thresh_amp_high))
            r = float(amp_plot_pad_ratio)
            if r < 0:
                raise ValueError(f"amp_plot_pad_ratio must be >= 0; got {amp_plot_pad_ratio}")
            low_use = max(floor, low_raw / (1.0 + r))
            high_use = high_raw * (1.0 + r)
            fig.update_yaxes(range=[np.log10(low_use), np.log10(high_use)])
        else:
            fig.update_yaxes(range=[a_low_p, a_high_p])
    
    if plotly_layout:
        fig.update_layout(**plotly_layout)

    _show_plotly(fig)


def plot_windowed_waveform(
    signal,
    sr,
    *,
    window_type: str | None = None,
    plot_title_name: str | None = None,
    plot_title_mode: str = "basename",
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict | None = None,
    show: bool = True,
):
    """
    Plot the waveform with the selected window applied, and overlay the window curve in red.

    The window curve is shown on a secondary y-axis (0..1) to keep it readable regardless of audio amplitude.
    """
    y = np.asarray(signal, dtype=float).reshape(-1)
    n = int(y.size)
    sr = float(sr) if sr else 0.0
    t = (np.arange(n, dtype=float) / sr) if (sr > 0 and n > 0) else np.array([0.0])

    w = _compute_window(window_type, n)
    y_win = y * w

    name = plot_title_name or ""
    name_fmt = _format_title_name(name, plot_title_mode) if name else ""
    win_label = str(window_type) if window_type else "none"
    title = f"Waveform (window={win_label})"
    if name_fmt:
        title = f"{title} {name_fmt}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y_win, mode="lines", name="Waveform"))
    fig.add_trace(go.Scatter(x=t, y=w, mode="lines", name="Window", line=dict(color="red"), yaxis="y2"))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        yaxis2=dict(title="Window", range=[0, 1], side="right", overlaying="y", showgrid=False, zeroline=False),
        autosize=False,
        width=(900 if plot_width is None else int(plot_width)),
        height=(350 if plot_height is None else int(plot_height)),
        showlegend=True,
    )

    if plotly_layout:
        fig.update_layout(**plotly_layout)

    if show:
        _show_plotly(fig)
    return fig

def analyze_signal(
    signal,
    sr,
    filename,
    window_type='boxcar',
    thresh_amp_low=0.2,
    thresh_amp_high=0.4,
    thresh_freq_low=0,
    thresh_freq_high=None,
    prominence=None,
    width=None,
    prominence_rel=None,
    width_hz=None,
    distance_hz=None,
    freq_axis_mode="linear",
    freq_axis_mix=0.5,
    mixed_log_floor_hz=1.0,
    amp_axis_mode="linear",
    amp_axis_mix=0.5,
    amp_log_floor=1e-12,
    auto_plot_range=False,
    freq_plot_pad_hz=None,
    freq_plot_pad_frac=0.05,
    amp_plot_pad=None,
    amp_plot_pad_frac=0.10,
    amp_plot_pad_ratio=0.15,
    show_peaks=False,
    show_plot=True,
    *,
    plot_width: int | None = None,
    plot_height: int | None = None,
    plot_title_name: str | None = None,
    plot_title_mode: str = "basename",
    plotly_layout: dict | None = None,
    show_windowed_waveform: bool = False,
    partial_tracking: bool = False,
    f0: float | list[float] | None = None,
    partial_bandwidth_hz: float = 20.0,
    plot_partials: bool = False,
    spectrum_color: str | None = None,
    peaks_color: str | None = None,
    threshold_color: str = "Red",
    partials_color: str | list[str] = "rgba(0, 180, 0, 0.65)",
    partials_band_fill: str | list[str] = "rgba(0, 180, 0, 0.10)",
):
    # Normalize input signal once; apply the window for FFT/analysis.
    y = np.asarray(signal, dtype=float).reshape(-1)
    w = _compute_window(window_type, int(y.size))
    y_win = y * w

    # Render waveform first (so it appears before prints/DF display/spectrum plot in notebooks).
    if show_windowed_waveform:
        plot_windowed_waveform(
            y,
            sr,
            window_type=window_type,
            plot_title_name=(plot_title_name if plot_title_name is not None else filename),
            plot_title_mode=plot_title_mode,
            plot_width=plot_width,
            # Use the same plot_height unless the caller overrides via plotly_layout; keep a sensible default.
            plot_height=plot_height,
            plotly_layout=plotly_layout,
            show=True,
        )
    
    freqs, spec = compute_fft(y_win, sr)
    peaks = filter_peaks(
        spec,
        freqs,
        thresh_amp_low,
        thresh_amp_high,
        thresh_freq_low,
        thresh_freq_high,
        prominence=prominence,
        width=width,
        prominence_rel=prominence_rel,
        width_hz=width_hz,
        distance_hz=distance_hz,
    )

    # Optional: partial tracking relative to f0 (applies AFTER initial peak filtering).
    # f0 can be a single float or a list of fundamentals (e.g. chord).
    partials_hz = None
    partials_by_f0 = None
    f0_hz = None  # passed to plot: float or list[float]
    if partial_tracking:
        if f0 is None:
            raise ValueError("partial_tracking=True requires f0 (Hz) or list of f0")
        f0_list = [float(x) for x in (f0 if isinstance(f0, (list, tuple)) else [f0])]
        if not f0_list or any(x <= 0 for x in f0_list):
            raise ValueError(f"each f0 must be > 0; got {f0}")
        f0_hz = f0_list[0] if len(f0_list) == 1 else f0_list
        bw = float(partial_bandwidth_hz)
        if bw < 0:
            raise ValueError(f"partial_bandwidth_hz must be >= 0; got {partial_bandwidth_hz}")

        # For each peak: find closest partial across all f0s; keep if within bw.
        peak_freqs = freqs[peaks] if len(peaks) else np.array([], dtype=float)
        if peak_freqs.size:
            best_k = np.zeros(peak_freqs.size, dtype=int)
            best_f0_idx = np.zeros(peak_freqs.size, dtype=int)
            best_delta = np.full(peak_freqs.size, np.inf)
            for fi, f0_i in enumerate(f0_list):
                k = np.rint(peak_freqs / f0_i).astype(int)
                k = np.maximum(k, 0)
                target = k.astype(float) * f0_i
                delta = np.abs(peak_freqs - target)
                # for each peak, see if this (f0_i, k) is better
                better = (k >= 1) & (delta < best_delta)
                best_delta = np.where(better, delta, best_delta)
                best_k = np.where(better, k, best_k)
                best_f0_idx = np.where(better, fi, best_f0_idx)
            mask = (best_k >= 1) & (best_delta <= bw)
            peaks = [p for p, keep in zip(peaks, mask) if bool(keep)]
            k_kept = best_k[mask]
            f0_vals = np.array([f0_list[i] for i in best_f0_idx], dtype=float)
            delta_kept = (peak_freqs - best_k.astype(float) * f0_vals)[mask]
            f0_kept = f0_vals[mask]
        else:
            k_kept = np.array([], dtype=int)
            delta_kept = np.array([], dtype=float)
            f0_kept = np.array([], dtype=float)

        # Build partial grid for overlay, grouped by f0 so each f0 can get its own style.
        fmax_partials = float(thresh_freq_high) if thresh_freq_high is not None else float(freqs[-1] if len(freqs) else 0.0)
        partials_by_f0 = {}
        partial_set = set()
        for f0_i in f0_list:
            n_partials = int(np.floor(fmax_partials / f0_i)) if fmax_partials > 0 else 0
            partials_i = [f0_i * k_ for k_ in range(1, n_partials + 1)]
            if partials_i:
                partials_by_f0[float(f0_i)] = partials_i
                partial_set.update(partials_i)
        if partial_set:
            partials_hz = sorted(partial_set)

    peaks_df = pd.DataFrame({'Frequency (Hz)': freqs[peaks], 'Amplitude': spec[peaks]})
    if partial_tracking:
        # Attach partial info if available (align to the kept peaks order).
        peaks_df["Partial #"] = k_kept[: len(peaks_df)]
        peaks_df["Delta from partial (Hz)"] = delta_kept[: len(peaks_df)]
        peaks_df["f0 (Hz)"] = f0_kept[: len(peaks_df)]

    # Print results
    print('File name:', filename)
    duration = (len(y) / sr) if sr else 0.0
    print('Duration (s):', round(duration, 6))
    print('Sampling rate (Hz):', sr)
    print()

    max_amp = np.max(spec)
    print('Maximum amplitude value:', round(max_amp, 6))

    num_bands = len(freqs)
    freq_resolution = freqs[1] - freqs[0]
    print('Total number of bands:', num_bands)
    print('Frequency resolution (Hz):', round(freq_resolution, 6))
    print()

    print('Amplitude Threshold 1:', thresh_amp_low)
    print('Amplitude Threshold 2:', thresh_amp_high)
    print('Frequency Threshold 1 (Hz):', thresh_freq_low)
    if thresh_freq_high is not None:
        print('Frequency Threshold 2 (Hz):', thresh_freq_high)
    print()

    if show_peaks:
        print('Peaks:')
        if ipy_display is not None:
            ipy_display(peaks_df)
        else:
            print(peaks_df.to_string(index=False))

    if show_plot:
        title_name = plot_title_name if plot_title_name is not None else filename
        title_fmt = _format_title_name(title_name, plot_title_mode) if title_name else ""
        win_label = str(window_type) if window_type else "none"
        spec_title = "Spectrum and Peaks"
        if title_fmt:
            spec_title = f"{spec_title} {title_fmt}"
        if window_type:
            spec_title = f"{spec_title} (window={win_label})"

        plot_spectrum(
            freqs,
            spec,
            peaks,
            thresh_amp_low,
            thresh_amp_high,
            thresh_freq_low,
            thresh_freq_high,
            freq_axis_mode=freq_axis_mode,
            freq_axis_mix=freq_axis_mix,
            mixed_log_floor_hz=mixed_log_floor_hz,
            amp_axis_mode=amp_axis_mode,
            amp_axis_mix=amp_axis_mix,
            amp_log_floor=amp_log_floor,
            auto_plot_range=auto_plot_range,
            freq_plot_pad_hz=freq_plot_pad_hz,
            freq_plot_pad_frac=freq_plot_pad_frac,
            amp_plot_pad=amp_plot_pad,
            amp_plot_pad_frac=amp_plot_pad_frac,
            amp_plot_pad_ratio=amp_plot_pad_ratio,
            plot_width=plot_width,
            plot_height=plot_height,
            plotly_layout=plotly_layout,
            title=spec_title,
            plot_partials=bool(plot_partials),
            partials_hz=partials_hz,
            partials_by_f0=partials_by_f0,
            partial_bandwidth_hz=partial_bandwidth_hz,
            f0_hz=f0_hz,
            spectrum_color=spectrum_color,
            peaks_color=peaks_color,
            threshold_color=threshold_color,
            partials_color=partials_color,
            partials_band_fill=partials_band_fill,
        )

    return peaks_df
