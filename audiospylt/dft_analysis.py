import numpy as np
import pandas as pd
from scipy.signal import find_peaks, get_window
import plotly.graph_objects as go

try:
    # Optional (works in notebooks). Falls back gracefully in non-IPython contexts.
    from IPython.display import display as ipy_display
except Exception:  # pragma: no cover
    ipy_display = None

def apply_window(signal, window_type):
    window = get_window(window_type, len(signal))
    return signal * window

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

    fig.add_trace(go.Scatter(x=x_vals, y=spec, mode='lines', name='Spectrum'))
    if len(peaks):
        fig.add_trace(go.Scatter(x=peak_x_vals, y=spec[peaks], mode='markers', name='Peaks'))

    # Amplitude-axis scaling (plot only; peak detection stays linear).
    y_axis_type = "linear"
    spec_for_plot = spec
    peaks_y_for_plot = spec[peaks] if len(peaks) else np.array([])
    thresh_amp_low_for_plot = thresh_amp_low
    thresh_amp_high_for_plot = thresh_amp_high

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
    elif amp_axis_mode != "linear":
        raise ValueError(f"amp_axis_mode must be linear|log|mixed; got {amp_axis_mode}")

    # Update traces with possibly-transformed amplitude values.
    fig.data[0].y = spec_for_plot
    if len(peaks) and len(fig.data) > 1:
        fig.data[1].y = peaks_y_for_plot

    # Add lines for amplitude threshold values
    fig.add_shape(
        type="line",
        x0=0,
        y0=thresh_amp_low_for_plot,
        x1=x_vals[-1] if len(x_vals) else 0,
        y1=thresh_amp_low_for_plot,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        ),
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=thresh_amp_high_for_plot,
        x1=x_vals[-1] if len(x_vals) else 0,
        y1=thresh_amp_high_for_plot,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        ),
    )
    
    # Add lines for frequency threshold values
    # Map threshold Hz to plot coordinates for non-linear axis modes.
    def _map_freq_for_axis(f_hz):
        if freq_axis_mode == "mel":
            return float(_hz_to_mel([f_hz])[0])
        if freq_axis_mode == "mixed":
            return float(_mixed_warp([f_hz], freq_axis_mix, fmax_hz=fmax_hz, log_floor_hz=mixed_log_floor_hz)[0])
        return float(f_hz)

    fig.add_shape(
        type="line",
        x0=_map_freq_for_axis(thresh_freq_low),
        y0=0,
        x1=_map_freq_for_axis(thresh_freq_low),
        y1=thresh_amp_high_for_plot + 0.05 * thresh_amp_high_for_plot,
        line=dict(color="Red", width=2, dash="dash"),
    )
    if thresh_freq_high is not None:
        fig.add_shape(
            type="line",
            x0=_map_freq_for_axis(thresh_freq_high),
            y0=0,
            x1=_map_freq_for_axis(thresh_freq_high),
            y1=thresh_amp_high_for_plot + 0.05 * thresh_amp_high_for_plot,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            ),
        )
    
    fig.update_layout(
        title='Spectrum and Peaks',
        xaxis_title=x_title,
        yaxis_title='Amplitude',
        autosize=False,
        width=900,
        height=600,
        showlegend=True
    )

    fig.update_xaxes(type=x_axis_type)
    if tickvals is not None and ticktext is not None:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    fig.update_yaxes(type=y_axis_type)

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
    
    try:  # pragma: no cover
        from .plotting import show_plotly
    except Exception:  # pragma: no cover
        from audiospylt.plotting import show_plotly
    show_plotly(fig)

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
):
    if window_type:
        signal = apply_window(signal, window_type)
    
    freqs, spec = compute_fft(signal, sr)
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
    peaks_df = pd.DataFrame({'Frequency (Hz)': freqs[peaks], 'Amplitude': spec[peaks]})

    # Print results
    print('File name:', filename)
    duration = len(signal) / sr
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
        )

    return peaks_df
