"""Waveform synthesis and plotting helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
from scipy.signal import get_window, sawtooth, square
import plotly.graph_objects as go

try:  # pragma: no cover
    from .plotting import show_plotly
except Exception:  # pragma: no cover
    from audiospylt.plotting import show_plotly


def _require_config(config: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in config]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")

def generate_waveforms(config: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray] | None:
    """
    Generate waveform data based on the provided configuration.

    Expected config keys:
      - dt: time step (seconds) between samples
      - t_max: duration in seconds
      - amp1, freq1, phase1
      - amp2, freq2, phase2
      - index_am, index_fm
      - (optional) square_duty: duty cycle in [0, 1] for 'square' (default 0.5)
      - (optional) saw_width: width in [0, 1] for 'saw'/'sawtooth' (default 1.0)

    Returns (waveform_data, t) or None on invalid parameters.
    """
    _require_config(
        config,
        (
            "t_max",
            "amp1",
            "freq1",
            "phase1",
            "amp2",
            "freq2",
            "phase2",
            "index_am",
            "index_fm",
        ),
    )
    if "dt" not in config and "n" not in config:
        raise KeyError("Config must include 'dt' (preferred) or legacy 'n'.")

    # Extract configuration parameters.
    if "dt" in config and "n" in config:
        raise ValueError("Provide only one of 'dt' or legacy 'n' in config, not both.")
    if "dt" in config:
        dt = float(config["dt"])
    elif "n" in config:
        dt = float(config["n"])
    else:
        raise KeyError("Config must include 'dt' (preferred) or legacy 'n'.")
    t_max = float(config["t_max"])
    amp1 = float(config["amp1"])
    freq1 = float(config["freq1"])
    phase1 = float(config["phase1"])
    amp2 = float(config["amp2"])
    freq2 = float(config["freq2"])
    phase2 = float(config["phase2"])
    index_am = float(config["index_am"])
    index_fm = float(config["index_fm"])

    if dt <= 0 or t_max <= 0:
        print("Error: Invalid dt or t_max value.")
        return None

    t = np.arange(0.0, t_max, dt, dtype=float)
    if t.size == 0:
        print("Error: Time vector is empty; check n (dt) and t_max.")
        return None

    waveform_data: Dict[str, np.ndarray] = {}
    
    # Generate waveforms
    waveform_data['sine1'] = amp1 * np.sin(2 * np.pi * freq1 * t + phase1)
    waveform_data['sine2'] = amp2 * np.sin(2 * np.pi * freq2 * t + phase2)

    # Additional (non-sinusoidal) waveforms derived from (amp1, freq1, phase1).
    # SciPy definitions:
    # - square(x, duty): duty in [0, 1]
    # - sawtooth(x, width): width in [0, 1] (0.5 yields triangle)
    square_duty = float(config.get("square_duty", 0.5))
    saw_width = float(config.get("saw_width", 1.0))
    waveform_data["square"] = amp1 * square(2 * np.pi * freq1 * t + phase1, duty=square_duty)
    waveform_data["saw"] = amp1 * sawtooth(2 * np.pi * freq1 * t + phase1, width=saw_width)
    
    # Amplitude Modulation (AM)
    waveform_data['am'] = (amp1 + index_am * waveform_data['sine2']) * np.sin(2 * np.pi * freq1 * t + phase1)
    
    # Frequency Modulation (FM)
    # Treat `index_fm * sine2` as a frequency deviation term (in Hz if sine2 is unitless),
    # then integrate the instantaneous frequency to obtain phase:
    #   f_inst(t) = freq1 + index_fm * sine2(t)
    #   phase[n] = phase1 + 2π * Σ_{k < n} f_inst[k] * dt
    f_inst = freq1 + index_fm * waveform_data["sine2"]
    phase_incr = 2.0 * np.pi * f_inst * dt
    phase = np.cumsum(phase_incr)
    phase = np.concatenate(([0.0], phase[:-1])) + phase1  # ensure phase(t=0) == phase1
    waveform_data["fm"] = amp1 * np.sin(phase)
    
    # Sum of Sine Waves
    waveform_data['sum'] = waveform_data['sine1'] + waveform_data['sine2']

    # Generate combined sine wave
    waveform_data['comb'] = np.zeros_like(t)
    half_period = max(1, int(round(t_max / (2 * dt))))
    for i in range(0, len(t), half_period):
        if (i // half_period) % 2 == 0:
            waveform_data['comb'][i:i+half_period] = waveform_data['sine1'][i:i+half_period]
        else:
            waveform_data['comb'][i:i+half_period] = waveform_data['sine2'][i:i+half_period]

    # Optionally apply a window function to all generated waveforms.
    # This is intentionally driven by config so notebooks can stay minimal.
    if bool(config.get("apply_window", False)):
        window_type = str(config.get("window_type", "tukey"))
        waveform_data = apply_window_to_waveforms(waveform_data, t, window_type)

    return waveform_data, t
    
def apply_window_to_waveforms(
    waveform_data: Dict[str, np.ndarray],
    t: np.ndarray,
    window_type: str,
) -> Dict[str, np.ndarray]:
    """Apply a window function in-place to all waveforms in the dict."""
    window = get_window(window_type, len(t))
    for waveform in waveform_data:
        waveform_data[waveform] *= window
    return waveform_data

def plot_waveforms(waveform_data: Mapping[str, np.ndarray], t: np.ndarray, config: Mapping[str, Any]) -> None:
    """Plot waveform data and their DFTs based on the provided configuration."""
    # Create the Plotly figure for waveforms
    fig_waveforms = go.Figure()

    selected_names = list(config["selected_waveforms"])
    selected_waveforms_data = [
        waveform_data[name] for name in selected_names if name in waveform_data
    ]

    if not selected_waveforms_data:
        print("Error: No valid waveform data available.")
        return  # Exit the function if no data is available

    stacked = np.concatenate([np.ravel(w) for w in selected_waveforms_data])
    waveform_max_min = {"max": float(np.max(stacked)), "min": float(np.min(stacked))}

    # Plot the selected waveforms
    for waveform in selected_names:
        if waveform not in waveform_data:
            print(f"Warning: requested waveform '{waveform}' not found. Skipping.")
            continue
        if waveform == "sine1":
            legend_name = f"{waveform} - {config['freq1']} Hz"
        elif waveform == "sine2":
            legend_name = f"{waveform} - {config['freq2']} Hz"
        elif waveform == "square":
            duty = config.get("square_duty", 0.5)
            legend_name = f"{waveform} - {config['freq1']} Hz (duty={duty})"
        elif waveform == "saw":
            width = config.get("saw_width", 1.0)
            legend_name = f"{waveform} - {config['freq1']} Hz (width={width})"
        else:
            legend_name = waveform

        fig_waveforms.add_trace(
            go.Scatter(x=t, y=waveform_data[waveform], mode="lines", name=legend_name)
        )

    # Plot the window function if enabled
    if config["apply_window"]:
        window = get_window(config["window_type"], len(t))
        fig_waveforms.add_trace(
            go.Scatter(
                x=t,
                y=window,
                mode="lines",
                name="window - " + config["window_type"],
                line=dict(dash="dash"),
            )
        )

    # Add vertical dotted lines to waveform plot
    if config["add_dotted_lines"]:
        range_25_percent = 0.25 * (waveform_max_min["max"] - waveform_max_min["min"])
        dt = float(config["dt"]) if "dt" in config else float(config["n"])
        num_lines_waveform = int(config["t_max"] / dt) + 1
        for x in np.linspace(0, config["t_max"], num_lines_waveform):
            fig_waveforms.add_shape(
                type="line",
                x0=x,
                y0=waveform_max_min["min"] - range_25_percent,
                x1=x,
                y1=waveform_max_min["max"] + range_25_percent,
                line=dict(color="Black", width=1, dash="dot"),
            )

    fig_waveforms.update_layout(title='Sine Waves', xaxis_title='Time (s)', yaxis_title='Amplitude', autosize=False, width=800, height=600, showlegend=True)
    show_plotly(fig_waveforms)

    # Create the DFT Magnitude and Phase Plots
    fig_dft_magnitude = go.Figure()
    fig_dft_phase = go.Figure()

    max_dft_amp = 0
    min_dft_phase = np.inf

    for waveform in selected_names:
        if waveform not in waveform_data:
            continue
        if waveform == "sine1":
            legend_name = f"{waveform} - {config['freq1']} Hz"
        elif waveform == "sine2":
            legend_name = f"{waveform} - {config['freq2']} Hz"
        else:
            legend_name = waveform

        y = waveform_data[waveform]

        Y = np.fft.fft(y)
        dt = float(config["dt"]) if "dt" in config else float(config["n"])
        freqs = np.fft.fftfreq(len(t), d=dt)
        inds = np.argsort(freqs)
        freqs = freqs[inds]
        Y = Y[inds]

        max_dft_amp = max(max_dft_amp, np.max(np.abs(Y)))
        min_dft_phase = min(min_dft_phase, np.min(np.angle(Y)))

        fig_dft_magnitude.add_trace(go.Scatter(x=freqs, y=np.abs(Y), mode='lines', name=legend_name))
        fig_dft_phase.add_trace(go.Scatter(x=freqs, y=np.angle(Y), mode='lines', name=legend_name))

    # Add vertical dotted lines to DFT plots
    if config["add_dotted_lines"]:
        range_25_percent_magnitude = 0.25 * max_dft_amp
        range_25_percent_phase = 0.25 * (np.pi - min_dft_phase)
        dt = float(config["dt"]) if "dt" in config else float(config["n"])
        num_lines_dft = int(config["t_max"] / dt)
        for x in np.linspace(np.min(freqs), np.max(freqs), num_lines_dft):
            fig_dft_magnitude.add_shape(type="line", x0=x, y0=0, x1=x, y1=max_dft_amp + range_25_percent_magnitude, line=dict(color="Black", width=1, dash="dot"))
            fig_dft_phase.add_shape(type="line", x0=x, y0=min_dft_phase - range_25_percent_phase, x1=x, y1=np.pi + range_25_percent_phase, line=dict(color="Black", width=1, dash="dot"))

    # Update layout for DFT Magnitude
    fig_dft_magnitude.update_layout(title='Magnitude Spectrum of the DFT', xaxis_title='Frequency (Hz)', yaxis_title='Magnitude', autosize=False, width=800, height=600, showlegend=True)

    # Update layout for DFT Phase
    fig_dft_phase.update_layout(title='Phase Spectrum of the DFT', xaxis_title='Frequency (Hz)', yaxis_title='Phase (radians)', autosize=False, width=800, height=600, showlegend=True)

    # Calculate and print sampling rate and Nyquist frequency
    dt = float(config["dt"]) if "dt" in config else float(config["n"])
    sampling_rate = 1 / dt
    nyquist_freq = sampling_rate / 2
    freq_resolution = sampling_rate / len(t)
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Nyquist frequency: {nyquist_freq} Hz")
    print(f"Total number of available sample points: {len(t)}")   
    print(f"Frequency Resolution: {freq_resolution} Hz")

    show_plotly(fig_dft_magnitude)
    show_plotly(fig_dft_phase)
