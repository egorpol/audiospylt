# multiplotter.py

import pandas as pd
import plotly.graph_objects as go
import random
import numpy as np
from typing import List, Optional, Tuple, Any, Iterator, Dict, Sequence

try:  # pragma: no cover
    from .plotting import show_plotly
except Exception:  # pragma: no cover
    from audiospylt.plotting import show_plotly

# Reuse the same axis-scaling utilities used by `analyze_signal()` in `py_scripts/dft_analysis.py`.
# We import them defensively so `multiplotter` can still be used even if optional deps for
# `dft_analysis` are not available in some environments.
try:  # pragma: no cover
    from .dft_analysis import _hz_to_mel, _mixed_warp, _mixed_warp_values, _default_tick_freqs_hz, _format_hz
except Exception:  # pragma: no cover
    def _hz_to_mel(hz):
        hz = np.asarray(hz, dtype=float)
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mixed_warp(freqs_hz, mix, fmax_hz, log_floor_hz):
        mix = float(mix)
        if not (0.0 <= mix <= 1.0):
            raise ValueError(f"freq_axis_mix must be in [0,1]; got {mix}")
        log_floor_hz = float(log_floor_hz)
        if log_floor_hz <= 0:
            raise ValueError(f"mixed_log_floor_hz must be > 0; got {log_floor_hz}")
        freqs_hz = np.asarray(freqs_hz, dtype=float)
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

# Predefined set of well-separated colors
PREDEFINED_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    # Added a few more common ones
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]

def _color_palette_generator() -> Iterator[str]:
    """
    Yields colors from a predefined list, then generates random unique RGB colors.
    """
    used_colors = set()
    for color in PREDEFINED_COLORS:
        if color not in used_colors:
            used_colors.add(color)
            yield color
    
    # Fallback to random colors if predefined are exhausted
    while True:
        color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
        if color not in used_colors:
            used_colors.add(color)
            yield color

def _load_and_prepare_data(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None
) -> List[Tuple[pd.DataFrame, str]]:
    """
    Loads data from files and combines with provided DataFrames.
    Generates labels if necessary.
    """
    all_data: List[Tuple[pd.DataFrame, str]] = []

    if files:
        for i, file_path in enumerate(files):
            try:
                df = pd.read_csv(file_path, delimiter='\t')
                # Use filename as label if no specific labels for files are handled elsewhere
                label = file_path # Or some derivation like os.path.basename(file_path)
                all_data.append((df, label))
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty: {file_path}")
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")


    if dfs:
        # Generate default labels for DataFrames if not provided or insufficient
        num_dfs = len(dfs)
        current_df_labels: List[str] = []
        if df_labels and len(df_labels) >= num_dfs :
            current_df_labels = df_labels[:num_dfs]
        elif df_labels: # some labels provided, but not enough
             current_df_labels = df_labels + [f"DataFrame {i}" for i in range(len(df_labels), num_dfs)]
        else: # no labels provided
            current_df_labels = [f"DataFrame {i}" for i in range(num_dfs)]

        for df, label in zip(dfs, current_df_labels):
            all_data.append((df, str(label))) # Ensure label is string

    return all_data


def plot_combined(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None
) -> None:
    """
    Plots data from multiple sources (files and/or DataFrames) on the same 2D canvas.
    Each row in a DataFrame is treated as a separate line segment.

    Parameters:
    - files (list, optional): List of file paths (TSV) to be read and plotted.
    - dfs (list, optional): List of DataFrames to be plotted.
    - df_labels (list, optional): List of labels corresponding to each DataFrame in `dfs`.
                                 If not provided, DataFrames will be labeled "DataFrame 0", "DataFrame 1", etc.
                                 File paths are used as labels for file-based sources.
    """
    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot.")
        return

    fig = go.Figure()
    color_gen = _color_palette_generator()

    for df, label in all_data:
        if df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping.")
            continue
        
        # Check for required columns
        required_cols = ['time_start', 'time_stop', 'freq_start', 'freq_stop']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing one or more required columns: {required_cols}. Skipping.")
            continue

        current_color = next(color_gen)
        for index, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['time_start'], row['time_stop']],
                y=[row['freq_start'], row['freq_stop']],
                mode='lines+markers',
                line=dict(width=2, color=current_color),
                marker=dict(size=5),
                name=str(label),
                legendgroup=str(label),
                showlegend=(index == 0)  # Show legend only for the first trace of this source
            ))

    fig.update_layout(
        title_text='Time vs Frequency for Each Event',
        xaxis_title_text='Time (s)',
        yaxis_title_text='Frequency (Hz)',
        template='plotly_white',
        width=900,
        height=700,
        legend_title_text='Sources'
    )
    show_plotly(fig)


def plot_combined_3d(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None,
    axis_order: Sequence[str] = ("time", "amp", "freq"),
    flip: Optional[Dict[str, bool]] = None,
    axis_pad_frac: float = 0.05,
    zero_based_axes: bool = True,
) -> None:
    """
    Plots data from multiple sources (files and/or DataFrames) on a 3D canvas.
    Each row in a DataFrame is treated as a separate line segment in 3D.

    Parameters:
    - files (list, optional): List of file paths (TSV) to be read and plotted.
    - dfs (list, optional): List of DataFrames to be plotted.
    - df_labels (list, optional): List of labels corresponding to each DataFrame in `dfs`.
                                 If not provided, DataFrames will be labeled "DataFrame 0", "DataFrame 1", etc.
                                 File paths are used as labels for file-based sources.
    - axis_order (Sequence[str]): Which dimensions to place on (x, y, z).
                                 Must be a permutation of ('time', 'amp', 'freq').
                                 Examples:
                                   ('time','amp','freq')  # default: x=time, y=amp, z=freq
                                   ('freq','time','amp')  # x=freq, y=time, z=amp
    - flip (dict[str,bool], optional): Flip/reverse axes. Keys may be dimension names
                                       ('time'|'amp'|'freq') and/or axis names ('x'|'y'|'z').
                                       Axis keys override dimension keys.
                                       Example: {'freq': True} or {'x': True}
    - axis_pad_frac (float): Fractional padding applied to axis ranges (when ranges are set).
    - zero_based_axes (bool): If True, range mins are clamped to <= 0 (useful for time/amp/freq plots).
    """
    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot for 3D.")
        return

    def _canon_dim(name: str) -> str:
        s = str(name).strip().lower()
        if s in ("t", "time", "times", "seconds", "sec", "s"):
            return "time"
        if s in ("a", "amp", "amplitude"):
            return "amp"
        if s in ("f", "freq", "frequency", "hz"):
            return "freq"
        raise ValueError(f"Unknown dimension '{name}'. Use time|amp|freq.")

    dims = tuple(_canon_dim(d) for d in axis_order)
    if len(dims) != 3:
        raise ValueError(f"axis_order must have 3 items (x,y,z). Got: {axis_order!r}")
    if set(dims) != {"time", "amp", "freq"}:
        raise ValueError(f"axis_order must be a permutation of ('time','amp','freq'). Got: {axis_order!r}")

    flip = flip or {}
    # Axis keys override dimension keys.
    flip_by_dim: Dict[str, bool] = {}
    flip_by_axis: Dict[str, bool] = {}
    for k, v in flip.items():
        ks = str(k).strip().lower()
        if ks in ("x", "y", "z"):
            flip_by_axis[ks] = bool(v)
        else:
            flip_by_dim[_canon_dim(ks)] = bool(v)

    dim_specs: Dict[str, Dict[str, str]] = {
        "time": {"start": "time_start", "stop": "time_stop", "title": "Time (s)"},
        "amp": {"start": "amp_min", "stop": "amp_max", "title": "Amplitude"},
        "freq": {"start": "freq_start", "stop": "freq_stop", "title": "Frequency (Hz)"},
    }

    x_dim, y_dim, z_dim = dims
    x_start, x_stop = dim_specs[x_dim]["start"], dim_specs[x_dim]["stop"]
    y_start, y_stop = dim_specs[y_dim]["start"], dim_specs[y_dim]["stop"]
    z_start, z_stop = dim_specs[z_dim]["start"], dim_specs[z_dim]["stop"]
    x_title, y_title, z_title = dim_specs[x_dim]["title"], dim_specs[y_dim]["title"], dim_specs[z_dim]["title"]

    flip_x = flip_by_axis.get("x", flip_by_dim.get(x_dim, False))
    flip_y = flip_by_axis.get("y", flip_by_dim.get(y_dim, False))
    flip_z = flip_by_axis.get("z", flip_by_dim.get(z_dim, False))

    fig_3d = go.Figure()
    color_gen = _color_palette_generator()

    # For axis range calculation
    dim_min: Dict[str, float] = {"time": np.inf, "amp": np.inf, "freq": np.inf}
    dim_max: Dict[str, float] = {"time": -np.inf, "amp": -np.inf, "freq": -np.inf}
    has_data_for_axes = False

    for df, label in all_data:
        if df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping for 3D plot.")
            continue
        
        required_cols = ['time_start', 'time_stop', 'amp_min', 'amp_max', 'freq_start', 'freq_stop']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing one or more required columns for 3D plot: {required_cols}. Skipping.")
            continue

        current_color = next(color_gen)
        for index, row in df.iterrows():
            fig_3d.add_trace(go.Scatter3d(
                x=[row[x_start], row[x_stop]],
                y=[row[y_start], row[y_stop]],
                z=[row[z_start], row[z_stop]],
                mode='lines+markers',
                line=dict(width=2, color=current_color),
                marker=dict(size=3, color=current_color),
                name=str(label),
                legendgroup=str(label),
                showlegend=(index == 0)
            ))
        
        # Update min/max values for axis ranges (across all dimensions, regardless of axis placement).
        try:
            has_data_for_axes = True
            for dim, spec in dim_specs.items():
                v0 = df[spec["start"]].to_numpy(dtype=float)
                v1 = df[spec["stop"]].to_numpy(dtype=float)
                v = np.concatenate([v0, v1])
                if v.size == 0:
                    continue
                dim_min[dim] = min(dim_min[dim], float(np.nanmin(v)))
                dim_max[dim] = max(dim_max[dim], float(np.nanmax(v)))
        except Exception:
            # If a source has non-numeric data, we still show traces but skip range computation for it.
            pass

    def _range_for_dim(dim: str, do_flip: bool) -> Optional[List[float]]:
        vmin = float(dim_min[dim])
        vmax = float(dim_max[dim])
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if zero_based_axes:
            vmin = min(vmin, 0.0)
        span = vmax - vmin
        pad = float(axis_pad_frac) * (span if span > 0 else max(1.0, abs(vmax), abs(vmin)))
        lo = vmin - pad
        hi = vmax + pad
        return [hi, lo] if do_flip else [lo, hi]

    x_range = _range_for_dim(x_dim, flip_x) if has_data_for_axes else None
    y_range = _range_for_dim(y_dim, flip_y) if has_data_for_axes else None
    z_range = _range_for_dim(z_dim, flip_z) if has_data_for_axes else None

    scene_dict: Dict[str, Any] = dict(
        xaxis=dict(title=x_title, range=x_range) if x_range is not None else dict(title=x_title),
        yaxis=dict(title=y_title, range=y_range) if y_range is not None else dict(title=y_title),
        zaxis=dict(title=z_title, range=z_range) if z_range is not None else dict(title=z_title),
        camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=0.5))
    )


    fig_3d.update_layout(
        title_text=f'{x_title} vs {y_title} vs {z_title} (3D)',
        scene=scene_dict,
        template='plotly_white',
        showlegend=True,
        width=900,
        height=700,
        legend_title_text='Sources'
    )
    show_plotly(fig_3d)


def plot_scatter(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None,
    mode: str = 'markers+lines',
    freq_axis_mode: str = "linear",
    freq_axis_mix: float = 0.5,
    mixed_log_floor_hz: float = 1.0,
    amp_axis_mode: str = "linear",
    amp_axis_mix: float = 0.5,
    amp_log_floor: float = 1e-12,
    auto_plot_range: bool = True,
    freq_plot_pad_hz: Optional[float] = None,
    freq_plot_pad_frac: float = 0.05,
    amp_plot_pad: Optional[float] = None,
    amp_plot_pad_frac: float = 0.10,
    amp_plot_pad_ratio: float = 0.15,
) -> None:
    """
    Plots data from multiple sources (files and/or DataFrames) on a 2D scatter plot.
    Each DataFrame/file contributes one trace to the scatter plot.

    Parameters:
    - files (list, optional): List of file paths (TSV) to be read and plotted.
                               Expected columns: 'Frequency (Hz)', 'Amplitude'.
    - dfs (list, optional): List of DataFrames to be plotted.
                            Expected columns: 'Frequency (Hz)', 'Amplitude'.
    - df_labels (list, optional): List of labels corresponding to each DataFrame in `dfs`.
                                 If not provided, "DataFrame 0", "DataFrame 1", etc.
                                 File paths are used as labels for file-based sources.
    - mode (str, optional): Plotly mode ('lines', 'markers', 'lines+markers'). Default 'markers+lines'.
    - freq_axis_mode (str): Frequency axis scaling, matching `analyze_signal()`:
                            'linear' | 'log' | 'mel' | 'mixed'
    - freq_axis_mix (float): For 'mixed' mode, blend factor in [0, 1] (0=linear, 1=log-like).
    - mixed_log_floor_hz (float): For 'mixed' mode, log floor in Hz (>0).
    - amp_axis_mode (str): Amplitude axis scaling, matching `analyze_signal()`:
                           'linear' | 'log' | 'mixed'
    - amp_axis_mix (float): For 'mixed' mode, blend factor in [0, 1] (0=linear, 1=log-like).
    - amp_log_floor (float): Floor for log/mixed amplitude handling (>0).
    - auto_plot_range (bool): If True, auto-zoom to data with padding (safer than forcing [0..max]).
    - freq_plot_pad_hz/freq_plot_pad_frac: Absolute/relative padding for x-range in Hz.
    - amp_plot_pad/amp_plot_pad_frac/amp_plot_pad_ratio: Padding for y-range (ratio used in log mode).
    """
    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot for scatter.")
        return

    scatter_fig = go.Figure()
    color_gen = _color_palette_generator()

    # Track global ranges for sensible auto-zoom.
    x_min_raw, x_max_raw = np.inf, -np.inf
    y_min_raw, y_max_raw = np.inf, -np.inf
    has_data_for_axes = False

    # Determine tick mapping for non-linear frequency axis modes.
    # We generate ticks against the max frequency across all sources.
    # Note: for 'log' we rely on plotly's native log axis formatting.
    global_fmax_hz = 0.0
    for df, _label in all_data:
        if df is None or df.empty or 'Frequency (Hz)' not in df.columns:
            continue
        try:
            global_fmax_hz = max(global_fmax_hz, float(np.nanmax(df['Frequency (Hz)'].to_numpy(dtype=float))))
        except Exception:
            continue

    x_axis_type = "linear"
    x_title = "Frequency (Hz)"
    x_tickvals = None
    x_ticktext = None
    if freq_axis_mode == "log":
        x_axis_type = "log"
    elif freq_axis_mode == "mel":
        x_title = "Frequency (mel)"
        tick_freqs = _default_tick_freqs_hz(global_fmax_hz)
        x_tickvals = _hz_to_mel(tick_freqs)
        x_ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode == "mixed":
        x_title = "Frequency"
        tick_freqs = _default_tick_freqs_hz(global_fmax_hz)
        x_tickvals = _mixed_warp(tick_freqs, freq_axis_mix, fmax_hz=global_fmax_hz, log_floor_hz=mixed_log_floor_hz)
        x_ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode != "linear":
        raise ValueError(f"freq_axis_mode must be linear|log|mel|mixed; got {freq_axis_mode}")

    y_axis_type = "linear"
    if amp_axis_mode == "log":
        y_axis_type = "log"
        if float(amp_log_floor) <= 0:
            raise ValueError(f"amp_log_floor must be > 0; got {amp_log_floor}")
    elif amp_axis_mode == "mixed":
        y_axis_type = "linear"
        if float(amp_log_floor) <= 0:
            raise ValueError(f"amp_log_floor must be > 0; got {amp_log_floor}")
    elif amp_axis_mode != "linear":
        raise ValueError(f"amp_axis_mode must be linear|log|mixed; got {amp_axis_mode}")

    for df, label in all_data:
        if df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping for scatter plot.")
            continue

        required_cols = ['Frequency (Hz)', 'Amplitude']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing one or more required columns for scatter: {required_cols}. Skipping.")
            continue

        # Build x/y arrays (and apply optional axis scaling in the same spirit as `analyze_signal()`).
        x_hz = df['Frequency (Hz)'].to_numpy(dtype=float)
        y_amp = df['Amplitude'].to_numpy(dtype=float)

        # Frequency axis mapping.
        if freq_axis_mode == "mel":
            x_vals = _hz_to_mel(x_hz)
        elif freq_axis_mode == "mixed":
            fmax = float(np.nanmax(x_hz)) if len(x_hz) else 0.0
            x_vals = _mixed_warp(x_hz, freq_axis_mix, fmax_hz=max(1e-12, fmax), log_floor_hz=mixed_log_floor_hz)
        else:
            x_vals = x_hz

        # Amplitude axis mapping.
        if amp_axis_mode == "log":
            floor = float(amp_log_floor)
            y_vals = np.maximum(y_amp, floor)
        elif amp_axis_mode == "mixed":
            floor = float(amp_log_floor)
            vmax = float(np.nanmax(y_amp)) if len(y_amp) else 0.0
            y_vals = _mixed_warp_values(
                np.maximum(y_amp, 0.0),
                mix=amp_axis_mix,
                vmax=max(1e-12, vmax),
                log_floor=floor,
                mix_param_name="amp_axis_mix",
                floor_param_name="amp_log_floor",
            )
        else:
            y_vals = y_amp

        current_color = next(color_gen)
        scatter_fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode=mode,
            name=str(label),
            line=dict(color=current_color), # For lines mode
            marker=dict(color=current_color) # For markers mode
        ))
        
        if len(x_vals) and len(y_vals):
            has_data_for_axes = True
            x_min_raw = min(x_min_raw, float(np.nanmin(x_vals)))
            x_max_raw = max(x_max_raw, float(np.nanmax(x_vals)))
            y_min_raw = min(y_min_raw, float(np.nanmin(y_vals)))
            y_max_raw = max(y_max_raw, float(np.nanmax(y_vals)))

    scatter_fig.update_layout(
        title_text="Scatter Plot of Frequency vs. Amplitude",
        xaxis_title_text=x_title,
        yaxis_title_text="Amplitude",
        template='plotly_white',
        width=900,
        height=700,
        legend_title_text='Sources'
    )

    scatter_fig.update_xaxes(type=x_axis_type)
    if x_tickvals is not None and x_ticktext is not None:
        scatter_fig.update_xaxes(tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext)
    scatter_fig.update_yaxes(type=y_axis_type)

    if has_data_for_axes and auto_plot_range:
        # X range padding.
        x0, x1 = float(x_min_raw), float(x_max_raw)
        if x_axis_type == "log":
            # Plotly expects log10 values for axis range when type='log'.
            # Our x values are raw Hz for log axis.
            # For mel/mixed we don't use log axis type.
            positive_vals = []
            for tr in scatter_fig.data:
                arr = np.asarray(tr.x, dtype=float)
                positive_vals.append(arr[arr > 0])
            positive_vals = np.concatenate(positive_vals) if len(positive_vals) else np.array([])
            min_pos = float(np.min(positive_vals)) if len(positive_vals) else 1e-6
            # Avoid invalid/degenerate ranges.
            x0_use = max(min_pos, x0 if np.isfinite(x0) else min_pos)
            x1_use = max(x0_use * (1.0 + 1e-6), x1 if np.isfinite(x1) else x0_use * 10.0)
            # Apply padding in Hz-space.
            span = max(1e-12, x1_use - x0_use)
            pad = float(freq_plot_pad_hz) if freq_plot_pad_hz is not None else float(freq_plot_pad_frac) * span
            x0_use = max(min_pos, x0_use - pad)
            x1_use = max(x0_use * (1.0 + 1e-6), x1_use + pad)
            scatter_fig.update_xaxes(range=[np.log10(x0_use), np.log10(x1_use)])
        else:
            span = max(1e-12, x1 - x0)
            pad = float(freq_plot_pad_hz) if freq_plot_pad_hz is not None else float(freq_plot_pad_frac) * span
            scatter_fig.update_xaxes(range=[x0 - pad, x1 + pad])

        # Y range padding.
        y0, y1 = float(y_min_raw), float(y_max_raw)
        if y_axis_type == "log":
            floor = float(amp_log_floor)
            y0_use = max(floor, y0 if np.isfinite(y0) else floor)
            y1_use = max(y0_use * (1.0 + 1e-9), y1 if np.isfinite(y1) else y0_use * 10.0)
            r = float(amp_plot_pad_ratio)
            if r < 0:
                raise ValueError(f"amp_plot_pad_ratio must be >= 0; got {amp_plot_pad_ratio}")
            y0_use = max(floor, y0_use / (1.0 + r))
            y1_use = y1_use * (1.0 + r)
            scatter_fig.update_yaxes(range=[np.log10(y0_use), np.log10(y1_use)])
        else:
            span = max(1e-12, y1 - y0)
            pad = float(amp_plot_pad) if amp_plot_pad is not None else float(amp_plot_pad_frac) * span
            scatter_fig.update_yaxes(range=[y0 - pad, y1 + pad])

    show_plotly(scatter_fig)


def plot_scatter_binned(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None,
    x_bin_size: float = 50.0,
    y_bin_size: float = 0.01,
    freq_axis_mode: str = "linear",
    freq_axis_mix: float = 0.5,
    mixed_log_floor_hz: float = 1.0,
    amp_axis_mode: str = "linear",
    amp_axis_mix: float = 0.5,
    amp_log_floor: float = 1e-12,
    title: str = "Binned Scatter (2D histogram) of Frequency vs. Amplitude",
) -> None:
    """
    "Bin-like" alternative to `plot_scatter`: aggregates points into 2D bins and visualizes density.

    Notes:
    - Bin sizes are defined in the *plot coordinate space*. For non-linear axis modes this means:
      - freq_axis_mode='mel'/'mixed': x bins are in mel/warped units
      - freq_axis_mode='log': x bins are in Hz (plotly log axis shows them log-scaled visually)
      - amp_axis_mode='mixed': y bins are in warped units
      - amp_axis_mode='log': y bins are in linear amplitude units (displayed log-scaled)
    """
    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot for binned scatter.")
        return

    # Determine tick mapping for non-linear frequency axis modes (same as `plot_scatter`).
    global_fmax_hz = 0.0
    for df, _label in all_data:
        if df is None or df.empty or 'Frequency (Hz)' not in df.columns:
            continue
        try:
            global_fmax_hz = max(global_fmax_hz, float(np.nanmax(df['Frequency (Hz)'].to_numpy(dtype=float))))
        except Exception:
            continue

    x_axis_type = "linear"
    x_title = "Frequency (Hz)"
    x_tickvals = None
    x_ticktext = None
    if freq_axis_mode == "log":
        x_axis_type = "log"
    elif freq_axis_mode == "mel":
        x_title = "Frequency (mel)"
        tick_freqs = _default_tick_freqs_hz(global_fmax_hz)
        x_tickvals = _hz_to_mel(tick_freqs)
        x_ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode == "mixed":
        x_title = "Frequency"
        tick_freqs = _default_tick_freqs_hz(global_fmax_hz)
        x_tickvals = _mixed_warp(tick_freqs, freq_axis_mix, fmax_hz=global_fmax_hz, log_floor_hz=mixed_log_floor_hz)
        x_ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode != "linear":
        raise ValueError(f"freq_axis_mode must be linear|log|mel|mixed; got {freq_axis_mode}")

    y_axis_type = "linear"
    if amp_axis_mode == "log":
        y_axis_type = "log"
        if float(amp_log_floor) <= 0:
            raise ValueError(f"amp_log_floor must be > 0; got {amp_log_floor}")
    elif amp_axis_mode == "mixed":
        y_axis_type = "linear"
        if float(amp_log_floor) <= 0:
            raise ValueError(f"amp_log_floor must be > 0; got {amp_log_floor}")
    elif amp_axis_mode != "linear":
        raise ValueError(f"amp_axis_mode must be linear|log|mixed; got {amp_axis_mode}")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for df, label in all_data:
        if df is None or df.empty:
            continue
        required_cols = ['Frequency (Hz)', 'Amplitude']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing one or more required columns for binned scatter: {required_cols}. Skipping.")
            continue

        x_hz = df['Frequency (Hz)'].to_numpy(dtype=float)
        y_amp = df['Amplitude'].to_numpy(dtype=float)

        if freq_axis_mode == "mel":
            x_vals = _hz_to_mel(x_hz)
        elif freq_axis_mode == "mixed":
            fmax = float(np.nanmax(x_hz)) if len(x_hz) else 0.0
            x_vals = _mixed_warp(x_hz, freq_axis_mix, fmax_hz=max(1e-12, fmax), log_floor_hz=mixed_log_floor_hz)
        else:
            x_vals = x_hz

        if amp_axis_mode == "log":
            floor = float(amp_log_floor)
            y_vals = np.maximum(y_amp, floor)
        elif amp_axis_mode == "mixed":
            floor = float(amp_log_floor)
            vmax = float(np.nanmax(y_amp)) if len(y_amp) else 0.0
            y_vals = _mixed_warp_values(
                np.maximum(y_amp, 0.0),
                mix=amp_axis_mix,
                vmax=max(1e-12, vmax),
                log_floor=floor,
                mix_param_name="amp_axis_mix",
                floor_param_name="amp_log_floor",
            )
        else:
            y_vals = y_amp

        xs.append(np.asarray(x_vals, dtype=float))
        ys.append(np.asarray(y_vals, dtype=float))

    if not xs or not ys:
        print("No valid data points to bin.")
        return

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=x_all,
        y=y_all,
        xbins=dict(size=float(x_bin_size)),
        ybins=dict(size=float(y_bin_size)),
        colorscale="Viridis",
        colorbar=dict(title="Count"),
    ))

    fig.update_layout(
        title_text=title,
        xaxis_title_text=x_title,
        yaxis_title_text="Amplitude",
        template="plotly_white",
        width=900,
        height=700,
    )
    fig.update_xaxes(type=x_axis_type)
    if x_tickvals is not None and x_ticktext is not None:
        fig.update_xaxes(tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext)
    fig.update_yaxes(type=y_axis_type)

    show_plotly(fig)


def plot_equalizer_bars(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None,
    x_bin_size: float = 50.0,
    agg: str = "max",
    mode: str = "group",
    opacity: float = 0.85,
    freq_axis_mode: str = "linear",
    freq_axis_mix: float = 0.5,
    mixed_log_floor_hz: float = 1.0,
    amp_axis_mode: str = "linear",
    amp_axis_mix: float = 0.5,
    amp_log_floor: float = 1e-12,
    title: str = "Equalizer-style bars (binned spectrum)",
) -> None:
    """
    Equalizer-like bar visualization for peak-list TSVs (Frequency/Amplitude).

    - Each TSV/DataFrame becomes its own bar trace (unique color per source).
    - Frequencies are binned into fixed-width bins (Hz) and amplitudes are aggregated per bin.
    - Background uses Plotly's white template.

    Parameters:
    - files/dfs/df_labels: same conventions as `plot_scatter`
    - x_bin_size: bin size in Hz (must be > 0)
    - agg: aggregation per bin: 'max' | 'sum' | 'mean'
    - mode: Plotly bar mode: 'group' | 'overlay' | 'stack'
    - opacity: bar opacity in [0,1] (useful in overlay mode)
    - freq_axis_mode (str): 'linear' | 'log' | 'mel' | 'mixed' (mirrors `analyze_signal()`)
    - freq_axis_mix (float): used only for freq_axis_mode='mixed' (0..1)
    - mixed_log_floor_hz (float): used only for freq_axis_mode='mixed' (>0)
    - amp_axis_mode (str): 'linear' | 'log' | 'mixed' (mirrors `analyze_signal()`)
    - amp_axis_mix (float): used only for amp_axis_mode='mixed' (0..1)
    - amp_log_floor (float): used for amp_axis_mode='log'/'mixed' (>0)
    """
    if x_bin_size is None or float(x_bin_size) <= 0:
        raise ValueError(f"x_bin_size must be > 0; got {x_bin_size}")
    if agg not in {"max", "sum", "mean"}:
        raise ValueError(f"agg must be one of: max|sum|mean; got {agg}")
    if mode not in {"group", "overlay", "stack"}:
        raise ValueError(f"mode must be one of: group|overlay|stack; got {mode}")
    if not (0.0 <= float(opacity) <= 1.0):
        raise ValueError(f"opacity must be in [0,1]; got {opacity}")

    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot for equalizer bars.")
        return

    # Global bin edges across all sources for consistent alignment.
    fmins: List[float] = []
    fmaxs: List[float] = []
    fposmins: List[float] = []
    for df, _label in all_data:
        if df is None or df.empty or 'Frequency (Hz)' not in df.columns:
            continue
        x = pd.to_numeric(df['Frequency (Hz)'], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x):
            fmins.append(float(np.min(x)))
            fmaxs.append(float(np.max(x)))
            xp = x[x > 0]
            if len(xp):
                fposmins.append(float(np.min(xp)))

    if not fmins or not fmaxs:
        print("No valid frequency values to bin.")
        return

    fmin = min(fmins)
    fmax = max(fmaxs)
    # Make edges start at 0 for a more "equalizer" feel unless data is negative.
    start = 0.0 if fmin >= 0 else fmin
    if freq_axis_mode == "log" and start <= 0:
        # Log axis cannot display non-positive frequencies.
        if not fposmins:
            raise ValueError("freq_axis_mode='log' requires positive frequencies.")
        # Keep the user-friendly 0-based bins in linear modes, but for log we must start > 0.
        start = min(fposmins)
    bin_size = float(x_bin_size)
    # Ensure at least 1 bin.
    edges = np.arange(start, fmax + bin_size, bin_size, dtype=float)
    if len(edges) < 2:
        edges = np.array([start, start + bin_size], dtype=float)

    centers = (edges[:-1] + edges[1:]) / 2.0
    # Plotly accepts per-bar widths in x-units; keep slightly narrower than bin to show separation.
    width = bin_size * 0.92

    fig = go.Figure()
    color_gen = _color_palette_generator()

    # X-axis scaling / ticks (reuse same conventions as `plot_scatter`/`analyze_signal`).
    x_axis_type = "linear"
    x_title = "Frequency (Hz)"
    x_tickvals = None
    x_ticktext = None
    if freq_axis_mode == "log":
        x_axis_type = "log"
    elif freq_axis_mode == "mel":
        x_title = "Frequency (mel)"
        tick_freqs = _default_tick_freqs_hz(fmax)
        x_tickvals = _hz_to_mel(tick_freqs)
        x_ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode == "mixed":
        x_title = "Frequency"
        tick_freqs = _default_tick_freqs_hz(fmax)
        x_tickvals = _mixed_warp(tick_freqs, freq_axis_mix, fmax_hz=fmax, log_floor_hz=mixed_log_floor_hz)
        x_ticktext = [_format_hz(v) for v in tick_freqs]
    elif freq_axis_mode != "linear":
        raise ValueError(f"freq_axis_mode must be linear|log|mel|mixed; got {freq_axis_mode}")

    # Y-axis scaling (plot only).
    y_axis_type = "linear"
    if amp_axis_mode == "log":
        y_axis_type = "log"
        if float(amp_log_floor) <= 0:
            raise ValueError(f"amp_log_floor must be > 0; got {amp_log_floor}")
    elif amp_axis_mode == "mixed":
        y_axis_type = "linear"
        if float(amp_log_floor) <= 0:
            raise ValueError(f"amp_log_floor must be > 0; got {amp_log_floor}")
    elif amp_axis_mode != "linear":
        raise ValueError(f"amp_axis_mode must be linear|log|mixed; got {amp_axis_mode}")

    for df, label in all_data:
        if df is None or df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping for equalizer bars.")
            continue
        required_cols = ['Frequency (Hz)', 'Amplitude']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing required columns {required_cols}. Skipping.")
            continue

        x = pd.to_numeric(df['Frequency (Hz)'], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df['Amplitude'], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if not len(x):
            print(f"Warning: No finite points for '{label}'. Skipping.")
            continue

        # Assign each point to a bin index in [0, nbins-1].
        bin_idx = np.digitize(x, edges, right=False) - 1
        nbins = len(edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < nbins)
        bin_idx = bin_idx[valid]
        y = y[valid]
        if not len(bin_idx):
            print(f"Warning: All points for '{label}' fell outside bins. Skipping.")
            continue

        # Aggregate per bin.
        vals = np.zeros(nbins, dtype=float)
        if agg == "max":
            vals[:] = 0.0
            for i in range(nbins):
                yi = y[bin_idx == i]
                vals[i] = float(np.max(yi)) if len(yi) else 0.0
        elif agg == "sum":
            for i in range(nbins):
                yi = y[bin_idx == i]
                vals[i] = float(np.sum(yi)) if len(yi) else 0.0
        else:  # mean
            for i in range(nbins):
                yi = y[bin_idx == i]
                vals[i] = float(np.mean(yi)) if len(yi) else 0.0

        # Apply amplitude scaling (plot-only) to the aggregated values.
        vals_for_plot = vals
        if amp_axis_mode == "log":
            floor = float(amp_log_floor)
            vals_for_plot = np.maximum(vals_for_plot, floor)
        elif amp_axis_mode == "mixed":
            floor = float(amp_log_floor)
            vmax = float(np.max(vals_for_plot)) if len(vals_for_plot) else 0.0
            vals_for_plot = _mixed_warp_values(
                np.maximum(vals_for_plot, 0.0),
                mix=amp_axis_mix,
                vmax=max(1e-12, vmax),
                log_floor=floor,
                mix_param_name="amp_axis_mix",
                floor_param_name="amp_log_floor",
            )

        # Apply frequency-axis mapping to bin centers.
        if freq_axis_mode == "mel":
            x_vals = _hz_to_mel(centers)
        elif freq_axis_mode == "mixed":
            x_vals = _mixed_warp(centers, freq_axis_mix, fmax_hz=fmax, log_floor_hz=mixed_log_floor_hz)
        else:
            x_vals = centers

        current_color = next(color_gen)
        fig.add_trace(go.Bar(
            x=x_vals,
            y=vals_for_plot,
            width=width,
            name=str(label),
            marker=dict(color=current_color),
            opacity=float(opacity),
        ))

    barmode = "group" if mode == "group" else ("overlay" if mode == "overlay" else "stack")
    fig.update_layout(
        title_text=title,
        xaxis_title_text=x_title,
        yaxis_title_text=f"Amplitude ({agg} per {x_bin_size} Hz bin)",
        template="plotly_white",
        width=900,
        height=700,
        barmode=barmode,
        legend_title_text="Sources",
    )
    fig.update_xaxes(type=x_axis_type)
    if x_tickvals is not None and x_ticktext is not None:
        fig.update_xaxes(tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext)
    fig.update_yaxes(type=y_axis_type)
    show_plotly(fig)