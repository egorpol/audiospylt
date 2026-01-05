import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, Optional, Tuple, Iterable, Sequence

try:  # pragma: no cover
    from .plotting import show_plotly
except Exception:  # pragma: no cover
    from audiospylt.plotting import show_plotly

def plot_waves(
    wave_params: pd.DataFrame,
    k: float = 0.001,
    edge_fade_s: float = 0.0,
    fade_in_s: Optional[float] = None,
    fade_out_s: Optional[float] = None,
    fade_shape: str = "cosine",
    interval_mode: str = "closed",
    phase_mode: str = "reset",
    show_waveform: bool = True,
    show_fft: bool = False,
    spectrogram_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    """
    Synthesize the combined waveform described by `wave_params` and optionally visualize it.

    This is a refactored version of the old helper that used `disable_*_plot` flags.

    Parameters
    - wave_params: DataFrame with columns:
      'time_start', 'time_stop', 'freq_start', 'freq_stop', 'amp_min', 'amp_max'
    - k: time step (seconds) for the synthesis grid (kept intact for waveform x-axis)
    - edge_fade_s: legacy convenience fade time (seconds) applied to *both* ends of each segment
                   (equivalent to fade_in_s=fade_out_s=edge_fade_s if per-row overrides are absent).
    - fade_in_s / fade_out_s: explicit fade times (seconds) for the start/end of each segment.
                              If None, defaults to edge_fade_s. Can also be provided per-row via
                              wave_params columns 'fade_in_s' / 'fade_out_s' for exact control.
    - fade_shape: fade curve: 'linear' | 'cosine' | 'sqrt_hann'
                  - 'cosine' is a raised-cosine (Hann half-window) amplitude ramp
                  - 'sqrt_hann' is useful for *constant-power* crossfades (w1^2 + w2^2 ≈ 1)
    - interval_mode: how to interpret [time_start, time_stop] for selecting samples:
        - "closed": include stop sample (legacy behavior; can double-count boundaries)
        - "half_open": exclude stop sample (often reduces boundary artifacts)
    - phase_mode: how to handle oscillator phase inside each event:
        - "reset": phase resets at each event start (simple, but most likely to click at boundaries)
        - "global": absolute-time referenced phase:
            - keeps phase continuous for constant-frequency bins across adjacent segments
            - keeps the intended instantaneous frequency for linear ramps within each event
            - does NOT guarantee phase continuity across events when the frequency changes between events
        - "chirp": integrate a linear freq ramp to get a continuous phase *within* the event
    - show_waveform: if True, show the synthesized waveform only
    - show_fft: if True, show a spectrogram using `py_scripts.audio_utils.plot_spectrogram`
    - spectrogram_kwargs: optional dict forwarded to `plot_spectrogram(...)`

    Backwards-compat:
    - Old kwargs like disable_freq_plot/disable_amp_plot/disable_combined_plot/disable_wave_plot
      are accepted but deprecated. Only disable_wave_plot is mapped (to show_waveform).

    Returns
    - y_combined: synthesized waveform samples (np.ndarray), aligned to t = arange(0, n+k, k)
    """
    # Backwards-compat for old call sites (kept strict: only known legacy keys are consumed).
    legacy_keys = {
        "disable_freq_plot",
        "disable_amp_plot",
        "disable_combined_plot",
        "disable_wave_plot",
    }
    legacy_present = {k_: kwargs.pop(k_) for k_ in list(kwargs.keys()) if k_ in legacy_keys}
    if "disable_wave_plot" in legacy_present:
        show_waveform = not bool(legacy_present["disable_wave_plot"])

    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"plot_waves() got unexpected keyword argument(s): {unknown}")

    required_cols = ["time_start", "time_stop", "freq_start", "freq_stop", "amp_min", "amp_max"]
    missing = [c for c in required_cols if c not in wave_params.columns]
    if missing:
        raise ValueError(f"wave_params is missing required columns: {missing}")

    k = float(k)
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")

    edge_fade_s = float(edge_fade_s)
    if edge_fade_s < 0:
        raise ValueError(f"edge_fade_s must be >= 0; got {edge_fade_s}")

    if fade_in_s is not None:
        fade_in_s = float(fade_in_s)
        if fade_in_s < 0:
            raise ValueError(f"fade_in_s must be >= 0; got {fade_in_s}")
    if fade_out_s is not None:
        fade_out_s = float(fade_out_s)
        if fade_out_s < 0:
            raise ValueError(f"fade_out_s must be >= 0; got {fade_out_s}")

    fade_shape = str(fade_shape).lower().strip()
    if fade_shape not in {"linear", "cosine", "sqrt_hann"}:
        raise ValueError(f"fade_shape must be one of: linear|cosine|sqrt_hann; got {fade_shape!r}")

    interval_mode = str(interval_mode).lower().strip().replace("-", "_")
    if interval_mode not in {"closed", "half_open"}:
        raise ValueError(f"interval_mode must be one of: closed|half_open; got {interval_mode!r}")

    phase_mode = str(phase_mode).lower().strip()
    if phase_mode not in {"reset", "global", "chirp"}:
        raise ValueError(f"phase_mode must be one of: reset|global|chirp; got {phase_mode!r}")

    # Find the biggest interval between all given time_start and time_stop values.
    n = float(pd.to_numeric(wave_params["time_stop"], errors="coerce").max())
    if not np.isfinite(n) or n <= 0:
        # No meaningful synthesis window.
        return np.array([], dtype=float)

    # Define the time vector (from 0 to n seconds, with a step of k seconds).
    # NOTE: This preserves the requested "k time vector resolution" exactly.
    t = np.arange(0.0, n + k, k, dtype=float)

    # Synthesize the combined wave without constructing large (t x events) matrices.
    y_combined = np.zeros_like(t)

    def _ramp(x01: np.ndarray) -> np.ndarray:
        """Map x in [0,1] to a fade curve in [0,1]."""
        if fade_shape == "linear":
            return x01
        # raised cosine (Hann half-window)
        base = 0.5 - 0.5 * np.cos(np.pi * x01)
        if fade_shape == "cosine":
            return base
        # constant-power friendly crossfade weight
        return np.sqrt(np.maximum(base, 0.0))

    def _apply_fades(amp_vec: np.ndarray, fade_in_sec: float, fade_out_sec: float) -> np.ndarray:
        """Apply in/out fades (in seconds) to a per-sample amplitude vector."""
        if amp_vec.size == 0:
            return amp_vec
        fi_n = int(round(max(0.0, float(fade_in_sec)) / k))
        fo_n = int(round(max(0.0, float(fade_out_sec)) / k))
        fi_n = max(0, min(fi_n, amp_vec.size))
        fo_n = max(0, min(fo_n, amp_vec.size))
        if fi_n == 0 and fo_n == 0:
            return amp_vec

        env = np.ones_like(amp_vec, dtype=float)
        if fi_n > 0:
            if fi_n == 1:
                env[0] *= 0.0
            else:
                x = np.linspace(0.0, 1.0, num=fi_n, endpoint=True, dtype=float)
                env[:fi_n] *= _ramp(x)
        if fo_n > 0:
            if fo_n == 1:
                env[-1] *= 0.0
            else:
                x = np.linspace(0.0, 1.0, num=fo_n, endpoint=True, dtype=float)
                env[-fo_n:] *= _ramp(x)[::-1]
        return amp_vec * env

    # Optional per-row fade overrides (seconds).
    has_fade_in_col = "fade_in_s" in wave_params.columns
    has_fade_out_col = "fade_out_s" in wave_params.columns
    iter_cols = required_cols + ([ "fade_in_s" ] if has_fade_in_col else []) + ([ "fade_out_s" ] if has_fade_out_col else [])

    # Iterate rows once; per-row computations are vectorized over the active time slice.
    for row in wave_params[iter_cols].itertuples(index=False, name=None):
        time_start, time_stop, freq_start, freq_stop, amp_min, amp_max = [float(v) for v in row[:6]]
        if not (np.isfinite(time_start) and np.isfinite(time_stop)):
            continue
        if time_stop <= time_start:
            continue

        if interval_mode == "half_open":
            idx = (t >= time_start) & (t < time_stop)
        else:  # "closed" (legacy)
            idx = (t >= time_start) & (t <= time_stop)
        if not np.any(idx):
            continue

        tau = t[idx] - time_start
        dur = max(1e-12, time_stop - time_start)

        freq = freq_start + (freq_stop - freq_start) * (tau / dur)
        amp = amp_min + (amp_max - amp_min) * (tau / dur)

        # Per-event fades (seconds): per-row overrides take precedence over function args.
        default_fi = edge_fade_s if fade_in_s is None else float(fade_in_s)
        default_fo = edge_fade_s if fade_out_s is None else float(fade_out_s)
        fi_val = default_fi
        fo_val = default_fo
        # Row layout: required 6 cols, then optional fade cols in order.
        off = 6
        if has_fade_in_col:
            v = row[off]
            off += 1
            try:
                vv = float(v)
                if np.isfinite(vv) and vv >= 0:
                    fi_val = vv
            except Exception:
                pass
        if has_fade_out_col:
            v = row[off] if off < len(row) else None
            try:
                vv = float(v)
                if np.isfinite(vv) and vv >= 0:
                    fo_val = vv
            except Exception:
                pass

        if (fi_val > 0 or fo_val > 0) and tau.size:
            amp = _apply_fades(amp, fade_in_sec=fi_val, fade_out_sec=fo_val)

        # Phase handling.
        if phase_mode == "reset":
            # Phase resets at each segment start.
            #
            # Note: for time-varying frequency, using phase = 2π f(tau) tau will *not*
            # preserve the intended instantaneous frequency (it introduces a +tau f'(tau)
            # term). Because our per-event frequency model is linear, we integrate it so
            # the synthesized tone follows freq_start->freq_stop as expected.
            f0 = float(freq_start)
            f1 = float(freq_stop)
            if f0 == f1:
                phase = 2.0 * np.pi * f0 * tau
            else:
                phase = 2.0 * np.pi * (f0 * tau + 0.5 * (f1 - f0) * (tau * tau) / dur)
        elif phase_mode == "global":
            # Absolute-time referenced phase.
            #
            # For constant-frequency segments (freq_start == freq_stop), this is the classic
            # phase = 2π f t, which keeps phase continuous across adjacent segments at the same f.
            #
            # IMPORTANT: If freq is time-varying and you do phase = 2π f(t) t, the instantaneous
            # frequency becomes f(t) + t f'(t) (product rule), which can produce "extra" partials
            # that are not in the input table. To keep the intended instantaneous frequency for
            # linear ramps, we integrate the linear model over the event and add an absolute-time
            # offset so that the constant-frequency case matches 2π f t.
            f0 = float(freq_start)
            f1 = float(freq_stop)
            phase = 2.0 * np.pi * (f0 * t[idx] + 0.5 * (f1 - f0) * (tau * tau) / dur)
        else:  # "chirp"
            # Continuous phase *within* an event with a linear frequency ramp:
            # phi(tau) = 2π * ∫ f(u) du = 2π*(f0*tau + 0.5*(f1-f0)*tau^2/dur)
            f0 = float(freq_start)
            f1 = float(freq_stop)
            phase = 2.0 * np.pi * (f0 * tau + 0.5 * (f1 - f0) * (tau * tau) / dur)

        y_combined[idx] += amp * np.sin(phase)

    if show_waveform:
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t, y=y_combined, mode="lines", name="Synthesized"))
        fig_wave.update_layout(
            title="Synthesized waveform",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
        )
        # Keep time axis consistent with the spectrogram visualization (0..duration).
        if t.size:
            fig_wave.update_xaxes(range=[0.0, float(t[-1])])
        show_plotly(fig_wave)

    if show_fft:
        # Lazy import: audio_utils has heavier deps (librosa, scipy).
        from .audio_utils import plot_spectrogram

        sr = 1.0 / k
        skw = dict(spectrogram_kwargs or {})
        # Ensure show=True unless user explicitly overrides.
        skw.setdefault("show", True)
        plot_spectrogram(y_combined, sample_rate=sr, **skw)

    return y_combined


def crossfade_adjacent_events(
    wave_params: pd.DataFrame,
    crossfade_s: float,
    group_cols: Sequence[str] = ("freq_start", "freq_stop"),
    boundary_tol_s: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create a *true crossfade* between adjacent (abuttting) events by overlapping them in time and
    adding per-row fade durations.

    This is most useful for "frame-based" representations where each (frequency-bin) component is
    split into consecutive time segments. A crossfade is different from a simple fade-in/out because
    it requires an *overlap region* where the outgoing segment fades out while the incoming fades in.

    Behavior:
    - For each group defined by `group_cols`, events are sorted by time_start.
    - If two consecutive events are adjacent (next.time_start ~= prev.time_stop), we:
      - overlap them by `crossfade_s` (half on each side of the boundary)
      - set/update per-row 'fade_out_s' on the previous event and 'fade_in_s' on the next event

    Notes:
    - The returned DataFrame is a copy.
    - Existing 'fade_in_s'/'fade_out_s' values are preserved, and only increased if needed.
    - Use `fade_shape="sqrt_hann"` in `plot_waves` for a constant-power style crossfade.
    """
    if crossfade_s is None:
        raise ValueError("crossfade_s must be provided")
    crossfade_s = float(crossfade_s)
    if crossfade_s <= 0:
        return wave_params.copy()

    df = wave_params.copy()
    for c in ("time_start", "time_stop"):
        if c not in df.columns:
            raise ValueError(f"wave_params is missing required column: {c}")
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"wave_params is missing group column: {c}")

    if boundary_tol_s is None:
        boundary_tol_s = max(1e-9, crossfade_s * 0.01)
    boundary_tol_s = float(boundary_tol_s)
    if boundary_tol_s < 0:
        raise ValueError(f"boundary_tol_s must be >= 0; got {boundary_tol_s}")

    if "fade_in_s" not in df.columns:
        df["fade_in_s"] = np.nan
    if "fade_out_s" not in df.columns:
        df["fade_out_s"] = np.nan

    half = 0.5 * crossfade_s

    # Work per group to avoid crossfading unrelated components.
    for _key, g in df.groupby(list(group_cols), sort=False):
        g = g.sort_values("time_start", kind="mergesort")
        idxs = g.index.to_numpy()
        if idxs.size < 2:
            continue

        for i in range(idxs.size - 1):
            a = idxs[i]
            b = idxs[i + 1]
            a_stop = float(df.at[a, "time_stop"])
            b_start = float(df.at[b, "time_start"])
            if not (np.isfinite(a_stop) and np.isfinite(b_start)):
                continue

            gap = b_start - a_stop
            if abs(gap) > boundary_tol_s:
                continue  # not adjacent, don't force overlap

            # Overlap centered on the boundary; clip to >= 0 on the left.
            df.at[a, "time_stop"] = a_stop + half
            df.at[b, "time_start"] = max(0.0, b_start - half)

            # Ensure per-row fades cover the overlap (increase only).
            prev_fo = df.at[a, "fade_out_s"]
            next_fi = df.at[b, "fade_in_s"]
            try:
                prev_fo_f = float(prev_fo)
            except Exception:
                prev_fo_f = np.nan
            try:
                next_fi_f = float(next_fi)
            except Exception:
                next_fi_f = np.nan

            df.at[a, "fade_out_s"] = max(crossfade_s, prev_fo_f) if np.isfinite(prev_fo_f) else crossfade_s
            df.at[b, "fade_in_s"] = max(crossfade_s, next_fi_f) if np.isfinite(next_fi_f) else crossfade_s

    return df


def estimate_sampling_frequency_and_time_vector(
    df: pd.DataFrame,
) -> Tuple[float, float, float]:
    """
    Estimate a safe sampling frequency and derived time-step for synthesizing from an event DataFrame.

    Heuristic:
    - sampling_frequency = 2 * max(freq_start, freq_stop)  (Nyquist)
    - delta_t = 1 / sampling_frequency
    - duration = max(time_stop)

    Expected columns: 'freq_start', 'freq_stop', 'time_stop'.

    Returns (sampling_frequency_hz, delta_t_seconds, duration_seconds).
    """
    required = ["freq_start", "freq_stop", "time_stop"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    highest_frequency = float(
        max(
            pd.to_numeric(df["freq_start"], errors="coerce").max(),
            pd.to_numeric(df["freq_stop"], errors="coerce").max(),
        )
    )
    if not np.isfinite(highest_frequency) or highest_frequency <= 0:
        raise ValueError(f"highest frequency must be finite and > 0; got {highest_frequency}")

    sampling_frequency = 2.0 * highest_frequency
    delta_t = 1.0 / sampling_frequency

    duration = float(pd.to_numeric(df["time_stop"], errors="coerce").max())
    if not np.isfinite(duration) or duration < 0:
        raise ValueError(f"duration (max time_stop) must be finite and >= 0; got {duration}")

    return sampling_frequency, delta_t, duration

# Example usage:
# wave_params = pd.DataFrame({
#     'freq_start': [100, 200],
#     'freq_stop': [200, 300],
#     'time_start': [0, 0.5],
#     'time_stop': [0.5, 1],
#     'amp_min': [0, 0],
#     'amp_max': [1, 1],
# })
# y_combined = plot_waves(wave_params, k=0.001)
