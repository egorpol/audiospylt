"""Utilities for f0 search from measured partial frequencies."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

__all__ = ["search_f0", "summarize_result"]


def _show_plotly(fig):
    try:  # pragma: no cover
        from .plotting import show_plotly
    except Exception:  # pragma: no cover
        from audiospylt.plotting import show_plotly
    return show_plotly(fig)


def _coerce_freqs_and_weights(freqs, weights=None) -> tuple[np.ndarray, np.ndarray | None]:
    freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
    if freqs_arr.size == 0:
        raise ValueError("freqs must contain at least one value")

    w_arr = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float).reshape(-1)
        if w_arr.size != freqs_arr.size:
            raise ValueError(
                f"weights length ({w_arr.size}) must match freqs length ({freqs_arr.size})"
            )

    valid = np.isfinite(freqs_arr) & (freqs_arr > 0.0)
    freqs_arr = freqs_arr[valid]
    if w_arr is not None:
        w_arr = w_arr[valid]

    if freqs_arr.size == 0:
        raise ValueError("freqs must contain at least one finite value > 0 Hz")
    return freqs_arr, w_arr


def _candidate_grid(f0_min: float, f0_max: float, step: float) -> np.ndarray:
    f0_min = float(f0_min)
    f0_max = float(f0_max)
    step = float(step)
    if f0_min <= 0 or f0_max <= 0:
        raise ValueError(f"f0_min and f0_max must be > 0; got {f0_min}, {f0_max}")
    if f0_max <= f0_min:
        raise ValueError(f"f0_max must be > f0_min; got {f0_max} <= {f0_min}")
    if step <= 0:
        raise ValueError(f"step must be > 0; got {step}")

    f0_candidates = np.arange(f0_min, f0_max, step, dtype=float)
    if f0_candidates.size == 0:
        raise ValueError(
            f"empty search grid for f0_min={f0_min}, f0_max={f0_max}, step={step}"
        )
    return f0_candidates


def _validate_trim_params(trim_outlier_cents, robust_trim_frac):
    if trim_outlier_cents is not None:
        trim_outlier_cents = float(trim_outlier_cents)
        if trim_outlier_cents <= 0:
            raise ValueError(
                f"trim_outlier_cents must be > 0 when provided; got {trim_outlier_cents}"
            )
    if robust_trim_frac is not None:
        robust_trim_frac = float(robust_trim_frac)
        if not (0.0 <= robust_trim_frac < 1.0):
            raise ValueError(
                f"robust_trim_frac must be in [0, 1); got {robust_trim_frac}"
            )
    return trim_outlier_cents, robust_trim_frac


def _normalize_weights(weights, n_partials: int) -> np.ndarray:
    if weights is None:
        return np.full(n_partials, 1.0 / float(n_partials), dtype=float)

    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n_partials:
        raise ValueError(
            f"weights length ({w.size}) must match number of partials ({n_partials})"
        )
    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be >= 0")
    s = float(np.sum(w))
    if s <= 0:
        raise ValueError("weights must have a positive sum")
    return w / s


def _evaluate_candidates(
    freqs_arr: np.ndarray,
    f0_candidates: np.ndarray,
    base_weights: np.ndarray,
    trim_outlier_cents: float | None,
    robust_trim_frac: float | None,
):
    # Broadcast over all candidates at once: (n_candidates, n_partials)
    f0s = f0_candidates[:, None]
    freqs_row = freqs_arr[None, :]

    n = np.rint(freqs_row / f0s)
    n = np.maximum(n, 1.0)
    recon = n * f0s
    cents = 1200.0 * np.log2(freqs_row / recon)

    mask = np.isfinite(cents)
    abs_cents = np.abs(cents)

    if trim_outlier_cents is not None:
        mask &= abs_cents <= float(trim_outlier_cents)

    if robust_trim_frac is not None and cents.shape[1] > 1:
        keep_n = max(1, int(np.ceil(cents.shape[1] * (1.0 - float(robust_trim_frac)))))
        sort_idx = np.argsort(abs_cents, axis=1)
        rows = np.arange(cents.shape[0])[:, None]
        keep_mask = np.zeros_like(mask, dtype=bool)
        keep_mask[rows, sort_idx[:, :keep_n]] = True
        mask &= keep_mask

    weights_2d = np.broadcast_to(base_weights.reshape(1, -1), cents.shape)
    eff_weights = np.where(mask, weights_2d, 0.0)
    den = np.sum(eff_weights, axis=1)

    rms_cents = np.full(cents.shape[0], np.inf, dtype=float)
    valid = den > 0
    if np.any(valid):
        weighted_sq = np.sum((cents**2) * eff_weights, axis=1)
        rms_cents[valid] = np.sqrt(weighted_sq[valid] / den[valid])

    return n, cents, rms_cents


def search_f0(
    freqs=None,
    f0_min: float = 35.0,
    f0_max: float = 150.0,
    step: float = 0.001,
    *,
    peaks_df=None,
    freqs_col: str = "Frequency (Hz)",
    weights=None,
    use_amp_weights: bool = False,
    amp_col: str = "Amplitude",
    top_k: int = 1,
    trim_outlier_cents: float | None = None,
    robust_trim_frac: float | None = None,
    coarse_step: float | None = None,
    refine_half_width_hz: float = 0.5,
    include_plot: bool = False,
    show_plot: bool = True,
    annotate_plot: bool = True,
    plot_title: str = "RMS cents error vs f0 candidate",
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Search the f0 that best explains measured frequencies as integer partials.

    Parameters
    ----------
    freqs:
        1D list/array of measured partial frequencies in Hz. If omitted, provide
        `peaks_df` + `freqs_col`.
    f0_min, f0_max, step:
        Search grid bounds and step in Hz.
    peaks_df:
        Optional dataframe-like object (e.g. output of `analyze_signal`) used as
        an alternative input source for frequencies.
    freqs_col:
        Frequency column name in `peaks_df`.
    weights:
        Optional per-frequency weights. Must match frequency length.
    use_amp_weights:
        If True, use `peaks_df[amp_col]` as weights (ignored when `weights` is provided).
    amp_col:
        Amplitude column name in `peaks_df` for automatic weighting.
    top_k:
        Number of best candidates to return (>= 1).
    trim_outlier_cents:
        Optional absolute-cents threshold. Residuals above this value are ignored
        in the objective.
    robust_trim_frac:
        Optional fraction [0, 1) of worst residuals to drop per candidate.
    coarse_step:
        Optional coarse global search step in Hz. If provided and > `step`, a
        coarse sweep is done first, then a fine sweep around the best coarse
        candidate.
    refine_half_width_hz:
        Half-width around the best coarse f0 used for fine refinement.
    include_plot:
        If True, build an RMS-cents-vs-f0 plot and include it in the result.
    show_plot:
        If True and include_plot=True, render the plot.
    annotate_plot:
        If True and include_plot=True, annotate the plot with search settings.
    plot_title, plot_width, plot_height, plotly_layout:
        Plot customization parameters.

    Returns
    -------
    dict
        Keys:
        - best_f0_hz
        - best_partial_numbers
        - best_cents_errors
        - best_rms_cents
        - top_candidates
        - f0_candidates_hz
        - rms_cents_all
        - search_params
        - fig (Plotly figure or None)
    """
    if freqs is None and peaks_df is None:
        raise ValueError("provide `freqs` or `peaks_df`")
    if freqs is not None and peaks_df is not None:
        raise ValueError("provide only one input source: `freqs` or `peaks_df`")

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}")

    trim_outlier_cents, robust_trim_frac = _validate_trim_params(
        trim_outlier_cents, robust_trim_frac
    )

    raw_weights = weights
    if freqs is None:
        if freqs_col not in peaks_df:
            raise ValueError(f"`peaks_df` missing frequency column: {freqs_col}")
        freqs = peaks_df[freqs_col]
        if raw_weights is None and use_amp_weights:
            if amp_col not in peaks_df:
                raise ValueError(f"`peaks_df` missing amplitude column: {amp_col}")
            raw_weights = peaks_df[amp_col]
    elif raw_weights is None and use_amp_weights:
        raise ValueError("use_amp_weights=True requires `peaks_df` input")

    freqs_arr, raw_weights_arr = _coerce_freqs_and_weights(freqs, raw_weights)
    base_weights = _normalize_weights(raw_weights_arr, n_partials=freqs_arr.size)

    if coarse_step is not None and float(coarse_step) > float(step):
        coarse_grid = _candidate_grid(f0_min, f0_max, float(coarse_step))
        _, _, coarse_rms = _evaluate_candidates(
            freqs_arr,
            coarse_grid,
            base_weights,
            trim_outlier_cents,
            robust_trim_frac,
        )
        coarse_best = float(coarse_grid[int(np.argmin(coarse_rms))])
        half = float(refine_half_width_hz)
        if half <= 0:
            raise ValueError(
                f"refine_half_width_hz must be > 0 when coarse_step is used; got {half}"
            )
        f_start = max(float(f0_min), coarse_best - half)
        f_stop = min(float(f0_max), coarse_best + half)
        fine_grid = np.arange(f_start, f_stop + float(step) * 0.5, float(step), dtype=float)
        if fine_grid.size == 0:
            fine_grid = np.array([coarse_best], dtype=float)
        f0_candidates = np.unique(np.concatenate([coarse_grid, fine_grid]))
    else:
        f0_candidates = _candidate_grid(f0_min, f0_max, step)

    n, cents, rms_cents_all = _evaluate_candidates(
        freqs_arr,
        f0_candidates,
        base_weights,
        trim_outlier_cents,
        robust_trim_frac,
    )

    best_idx = int(np.argmin(rms_cents_all))
    best_f0 = float(f0_candidates[best_idx])
    best_n = n[best_idx].astype(int)
    best_cents = cents[best_idx]
    best_rms_cents = float(rms_cents_all[best_idx])

    finite_mask = np.isfinite(rms_cents_all)
    sorted_idx = np.argsort(rms_cents_all[finite_mask])
    finite_indices = np.where(finite_mask)[0][sorted_idx]
    top_n = min(int(top_k), int(finite_indices.size))
    top_candidates = []
    for idx in finite_indices[:top_n]:
        top_candidates.append(
            {
                "f0_hz": float(f0_candidates[idx]),
                "rms_cents": float(rms_cents_all[idx]),
                "partial_numbers": n[idx].astype(int),
                "cents_errors": cents[idx],
            }
        )

    search_params = {
        "f0_min": float(f0_min),
        "f0_max": float(f0_max),
        "step": float(step),
        "coarse_step": (None if coarse_step is None else float(coarse_step)),
        "refine_half_width_hz": float(refine_half_width_hz),
        "top_k": int(top_k),
        "trim_outlier_cents": trim_outlier_cents,
        "robust_trim_frac": robust_trim_frac,
        "use_amp_weights": bool(use_amp_weights),
        "freqs_col": str(freqs_col),
        "amp_col": str(amp_col),
    }

    fig = None
    if include_plot:
        y_min = float(np.min(rms_cents_all))
        y_max = float(np.max(rms_cents_all))
        if y_max <= y_min:
            y_max = y_min + 1e-9

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=f0_candidates,
                y=rms_cents_all,
                mode="lines",
                name="RMS cents error",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[best_f0, best_f0],
                y=[y_min, y_max],
                mode="lines",
                name=f"best f0 = {best_f0:.3f} Hz",
                line=dict(color="red", dash="dash"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[best_f0],
                y=[best_rms_cents],
                mode="markers",
                name="best candidate",
                marker=dict(color="red", size=8),
            )
        )
        if annotate_plot:
            coarse_txt = "off" if coarse_step is None else f"{float(coarse_step):g}"
            trim_txt = (
                "off" if trim_outlier_cents is None else f"{float(trim_outlier_cents):g}"
            )
            robust_txt = (
                "off" if robust_trim_frac is None else f"{float(robust_trim_frac):g}"
            )
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                text=(
                    f"range={float(f0_min):g}..{float(f0_max):g} Hz"
                    f"<br>step={float(step):g} Hz, coarse_step={coarse_txt}"
                    f"<br>trim_outlier_cents={trim_txt}, robust_trim_frac={robust_txt}"
                ),
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
            )
        fig.update_layout(
            title=plot_title,
            xaxis_title="f0 candidate (Hz)",
            yaxis_title="RMS cents error",
            autosize=False,
            width=(900 if plot_width is None else int(plot_width)),
            height=(420 if plot_height is None else int(plot_height)),
            showlegend=True,
        )
        if plotly_layout:
            fig.update_layout(**plotly_layout)
        if show_plot:
            _show_plotly(fig)

    return {
        "best_f0_hz": best_f0,
        "best_partial_numbers": best_n,
        "best_cents_errors": best_cents,
        "best_rms_cents": best_rms_cents,
        "top_candidates": top_candidates,
        "f0_candidates_hz": f0_candidates,
        "rms_cents_all": rms_cents_all,
        "search_params": search_params,
        "fig": fig,
    }


def summarize_result(
    label: str,
    result: dict[str, Any],
    top_n: int = 3,
    *,
    precision: int = 3,
    suppress: bool = True,
    floatmode: str = "fixed",
) -> None:
    """
    Print a compact summary for a `search_f0(...)` result dict.

    Parameters
    ----------
    label:
        Text label printed before the summary.
    result:
        Dictionary returned by `search_f0(...)`.
    top_n:
        Number of top candidates to print.
    precision, suppress, floatmode:
        Forwarded to NumPy print settings for readability.
    """
    top_n = int(top_n)
    precision = int(precision)
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1; got {top_n}")
    if precision < 0:
        raise ValueError(f"precision must be >= 0; got {precision}")

    with np.printoptions(precision=precision, suppress=bool(suppress), floatmode=str(floatmode)):
        print(f"{label}")
        print(f"best f0: {result['best_f0_hz']:.{precision}f} Hz")
        print(f"best RMS cents: {result['best_rms_cents']:.{precision}f}")
        print(f"best partial numbers: {result['best_partial_numbers'].tolist()}")
        print(
            "best cents errors:",
            np.array2string(
                result["best_cents_errors"],
                precision=precision,
                suppress_small=bool(suppress),
            ),
        )
        print("top candidates:")
        for cand in result["top_candidates"][:top_n]:
            print(
                f"  f0={cand['f0_hz']:.{precision}f} Hz, "
                f"RMS={cand['rms_cents']:.{precision}f}"
            )
        print()
