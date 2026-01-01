"""
Self-similarity matrix (SSM) analysis.

Notes on the implementation:
- Audio loading is delegated to `audiospylt.audio_utils.load_audio_sample`, which supports
  both local paths and URLs.
- Feature extraction is done on the full signal (once) and then aggregated to `window_sec`
  segments for speed and consistency.
- Plotting uses Plotly and the repo-wide `show_plotly` helper for robust notebook rendering.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple

import os
import librosa
from scipy.signal import convolve2d

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

try:  # pragma: no cover
    from .plotting import show_plotly
except Exception:  # pragma: no cover
    from audiospylt.plotting import show_plotly

try:  # pragma: no cover
    from .audio_utils import load_audio_sample
except Exception:  # pragma: no cover
    from audiospylt.audio_utils import load_audio_sample

# Default customizable variables (can be overridden by main function arguments)
# These are not directly used by the main function if all arguments are provided,
# but serve as documentation or defaults if the script were run directly with a modified
# `if __name__ == '__main__':` block.
RESAMPLE_DEFAULT = False
CUSTOM_SR_DEFAULT = 44100
ANALYSIS_METHOD_DEFAULT = 'chroma'
FILE_PATH_DEFAULT = 'chant1.wav'
MFCC_START_DEFAULT = 0
MFCC_END_DEFAULT = 20 # Default is 20 to get MFCCs 0-19
WINDOW_DEFAULT = 1.0 # seconds
DIAGONAL_SMOOTH_WIDTH_DEFAULT = 1


def _load_audio_mono(
    wav_source: str,
    *,
    resample_audio: bool,
    target_sr: int,
) -> Tuple[NDArray[np.float32], float, Dict[str, Any]]:
    """
    Load audio from a local path or URL.

    Returns (y_mono, sr, info).
    """
    # Provide a more helpful error than "Failed to load WAV from URL" when a local
    # path is missing (common in notebooks).
    if not os.path.exists(wav_source) and not str(wav_source).startswith(("http://", "https://")):
        raise FileNotFoundError(f"Audio file not found: {wav_source!r}")

    desired_sr = int(target_sr) if resample_audio else None
    y, sr, info = load_audio_sample(
        wav_source,
        desired_sample_rate=desired_sr,
        convert_to_mono=True,
        verbose=False,
    )
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr = float(sr)
    return y, sr, info


def compute_features(
    y: NDArray[np.float32],
    sr: float,
    analysis_method: str,
    segment_duration_samples: int,
    mfcc_start_idx: int,
    mfcc_end_idx: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: # CHANGED np.float_ to np.float64
    """
    Compute chroma and/or MFCC features from audio signal.
    Features are computed for segments of `segment_duration_samples` length,
    by averaging finer-scale features (e.g. STFT frames) within each segment.

    Args:
        y: Audio waveform.
        sr: Sample rate.
        analysis_method: 'chroma', 'mfcc', or 'chroma+mfcc'.
        segment_duration_samples: Length of audio segments for feature extraction (in samples).
        mfcc_start_idx: Starting index for MFCC coefficient selection.
        mfcc_end_idx: Ending index (exclusive) for MFCC coefficient selection.

    Returns:
        A tuple (chroma_features, mfcc_features). Each is a NumPy array with
        features as rows and time segments as columns. If a feature type is not
        computed, its corresponding array will be empty with shape (N, 0) or (0,0).
    """
    # Aggregate full-resolution features to fixed windows for speed.
    # We keep librosa's default hop_length unless specified; it provides a stable
    # time grid that we can bin into `segment_duration_samples` windows.
    hop_length = 512

    num_chroma_bins = 12
    num_mfcc_coeffs_selected = mfcc_end_idx - mfcc_start_idx

    # Edge-case: too short audio.
    if y.size == 0 or segment_duration_samples <= 0:
        return (
            np.empty((num_chroma_bins, 0), dtype=np.float64),
            np.empty((max(0, num_mfcc_coeffs_selected), 0), dtype=np.float64),
        )

    # Convert segment size into seconds to bin STFT frames by time.
    window_sec = float(segment_duration_samples) / float(sr)
    if window_sec <= 0:
        raise ValueError("segment_duration_samples results in non-positive window_sec")

    chroma_features_arr = np.empty((num_chroma_bins, 0), dtype=np.float64)
    mfcc_features_arr = np.empty((max(0, num_mfcc_coeffs_selected), 0), dtype=np.float64)

    def _bin_mean(feat: NDArray[np.float64], frame_times: NDArray[np.float64]) -> NDArray[np.float64]:
        """feat: (n_feat, n_frames) -> (n_feat, n_segments) by mean within each segment."""
        if feat.size == 0 or feat.shape[1] == 0:
            return np.empty((feat.shape[0], 0), dtype=np.float64)
        seg_ids = np.floor(frame_times / window_sec).astype(int)
        seg_ids = np.maximum(seg_ids, 0)
        n_segments = int(seg_ids.max() + 1) if seg_ids.size else 0
        if n_segments <= 0:
            return np.empty((feat.shape[0], 0), dtype=np.float64)
        counts = np.bincount(seg_ids, minlength=n_segments).astype(np.float64)
        out = np.zeros((feat.shape[0], n_segments), dtype=np.float64)
        for d in range(feat.shape[0]):
            out[d, :] = np.bincount(seg_ids, weights=feat[d, :], minlength=n_segments)
        # Avoid divide-by-zero just in case (shouldn't happen).
        counts[counts == 0] = 1.0
        out /= counts[None, :]
        return out

    # Use a shared frame-time grid for all features by computing them on the full signal.
    if analysis_method in ["chroma", "chroma+mfcc"]:
        print("Computing chromagram (full signal)...")
        c = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        c = np.asarray(c, dtype=np.float64)
        frame_times = librosa.frames_to_time(np.arange(c.shape[1]), sr=sr, hop_length=hop_length).astype(np.float64)
        chroma_features_arr = _bin_mean(c, frame_times)

    if analysis_method in ["mfcc", "chroma+mfcc"]:
        print("Computing MFCCs (full signal)...")
        if mfcc_end_idx <= mfcc_start_idx:
            raise ValueError(f"mfcc_end must be > mfcc_start; got {mfcc_end_idx} <= {mfcc_start_idx}")
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=int(mfcc_end_idx), hop_length=hop_length)
        m = np.asarray(m, dtype=np.float64)
        m_selected = m[mfcc_start_idx:mfcc_end_idx, :]
        frame_times = librosa.frames_to_time(np.arange(m_selected.shape[1]), sr=sr, hop_length=hop_length).astype(np.float64)
        mfcc_features_arr = _bin_mean(m_selected, frame_times)

    return chroma_features_arr, mfcc_features_arr


def normalize_features(features: NDArray[np.float64]) -> NDArray[np.float64]: # CHANGED np.float_ to np.float64
    """
    Normalize feature vectors using StandardScaler.
    Input features: (num_feature_dimensions, num_segments)
    Output features_norm: (num_segments, num_feature_dimensions) for cosine_similarity
    """
    if features.shape[1] == 0:
        return np.empty((0, features.shape[0]), dtype=np.float64) # CHANGED

    scaler = StandardScaler()
    features_norm_transposed = scaler.fit_transform(features.T)
    # StandardScaler usually returns float64
    return features_norm_transposed.astype(np.float64, copy=False) # Ensure it is float64


def compute_ssm(features_normalized_transposed: NDArray[np.float64]) -> NDArray[np.float64]: # CHANGED np.float_ to np.float64
    """
    Compute the self-similarity matrix using cosine similarity.
    Input features_normalized_transposed: (num_segments, num_feature_dimensions)
    Output SSM: (num_segments, num_segments)
    """
    if features_normalized_transposed.shape[0] == 0:
        return np.array([[]], dtype=np.float64) # CHANGED

    ssm = cosine_similarity(features_normalized_transposed)
    # cosine_similarity usually returns float64
    return ssm.astype(np.float64, copy=False) # Ensure it is float64


def diagonal_smooth(matrix: NDArray[np.float64], width: int) -> NDArray[np.float64]: # CHANGED np.float_ to np.float64
    """
    Apply diagonal smoothing to a matrix.
    """
    if not isinstance(width, int) or width <= 1:
        return matrix.astype(np.float64, copy=False) # Ensure type if no smoothing
    if matrix.size == 0:
        return np.array([[]], dtype=np.float64) # Or matrix.astype(np.float64) if it must match shape

    kernel = np.eye(width, dtype=np.float64) # CHANGED
    kernel /= kernel.sum()

    smoothed_matrix = convolve2d(matrix, kernel, mode='same', boundary='symm')
    return smoothed_matrix.astype(np.float64, copy=False) # convolve2d can change dtype


def plot_heatmap(
    ssm: NDArray[np.float64],
    title: str,
    total_duration_sec: float,
    window_size_sec: float,
) -> Optional[go.Figure]:
    """
    Plot a self-similarity matrix with numeric time axes (seconds).

    Returns the Plotly figure (or None if empty).
    """
    num_segments = int(ssm.shape[0]) if ssm.ndim == 2 else 0
    if num_segments <= 0:
        print("SSM is empty, cannot plot heatmap.")
        return None

    time_sec = (np.arange(num_segments, dtype=float) * float(window_size_sec)).astype(float)

    # Use px.imshow for consistent heatmap defaults and fast rendering.
    fig = px.imshow(
        ssm,
        x=time_sec,
        y=time_sec,
        origin="lower",
        aspect="equal",
        color_continuous_scale="Hot",
        zmin=-1.0,
        zmax=1.0,
        labels={"x": "Time (s)", "y": "Time (s)", "color": "Cosine similarity"},
        title=title,
    )

    # Reasonable ticks in mm:ss when audio is longer.
    desired_ticks = 10
    if num_segments <= desired_ticks:
        tickvals = time_sec
    else:
        step = max(1, num_segments // desired_ticks)
        tickvals = time_sec[::step]
        if tickvals.size and tickvals[-1] != time_sec[-1]:
            tickvals = np.concatenate([tickvals, [time_sec[-1]]])

    def _fmt_mmss(t: float) -> str:
        t = float(t)
        m = int(t // 60.0)
        s = int(round(t - 60.0 * m))
        if s == 60:
            m += 1
            s = 0
        return f"{m}:{s:02d}"

    ticktext = [_fmt_mmss(v) for v in tickvals]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0.0, max(time_sec[-1], float(total_duration_sec))])
    fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0.0, max(time_sec[-1], float(total_duration_sec))])
    fig.update_layout(width=820, height=820)

    show_plotly(fig)
    return fig

def main(
    *,
    file_path: Optional[str] = None,
    wav_source: Optional[str] = None,
    resample: bool,
    custom_sr: int,
    analysis_method: str,
    window_sec: float,
    mfcc_start: int,
    mfcc_end: int,
    diagonal_smooth_width: int,
) -> Dict[str, Any]:
    """
    Run SSM analysis.

    Inputs:
    - file_path: backwards-compatible alias (local path or URL)
    - wav_source: preferred name (local path or URL)

    Returns a dict with:
    - y, sr, ssm, fig, features_shape, info (load_audio_sample info)
    """
    src = wav_source or file_path
    if not src:
        raise ValueError("Provide `wav_source` (preferred) or `file_path`.")

    y, sr, info = _load_audio_mono(src, resample_audio=resample, target_sr=custom_sr)

    if y.size == 0:
        print("Audio data is empty. Cannot proceed.")
        return {"y": y, "sr": sr, "ssm": np.array([[]], dtype=np.float64), "fig": None, "features_shape": (0, 0), "info": info}

    segment_duration_samples = int(sr * window_sec)
    if segment_duration_samples == 0:
        raise ValueError("Window size is too small, results in zero samples per segment.")

    chroma_features, mfcc_features = compute_features(
        y, sr, analysis_method, segment_duration_samples, mfcc_start, mfcc_end
    )
    # At this point, chroma_features and mfcc_features should be np.float64

    valid_chroma = chroma_features.ndim == 2 and chroma_features.shape[1] > 0
    valid_mfcc = mfcc_features.ndim == 2 and mfcc_features.shape[1] > 0

    features: NDArray[np.float64] # Explicitly type the features variable

    if analysis_method == 'chroma+mfcc':
        if valid_chroma and valid_mfcc:
            if chroma_features.shape[1] != mfcc_features.shape[1]:
                raise ValueError("Chroma and MFCC features have different number of segments.")
            features = np.concatenate((chroma_features, mfcc_features), axis=0)
            title = 'Self-Similarity Matrix (Chroma + MFCC features)'
        elif valid_chroma:
            print("Warning: MFCC features were empty, using only Chroma for 'chroma+mfcc'.")
            features = chroma_features
            title = 'Self-Similarity Matrix (Chroma features only)'
        elif valid_mfcc:
            print("Warning: Chroma features were empty, using only MFCC for 'chroma+mfcc'.")
            features = mfcc_features
            title = 'Self-Similarity Matrix (MFCC features only)'
        else:
            # Define num_feature_dims for empty case correctly
            # For chroma+mfcc, if both are empty, what should num_feature_dims be? Let's assume 0 for now.
            features = np.empty((0, 0), dtype=np.float64)
            title = 'Self-Similarity Matrix (No features)'
    elif analysis_method == 'chroma':
        num_chroma_bins = 12 # Standard
        features = chroma_features if valid_chroma else np.empty((num_chroma_bins, 0), dtype=np.float64)
        title = 'Self-Similarity Matrix (Chroma features)'
    elif analysis_method == 'mfcc':
        num_mfcc_coeffs_selected = mfcc_end - mfcc_start
        features = mfcc_features if valid_mfcc else np.empty((num_mfcc_coeffs_selected, 0), dtype=np.float64)
        title = 'Self-Similarity Matrix (MFCC features)'
    else:
        raise ValueError(
            f"Invalid analysis method: {analysis_method}. "
            "Choose 'chroma', 'mfcc', or 'chroma+mfcc'."
        )

    if features.size == 0 or features.shape[1] == 0 :
        print(
            "No features were extracted or no segments found. "
            "Audio might be too short for the given window size, or an issue with feature extraction. Cannot compute SSM."
        )
        fig = plot_heatmap(np.array([[]], dtype=np.float64), "Self-Similarity Matrix (No data)", 0, window_sec)
        return {"y": y, "sr": sr, "ssm": np.array([[]], dtype=np.float64), "fig": fig, "features_shape": tuple(features.shape), "info": info}

    features_norm_transposed = normalize_features(features)
    ssm = compute_ssm(features_norm_transposed)

    if diagonal_smooth_width > 1 :
        ssm = diagonal_smooth(ssm, width=diagonal_smooth_width)

    total_duration_sec = librosa.get_duration(y=y, sr=sr)
    fig = plot_heatmap(ssm, title, total_duration_sec, window_sec)
    return {"y": y, "sr": sr, "ssm": ssm, "fig": fig, "features_shape": tuple(features.shape), "info": info}


def run_notebook(
    wav_source: str,
    *,
    resample: bool = RESAMPLE_DEFAULT,
    custom_sr: int = CUSTOM_SR_DEFAULT,
    analysis_method: str = ANALYSIS_METHOD_DEFAULT,
    window_sec: float = WINDOW_DEFAULT,
    mfcc_start: int = MFCC_START_DEFAULT,
    mfcc_end: int = MFCC_END_DEFAULT,
    diagonal_smooth_width: int = DIAGONAL_SMOOTH_WIDTH_DEFAULT,
    print_params: bool = True,
    catch_exceptions: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Notebook-friendly wrapper around :func:`main`.

    Keeps notebooks clean: a single function call handles printing + friendly errors.

    Returns the same dict as :func:`main`, or None when an error is caught.
    """
    params: Dict[str, Any] = {
        "wav_source": wav_source,
        "resample": resample,
        "custom_sr": custom_sr,
        "analysis_method": analysis_method,
        "window_sec": window_sec,
        "mfcc_start": mfcc_start,
        "mfcc_end": mfcc_end,
        "diagonal_smooth_width": diagonal_smooth_width,
    }

    if print_params:
        print("Analysis Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print("-" * 30)

    if not catch_exceptions:
        return main(**params)

    try:
        return main(**params)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return None
    except ValueError as ve:
        print(f"VALUE ERROR: {ve}")
        return None
    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURRED: {e}")
        # For debugging:
        # import traceback
        # traceback.print_exc()
        return None

# Ensure the script can be imported without executing the main function immediately
# (Example of how you might run it with default parameters if not importing)
# if __name__ == '__main__':
#     main(
#         file_path=FILE_PATH_DEFAULT, 
#         resample=RESAMPLE_DEFAULT, 
#         custom_sr=CUSTOM_SR_DEFAULT, 
#         analysis_method=ANALYSIS_METHOD_DEFAULT, 
#         window_sec=WINDOW_DEFAULT, 
#         mfcc_start=MFCC_START_DEFAULT, 
#         mfcc_end=MFCC_END_DEFAULT, 
#         diagonal_smooth_width=DIAGONAL_SMOOTH_WIDTH_DEFAULT
#     )