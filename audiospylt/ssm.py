# Importing necessary libraries
import numpy as np
from numpy.typing import NDArray # For type hinting
from typing import Tuple         # For type hinting
import librosa
from scipy.signal import convolve2d
# scipy.spatial.distance pdist, squareform not used, cosine_similarity is
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
# import plotly.io as pio # Not strictly needed in the .py file if plots are shown via fig.show()

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


def load_audio(file_path: str, resample_audio: bool, target_sr: int) -> Tuple[NDArray[np.float32], float]:
    # ... (function body remains the same) ...
    if resample_audio:
        print(f"Resampling to {target_sr} Hz...")
        y, sr = librosa.load(file_path, mono=True, sr=target_sr)
    else:
        y, sr = librosa.load(file_path, mono=True, sr=None)
        print(f"Using original sample rate of {sr} Hz...")
    return y, sr


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
    chroma_list = []
    mfcc_list = []

    num_audio_samples = y.shape[0]

    if analysis_method in ['chroma', 'chroma+mfcc']:
        print("Computing chromagram...")
        for i in tqdm(range(0, num_audio_samples, segment_duration_samples), desc="Chroma Segments"):
            frame = y[i : i + segment_duration_samples]
            if frame.size == 0: continue
            c = librosa.feature.chroma_stft(y=frame, sr=sr)
            chroma_list.append(np.mean(c, axis=1))

    if analysis_method in ['mfcc', 'chroma+mfcc']:
        print("Computing MFCCs...")
        num_mfccs_to_extract = mfcc_end_idx
        for i in tqdm(range(0, num_audio_samples, segment_duration_samples), desc="MFCC Segments"):
            frame = y[i : i + segment_duration_samples]
            if frame.size == 0: continue
            m = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=num_mfccs_to_extract)
            m_selected = m[mfcc_start_idx:mfcc_end_idx, :]
            mfcc_list.append(np.mean(m_selected, axis=1))

    chroma_features_arr = np.array(chroma_list, dtype=np.float64).T # Ensure dtype if list is not empty
    mfcc_features_arr = np.array(mfcc_list, dtype=np.float64).T   # Ensure dtype if list is not empty

    # Handle empty arrays explicitly to match type hint if necessary
    if not chroma_list:
        chroma_features_arr = np.empty((0,0), dtype=np.float64) # Example, adjust shape if needed
        # Or more precisely, if you know the number of chroma bins (e.g., 12)
        # chroma_features_arr = np.empty((12,0), dtype=np.float64)

    if not mfcc_list:
        mfcc_features_arr = np.empty((0,0), dtype=np.float64)
        # Or, if you know num MFCCs (mfcc_end_idx - mfcc_start_idx)
        # num_selected_mfccs = mfcc_end_idx - mfcc_start_idx
        # mfcc_features_arr = np.empty((num_selected_mfccs, 0), dtype=np.float64)

    # A simpler approach that often works and relies on np.array behavior:
    # Convert lists of feature vectors to 2D NumPy arrays (features x time_segments)
    # np.array will infer dtype, but if the list is empty, it will be object.
    # So, specify dtype=np.float64, and handle the shape of empty arrays.

    # If chroma_list is empty, np.array(chroma_list).T results in array([]) with shape (0,).
    # To match NDArray[np.float64] with 2 dimensions, we might need to reshape if it's going to be (N,0)
    
    # Let's refine the creation of empty arrays to match typical feature dimensions
    num_chroma_bins = 12 # Standard for chroma
    num_mfcc_coeffs_selected = mfcc_end_idx - mfcc_start_idx

    if chroma_list:
        chroma_features_arr = np.array(chroma_list, dtype=np.float64).T
    else:
        chroma_features_arr = np.empty((num_chroma_bins, 0), dtype=np.float64)

    if mfcc_list:
        mfcc_features_arr = np.array(mfcc_list, dtype=np.float64).T
    else:
        mfcc_features_arr = np.empty((num_mfcc_coeffs_selected, 0), dtype=np.float64)
        
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


def plot_heatmap(ssm: NDArray[np.float64], title: str, total_duration_sec: float, window_size_sec: float): # CHANGED np.float_ to np.float64
    # ... (function body remains the same, assuming ssm is now float64) ...
    num_segments = ssm.shape[0]
    if num_segments == 0:
        print("SSM is empty, cannot plot heatmap.")
        return

    time_stamps_sec = np.arange(num_segments) * window_size_sec
    time_labels_min_sec = [f"{int(t // 60)}:{int(t % 60):02d}" for t in time_stamps_sec]

    num_labels = len(time_labels_min_sec)
    if num_labels == 0:
        tickvals, ticktext_labels = [], []
    else:
        desired_ticks = 10
        if num_labels <= desired_ticks:
            step_size = 1
        else:
            step_size = max(1, num_labels // desired_ticks)
        
        tickvals = list(range(0, num_labels, step_size))
        if num_labels > 0 and (num_labels - 1) not in tickvals and step_size > 1 :
            if (num_labels - 1) - tickvals[-1] >= step_size / 2 :
                 tickvals.append(num_labels - 1)
        
        ticktext_labels = [time_labels_min_sec[i] for i in tickvals]

    heatmap = go.Heatmap(
        x=time_labels_min_sec,
        y=time_labels_min_sec,
        z=ssm,
        colorscale='Hot',
        colorbar=dict(title='Cosine similarity'),
        zmin=-1.0, zmax=1.0 # Cosine similarity is bounded by [-1, 1]
    )

    layout = go.Layout(
        title=title,
        xaxis=dict(title=f'Time (segment start, window={window_size_sec}s)', tickvals=tickvals, ticktext=ticktext_labels),
        yaxis=dict(title=f'Time (segment start, window={window_size_sec}s)', tickvals=tickvals, ticktext=ticktext_labels),
        autosize=False,
        width=800,
        height=800,
        xaxis_showgrid=False, yaxis_showgrid=False
    )

    fig = go.Figure(data=[heatmap], layout=layout)
    fig.show()

def main(
    file_path: str,
    resample: bool,
    custom_sr: int,
    analysis_method: str,
    window_sec: float,
    mfcc_start: int,
    mfcc_end: int,
    diagonal_smooth_width: int
):
    y, sr = load_audio(file_path, resample_audio=resample, target_sr=custom_sr)

    if y.size == 0:
        print("Audio data is empty. Cannot proceed.")
        return

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
        plot_heatmap(np.array([[]], dtype=np.float64), "Self-Similarity Matrix (No data)", 0, window_sec)
        return

    features_norm_transposed = normalize_features(features)
    ssm = compute_ssm(features_norm_transposed)

    if diagonal_smooth_width > 1 :
        ssm = diagonal_smooth(ssm, width=diagonal_smooth_width)

    total_duration_sec = librosa.get_duration(y=y, sr=sr)
    plot_heatmap(ssm, title, total_duration_sec, window_sec)

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