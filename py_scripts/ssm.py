# Importing necessary libraries
import numpy as np
from scipy.spatial.distance import pdist, squareform
import librosa
from scipy.signal import convolve2d
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.io as pio

# Customizable variables for the script
resample = False  # Set this to False to preserve original sr
custom_sr = 44100  # Modify this to change the custom sr
analysis_method = 'chroma'  # Modify this to 'chroma', 'mfcc', or 'chroma+mfcc'
file_path = 'chant1.wav'  # Modify this to change the audio file path
mfcc_start = 0  # start of interval (included)
mfcc_end = 19  # end of interval (excluded)
window = 1  # Modify this to change the window size (s)
diagonal_smooth_width = 1  # Modify this to change the diagonal smoothing width

def load_audio(file_path, resample, custom_sr):
    """
    Load an audio file, with optional resampling.
    """
    if resample:
        print(f"Downsampling to {custom_sr} Hz...")
        y, sr = librosa.load(file_path, mono=True, sr=custom_sr)
    else:
        y, sr = librosa.load(file_path, mono=True, sr=None)
        print(f"Preserving original sample rate of {sr} Hz...")
    return y, sr

def compute_features(y, sr, analysis_method, hop_length, mfcc_start, mfcc_end):
    """
    Compute chroma and/or MFCC features from audio signal.
    """
    chroma_features, mfcc_features = [], []
    if analysis_method in ['chroma', 'chroma+mfcc']:
        print("Computing chromagram...")
        for i in tqdm(range(0, y.shape[0], hop_length), desc="Chroma"):
            frame = y[i:i+hop_length]
            c = librosa.feature.chroma_stft(y=frame, sr=sr)
            chroma_features.append(np.mean(c, axis=1))
        chroma_features = np.array(chroma_features).T

    if analysis_method in ['mfcc', 'chroma+mfcc']:
        print("Computing MFCCs...")
        for i in tqdm(range(0, y.shape[0], hop_length), desc="MFCC"):
            frame = y[i:i+hop_length]
            m = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=mfcc_end)
            m = m[mfcc_start:mfcc_end, :]
            mfcc_features.append(np.mean(m, axis=1))
        mfcc_features = np.array(mfcc_features).T

    return chroma_features, mfcc_features

def normalize_features(features):
    """
    Normalize feature vectors using StandardScaler.
    """
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features.T)
    return features_norm

def compute_ssm(features_norm):
    """
    Compute the self-similarity matrix using cosine similarity.
    """
    ssm = cosine_similarity(features_norm)
    return ssm

def diagonal_smooth(mat, width):
    """
    Apply diagonal smoothing to a matrix.
    """
    if width <= 1:
        return mat
    kernel = np.eye(width)
    kernel /= kernel.sum()
    smoothed = convolve2d(mat, kernel, mode='same', boundary='symm')
    return smoothed

def plot_heatmap(ssm, title, duration_sec):
    """
    Plot the self-similarity matrix as a heatmap using Plotly.
    """
    time_data_sec = [i*duration_sec/ssm.shape[0] for i in range(ssm.shape[0])]
    time_data_min_sec = [f"{int(t // 60)}:{int(t % 60):02}" for t in time_data_sec]

    # Set the step size to have 10 ticks per axis
    step_size = len(time_data_min_sec) // 10
    tickvals = [i for i in range(0, len(time_data_min_sec), step_size)]

    heatmap = go.Heatmap(
        x=time_data_min_sec,
        y=time_data_min_sec,
        z=ssm,
        colorscale='Hot',
        colorbar=dict(title='Cosine similarity'),
    )

    layout = go.Layout(
        title=title,
        xaxis=dict(title='Time (minute:second)', tickvals=tickvals, ticktext=[time_data_min_sec[i] for i in tickvals]),
        yaxis=dict(title='Time (minute:second)', tickvals=tickvals, ticktext=[time_data_min_sec[i] for i in tickvals]),
        autosize=False,
        width=1000,
        height=1000,
    )

    fig = go.Figure(data=[heatmap], layout=layout)
    fig.show()

# Main function to execute the analysis and plotting
def main(file_path, resample, custom_sr, analysis_method, window, mfcc_start, mfcc_end, diagonal_smooth_width):
    y, sr = load_audio(file_path, resample, custom_sr)
    hop_length = int(sr * window)
    chroma_features, mfcc_features = compute_features(y, sr, analysis_method, hop_length, mfcc_start, mfcc_end)
    
    # Determine which features to use based on analysis method
    if analysis_method == 'chroma+mfcc':
        features = np.concatenate((chroma_features, mfcc_features), axis=0)
        title = 'Self-Similarity Matrix (chroma+mfcc features)'
    elif analysis_method == 'chroma':
        features = chroma_features
        title = 'Self-Similarity Matrix (chroma features)'
    elif analysis_method == 'mfcc':
        features = mfcc_features
        title = 'Self-Similarity Matrix (MFCC features)'
    else:
        raise ValueError(f"Invalid analysis method: {analysis_method}. Choose 'chroma', 'mfcc', or 'chroma+mfcc'.")

    features_norm = normalize_features(features)
    ssm = compute_ssm(features_norm)
    ssm = diagonal_smooth(ssm, width=diagonal_smooth_width)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    plot_heatmap(ssm, title, duration_sec)

# Ensure the script can be imported without executing the main function immediately
# if __name__ == '__main__':
#     main(file_path, resample, custom_sr, analysis_method, window, mfcc_start, mfcc_end, diagonal_smooth_width)
