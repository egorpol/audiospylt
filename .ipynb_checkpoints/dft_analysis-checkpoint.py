import numpy as np
import pandas as pd
from scipy.signal import find_peaks, get_window
import plotly.graph_objects as go

def apply_window(signal, window_type):
    window = get_window(window_type, len(signal))
    return signal * window

def compute_fft(signal, sr):
    fft = np.fft.rfft(signal) / len(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    spec = np.abs(fft)
    return freqs, spec

def filter_peaks(spec, freqs, thresh_amp_low, thresh_amp_high, thresh_freq_low, thresh_freq_high):
    peaks, _ = find_peaks(spec, height=(thresh_amp_low, thresh_amp_high))
    peaks = [peak for peak in peaks if thresh_freq_low <= freqs[peak] and (thresh_freq_high is None or freqs[peak] <= thresh_freq_high)]
    return peaks

def plot_spectrum(freqs, spec, peaks, thresh_amp_low, thresh_amp_high, thresh_freq_low, thresh_freq_high):
    max_amp = np.max(spec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=spec, mode='lines', name='Spectrum'))
    fig.add_trace(go.Scatter(x=freqs[peaks], y=spec[peaks], mode='markers', name='Peaks'))

    # Add lines for amplitude threshold values
    fig.add_shape(
        type="line",
        x0=0,
        y0=thresh_amp_low,
        x1=freqs[-1],
        y1=thresh_amp_low,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        ),
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=thresh_amp_high,
        x1=freqs[-1],
        y1=thresh_amp_high,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        ),
    )
    
    # Add lines for frequency threshold values
    fig.add_shape(
        type="line",
        x0=thresh_freq_low,
        y0=0,
        x1=thresh_freq_low,
        y1=max_amp + 0.2 * max_amp,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        ),
    )
    if thresh_freq_high is not None:
        fig.add_shape(
            type="line",
            x0=thresh_freq_high,
            y0=0,
            x1=thresh_freq_high,
            y1=max_amp + 0.2 * max_amp,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            ),
        )
    
    fig.update_layout(
        title='Spectrum and Peaks',
        xaxis_title='Frequency',
        yaxis_title='Amplitude',
        autosize=False,
        width=900,
        height=600,
        showlegend=True
    )
    
    fig.show()

def analyze_signal(signal, sr, filename, window_type='boxcar', thresh_amp_low=0.2, thresh_amp_high=0.4, thresh_freq_low=0, thresh_freq_high=None, show_peaks=False, show_plot=True):
    if window_type:
        signal = apply_window(signal, window_type)
    
    freqs, spec = compute_fft(signal, sr)
    peaks = filter_peaks(spec, freqs, thresh_amp_low, thresh_amp_high, thresh_freq_low, thresh_freq_high)
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
        display(peaks_df)

    if show_plot:
        plot_spectrum(freqs, spec, peaks, thresh_amp_low, thresh_amp_high, thresh_freq_low, thresh_freq_high)

    return peaks_df
