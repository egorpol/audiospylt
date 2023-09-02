import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

def plot_waves(wave_params, k=0.001, disable_freq_plot=False, disable_amp_plot=False, disable_combined_plot=False, disable_wave_plot=False):
    # find the biggest interval between all given time_start and time_stop values
    n = wave_params['time_stop'].max()

    # define the time vector (from 0 to n seconds, with a step of k seconds)
    t = np.arange(0, n + k, k)

    # initialize an empty array for the combined wave
    y_combined = np.zeros_like(t)

    # calculate the y values for all waves at once
    time_start = wave_params['time_start'].values
    time_stop = wave_params['time_stop'].values
    freq_start = wave_params['freq_start'].values
    freq_stop = wave_params['freq_stop'].values
    amp_min = wave_params['amp_min'].values
    amp_max = wave_params['amp_max'].values

    freq = np.where((t[:, None] >= time_start) & (t[:, None] <= time_stop),
                    freq_start + (freq_stop - freq_start) * (t[:, None] - time_start) / (time_stop - time_start),
                    0)
    amp = np.where((t[:, None] >= time_start) & (t[:, None] <= time_stop),
                   amp_min + (amp_max - amp_min) * (t[:, None] - time_start) / (time_stop - time_start),
                   0)
    y = amp * np.sin(2 * np.pi * freq * (t[:, None] - time_start))
    
    # add all the waves to the combined wave
    y_combined = y.sum(axis=1)
    
    # create the frequency plot
    if not disable_freq_plot:
        fig_freq = go.Figure()
        for index in tqdm(range(wave_params.shape[0])):
            fig_freq.add_trace(go.Scatter(x=t, y=freq[:, index], mode='lines', name=f'Wave {index + 1}'))
        fig_freq.update_layout(title='Frequency vs Time', xaxis_title='Time (s)', yaxis_title='Frequency')
        fig_freq.show()

    # create the amplitude plot
    if not disable_amp_plot:
        fig_amp = go.Figure()
        for index in tqdm(range(wave_params.shape[0])):
            fig_amp.add_trace(go.Scatter(x=t, y=amp[:, index], mode='lines', name=f'Wave {index + 1}'))
        fig_amp.update_layout(title='Amplitude vs Time', xaxis_title='Time (s)', yaxis_title='Amplitude')
        fig_amp.show()

    # create the combined wave plot
    if not disable_combined_plot:
        fig_combined = go.Figure()
        for index in tqdm(range(wave_params.shape[0])):
            fig_combined.add_trace(go.Scatter3d(x=t, y=freq[:, index], z=amp[:, index], mode='lines', name=f'Wave {index + 1}'))
        fig_combined.update_layout(title='Amplitude vs Time and Frequency', scene=dict(xaxis_title='Time (s)', yaxis_title='Frequency', zaxis_title='Amplitude'))
        fig_combined.show()
    
    # create the wave plot
    if not disable_wave_plot:
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t, y=y_combined, mode='lines', name='Combined Wave'))
        fig_wave.update_layout(title='Waveform', xaxis_title='Time (s)', yaxis_title='Amplitude')
        fig_wave.show()

    # Return y_combined for further use
    return y_combined

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
