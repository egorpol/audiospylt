import numpy as np
from scipy.signal import get_window
import plotly.graph_objects as go

def generate_waveforms(config):
    """
    Generates waveform data based on the provided configuration.
    """
    # Extract configuration parameters
    n = config['n']
    t_max = config['t_max']
    amp1 = config['amp1']
    freq1 = config['freq1']
    phase1 = config['phase1']
    amp2 = config['amp2']
    freq2 = config['freq2']
    phase2 = config['phase2']
    index_am = config['index_am']
    index_fm = config['index_fm']
    
    # Calculate time vector and initialize waveform data dictionary
    t = np.arange(0, t_max, n)
    waveform_data = {
        't': t
    }
    
    # Generate waveforms
    waveform_data['sine1'] = amp1 * np.sin(2*np.pi*freq1*t + phase1)
    waveform_data['sine2'] = amp2 * np.sin(2*np.pi*freq2*t + phase2)
    waveform_data['am'] = (amp1 + index_am * waveform_data['sine2']) * np.sin(2*np.pi*freq1*t + phase1)
    waveform_data['fm'] = amp1 * np.sin(2*np.pi*(freq1 + index_fm * waveform_data['sine2'])*t + phase1)
    waveform_data['sum'] = waveform_data['sine1'] + waveform_data['sine2']

    # Generate combined sine wave
    waveform_data['comb'] = np.zeros_like(t)
    half_period = int(t_max / (2 * n))
    for i in range(0, len(t), half_period):
        if (i // half_period) % 2 == 0:
            waveform_data['comb'][i:i+half_period] = waveform_data['sine1'][i:i+half_period]
        else:
            waveform_data['comb'][i:i+half_period] = waveform_data['sine2'][i:i+half_period]
    
    return waveform_data

def apply_window_to_waveforms(waveform_data, window_type):
    """
    Applies window function to the waveform data.
    """
    t = waveform_data['t']
    window = get_window(window_type, len(t))
    for waveform, data in waveform_data.items():
        if waveform != 't':
            waveform_data[waveform] *= window
    return waveform_data

def plot_waveforms(waveform_data, config):
    """
    Plots waveform data based on the provided configuration.
    """
    t = waveform_data['t']
    
    # Create the Plotly figure for waveforms
    fig_waveforms = go.Figure()

    waveform_max_min = {'max': np.amax([waveform_data[waveform] for waveform in config['selected_waveforms']]),
                        'min': np.amin([waveform_data[waveform] for waveform in config['selected_waveforms']])}

    # Plot the selected waveforms
    for waveform in config['selected_waveforms']:
        fig_waveforms.add_trace(go.Scatter(x=t, y=waveform_data[waveform], mode='lines', name=waveform))

    # Plot the window function if enabled
    if config['apply_window']:
        window = get_window(config['window_type'], len(t))
        fig_waveforms.add_trace(go.Scatter(x=t, y=window, mode='lines', name='window - ' + config['window_type'], line=dict(dash='dash')))

    # Add vertical dotted lines to waveform plot
    if config['add_dotted_lines']:
        range_25_percent = 0.25 * (waveform_max_min['max'] - waveform_max_min['min'])
        num_lines_waveform = int(config['t_max'] / config['n']) + 1
        for x in np.linspace(0, config['t_max'], num_lines_waveform):
            fig_waveforms.add_shape(type="line", x0=x, y0=waveform_max_min['min'] - range_25_percent,
                                    x1=x, y1=waveform_max_min['max'] + range_25_percent,
                                    line=dict(color="Black", width=1, dash="dot"))

    fig_waveforms.update_layout(title='Sine Waves', xaxis_title='Time (s)', yaxis_title='Amplitude', autosize=False, width=800, height=600)
    fig_waveforms.show()

    # Create the DFT Magnitude and Phase Plots
    fig_dft_magnitude = go.Figure()
    fig_dft_phase = go.Figure()

    max_dft_amp = 0
    min_dft_phase = np.inf

    for waveform in config['selected_waveforms']:
        y = waveform_data[waveform]

        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(t), d=config['n'])
        inds = np.argsort(freqs)
        freqs = freqs[inds]
        Y = Y[inds]

        max_dft_amp = max(max_dft_amp, np.max(np.abs(Y)))
        min_dft_phase = min(min_dft_phase, np.min(np.angle(Y)))

        fig_dft_magnitude.add_trace(go.Scatter(x=freqs, y=np.abs(Y), mode='lines', name=waveform))
        fig_dft_phase.add_trace(go.Scatter(x=freqs, y=np.angle(Y), mode='lines', name=waveform))

    # Add vertical dotted lines to DFT plots
    if config['add_dotted_lines']:
        range_25_percent_magnitude = 0.25 * max_dft_amp
        range_25_percent_phase = 0.25 * (np.pi - min_dft_phase)
        num_lines_dft = int(config['t_max'] / config['n'])
        for x in np.linspace(np.min(freqs), np.max(freqs), num_lines_dft):
            fig_dft_magnitude.add_shape(type="line", x0=x, y0=0, x1=x, y1=max_dft_amp + range_25_percent_magnitude, line=dict(color="Black", width=1, dash="dot"))
            fig_dft_phase.add_shape(type="line", x0=x, y0=min_dft_phase - range_25_percent_phase, x1=x, y1=np.pi + range_25_percent_phase, line=dict(color="Black", width=1, dash="dot"))

    # Update layout for DFT Magnitude
    fig_dft_magnitude.update_layout(title='Magnitude Spectrum of the DFT', xaxis_title='Frequency (Hz)', yaxis_title='Magnitude', autosize=False, width=800, height=600)

    # Update layout for DFT Phase
    fig_dft_phase.update_layout(title='Phase Spectrum of the DFT', xaxis_title='Frequency (Hz)', yaxis_title='Phase (radians)', autosize=False, width=800, height=600)

    # Calculate and print sampling rate and Nyquist frequency
    sampling_rate = 1 / config['n']
    nyquist_freq = sampling_rate / 2
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Nyquist frequency: {nyquist_freq} Hz")
    print(f"Total number of available sample points: {len(t)}")   

    fig_dft_magnitude.show()
    fig_dft_phase.show()
