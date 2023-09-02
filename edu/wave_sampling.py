import numpy as np
import plotly.graph_objects as go

def plot_waveforms(n=0.05, t_max=2.0, selected_waveforms=['sine1', 'sine2'], amp1=1.0, freq1=9.0, phase1=0*np.pi/2, amp2=1.0, freq2=11.0, phase2=2*np.pi/2):
    # Calculate time vector
    t = np.arange(0, t_max, n)

    # Calculate sine wave values
    y1 = amp1 * np.sin(2*np.pi*freq1*t + phase1)
    y2 = amp2 * np.sin(2*np.pi*freq2*t + phase2)
    y_am = (amp1 + y2) * np.sin(2*np.pi*freq1*t + phase1)
    y_fm = amp1 * np.sin(2*np.pi*(freq1 + y2)*t + phase1)
    y_sum = y1 + y2

    # Create the Plotly figure for waveforms
    fig_waveforms = go.Figure()

    # Plot the selected waveforms
    waveform_data = {'sine1': y1, 'sine2': y2, 'am': y_am, 'fm': y_fm, 'sum': y_sum}
    waveform_max_min = {'max': -np.inf, 'min': np.inf}
    for waveform in selected_waveforms:
        data = waveform_data.get(waveform, [])
        fig_waveforms.add_trace(go.Scatter(x=t, y=data, mode='lines', name=waveform))
        waveform_max_min['max'] = max(waveform_max_min['max'], np.max(data))
        waveform_max_min['min'] = min(waveform_max_min['min'], np.min(data))

    # Add vertical dotted lines to waveform plot
    range_25_percent = 0.25 * (waveform_max_min['max'] - waveform_max_min['min'])
    for x in np.arange(0, t_max + n, n):
        fig_waveforms.add_shape(type="line", x0=x, y0=waveform_max_min['min'] - range_25_percent,
                                x1=x, y1=waveform_max_min['max'] + range_25_percent,
                                line=dict(color="Black", width=1, dash="dot"))

    # Calculate and print sampling rate and Nyquist frequency
    sampling_rate = 1 / n
    nyquist_freq = sampling_rate / 2
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Nyquist frequency: {nyquist_freq} Hz")

    # Display the waveform plot
    fig_waveforms.update_layout(title='Sine Waves', xaxis_title='Time (s)', yaxis_title='Amplitude',
                                autosize=False, width=800, height=600, legend=dict(x=0, y=1))
    fig_waveforms.show()

    # Create the DFT Magnitude and Phase Plots
    fig_dft_magnitude = go.Figure()
    fig_dft_phase = go.Figure()

    # Find the maximum amplitude and minimum phase of the DFT across selected waveforms
    max_dft_amp = 0
    min_dft_phase = np.inf

    for waveform in selected_waveforms:
        y = waveform_data.get(waveform, [])
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(t), d=n)
        inds = np.argsort(freqs)
        freqs = freqs[inds]
        Y = Y[inds]

        max_dft_amp = max(max_dft_amp, np.max(np.abs(Y)))
        min_dft_phase = min(min_dft_phase, np.min(np.angle(Y)))

        # Add to Magnitude and Phase plots of the DFT
        fig_dft_magnitude.add_trace(go.Scatter(x=freqs, y=np.abs(Y), mode='lines', name=waveform))
        fig_dft_phase.add_trace(go.Scatter(x=freqs, y=np.angle(Y), mode='lines', name=waveform))

    # Add vertical dotted lines to DFT plots
    range_25_percent_magnitude = 0.25 * max_dft_amp
    range_25_percent_phase = 0.25 * (np.pi - min_dft_phase)
    for i in range(len(t)):
        fig_dft_magnitude.add_shape(type="line", x0=freqs[i], y0=0, x1=freqs[i], y1=max_dft_amp + range_25_percent_magnitude, line=dict(color="Black", width=1, dash="dot"))
        fig_dft_phase.add_shape(type="line", x0=freqs[i], y0=min_dft_phase - range_25_percent_phase, x1=freqs[i], y1=np.pi + range_25_percent_phase, line=dict(color="Black", width=1, dash="dot"))

    # Update layout for DFT Magnitude
    fig_dft_magnitude.update_layout(title='Magnitude Spectrum of the DFT', xaxis_title='Frequency (Hz)', yaxis_title='Magnitude',
                                    autosize=False, width=800, height=600)

    # Update layout for DFT Phase
    fig_dft_phase.update_layout(title='Phase Spectrum of the DFT', xaxis_title='Frequency (Hz)', yaxis_title='Phase (radians)',
                                autosize=False, width=800, height=600)

    # Print total number of available sample points
    print(f"Total number of available sample points: {len(t)}")

    # Display the DFT Magnitude plot
    fig_dft_magnitude.show()

    # Display the DFT Phase plot
    fig_dft_phase.show()
