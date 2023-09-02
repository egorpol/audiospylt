import requests
import numpy as np
import plotly.graph_objs as go
import os
import io
import librosa

def load_wav_from_source(wav_source):
    """Load a WAV file either from a local path or a URL."""
    if os.path.exists(wav_source):
        wav_filename = os.path.basename(wav_source)
        print(f"WAV file loaded from local path: {os.path.abspath(wav_source)}")
    else:
        try:
            response = requests.get(wav_source)
            response.raise_for_status()

            wav_buffer = io.BytesIO(response.content)
            wav_filename = wav_source.split('/')[-1]

            with open(wav_filename, 'wb') as f:
                f.write(wav_buffer.read())
        except requests.RequestException as e:
            raise ValueError(f"Failed to load WAV from URL. Error: {e}")
    return wav_filename

def load_audio_data(wav_filename, desired_sample_rate, convert_to_mono):
    """Load audio data, convert it to mono (if desired), and resample it (if a desired sample rate is provided)."""
    return librosa.load(wav_filename, sr=desired_sample_rate, mono=convert_to_mono)

def display_audio_properties(audio_data, sample_rate, wav_source):
    """Display the properties of the loaded audio."""
    num_channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
    print(f"Number of audio channels: {num_channels}")
    print(f"Sampling rate: {sample_rate} Hz")
    print(f"WAV file loaded from {wav_source}")

    return num_channels

def plot_waveform(audio_data, sample_rate, wav_filename):
    """Plot the waveform of the audio."""
    duration = len(audio_data) / float(sample_rate)
    print(f"Total duration: {duration:.3f} seconds")

    time_axis = np.linspace(0, duration, len(audio_data))
    fig = go.Figure(go.Scatter(x=time_axis, y=audio_data))
    fig.update_layout(title=f'Waveform of {wav_filename}', xaxis_title='Time (seconds)', yaxis_title='Amplitude')
    fig.show()

    return duration
       
def trim_and_fade_audio(audio_data, sample_rate, num_channels, duration, wav_source, 
                        start_time=0.4, end_time=1.6, add_fades=True, fade_in_duration=0.2, 
                        fade_out_duration=0.3, fade_in_exponent=0.8, fade_out_exponent=1.5):
    
    # Ensure start_time and end_time are within the duration of the audio
    start_time = max(0, start_time)
    end_time = min(duration, end_time)

    # Calculate start and end frame indices
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)

    # Cut the audio data
    cut_audio_data = audio_data[start_frame:end_frame].copy()

    # Adjust the time axis for the cut waveform
    cut_duration = end_time - start_time
    cut_time_axis = np.linspace(0, cut_duration, len(cut_audio_data))

    if add_fades:
        # Calculate the number of frames for the fade-in and fade-out
        fade_in_frames = int(fade_in_duration * sample_rate)
        fade_out_frames = int(fade_out_duration * sample_rate)

        # Create exponential fade-in and fade-out curves
        fade_in_curve = np.linspace(0, 1, fade_in_frames) ** fade_in_exponent
        fade_out_curve = np.linspace(1, 0, fade_out_frames) ** fade_out_exponent

        # Apply the fade-in and fade-out curves to the cut audio data
        if num_channels > 1:
            for channel in range(num_channels):
                channel_data = cut_audio_data[channel::num_channels]
                channel_data[:fade_in_frames] *= fade_in_curve
                channel_data[-fade_out_frames:] *= fade_out_curve
        else:
            cut_audio_data[:fade_in_frames] *= fade_in_curve
            cut_audio_data[-fade_out_frames:] *= fade_out_curve

        # Plot the waveform with fade-in and fade-out applied
        if num_channels > 1:
            overlay_fig = go.Figure()
            for channel in range(num_channels):
                channel_data = cut_audio_data[channel::num_channels]
                overlay_fig.add_trace(go.Scatter(x=cut_time_axis, y=channel_data, name=f'Channel {channel+1}', yaxis='y1'))
        else:
            overlay_fig = go.Figure(go.Scatter(x=cut_time_axis, y=cut_audio_data, name='Waveform', yaxis='y1'))
        overlay_fig.update_layout(title=f'Waveform with Fade-in and Fade-out {os.path.basename(wav_source)}', xaxis_title='Time (seconds)', yaxis_title='Amplitude', showlegend=False)

        # Add fade-in and fade-out curve traces to the plot
        fade_in_trace = go.Scatter(x=cut_time_axis[:fade_in_frames],
                                   y=fade_in_curve,
                                   name='Fade-in Curve',
                                   yaxis='y2',
                                   fill='tozeroy',
                                   fillcolor='rgba(255,0,0,0.2)',
                                   line=dict(color='rgba(255,0,0,0.8)'))

        fade_out_trace = go.Scatter(x=cut_time_axis[-fade_out_frames:],
                                     y=fade_out_curve,
                                     name='Fade-out Curve',
                                     yaxis='y2',
                                     fill='tozeroy',
                                     fillcolor='rgba(255,0,0,0.2)',
                                     line=dict(color='rgba(255,0,0,0.8)'))

        overlay_fig.add_trace(fade_in_trace)
        overlay_fig.add_trace(fade_out_trace)

        # Configure the layout for dual y-axes
        overlay_fig.update_layout(
            yaxis1=dict(
                title='Amplitude',
                side='left',
                showgrid=False,
                zeroline=False
            ),
            yaxis2=dict(
                title='Fade-in and Fade-out',
                range=[0, 1],
                side='right',
                overlaying='y',
                showgrid=False,
                zeroline=False
            )
        )

        # Display the waveform plot with fade-in and fade-out applied
        overlay_fig.show()

    else:
        # Plot the cut waveform only
        if num_channels > 1:
            cut_fig = go.Figure()
            for channel in range(num_channels):
                channel_data = cut_audio_data[channel::num_channels]
                cut_fig.add_trace(go.Scatter(x=cut_time_axis, y=channel_data, name=f'Channel {channel+1}'))
        else:
            cut_fig = go.Figure(go.Scatter(x=cut_time_axis, y=cut_audio_data))
        cut_fig.update_layout(title=f'Cut Waveform of {os.path.basename(wav_source)}', xaxis_title='Time (seconds)', yaxis_title='Amplitude')
        # Display the cut waveform plot
        cut_fig.show()

    return cut_audio_data, cut_duration

