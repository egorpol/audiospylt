import numpy as np
import os
from scipy.signal import resample
from datetime import datetime
import soundfile as sf

def scale_samples(samples, bit_rate):
    if bit_rate == 16:
        return np.int16(samples * (2**15 - 1))
    elif bit_rate == 24:
        return (samples * (2**23 - 1)).astype(np.int32) << 8
    else:
        raise ValueError(f"Unsupported bit rate: {bit_rate}")

def generate_wave_file(y_combined, fs_initial, fs_target_name='44.1kHz', bit_rate=16, custom_filename=None):
    # dictionary of common sampling rates
    fs_target_dict = {
        '44.1kHz': 44100,
        '48kHz': 48000,
        '88.2kHz': 88200,
        '96kHz': 96000,
        '192kHz': 192000
    }

    # get the target sampling rate from the dictionary
    if fs_target_name in fs_target_dict:
        fs_target = fs_target_dict[fs_target_name]
    else:
        raise ValueError(f"Unsupported sampling rate: {fs_target_name}")

    # calculate the number of samples in the resampled signal
    num_samples_resampled = int(len(y_combined) * fs_target / fs_initial)

    # resample the signal to the target sample rate
    y_resampled = resample(y_combined, num_samples_resampled)

    # normalize the resampled signal to the range of -1 to 1
    y_normalized = y_resampled / np.max(np.abs(y_resampled))

    # scale the resampled signal to the desired bit rate range
    y_scaled = scale_samples(y_normalized, bit_rate)

    # get the current timestamp
    timestamp = datetime.now().strftime("%H_%M_%S")

    # set the output file name
    if custom_filename is None:
        output_filename = f"generated_wave_file_{fs_target_name}_{bit_rate}bit_{timestamp}.wav"
    else:
        output_filename = custom_filename

    # write the resampled waveform to a wave audio file
    sf.write(output_filename, y_scaled, fs_target, subtype=f'PCM_{bit_rate}')

    # get the current working directory
    current_dir = os.getcwd()

    # construct the file path
    file_path = os.path.join(current_dir, output_filename)

    # get the current timestamp for log
    timestamp_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # print a message about the wave file being saved successfully with a timestamp
    print(f"[{timestamp_log}] {bit_rate}-bit wave file with {fs_target_name} sampling rate saved successfully to: {file_path}")
