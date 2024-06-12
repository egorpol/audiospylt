import numpy as np
import librosa

def itakura_saito_distance(params, target_freqs, target_amps, waveforms, duration, sample_rate):
    """
    Compute the Itakura-Saito distance between the target spectrum and the generated spectrum.
    """
    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    max_val = np.max(np.abs(combined_signal))
    if max_val != 0:
        combined_signal /= max_val
    
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    power_spectrum = np.abs(fft_result) ** 2
    power_spectrum /= np.sum(power_spectrum)
    
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    target_power_spectrum = np.abs(target_spectrum) ** 2
    target_power_spectrum /= np.sum(target_power_spectrum)
    
    epsilon = 1e-10
    power_spectrum += epsilon
    target_power_spectrum += epsilon
    
    itakura_saito_distance = np.sum(target_power_spectrum * np.log(target_power_spectrum / power_spectrum) - target_power_spectrum + power_spectrum)
    
    return itakura_saito_distance

def cosine_similarity(params, target_freqs, target_amps, waveforms, duration, sample_rate):
    """
    Compute the cosine similarity between the target spectrum and the generated spectrum.
    """
    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    max_val = np.max(np.abs(combined_signal))
    if max_val != 0:
        combined_signal /= max_val
    
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    power_spectrum = np.abs(fft_result) ** 2
    power_spectrum /= np.sum(power_spectrum)
    
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    target_power_spectrum = np.abs(target_spectrum) ** 2
    target_power_spectrum /= np.sum(target_power_spectrum)
    
    epsilon = 1e-10
    power_spectrum += epsilon
    target_power_spectrum += epsilon
    
    cosine_similarity = np.dot(power_spectrum, target_power_spectrum) / (np.linalg.norm(power_spectrum) * np.linalg.norm(target_power_spectrum))
    
    return -cosine_similarity  # Negate because we want to minimize the distance

def spectral_convergence_distance(params, target_freqs, target_amps, waveforms, duration, sample_rate):

    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    combined_signal /= np.max(np.abs(combined_signal))
    
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    fft_magnitudes = np.abs(fft_result)
    
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    
    target_magnitudes = np.abs(target_spectrum)
    
    spectral_convergence = np.linalg.norm(fft_magnitudes - target_magnitudes) / np.linalg.norm(target_magnitudes)
    
    return spectral_convergence

def euclidean_distance(params, target_freqs, target_amps, waveforms, duration, sample_rate):
    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    combined_signal /= np.max(np.abs(combined_signal))
    
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    fft_magnitudes = np.abs(fft_result)
    
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    
    target_magnitudes = np.abs(target_spectrum)
    
    euclidean_dist = np.linalg.norm(fft_magnitudes - target_magnitudes)
    
    return euclidean_dist

def manhattan_distance(params, target_freqs, target_amps, waveforms, duration, sample_rate):
    """
    Compute the Manhattan distance between the target spectrum and the generated spectrum.
    """
    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    max_val = np.max(np.abs(combined_signal))
    if max_val != 0:
        combined_signal /= max_val
    
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    fft_magnitudes = np.abs(fft_result)
    
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    
    target_magnitudes = np.abs(target_spectrum)
    
    manhattan_dist = np.sum(np.abs(fft_magnitudes - target_magnitudes))
    
    return manhattan_dist

def kullback_leibler_divergence(params, target_freqs, target_amps, waveforms, duration, sample_rate):
    """
    Compute the Kullback-Leibler divergence between the target spectrum and the generated spectrum.
    """
    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    max_val = np.max(np.abs(combined_signal))
    if max_val != 0:
        combined_signal /= max_val
    
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    power_spectrum = np.abs(fft_result) ** 2
    power_spectrum /= np.sum(power_spectrum)
    
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    
    target_power_spectrum = np.abs(target_spectrum) ** 2
    target_power_spectrum /= np.sum(target_power_spectrum)
    
    epsilon = 1e-10
    power_spectrum += epsilon
    target_power_spectrum += epsilon
    
    kl_divergence = np.sum(target_power_spectrum * np.log(target_power_spectrum / power_spectrum))
    
    return kl_divergence

def pearson_correlation_coefficient(params, target_freqs, target_amps, waveforms, duration, sample_rate):
    """
    Compute the Pearson correlation coefficient between the target spectrum and the generated spectrum.
    """
    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    max_val = np.max(np.abs(combined_signal))
    if max_val != 0:
        combined_signal /= max_val
    
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    fft_magnitudes = np.abs(fft_result)
    
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    
    target_magnitudes = np.abs(target_spectrum)
    
    correlation_matrix = np.corrcoef(fft_magnitudes, target_magnitudes)
    pearson_corr = correlation_matrix[0, 1]
    
    return -pearson_corr  # Negate because we want to minimize the distance

def mfcc_distance(params, target_freqs, target_amps, waveforms, duration, sample_rate):
    """
    Compute the distance between the MFCCs of the target signal and the generated signal.
    """
    num_oscillators = len(params) // 3
    combined_signal = np.zeros(int(sample_rate * duration))
    
    # Generate the combined signal using the given parameters
    for i in range(num_oscillators):
        freq = params[i * 3]
        closest_freq_index = np.argmin(np.abs(target_freqs - freq))
        matched_freq = target_freqs[closest_freq_index]
        amp = params[i * 3 + 1]
        waveform_index = int(params[i * 3 + 2])
        waveform = waveforms[waveform_index]
        combined_signal += waveform(matched_freq, amp, duration, sample_rate)
    
    # Normalize the combined signal
    max_val = np.max(np.abs(combined_signal))
    if max_val != 0:
        combined_signal /= max_val
    
    # Calculate MFCCs for the generated signal
    generated_mfcc = librosa.feature.mfcc(y=combined_signal, sr=sample_rate, n_mfcc=20)
    generated_mfcc_mean = np.mean(generated_mfcc, axis=1)
    
    # Generate the target signal
    target_signal = np.zeros(int(sample_rate * duration))
    for freq, amp in zip(target_freqs, target_amps):
        target_signal += np.sin(2 * np.pi * freq * np.linspace(0, duration, int(sample_rate * duration))) * amp
    
    # Normalize the target signal
    max_val_target = np.max(np.abs(target_signal))
    if max_val_target != 0:
        target_signal /= max_val_target
    
    # Calculate MFCCs for the target signal
    target_mfcc = librosa.feature.mfcc(y=target_signal, sr=sample_rate, n_mfcc=20)
    target_mfcc_mean = np.mean(target_mfcc, axis=1)
    
    # Compute the distance between the MFCCs
    mfcc_dist = np.linalg.norm(generated_mfcc_mean - target_mfcc_mean)
    
    return mfcc_dist
