import numpy as np

def sine_wave(frequency, amplitude, duration, sample_rate):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    return amplitude * np.sin(2 * np.pi * frequency * time_vector)

def square_wave(frequency, amplitude, duration, sample_rate):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * time_vector))

def triangle_wave(frequency, amplitude, duration, sample_rate):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    return amplitude * (2 * np.abs(np.arcsin(np.sin(2 * np.pi * frequency * time_vector))) / np.pi)

def sawtooth_wave(frequency, amplitude, duration, sample_rate):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    return amplitude * (2 * (time_vector * frequency - np.floor(0.5 + time_vector * frequency)))

def noise_wave(frequency, amplitude, duration, sample_rate):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    return amplitude * np.random.normal(0, 1, size=len(time_vector))
