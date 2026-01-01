
import numpy as np
import sys
import os

# Ensure we can import the local package
sys.path.append(os.getcwd())

from audiospylt.audio_utils import plot_spectrogram

def debug_framing(N, n_fft, overlap, boundary="zeros_end", padded=True):
    sr = 1000  # Use 1kHz for easy math (1 sample = 1ms)
    y = np.ones(N) # Signal
    
    print(f"\n=== DEBUG: N={N}, n_fft={n_fft}, overlap={overlap} ===")
    
    # Run spectrogram (suppress plot)
    fig, info = plot_spectrogram(
        y, sr,
        n_fft=n_fft,
        overlap=overlap,
        boundary=boundary,
        padded=padded,
        time_reference="start",
        time_range="signal",
        show=False
    )
    
    times = info["times"]
    if len(times) == 0:
        print("  NO FRAMES GENERATED")
        return

    hop_length = int(n_fft * (1 - overlap))
    
    # "times" in info dict are shifted for Plotly centering
    # recover actual frame start times
    # plot_spectrogram shifts by hop_length / (2 * sr) for "start" reference
    times_for_plot = times
    frame_starts_sec = times_for_plot - (hop_length / (2.0 * sr))
    
    last_frame_start_time = frame_starts_sec[-1]
    last_frame_start_sample = int(round(last_frame_start_time * sr))
    
    last_frame_end_time = last_frame_start_time + n_fft/sr
    last_frame_end_sample = last_frame_start_sample + n_fft
    
    signal_duration = N/sr
    
    print(f"  Signal Length:       {N} samples ({signal_duration:.4f} s)")
    print(f"  Generated Frames:    {len(times)}")
    print(f"  Last Frame Start:    {last_frame_start_sample} samples ({last_frame_start_time:.4f} s)")
    print(f"  Last Frame End:      {last_frame_end_sample} samples ({last_frame_end_time:.4f} s)")
    
    # Analyze the last frame content
    valid_samples = N - last_frame_start_sample
    padding_samples = n_fft - valid_samples
    
    print(f"  Content of Last Frame:")
    print(f"    - Valid Signal:    {valid_samples} samples")
    print(f"    - Padding:         {padding_samples} samples")
    
    if valid_samples <= 0:
        print("  [ERROR] Last frame starts AFTER signal ends (Empty Frame)")
    elif padding_samples > 0:
        print(f"  [INFO]  Extended by {padding_samples} samples ({padding_samples/sr:.4f} s) to fill block")
    else:
        print("  [OK]    Perfect fit")

if __name__ == "__main__":
    # Test cases relevant to your issue
    print("Checking overlap=0 behavior:")
    debug_framing(N=2000, n_fft=2048, overlap=0) # N < n_fft
    debug_framing(N=2048, n_fft=2048, overlap=0) # Exact fit
    debug_framing(N=2049, n_fft=2048, overlap=0) # 1 sample over (should trigger new frame)

