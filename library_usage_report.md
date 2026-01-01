# audiospylt: library usage per function (AST-derived)

Notes:
- Reported libraries are **only those referenced** in each function body (not just imported at module level).
- `internal` includes `audiospylt` and same-folder modules imported without the package prefix (e.g., `audio_utils`).
- Dynamic imports / reflection-based usage will not be detected.

## Aggregate: external libraries -> functions

### IPython
- `audiospylt/generate_wave_file.py` -> `render_audio()`
- `audiospylt/mei.py` -> `_render_and_save_mei()`
- `audiospylt/mei.py` -> `process_and_visualize()`
- `audiospylt/mei.py` -> `process_temporal_chords()`
- `audiospylt/mei_old.py` -> `process_and_visualize()`

### librosa
- `audiospylt/audio_utils.py` -> `load_audio_data()`
- `audiospylt/audio_utils.py` -> `plot_spectrogram()`
- `audiospylt/ssm.py` -> `compute_features()`
- `audiospylt/ssm.py` -> `main()`

### numpy
- `audiospylt/audio_utils.py` -> `ensure_finite_audio()`
- `audiospylt/audio_utils.py` -> `load_audio_sample()`
- `audiospylt/audio_utils.py` -> `plot_spectrogram()`
- `audiospylt/audio_utils.py` -> `plot_spectrogram._default_tick_freqs_hz()`
- `audiospylt/audio_utils.py` -> `plot_spectrogram._mixed_warp()`
- `audiospylt/audio_utils.py` -> `plot_waveform()`
- `audiospylt/audio_utils.py` -> `trim_and_fade_audio()`
- `audiospylt/dft_analysis.py` -> `_default_tick_freqs_hz()`
- `audiospylt/dft_analysis.py` -> `_hz_to_mel()`
- `audiospylt/dft_analysis.py` -> `_mixed_warp()`
- `audiospylt/dft_analysis.py` -> `_mixed_warp_values()`
- `audiospylt/dft_analysis.py` -> `analyze_signal()`
- `audiospylt/dft_analysis.py` -> `compute_fft()`
- `audiospylt/dft_analysis.py` -> `filter_peaks()`
- `audiospylt/dft_analysis.py` -> `plot_spectrum()`
- `audiospylt/generate_wave_file.py` -> `ensure_finite_audio()`
- `audiospylt/generate_wave_file.py` -> `generate_wave_file()`
- `audiospylt/generate_wave_file.py` -> `scale_samples()`
- `audiospylt/mei.py` -> `convert_frequencies()`
- `audiospylt/mei_old.py` -> `convert_frequencies()`
- `audiospylt/multiplotter.py` -> `_default_tick_freqs_hz()`
- `audiospylt/multiplotter.py` -> `_hz_to_mel()`
- `audiospylt/multiplotter.py` -> `_mixed_warp()`
- `audiospylt/multiplotter.py` -> `_mixed_warp_values()`
- `audiospylt/multiplotter.py` -> `plot_equalizer_bars()`
- `audiospylt/multiplotter.py` -> `plot_scatter()`
- `audiospylt/multiplotter.py` -> `plot_scatter_binned()`
- `audiospylt/plot_wave.py` -> `crossfade_adjacent_events()`
- `audiospylt/plot_wave.py` -> `estimate_sampling_frequency_and_time_vector()`
- `audiospylt/plot_wave.py` -> `plot_waves()`
- `audiospylt/plot_wave.py` -> `plot_waves._apply_fades()`
- `audiospylt/plot_wave.py` -> `plot_waves._ramp()`
- `audiospylt/ssm.py` -> `_load_audio_mono()`
- `audiospylt/ssm.py` -> `compute_features()`
- `audiospylt/ssm.py` -> `compute_features._bin_mean()`
- `audiospylt/ssm.py` -> `compute_ssm()`
- `audiospylt/ssm.py` -> `diagonal_smooth()`
- `audiospylt/ssm.py` -> `main()`
- `audiospylt/ssm.py` -> `normalize_features()`
- `audiospylt/ssm.py` -> `plot_heatmap()`
- `audiospylt/waveform_utils.py` -> `generate_waveforms()`
- `audiospylt/waveform_utils.py` -> `plot_waveforms()`

### pandas
- `audiospylt/dft_analysis.py` -> `analyze_signal()`
- `audiospylt/io_utils.py` -> `events_df_from_peaks_by_interval()`
- `audiospylt/mei.py` -> `_load_dataframe_from_input()`
- `audiospylt/mei.py` -> `convert_frequencies()`
- `audiospylt/mei.py` -> `process_and_visualize()`
- `audiospylt/mei.py` -> `process_temporal_chords()`
- `audiospylt/mei_old.py` -> `convert_frequencies()`
- `audiospylt/mei_old.py` -> `process_and_visualize()`
- `audiospylt/multiplotter.py` -> `_load_and_prepare_data()`
- `audiospylt/multiplotter.py` -> `plot_combined()`
- `audiospylt/multiplotter.py` -> `plot_combined_3d()`
- `audiospylt/multiplotter.py` -> `plot_equalizer_bars()`
- `audiospylt/multiplotter.py` -> `plot_scatter()`
- `audiospylt/multiplotter.py` -> `plot_scatter_binned()`
- `audiospylt/plot_wave.py` -> `crossfade_adjacent_events()`
- `audiospylt/plot_wave.py` -> `estimate_sampling_frequency_and_time_vector()`
- `audiospylt/plot_wave.py` -> `plot_waves()`

### plotly
- `audiospylt/audio_utils.py` -> `plot_spectrogram()`
- `audiospylt/audio_utils.py` -> `plot_waveform()`
- `audiospylt/audio_utils.py` -> `trim_and_fade_audio()`
- `audiospylt/dft_analysis.py` -> `plot_spectrum()`
- `audiospylt/multiplotter.py` -> `plot_combined()`
- `audiospylt/multiplotter.py` -> `plot_combined_3d()`
- `audiospylt/multiplotter.py` -> `plot_equalizer_bars()`
- `audiospylt/multiplotter.py` -> `plot_scatter()`
- `audiospylt/multiplotter.py` -> `plot_scatter_binned()`
- `audiospylt/plot_wave.py` -> `plot_waves()`
- `audiospylt/ssm.py` -> `plot_heatmap()`
- `audiospylt/waveform_utils.py` -> `plot_waveforms()`

### requests
- `audiospylt/audio_utils.py` -> `load_wav_from_source()`

### scipy
- `audiospylt/audio_utils.py` -> `plot_spectrogram()`
- `audiospylt/dft_analysis.py` -> `apply_window()`
- `audiospylt/dft_analysis.py` -> `filter_peaks()`
- `audiospylt/generate_wave_file.py` -> `generate_wave_file()`
- `audiospylt/ssm.py` -> `diagonal_smooth()`
- `audiospylt/waveform_utils.py` -> `apply_window_to_waveforms()`
- `audiospylt/waveform_utils.py` -> `plot_waveforms()`

### sklearn
- `audiospylt/ssm.py` -> `compute_ssm()`
- `audiospylt/ssm.py` -> `normalize_features()`

### soundfile
- `audiospylt/generate_wave_file.py` -> `generate_wave_file()`

### verovio
- `audiospylt/mei.py` -> `_render_and_save_mei()`
- `audiospylt/mei_old.py` -> `process_and_visualize()`

## Aggregate: stdlib modules -> functions

### datetime
- `audiospylt/generate_wave_file.py` -> `_build_output_filename()`
- `audiospylt/generate_wave_file.py` -> `generate_wave_file()`
- `audiospylt/io_utils.py` -> `save_df_tsv()`

### io
- `audiospylt/audio_utils.py` -> `load_wav_from_source()`

### logging
- `audiospylt/mei.py` -> `_add_chord_content_to_measure()`
- `audiospylt/mei.py` -> `_add_sequence_content_to_section()`
- `audiospylt/mei.py` -> `_load_dataframe_from_input()`
- `audiospylt/mei.py` -> `_render_and_save_mei()`
- `audiospylt/mei.py` -> `convert_frequencies()`
- `audiospylt/mei.py` -> `create_mei_string()`
- `audiospylt/mei.py` -> `create_temporal_mei_string()`
- `audiospylt/mei.py` -> `pitch_name_and_deviation_to_step_octave_alter()`
- `audiospylt/mei.py` -> `process_and_visualize()`
- `audiospylt/mei.py` -> `process_temporal_chords()`

### os
- `audiospylt/audio_utils.py` -> `load_wav_from_source()`
- `audiospylt/audio_utils.py` -> `trim_and_fade_audio()`
- `audiospylt/generate_wave_file.py` -> `generate_wave_file()`
- `audiospylt/io_utils.py` -> `save_df_tsv()`
- `audiospylt/ssm.py` -> `_load_audio_mono()`

### pathlib
- `audiospylt/generate_wave_file.py` -> `_build_output_filename()`

### random
- `audiospylt/multiplotter.py` -> `_color_palette_generator()`

### typing
- `audiospylt/io_utils.py` -> `events_df_from_peaks_by_interval()`
- `audiospylt/mei.py` -> `_add_chord_content_to_measure()`
- `audiospylt/mei.py` -> `_add_score_def_to_score()`
- `audiospylt/mei.py` -> `_add_sequence_content_to_section()`
- `audiospylt/mei.py` -> `_create_staff_defs_and_map()`
- `audiospylt/mei.py` -> `_init_mei_structure_elements()`
- `audiospylt/mei.py` -> `_load_dataframe_from_input()`
- `audiospylt/mei.py` -> `_render_and_save_mei()`
- `audiospylt/mei.py` -> `convert_frequencies()`
- `audiospylt/mei.py` -> `create_mei_string()`
- `audiospylt/mei.py` -> `create_temporal_mei_string()`
- `audiospylt/mei.py` -> `map_alteration_to_mei_accid()`
- `audiospylt/mei.py` -> `pitch_name_and_deviation_to_step_octave_alter()`
- `audiospylt/mei.py` -> `process_and_visualize()`
- `audiospylt/mei.py` -> `process_temporal_chords()`
- `audiospylt/mei_old.py` -> `convert_frequencies()`
- `audiospylt/mei_old.py` -> `pitch_name_and_cent_deviation_to_step_octave_alter()`
- `audiospylt/multiplotter.py` -> `_color_palette_generator()`
- `audiospylt/multiplotter.py` -> `_load_and_prepare_data()`
- `audiospylt/multiplotter.py` -> `plot_combined()`
- `audiospylt/multiplotter.py` -> `plot_combined_3d()`
- `audiospylt/multiplotter.py` -> `plot_equalizer_bars()`
- `audiospylt/multiplotter.py` -> `plot_scatter()`
- `audiospylt/multiplotter.py` -> `plot_scatter_binned()`
- `audiospylt/plot_wave.py` -> `crossfade_adjacent_events()`
- `audiospylt/plot_wave.py` -> `estimate_sampling_frequency_and_time_vector()`
- `audiospylt/plot_wave.py` -> `plot_waves()`
- `audiospylt/plotting.py` -> `show_plotly()`
- `audiospylt/ssm.py` -> `_load_audio_mono()`
- `audiospylt/ssm.py` -> `compute_features()`
- `audiospylt/ssm.py` -> `main()`
- `audiospylt/ssm.py` -> `plot_heatmap()`
- `audiospylt/ssm.py` -> `run_notebook()`

### xml
- `audiospylt/mei.py` -> `_add_chord_content_to_measure()`
- `audiospylt/mei.py` -> `_add_score_def_to_score()`
- `audiospylt/mei.py` -> `_add_sequence_content_to_section()`
- `audiospylt/mei.py` -> `_init_mei_structure_elements()`
- `audiospylt/mei.py` -> `_render_and_save_mei()`
- `audiospylt/mei.py` -> `create_mei_string()`
- `audiospylt/mei.py` -> `create_temporal_mei_string()`
- `audiospylt/mei_old.py` -> `create_mei_element()`

## Aggregate: internal modules -> functions

### audio_utils
- `audiospylt/plot_wave.py` -> `plot_waves()`

### audiospylt
- `audiospylt/dft_analysis.py` -> `plot_spectrum()`

## Per file: function -> libraries

### audiospylt/__init__.py

_No functions detected._

### audiospylt/audio_utils.py

- `_infer_num_channels()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `_infer_num_samples()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `display_audio_properties()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `ensure_finite_audio()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `load_audio_data()`
  - external: `librosa`
  - stdlib: _none_
  - internal: _none_
- `load_audio_sample()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `load_wav_from_source()`
  - external: `requests`
  - stdlib: `io`, `os`
  - internal: _none_
- `plot_spectrogram()`
  - external: `librosa`, `numpy`, `plotly`, `scipy`
  - stdlib: _none_
  - internal: _none_
- `plot_spectrogram._default_tick_freqs_hz()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `plot_spectrogram._format_hz()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `plot_spectrogram._mixed_warp()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `plot_waveform()`
  - external: `numpy`, `plotly`
  - stdlib: _none_
  - internal: _none_
- `trim_and_fade_audio()`
  - external: `numpy`, `plotly`
  - stdlib: `os`
  - internal: _none_

### audiospylt/dft_analysis.py

- `_default_tick_freqs_hz()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `_format_hz()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `_hz_to_mel()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `_mixed_warp()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `_mixed_warp_values()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `analyze_signal()`
  - external: `numpy`, `pandas`
  - stdlib: _none_
  - internal: _none_
- `apply_window()`
  - external: `scipy`
  - stdlib: _none_
  - internal: _none_
- `compute_fft()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `filter_peaks()`
  - external: `numpy`, `scipy`
  - stdlib: _none_
  - internal: _none_
- `plot_spectrum()`
  - external: `numpy`, `plotly`
  - stdlib: _none_
  - internal: `audiospylt`
- `plot_spectrum._map_freq_for_axis()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `plot_spectrum._map_freq_for_axis_range()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_

### audiospylt/generate_wave_file.py

- `_build_output_filename()`
  - external: _none_
  - stdlib: `datetime`, `pathlib`
  - internal: _none_
- `_sanitize_filename()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `ensure_finite_audio()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `generate_wave_file()`
  - external: `numpy`, `scipy`, `soundfile`
  - stdlib: `datetime`, `os`
  - internal: _none_
- `render_audio()`
  - external: `IPython`
  - stdlib: _none_
  - internal: _none_
- `scale_samples()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_

### audiospylt/io_utils.py

- `events_df_from_peaks_by_interval()`
  - external: `pandas`
  - stdlib: `typing`
  - internal: _none_
- `save_df_tsv()`
  - external: _none_
  - stdlib: `datetime`, `os`
  - internal: _none_

### audiospylt/mei.py

- `_add_chord_content_to_measure()`
  - external: _none_
  - stdlib: `logging`, `typing`, `xml`
  - internal: _none_
- `_add_score_def_to_score()`
  - external: _none_
  - stdlib: `typing`, `xml`
  - internal: _none_
- `_add_sequence_content_to_section()`
  - external: _none_
  - stdlib: `logging`, `typing`, `xml`
  - internal: _none_
- `_create_staff_defs_and_map()`
  - external: _none_
  - stdlib: `typing`
  - internal: _none_
- `_init_mei_structure_elements()`
  - external: _none_
  - stdlib: `typing`, `xml`
  - internal: _none_
- `_load_dataframe_from_input()`
  - external: `pandas`
  - stdlib: `logging`, `typing`
  - internal: _none_
- `_render_and_save_mei()`
  - external: `IPython`, `verovio`
  - stdlib: `logging`, `typing`, `xml`
  - internal: _none_
- `acoustical_pitch_name()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `convert_frequencies()`
  - external: `numpy`, `pandas`
  - stdlib: `logging`, `typing`
  - internal: _none_
- `create_mei_string()`
  - external: _none_
  - stdlib: `logging`, `typing`, `xml`
  - internal: _none_
- `create_temporal_mei_string()`
  - external: _none_
  - stdlib: `logging`, `typing`, `xml`
  - internal: _none_
- `map_alteration_to_mei_accid()`
  - external: _none_
  - stdlib: `typing`
  - internal: _none_
- `midi_cent_value()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `midi_value()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `pitch_name_and_deviation_to_step_octave_alter()`
  - external: _none_
  - stdlib: `logging`, `typing`
  - internal: _none_
- `process_and_visualize()`
  - external: `IPython`, `pandas`
  - stdlib: `logging`, `typing`
  - internal: _none_
- `process_temporal_chords()`
  - external: `IPython`, `pandas`
  - stdlib: `logging`, `typing`
  - internal: _none_

### audiospylt/mei_old.py

- `acoustical_pitch_name()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `convert_frequencies()`
  - external: `numpy`, `pandas`
  - stdlib: `typing`
  - internal: _none_
- `create_mei_element()`
  - external: _none_
  - stdlib: `xml`
  - internal: _none_
- `map_alteration_to_mei_accid()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `midi_cent_value()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `midi_value()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `pitch_name_and_cent_deviation_to_step_octave_alter()`
  - external: _none_
  - stdlib: `typing`
  - internal: _none_
- `process_and_visualize()`
  - external: `IPython`, `pandas`, `verovio`
  - stdlib: _none_
  - internal: _none_

### audiospylt/multiplotter.py

- `_color_palette_generator()`
  - external: _none_
  - stdlib: `random`, `typing`
  - internal: _none_
- `_default_tick_freqs_hz()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `_format_hz()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `_hz_to_mel()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `_load_and_prepare_data()`
  - external: `pandas`
  - stdlib: `typing`
  - internal: _none_
- `_mixed_warp()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `_mixed_warp_values()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `plot_combined()`
  - external: `pandas`, `plotly`
  - stdlib: `typing`
  - internal: _none_
- `plot_combined_3d()`
  - external: `pandas`, `plotly`
  - stdlib: `typing`
  - internal: _none_
- `plot_equalizer_bars()`
  - external: `numpy`, `pandas`, `plotly`
  - stdlib: `typing`
  - internal: _none_
- `plot_scatter()`
  - external: `numpy`, `pandas`, `plotly`
  - stdlib: `typing`
  - internal: _none_
- `plot_scatter_binned()`
  - external: `numpy`, `pandas`, `plotly`
  - stdlib: `typing`
  - internal: _none_

### audiospylt/plot_wave.py

- `crossfade_adjacent_events()`
  - external: `numpy`, `pandas`
  - stdlib: `typing`
  - internal: _none_
- `estimate_sampling_frequency_and_time_vector()`
  - external: `numpy`, `pandas`
  - stdlib: `typing`
  - internal: _none_
- `plot_waves()`
  - external: `numpy`, `pandas`, `plotly`
  - stdlib: `typing`
  - internal: `audio_utils`
- `plot_waves._apply_fades()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `plot_waves._ramp()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_

### audiospylt/plotting.py

- `show_plotly()`
  - external: _none_
  - stdlib: `typing`
  - internal: _none_

### audiospylt/ssm.py

- `_load_audio_mono()`
  - external: `numpy`
  - stdlib: `os`, `typing`
  - internal: _none_
- `compute_features()`
  - external: `librosa`, `numpy`
  - stdlib: `typing`
  - internal: _none_
- `compute_features._bin_mean()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `compute_ssm()`
  - external: `numpy`, `sklearn`
  - stdlib: _none_
  - internal: _none_
- `diagonal_smooth()`
  - external: `numpy`, `scipy`
  - stdlib: _none_
  - internal: _none_
- `main()`
  - external: `librosa`, `numpy`
  - stdlib: `typing`
  - internal: _none_
- `normalize_features()`
  - external: `numpy`, `sklearn`
  - stdlib: _none_
  - internal: _none_
- `plot_heatmap()`
  - external: `numpy`, `plotly`
  - stdlib: `typing`
  - internal: _none_
- `plot_heatmap._fmt_mmss()`
  - external: _none_
  - stdlib: _none_
  - internal: _none_
- `run_notebook()`
  - external: _none_
  - stdlib: `typing`
  - internal: _none_

### audiospylt/waveform_utils.py

- `apply_window_to_waveforms()`
  - external: `scipy`
  - stdlib: _none_
  - internal: _none_
- `generate_waveforms()`
  - external: `numpy`
  - stdlib: _none_
  - internal: _none_
- `plot_waveforms()`
  - external: `numpy`, `plotly`, `scipy`
  - stdlib: _none_
  - internal: _none_

