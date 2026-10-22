# Migrating from AudioSpylt 0.6 to 1.0

## What changes when you upgrade

`import audiospylt` no longer loads any 0.6 module or its notebook dependencies.
A 0.6 module loads on first access to its module or one of its old top-level
names. Its functions keep their 0.6 signatures, numbers, return values, printing,
displays and file writes. Each user call emits one
`audiospylt._compat.AudiospyltDeprecationWarning`, a `DeprecationWarning`, naming
its v1 replacement (or stating that none exists). Calls made inside another
wrapped 0.6 call stay silent. Warning filters still control which emissions you
see; accessing or importing a name does not warn. The 0.6 functions are removed
in **1.1**.

Seven of the eleven 0.6 modules import IPython, plotly, librosa or verovio at
module level, so the 0.6 API needs `pip install "audiospylt[all]"`. With a
core-only install (`pip install audiospylt`), an old top-level name whose module
needs a missing package raises an `ImportError` naming `audiospylt[all]`, e.g.
`search_f0 is 0.6 API and needs plotly; install audiospylt[all]`, and a direct
`from audiospylt.<module> import …` raises Python's plain `ModuleNotFoundError`.
`render_audio` and the `df_transforms` names need only core packages and still
resolve there. `from audiospylt import *` now binds only the v1 names; import a
0.6 name explicitly (`from audiospylt import search_f0`) while you migrate.

To see warnings in scripts, run `python -W default::DeprecationWarning script.py`.
To silence only these warnings, first use
`from audiospylt._compat import AudiospyltDeprecationWarning`, then
`warnings.filterwarnings("ignore", category=AudiospyltDeprecationWarning)` after
`import warnings`. This does not silence the separate `LegacyFormatWarning`
used when reading old tables.

## Conventions that changed

- **Side effects:** v1 compute functions never display, print or write files.
  `viz` functions return figures; `nb.display` handles display and playback.
  Files are written only when you pass a destination path or a download directory.
  In 0.6, URL loading defaults to `download_mode="cwd"`, notation save options
  default to paths such as `output.mei` / `output.svg`, and WAV exporters can
  choose timestamped filenames. In v1, saving notation is an explicit string or
  SVG write by your application.
- **Audio:** float64 samples throughout, mono `(samples,)`, multichannel
  `(channels, samples)`. The channel axis is always first when present. 0.6
  mixed layouts; its channel-first stereo trim sliced the wrong axis. v1
  `io.audio.trim` slices the sample axis and checks bounds instead of silently
  clamping them. Trim and fade durations round to samples rather than truncating. `load_audio` returns `(audio, sr)`, without the 0.6 info dict.
- **Amplitude:** amplitudes are linear peak amplitudes relative to full scale: an
  isolated on-bin sine of amplitude *a* reads *a*. dB is 20·log10 relative to
  1.0. The unit-sine known-issue tests give 0.5 with the 0.6 boxcar spectrum
  (−6.0206 dB) and 0.25 with Hann (−12.0412 dB), versus 1.0 in v1. Thus interior
  boxcar amplitudes double and Hann on-bin amplitudes quadruple; old thresholds
  cannot be copied numerically. The 0.6 magnitude spectrogram's default dB
  conversion reports a 3.0103 dB drop when amplitude halves; v1 reports 6.0206 dB.
- **Peaks:** 0.6 linear `thresh_amp_low`, `prominence` / `prominence_rel` and
  `distance_hz` correspond to v1 `min_db`, `prominence_db` and `min_distance_hz`,
  in dB, so retune them; v1 adds `rel_db`, a floor relative to the loudest peak.
  There is no upper-amplitude threshold (`thresh_amp_high`) or width filter in v1. Retune the notebooks with
  `spectrum(..., zero_pad=4)` and `find_peaks(prominence_db=..., rel_db=...)`
  when reproducing their peak sets; a literal transfer of linear thresholds
  will not reproduce them. v1 interpolates frequencies using QIFFT on dB
  magnitudes; 0.6 reports bin centres. Use `interpolate=False` to keep bin centres.
- **Tables:** v1 uses `freq`, `amp`, `amp_start`, `amp_stop`, `time_start`,
  `time_stop`, etc.; see [the table format](table-format.md).
  `io.tables.read_table` and `model.schema.upgrade` convert 0.6 column names
  with `model.schema.LegacyFormatWarning` (a `FutureWarning`). v1 compute and
  plotting functions validate v1 names; they do **not** automatically upgrade
  old names. Upgrade first, including `amp_min` → `amp_start` and `amp_max` →
  `amp_stop`. Wide `freq1` / `amp1` snapshots are not automatically reshaped: convert them
  explicitly into a canonical snapshot, events or tracks table.
- **Synthesis:** row intervals are always half-open `[time_start, time_stop)`;
  0.6 defaults to `interval_mode="closed"` and can double a shared boundary
  sample. 0.6 `phase_mode="reset"` and `"chirp"` both map to v1 `phase="reset"`
  (the two old modes differ only by rounding error, about 1e−14). `"global"` maps to `"absolute"`. v1 defaults to the new
  `"continuous"`, linking abutting rows into phase-continuous voices.
  v1 export never resamples: render at the target `sr`; 0.6 `fs_target_name`
  resampled after synthesis and its exporter always normalised to full scale. `io.audio.write_audio` raises if integer PCM
  would clip; `normalize=` explicitly opts into a peak level in dBFS.
  FLOAT/DOUBLE export can retain samples above full scale.
- **Morphing:** 0.6 `create_time_transition_df(scale_amp=True)` normalises
  amplitudes. v1 `transform.morph.morph` uses them as given. Where several rows
  share one partial at an end, one row carries its full amplitude and the others
  start or end at zero, so both ends match the source and target levels.
  v1 pairings carry primary-row flags; use `morph(source, target, duration, ...)`
  to emit events rather than feeding a pairing into the old transition helper.
- **Notation:** `half_tone` → `semitone`, `quarter_tone` → `quarter`,
  `eighth_tone` → `eighth`; `half_tone_deviation` → `semitone` with `labels=True`.
  v1 uses standard and Stein–Zimmermann quarter-tone accidentals, with no
  arrows. Eighth-tone accuracy uses the nearest quarter-tone accidental plus a
  mandatory cents label. Sharps are the default; choose `spelling="flat"` for
  flats. Labels use MEI `<dir>` elements. Documents are MEI 5.1 with a header.
- **Performance:** `find_peaks(min_distance_hz=...)` takes O(n·k) time for n
  candidates and k kept peaks. Unrestricted continuous linking with
  `synthesize(link_cents=None)` takes O(m·k) for m rows and k candidate voices.

## Name by name

Names below are relative to `audiospylt`. “None” means there is no v1 replacement;
a closest tool is only a starting point, not a promise of identical behavior.
The old top-level function aliases have the same replacements as their module
entries. The four constants in `mei` stay readable without warnings.

### audio_utils

Display-only sizes, colours, titles and layout options move to returned figure layout updates; call `nb.display.show` / `play` explicitly.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `audio_utils.ensure_finite_audio` | None | None; closest: `numpy.isfinite` / `numpy.nan_to_num`. v1 rejects nonfinite samples; sanitize explicitly if needed. |
| `audio_utils.load_wav_from_source` | `io.audio.load_audio` | Returns decoded samples and rate rather than a local filename. `download_mode` is gone: memory by default; pass an existing `download_dir` to keep URL bytes. `download_timeout_s` → `timeout`; no progress bar or chunk-size option. |
| `audio_utils.load_audio_data` | `io.audio.load_audio` (sr=, mono=) | `desired_sample_rate` → `sr`; `convert_to_mono` → `mono`. Same two-value return shape, with the v1 channel convention. |
| `audio_utils.display_audio_properties` | None | None; closest: inspect `audio.shape` and `sr` returned by `io.audio.load_audio`; print only what you need. |
| `audio_utils.plot_waveform` | `viz.signal.plot_waveform` with `nb.display.show` | Returns a figure; explicitly show it. Input is audio and `sr`; no 0.6 info dict required. The time axis uses exact sample spacing. |
| `audio_utils.trim_and_fade_audio` | `io.audio.trim` and `io.audio.fade` | Call `trim(audio, sr, start, stop)` then `fade(audio, sr, fade_in, fade_out)`. `start_time` / `end_time` → `start` / `stop`; `fade_in_duration` / `fade_out_duration` → `fade_in` / `fade_out`; `fade_in_exponent` / `fade_out_exponent` → `in_exponent` / `out_exponent`. Old defaults trim 0.4–1.6 s and fade 0.2/0.3 s; v1 requires explicit trim bounds and defaults to no fades. Skip `fade` when `add_fades=False`. v1 rounds durations to samples, rejects invalid bounds and overlapping fades, and returns samples only. Fade exponents retain the old curves. |
| `audio_utils.trim_and_fade_and_render` | `io.audio.trim` and `io.audio.fade`, then `io.audio.write_audio` or `nb.display.play` | Separate trimming, fades, export and playback. No automatic filenames. `save_audio` / `player` become explicit calls; export options follow `generate_wave_file` below. `cut_sanitize` / `sanitize_cut` and `sanitize` need explicit NumPy cleanup; v1 rejects nonfinite audio. |
| `audio_utils.load_audio_sample` | `io.audio.load_audio` | `desired_sample_rate` → `sr`, `convert_to_mono` → `mono`; drops the info dict. `download_mode="cwd"` / `"dir"` / `"temp"` → memory by default or explicit existing `download_dir`; kept files have hashed names. Use `refresh=True` to redownload. |
| `audio_utils.load_audio_sample_and_preview` | `io.audio.load_audio`, then `viz.signal.plot_waveform` and `nb.display.play` | Load as above, then explicitly plot and play. Playback and loading no longer happen in one call; `download_mode` / `download_dir` follow `load_audio_sample`. `playback_fs_target_name` requires rendering/resampling beforehand; `playback_bit_rate` applies only to an explicit writer `subtype`. `playback_save_audio` → a writer call, and `playback_sanitize` → explicit cleanup. |
| `audio_utils.plot_spectrogram` | `analysis.spectrogram.spectrogram` (or `analysis.spectrogram.mel_spectrogram`) with `viz.signal.plot_spectrogram` | `n_fft` → `frame_length`; `overlap` → `hop=max(1, int(n_fft * (1 - overlap)))`; `window_type` → `window`; integer `oversample_factor` → `zero_pad` (v1 requires an integer). `boundary=None` → `center=False`, zero-padding → `center=True`; v1 drops trailing partial frames, with no `padded` switch or `time_range` / `time_reference` alternatives. `mel_bins` / `mel_fmax` → `mel_spectrogram(n_mels=..., fmax=...)` on the computed STFT. Old `scaling` / `mode` density or PSD choices have no D4 switch; retune levels. `amp_scale` → figure `db=`, with no `linear_norm` / `power_law_gamma` equivalent. No old info-dict return. |
| `audio_utils.plot_spectrogram_with_waveform` | `analysis.spectrogram.spectrogram` with `viz.signal.plot_spectrogram`(..., audio=...) | Compute the spectrogram separately; pass waveform samples through the figure function’s `audio=` option. STFT and dB changes are as above. |

### dft_analysis

Display-only sizes, colours, titles, axis warps and layout options belong on returned figures; v1 supports linear/log frequency axes, with no automatic mixed-axis or partial-grid overlay.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `dft_analysis.apply_window` | `analysis.spectrum.spectrum`(..., window=...) | Pass `window_type` as `window`; scaling includes coherent-gain correction. Use NumPy/SciPy directly if you need the windowed samples themselves. |
| `dft_analysis.compute_fft` | `analysis.spectrum.spectrum` | Returns a `Spectrum` object instead of `(freqs, spec)`; use `.freqs` / `.amp`. For the old grid choose `window="boxcar", zero_pad=1`; interior amplitudes still double. |
| `dft_analysis.filter_peaks` | `analysis.peaks.find_peaks` | Takes a `Spectrum`, returns a snapshot instead of bin indices. Frequency limits → `fmin` / `fmax`; `distance_hz` → `min_distance_hz`. Retune linear height/prominence thresholds in dB; `width` / `width_hz` and an upper height limit have no equivalent. |
| `dft_analysis.plot_spectrum` | `viz.signal.plot_spectrum` | Pass the `Spectrum` and optional peak snapshot; `db=True` by default. Plot limits do not filter the computed table. |
| `dft_analysis.plot_windowed_waveform` | None | None. Multiply by a SciPy window yourself, then plot the resulting samples if required. |
| `dft_analysis.analyze_signal` | `analysis.spectrum.spectrum`, `analysis.peaks.find_peaks` and `viz.signal.plot_spectrum` | `window_type` → `spectrum(window=...)` (old default boxcar, v1 Hann); use `zero_pad=4`. `thresh_freq_low/high` → `find_peaks(fmin/fmax)`. Retune `thresh_amp_low` as `min_db`; `thresh_amp_high` needs a manual table filter. Retune `prominence` / `prominence_rel` as `prominence_db` and use `rel_db` for a floor relative to the loudest survivor. `distance_hz` → `min_distance_hz`; `width` / `width_hz`, `partial_tracking`, `f0` / `partial_bandwidth_hz` filtering have no automatic equivalent. Separate `show_peaks`, `show_plot`, `show_windowed_waveform` into explicit display/window code. |

### df_transforms

Display-only `plot`, plot sizes and titles become separate `viz.morph.plot_morph_map` and `nb.display.show` calls.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `df_transforms.detect_dataframe_type` | `model.schema.infer_kind` | Returns a kind string, not the 0.6 description dict. Requires canonical columns. |
| `df_transforms.process_spectral_dataframe` | `transform.scale.transpose` and `transform.scale.stretch` | `pitch_factor` → `transpose(factor=...)`; `pitch_cents` → `cents`; `pitch_min_hz` / `pitch_max_hz` → `fmin` / `fmax`; `pitch_overshoot="clip"/"allow"/"threshold"` → `out_of_range="clip"/"keep"/"drop"`. `stretch_factor` → `stretch(factor=...)`. Both return copies; stretch factors must be positive. Reshape wide `freq1` / `amp1` frames explicitly before v1 use. |
| `df_transforms.expand_to_multi` | `model.convert.snapshot_to_events` | Pass a canonical snapshot plus `time_start=0` and `time_stop=duration`. Amplitudes become `amp_start` / `amp_stop`. |
| `df_transforms.add_time_cues` | `transform.scale.split_events` | `cues` → `times`; split canonical events before per-cue transpose/stretch. |
| `df_transforms.merge_equal_split` | `transform.morph.pair`(..., method="equal") | Canonical snapshot inputs; returns a pairing with primary flags and endpoint gains rather than old `freq_data1` / `amp_data1` columns. |
| `df_transforms.create_time_transition_df` | `transform.morph.morph` | Pass source and target snapshots directly, with `duration` and the selected `method`. `scale_amp=True` and `override_amp` have no v1 switches: scale explicitly. Endpoint gains keep a partial that appears in several rows from inflating the levels. |
| `df_transforms.merge_cdf_analysis` | `transform.morph.parameter_range`(..., method="power") with `viz.morph.plot_morph_map` | Returns a tuple of covering bounds. To plot, build a `pair` at the chosen power, then pass that pairing to the figure function. |
| `df_transforms.merge_cdf` | `transform.morph.pair`(..., method="power", power=...) | `power_factor` → `power`; `None` still chooses the mean covering range. Pass canonical snapshots; output is a v1 pairing. Plot it separately. |
| `df_transforms.merge_sigmoid_analysis` | `transform.morph.parameter_range`(..., method="sigmoid") with `viz.morph.plot_morph_map` | Returns a tuple of covering bounds. Build a `pair` at the chosen range before plotting the map. |
| `df_transforms.merge_sigmoid` | `transform.morph.pair`(..., method="sigmoid", sigmoid_range=...) | `sigmoid_range` retains its name; `None` chooses the mean covering range. Output is a v1 pairing; plot separately. |

### f0_utils

Display-only summary plot sizes, titles and layouts stay in 0.6 until you replace that workflow.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `f0_utils.search_f0` | None | None. The 0.6 search, scoring, trimming and refinement options have no v1 equivalent. |
| `f0_utils.summarize_result` | None | None. Inspect stored 0.6 search results yourself when retiring the old search. |

### generate_wave_file

Display-only `player` and verbosity options become explicit playback or application logging.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `generate_wave_file.ensure_finite_audio` | None | None; closest: `numpy.isfinite` / `numpy.nan_to_num`. v1 validates finite samples instead of silently sanitizing. |
| `generate_wave_file.scale_samples` | None | None; closest: `io.audio.write_audio(..., subtype=...)`. Pass floating-point samples; do not preconvert to PCM integers. |
| `generate_wave_file.generate_wave_file` | `io.audio.write_audio` | `fs_target_name` has no writer equivalent: synthesize at the target `sr`. `bit_rate=16/24` → `subtype="PCM_16"/"PCM_24"` (old generator default 16, v1 writer default 24). Old export always peak-normalised; use `normalize=0` to opt into full-scale normalisation, or omit it to preserve levels. Give an explicit path; no filename templates/timestamps. `alias_above_nyquist=True` → `synthesize(nyquist="alias")` when rendering oscillators, not a resampling option. With `save_to_file=False`, retain the array instead. |
| `generate_wave_file.render_audio` | `io.audio.write_audio` and `nb.display.play` | `fs_target_name` and `alias_above_nyquist` migrate as above; `bit_rate` (0.6 default 24 here) → `subtype` (v1 default `"PCM_24"`). `save_audio` / `player` → explicit writer / playback calls. `sanitize` becomes explicit NumPy cleanup; choose `normalize=` separately for export or playback. |
| `generate_wave_file.mix_selected_waveforms` | None | None; closest: sum selected arrays with NumPy. Align lengths explicitly. |
| `generate_wave_file.render_selected_waveforms` | None | None; closest: `io.audio.write_audio` and `nb.display.play` on the summed array. Choose the output rate during generation. |

### io_utils

Display-only `verbose` logging is application-owned in v1.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `io_utils.save_df_tsv` | `io.tables.write_table` | Canonical table required; a versioned TSV header is written. `index` / `verbose` / `make_dirs` are gone; create the destination directory explicitly. |
| `io_utils.events_df_from_peaks_by_interval` | `model.convert.snapshot_to_events` per interval, then pandas.concat | Upgrade each `peaks_df`, pass its interval bounds to `snapshot_to_events`, then concatenate. `freq_col` / `amp_col` custom names must be renamed first. |

### mei

Display-only `display_df` becomes an explicit display; `notebook_white_background` → `nb.display.show_mei(background=...)`. Write MEI/SVG explicitly for save options; sizes and colours belong to renderer/display options.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `mei.acoustical_pitch_name` | `notation.pitch.pitch_table` | Use the table’s `step`, `octave` and `accid` fields instead of a pitch-name string; pass frequencies rather than rounded MIDI numbers. |
| `mei.midi_value` | `notation.pitch.pitch_table` | Use `midi` (unrounded) or `written_midi` (quantised); input is Hz, not a log-frequency ratio. |
| `mei.midi_cent_value` | `notation.pitch.pitch_table` | Use `residual` for unrounded cents relative to the written pitch; rounding is a presentation choice. |
| `mei.convert_frequencies` | `notation.pitch.pitch_table` | Pass the frequency vector; returns a pitch table, not three lists. |
| `mei.pitch_name_and_deviation_to_step_octave_alter` | `notation.pitch.pitch_table` | Pass absolute Hz. `half_tone` / `half_tone_deviation` → `semitone`; `quarter_tone` → `quarter`; `eighth_tone` → `eighth`. Read `step`, `octave`, `accid`, `residual`; no arrow accidentals. |
| `mei.map_alteration_to_mei_accid` | `notation.pitch.pitch_table` (accid column) | Use `accid` from the pitch table rather than an alteration string. Natural-sign display is chosen when writing MEI. |
| `mei.create_mei_string` | `notation.mei.snapshot_to_mei` | Pass a canonical snapshot rather than precomputed note tuples. `clef_config` → `clef`; `resolution_type` uses the new resolution names; `display_mode` → `layout`; `display_cent_deviation` → `labels`; `include_natural_accidentals` → `naturals`. |
| `mei.create_temporal_mei_string` | `notation.mei.events_to_mei` | Pass canonical events rather than grouped note tuples. `measure_represents_sec` → `measure_seconds`; `display_time_start` → `time_labels`; `time_label_format` → `time_format`. `time_label_mode` / `time_label_place` have no switches: v1 uses `<dir>` above the uppermost occupied staff. `clef_config` → `clef`; `resolution_type` → `resolution`; `include_natural_accidentals` → `naturals`. |
| `mei.process_and_visualize` | `notation.mei.snapshot_to_mei` with `nb.display.show_mei` | `resolution="half_tone_deviation"` (old default) → `resolution="semitone", labels=True`. `display_mode="sequence"` → `layout="sequence"` (v1 defaults to chord); `note_order` needs explicit input sorting. `clef_config` → `clef`; `display_cent_deviation` → `labels`; `include_natural_accidentals=True` (old default) → `naturals=True` (v1 default False). Save strings/rendered SVG explicitly; display the table separately. |
| `mei.process_temporal_chords` | `notation.mei.events_to_mei` with `nb.display.show_mei` | `measure_represents_sec` → `measure_seconds`; `display_time_start` → `time_labels`; `time_label_format` → `time_format`. `time_label_mode` / `time_label_place` → fixed `<dir>` above staff. `include_natural_accidentals=True` (old default) → `naturals=True` (v1 default False); `clef_config` → `clef`. Rename custom frequency/time columns before validating; v1 notates `freq_start`. Use new resolution names; labels are mandatory for eighth mode. |
| `mei.A4_FREQUENCY` | `notation.pitch.pitch_table`(..., a4=...) | Pass the reference frequency explicitly; this constant stays unwrapped. |
| `mei.MIDI_BASE` | None | Internal numeric/XML convention; no v1 public constant. Stays unwrapped. |
| `mei.CENT_FACTOR` | None | Internal numeric/XML convention; no v1 public constant. Stays unwrapped. |
| `mei.XML_NS` | None | Internal numeric/XML convention; no v1 public constant. Stays unwrapped. |

### multiplotter

Display-only sizes, colours, titles, axis padding and range settings move to returned figures; linear/log axes replace mixed-axis controls.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `multiplotter.plot_combined` | `viz.tables.plot_events` (snapshots: `viz.tables.plot_snapshot`) | 0.6 expects event columns (`time_start/stop`, `freq_start/stop`). Upgrade then use `plot_events`; for snapshots choose the separate snapshot function. Pass tables, not file lists; call `read_table` first. |
| `multiplotter.plot_combined_3d` | `viz.tables.plot_events_3d` | Upgrade events first. `axis_order` / `flip` have no v1 switches: time, frequency and amplitude are fixed on x, y and z. Pass tables instead of file lists. |
| `multiplotter.plot_scatter` | `viz.tables.plot_snapshot` (events, tracks: `viz.tables.plot_events`) | 0.6 expects snapshot columns (`Frequency (Hz)`, `Amplitude`). Upgrade then use `plot_snapshot`; events and tracks use the separate event function. |
| `multiplotter.plot_scatter_binned` | None | None. The frequency/amplitude bin sizes and density aggregation have no v1 equivalent. |
| `multiplotter.plot_equalizer_bars` | None | None; closest: `viz.tables.plot_snapshot`. A snapshot stem plot does not reproduce `agg` / frequency-bin aggregation. |

### plotting

Display-only renderer choice is passed to `nb.display.show(renderer=...)`.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `plotting.show_plotly` | `nb.display.show` | Explicit notebook display; accepts a figure and optional `renderer`. |

### plot_wave

Display-only sizes, colours, titles and spectrogram presentation options move to the returned figures.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `plot_wave.plot_waves` | `synth.additive.synthesize` with `viz.signal.plot_waveform` | `k` → `sr=1/k` (choose an integer rate). `phase_mode="reset"/"chirp"/"global"` → `phase="reset"/"reset"/"absolute"`; v1 defaults to `continuous`. `interval_mode="closed"` is gone; only half-open rows. `edge_fade_s` → `fade` for symmetric raised-cosine voice fades. Asymmetric/per-row `fade_in_s` / `fade_out_s`, linear and sqrt-Hann `fade_shape` need explicit processing; linked voice interiors are not faded. `show_waveform` / `show_fft` become separate plotting calls. |
| `plot_wave.crossfade_adjacent_events` | None | None; closest: `synth.additive.synthesize(..., phase="continuous")` joins abutting events without phase jumps. It does not recreate overlapping crossfades. |
| `plot_wave.estimate_sampling_frequency_and_time_vector` | None | None; closest: choose `sr` in `synth.additive.synthesize`. The renderer uses exact sample spacing; choose a rate above twice the highest frequency to retain that partial. |

### waveform_utils

Display-only sizes, colours, titles and layouts move to returned figures.

| 0.6 | v1 | notes |
| --- | --- | --- |
| `waveform_utils.generate_waveforms` | None | None. The oscillator dictionary, phase, AM/FM, duty and saw-width configuration has no v1 equivalent; additive synthesis covers sinusoidal event tables only. |
| `waveform_utils.apply_window_to_waveforms` | None | None. Apply a NumPy/SciPy window explicitly; this old helper mutates the waveform dictionary in place. |
| `waveform_utils.plot_waveforms` | None | None; closest: `viz.signal.plot_waveform` and `viz.signal.plot_spectrum`. Compute the spectrum explicitly; no combined oscillator-dictionary report. |

## Known 0.6 issues

These are the 26 strict expected failures retained in the 0.6 tests. The v1
import-without-notebook-dependencies case now passes. The numbers come from
those tests.

1. Boxcar analysis reads a unit sine as 0.5; v1 reads 1.0.
2. Hann analysis reads a unit sine as 0.25; v1 corrects the window gain and reads 1.0.
3. Off-bin Hann frequencies are reported at bin centres (1.455 cents error in the fixture); v1 uses QIFFT interpolation.
4. The standalone magnitude spectrogram shows only a 3.0103 dB amplitude-halving drop; v1 shows 6.0206 dB.
5. The waveform-plus-spectrogram entry point has the same 3.0103 dB drop; v1 uses the shared 20·log10 conversion.
6. The estimated sampling rate can put a zero-phase highest partial at Nyquist and erase it; v1 lets you choose a rate above twice that frequency.
7. Quarter-tone spelling misrepresents some base pitches and deviations; v1 quantises absolute pitch before spelling with quarter-tone accidentals.
8. Eighth-tone spelling misrepresents some pitches with arrow accidentals; v1 uses a quarter-tone accidental plus a mandatory residual cents label.
9. Channel-first stereo trimming returns `(0, 1000)` instead of `(2, 200)` in the fixture; v1 slices the sample axis and keeps both channels.
10. Default closed intervals double a shared event-boundary sample; v1 half-open intervals contribute once.
11. Non-antialiased export resampling uses 0.0015714 s spacing instead of 1/600 s in the ramp fixture; v1 synthesizes directly on the declared-rate grid without export resampling.
12. Equal pairing, smaller snapshot supplied first, repeats its amplitude at the start; v1 gives one primary row its full start amplitude and other rows zero.
13. Power/CDF pairing, smaller snapshot supplied first, repeats its amplitude at the start; v1 primary start gains preserve the source level.
14. Sigmoid pairing, smaller snapshot supplied first, repeats its amplitude at the start; v1 primary start gains preserve the source level.
15. Equal pairing, smaller snapshot supplied second, still repeats amplitudes when the pairing is oriented with that snapshot at the start; v1 primary start gains preserve its level.
16. Power/CDF pairing, smaller snapshot supplied second, repeats amplitudes when oriented at the start; v1 primary start gains preserve its level.
17. Sigmoid pairing, smaller snapshot supplied second, repeats amplitudes when oriented at the start; v1 primary start gains preserve its level.
18. Equal pairing, smaller snapshot supplied first, repeats amplitudes when oriented at the stop; v1 gives one primary row its full stop amplitude and other rows zero.
19. Power/CDF pairing, smaller snapshot supplied first, repeats amplitudes when oriented at the stop; v1 primary stop gains preserve its level.
20. Sigmoid pairing, smaller snapshot supplied first, repeats amplitudes when oriented at the stop; v1 primary stop gains preserve its level.
21. Equal pairing, smaller snapshot supplied second, repeats its amplitude at the stop; v1 primary stop gains preserve the target level.
22. Power/CDF pairing, smaller snapshot supplied second, repeats its amplitude at the stop; v1 primary stop gains preserve the target level.
23. Sigmoid pairing, smaller snapshot supplied second, repeats its amplitude at the stop; v1 primary stop gains preserve the target level.
24. Wide-snapshot pitch scaling leaves the second frequency column unchanged; after explicit reshaping into a v1 table, transpose scales every frequency column.
25. A scalar negative stretch silently reverses events; v1 rejects nonpositive stretch factors.
26. A per-cue stretch list containing a negative factor also reverses events; v1 validates every factor and rejects it.

## Examples

Run these from the repository root so `samples/` and `data/` resolve. Each block
is independent. Figures need `audiospylt[plot]`, display `audiospylt[notebook]`
and MEI rendering `audiospylt[notation]`; `pip install "audiospylt[all]"`
installs all three. Loading the headerless 0.6 TSVs in the last example
emits `LegacyFormatWarning`; values are preserved when their columns are renamed.

0.6 counterparts: `audio_utils.load_audio_sample_and_preview` and
`audio_utils.trim_and_fade_and_render`.

```python
from audiospylt.io.audio import load_audio, trim, fade
from audiospylt.viz.signal import plot_waveform
from audiospylt.nb.display import show, play

audio, sr = load_audio("samples/cello_arco_d_sul_c_01.wav")
excerpt = fade(trim(audio, sr, 2.0, 2.5), sr, fade_in=0.02, fade_out=0.03)
show(plot_waveform(excerpt, sr))
play(excerpt, sr)
```

0.6 counterparts: `dft_analysis.analyze_signal` and `mei.process_and_visualize`.
This uses the same faded 2.0–2.5 s excerpt. The thresholds are v1 choices, rather
than a numeric translation of old linear thresholds.

```python
from audiospylt.io.audio import load_audio, trim, fade
from audiospylt.analysis.spectrum import spectrum
from audiospylt.analysis.peaks import find_peaks
from audiospylt.viz.signal import plot_spectrum
from audiospylt.notation.mei import snapshot_to_mei
from audiospylt.nb.display import show, show_mei

audio, sr = load_audio("samples/cello_arco_d_sul_c_01.wav")
excerpt = fade(trim(audio, sr, 2.0, 2.5), sr, fade_in=0.02, fade_out=0.03)
spec = spectrum(excerpt, sr, window="hann", zero_pad=4)
peaks = find_peaks(spec, min_db=-60, rel_db=-35, prominence_db=10,
                   fmin=30, fmax=5000)
show(plot_spectrum(spec, peaks))
mei = snapshot_to_mei(peaks, resolution="quarter", labels=True)
show_mei(mei)
```

0.6 counterparts: `df_transforms.merge_equal_split` / `merge_cdf` / `merge_sigmoid`,
`df_transforms.create_time_transition_df`, `plot_wave.plot_waves` and
`generate_wave_file.render_audio`. Transport is a new pairing method. This
example writes `morph.wav` in the current directory at an explicitly chosen rate.

```python
from audiospylt.io.tables import read_table
from audiospylt.transform.morph import morph
from audiospylt.synth.additive import synthesize
from audiospylt.io.audio import write_audio

speech = read_table("data/tables/speech1.tsv")
conga = read_table("data/tables/conga3.tsv")
events = morph(speech, conga, duration=2.0, method="transport")
audio = synthesize(events, sr=44100, fade=0.01)
write_audio("morph.wav", audio, 44100, subtype="PCM_24")
```
