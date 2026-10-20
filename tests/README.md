# AudioSpylt 0.6.3 characterization safety net

Passing tests capture the checked-out code's current numerical results, schemas,
metadata, and return conventions. Desired fixes are separate strict xfails in
`test_known_bugs.py`; they do not calculate expectations from current-code goldens.
Each xfail also specifies `raises=AssertionError`, so unexpected non-assertion
exceptions (including setup errors and forbidden network calls) cannot masquerade
as a reproduced DSP finding. Fixing a finding
produces a failing XPASS until its marker is deliberately removed.

## Coverage

All public functions requested in Step 0.2 are exercised. This is a refactoring
safety net, not exhaustive input validation or a quality score.

| Test file | Functions covered | Finding IDs |
| --- | --- | --- |
| `test_audio_utils.py` | `load_audio_data`, `load_audio_sample`, `load_wav_from_source`, `trim_and_fade_audio`, `trim_and_fade_and_render`, `plot_waveform`, `plot_spectrogram`, `plot_spectrogram_with_waveform`, `load_audio_sample_and_preview` | DSP-3, DSP-6 (desired behaviour in `test_known_bugs.py`) |
| `test_dft_analysis.py` | `compute_fft`, `apply_window`, `filter_peaks`, `analyze_signal`, `plot_spectrum`, `plot_windowed_waveform` | DSP-1, DSP-2 (desired behaviour in `test_known_bugs.py`) |
| `test_f0_utils.py` | `search_f0`, `summarize_result`; harmonic/inharmonic inputs, explicit/amplitude weights, `top_k`, cents/fraction trimming, `peaks_df`, coarse refinement | — |
| `test_df_transforms.py` | `process_spectral_dataframe`, `create_time_transition_df`, `merge_equal_split`, `merge_cdf`, `merge_cdf_analysis`, `merge_sigmoid`, `merge_sigmoid_analysis`, `add_time_cues`, `expand_to_multi`, `detect_dataframe_type` | DSP-9 (desired behaviour in `test_known_bugs.py`) |
| `test_transform_scale.py` | v1 `transpose`, `stretch`, `split_events`; all kinds, bounds, per-cue factors, 0.6 characterization, validation and metadata | DSP-9 |
| `test_transform_morph.py` | v1 `parameter_range`, `pair`, `morph`; 0.6 mappings, D13 gains and emission order, transport marginals, endpoint and chained render levels | DSP-9, POS-6 |
| `test_plot_wave.py` | `plot_waves`: all `chirp`/`reset`/`global` × closed/half-open intervals × no/linear/cosine/sqrt-Hann fades; per-row fades and overlapping crossfades via `crossfade_adjacent_events` | DSP-4, DSP-7 (desired behaviour in `test_known_bugs.py`) |
| `test_generate_wave_file.py` | `generate_wave_file`, `render_audio`, `render_selected_waveforms`, `mix_selected_waveforms`, `scale_samples`; native 16/24-bit WAVs, bandlimited/intentional-alias resampling, silence, mixed lengths, legacy `n`/preferred `dt` | DSP-8 (desired behaviour in `test_known_bugs.py`) |
| `test_io_utils.py` | `save_df_tsv`, `events_df_from_peaks_by_interval` | — |
| `test_mei.py` | `midi_value`, `midi_cent_value`, `acoustical_pitch_name`, `pitch_name_and_deviation_to_step_octave_alter`, `map_alteration_to_mei_accid`, `convert_frequencies`, `create_mei_string`, `create_temporal_mei_string`, `process_and_visualize`, `process_temporal_chords` | DSP-5 (desired behaviour in `test_known_bugs.py`) |
| `test_notation_pitch.py` | v1 `pitch_table`, `cents_label`; exhaustive C1–B8 spelling/quantisation sweep, half-up ties, eighth-mode accuracy, input validation | DSP-5, CLM-8 |
| `test_notation_mei.py` | v1 `snapshot_to_mei`, `events_to_mei`, `render_svg`, `render_svg_pages`; MEI 5.1 schema, labels/staves, temporal gaps, range, fd-level warning capture, lazy/display-free imports | DSP-5, CLM-8, ENG-4 |
| `test_viz_signal.py` | v1 `plot_waveform`, `plot_spectrum`, `plot_spectrogram`; theme palette, `_minmax`, WebGL opt-in, mel axis, overlays, side-effect free | CLM-4, ENG-4 |
| `test_viz_tables.py` | v1 `plot_snapshot`, `plot_events`, `plot_events_3d`; stems, opacity bins, tracks, log-axis warnings | ENG-4, TXT-9 |
| `test_viz_morph.py` | v1 `plot_morph_map`; speech1↔conga3 category counts and widths for equal/power/sigmoid/transport | TXT-9, DSP-9 |
| `test_nb_display.py` | v1 `show`, `show_svg`, `show_mei`, `play`; lazy IPython, SVG white background, peak policy | ENG-4 |
| `test_layering.py` | AST: core/viz/nb import boundaries; `__init__` docstring-only; librosa/IPython laziness | ENG-4 |
| `test_multiplotter.py` | `plot_combined`, `plot_combined_3d`, `plot_equalizer_bars`; local small TSVs, all three bar aggregations | — |
| `test_known_bugs.py` | `analyze_signal`, both spectrogram entry points, `plot_waves`, `estimate_sampling_frequency_and_time_vector`, pitch spelling, stereo trimming, `_resample_by_sampling`, all three morph pairings, transitions, wide pitch scaling, negative stretch, v1 import without notebook stack | DSP-1 through DSP-9, ENG-4; 27 strict xfail cases |

## Goldens and tolerances

`golden/current.json` contains 78 small snapshots. `_cases.py` provides deterministic
synthetic inputs and actual-result producers, shared with regeneration;
`_support.py` serializes them and compares fixed expectations from disk. Tests
never regenerate goldens implicitly. Numeric arrays, DataFrame column order and
attributes, f0 candidate scores, pitch conversions, normalized MEI, and printed/
TSV text are captured. Frame dtypes are deliberately not pinned across pandas
versions; column names, order, values and attributes are pinned.

The normal float tolerance is `rtol=1e-5, atol=1e-8`. Spectrograms use
`rtol=1e-4, atol=1e-7`; decoded FLOAT WAVs use `atol=1e-7`. Native PCM conversion
is exact. Resampled 16-bit PCM allows one LSB (`rtol=0, atol=1`) for scipy rounding.
No golden uses librosa resampling: WAV loading retains the original sample rate.

MEI comparisons canonicalize generated `xml:id`s and their references together,
normalize creation/modification timestamps and date metadata, and retain musical
`tstamp` values and deterministic temporal labels. Real Verovio rendering runs
locally and saved SVGs must be valid, nonempty SVG documents. SVG geometry is
smoke-tested because renderer/font versions may change layout independently of
AudioSpylt's MEI.

Pure display functions (`plot_waveform`, `plot_spectrum`, `plot_windowed_waveform`,
`plot_combined`, `plot_combined_3d`, `plot_equalizer_bars`) and the display side of
`summarize_result` get smoke/return-contract checks, with trace data checks where
useful. `plot_windowed_waveform` additionally compares its windowed samples to a
golden; `summarize_result` pins printed text. Spectrograms and notation processing
also compare returned data/generated MEI. No requested function is untested.
Browser/kernel rendering and audible playback are intentionally inert; WAVs,
Plotly figures, Audio widgets, and Verovio notation still get constructed.

## Running

From the repository root, with dependencies already installed:

```bash
PYTHONPATH=. NUMBA_CACHE_DIR=.delegate/numba MPLBACKEND=Agg \
PLOTLY_RENDERER=json PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -p no:cacheprovider -o xfail_strict=true tests
```

`conftest.py` registers `slow` and `network`; network-marked tests are skipped.
It patches IPython display aliases and Plotly display sinks, forbids socket/HTTP
connections, and changes the working directory to each test's `tmp_path` so
wrappers without an `output_dir` cannot write elsewhere. Unless explicitly
specified, pytest's base temp directory is `tests/.pytest-tmp` to keep transient
files within this worktree. For concurrent runs, use a distinct `--basetemp` per
process and create its parent directory first.

To audit failures without xfail handling (exit 1 is expected on 0.6.3):

```bash
PYTHONPATH=. NUMBA_CACHE_DIR=.delegate/numba MPLBACKEND=Agg \
PLOTLY_RENDERER=json PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -p no:cacheprovider tests/test_known_bugs.py --runxfail --tb=short
```

The initial audit on Python 3.12 confirmed 20 failures, all at intended assertions:
0.5/0.25 analysis amplitudes; 1.455-cent off-bin error; 3.0103-dB amplitude-halving
drop; near-zero top-partial RMS; inaccurate microtone spelling; `(0, 1000)` stereo
trim instead of `(2, 200)`; doubled boundary amplitude; 0.0015714-s resampling
spacing instead of 1/600 s; inflated morph energy; unchanged second frequency
column; and no rejection of negative stretch.

## Deliberate regeneration

Run from the repository root against the version whose behaviour you intend to
record. Review the golden diff alongside the implementation diff.

```bash
PYTHONPATH=. NUMBA_CACHE_DIR=.delegate/numba MPLBACKEND=Agg \
PLOTLY_RENDERER=json PYTHONDONTWRITEBYTECODE=1 \
python tests/golden/regenerate.py
```

The script uses the same offline/display guards, writes only `golden/current.json`,
and creates then removes its synthetic WAV/MEI/SVG scratch directory inside
`tests/golden/`. It does not rewrite the desired-behaviour xfail assertions.
When fixing a DSP finding, remove the corresponding xfail marker (or split out
the fixed parameter cases) and deliberately update only affected current-behaviour
goldens. Existing quirks such as closed intervals remain characterized explicitly.

## Verified environments

Final matrix runs used the commands above with distinct `--basetemp=tests/.tmp/pyNNN`,
`-ra --tb=short`, and no pytest cache. Bash `time` measured whole-process wall time;
pytest's elapsed time is also recorded. Python 3.11 had three upstream audioread
stdlib deprecation warnings; no failures or skips occurred.

- Python 3.12.14 / librosa 1.0.0: **110 passed, 20 xfailed**, 5.934 s wall (pytest: 4.46 s).
- Python 3.11.16 / librosa 0.11.0: **110 passed, 20 xfailed**, 16.591 s wall (pytest: 14.83 s).
- Python 3.14.7 / librosa 1.0.0: **110 passed, 20 xfailed**, 6.318 s wall (pytest: 4.71 s).
- Python 3.12, `--runxfail` audit: **20 failed**, 2.64 s (expected).

The exact invocation without `--basetemp` also passed on Python 3.12: 110 passed,
20 xfailed, 5.444 s wall (pytest: 4.13 s), confirming the default temp isolation.
Golden regeneration was byte-identical (SHA-256
`f94a20432f8131f7e0df3ff8d54504c0495255a4b076453bd66b378e5956749b`);
the complete golden directory is 186,062 bytes, including the regeneration script.
