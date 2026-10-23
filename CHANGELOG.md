# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Deprecated
- The 0.6 modules and their top-level function names now warn on calls with `AudiospyltDeprecationWarning`; they are removed in 1.1. See `docs/migration.md` for replacements and changed conventions.

### Changed
- `import audiospylt` no longer imports the 0.6 modules or IPython; old names load lazily on first access.
- Dependencies are split into a lean core (`numpy`, `pandas`, `requests`, `scipy`, `soundfile`, `soxr`) and extras: `[plot]` (`plotly`), `[notebook]` (`ipython`, `ipykernel`, `ipywidgets`, `nbformat`, `tqdm`), `[notation]` (`verovio`), `[librosa]` (`librosa`), `[all]` (all four; the 0.6 API and the notebooks need it) and `[dev]` (`[all]` plus build, lint, type-check and test tools).
- librosa is optional: `load_audio(sr=...)` resamples with soxr (HQ) directly, and its output length is exactly `ceil(n * sr / rate)` (librosa's rounding sometimes added one zero sample). `mel_spectrogram()` and the mel axes in `audiospylt.viz` raise an `ImportError` naming `audiospylt[librosa]` when it is missing.
- `import audiospylt as asp` exposes the 47 v1 compute names lazily at the top level (`asp.load_audio`, `asp.spectrum`, `asp.find_peaks`, `asp.morph`, …); figures and notebook display stay namespaced as `asp.viz` and `asp.nb`. The 0.6 names leave `__all__` and `dir()` but still resolve.
- Version 1.0.0.dev0, with PEP 639 license metadata (`License-Expression: MIT`). The sdist ships the package only; tests run from a git checkout.

### Added
- v1 subpackage layout (`io`, `analysis`, `model`, `transform`, `synth`, `notation`, `viz`, `nb`); the 0.6 modules are unchanged.
- `audiospylt.model.schema`: column contracts for `snapshot`, `events` and `tracks` tables, with `validate()` reporting every problem at once, `infer_kind()` and `upgrade()` for 0.6 column names (`LegacyFormatWarning`).
- `audiospylt.io.tables`: `read_table()` / `write_table()` for versioned TSV files (`# audiospylt-table: <kind> v1`, optional metadata lines, exact float round trips); headerless 0.6 files load with their columns upgraded.
- `audiospylt.io.audio`: `load_audio()` for paths, binary streams and URLs (memory by default, optional atomic kept copies), with float64 channel-first audio, optional mono conversion and resampling; pure `to_mono()`, `trim()` and `fade()` helpers with explicit bounds checks.
- `audiospylt.analysis`: compute-only `spectrum()` scaled to peak amplitude (a sine of amplitude *a* reads *a*), `find_peaks()` with quadratic interpolation on dB magnitudes (QIFFT) returning snapshot tables with provenance, a matching vectorized `spectrogram()` and an energy-normalized `mel_spectrogram()`; dB is 20·log10 re full scale.
- `audiospylt.synth.additive.synthesize()`: additive synthesis of events tables directly at the output rate, with half-open row intervals (abutting rows never share a sample), three distinct phase modes (`continuous` links rows into phase-continuous voices, `reset`, `absolute`), voice-level raised-cosine fades and a Nyquist gate (`drop`, or `alias` for exact folding); `audiospylt.model.convert.snapshot_to_events()`; `audiospylt.io.audio.write_audio()` writes levels unchanged and refuses to clip PCM unless asked to `normalize` to a peak level.
- `audiospylt.notation`: `pitch_table()` quantises absolute pitch to semitone, quarter-tone or eighth-tone resolution and spells it with sharps (default) or flats using standard and Stein–Zimmermann quarter-tone accidentals, keeping the unrounded cents residual; `snapshot_to_mei()` (chord or sequence) and `events_to_mei()` (temporal chords) write MEI 5.1 documents that validate against the MEI schema, with optional cents labels (always on at eighth-tone resolution, where the label carries the eighth-tone accuracy); `render_svg()` / `render_svg_pages()` render them with Verovio (`audiospylt[notation]` extra) without display side effects.
- Table format specification in `docs/table-format.md`.
- `audiospylt.analysis.f0`: `pyin()` tracks f0 over time with librosa's pYIN (`[librosa]`) and by default refines each voiced frame with a fit to its QIFFT harmonics, which removes pYIN's 10-cent grid and its low-f0 bias (a 73.8 Hz sine: +21 cents before, at most 1.2 cents after). It returns an `F0Contour` on the `spectrogram()` frame grid and keeps pYIN's own values; `model.convert.f0_to_tracks()` turns the voiced frames into a `tracks` table. `harmonic_f0()` replaces 0.6 `search_f0`. It uses the same objective, optimised exactly, and returns the highest candidate within `tol_cents` (3) of the lowest error, because a subharmonic always fits at least as well. See `docs/f0.md`.
- `model.convert.tracks_to_events()`: each pair of consecutive points of a track becomes one events row that keeps its `track_id`, so `synthesize()` plays every track of a `tracks` table (from `f0_to_tracks()`, later from Partiels, SPEAR or SDIF) as one phase-continuous voice and never joins two tracks. A pyin contour of a vibrato sine, resynthesised this way, reads within 1.2 cents of the original. See `docs/synthesis.md`.
- `read_partiels()` reads Partiels exports of Ircam PM2 partial tracking (a folder, a file or several; CSV with any header, or JSON) into a `tracks` table. Every run of frames between gaps of one exported track is one track. `max_step_cents=` also splits at frequency jumps. Points that inharmonic PM2 copies into several slots are kept once, in the longest run (`dedupe=True`). PM2 amplitudes are already peak amplitudes, like `spectrum()`, and phases are sine phases at the frame time. A `partial_id` column keeps PM2's partial ID (the harmonic number) from JSON exported with `--description`. See `docs/interop.md`, which also lists measured limits of PM2 in Partiels.
- `read_sdif()` reads SDIF partial tracks (`1TRC` from AudioSculpt/PM2, the CNMAT tools and loristrck; `1HRM` from PM2 harmonic analysis; `RBEP` from Loris and SPEAR) into a `tracks` table, and `write_sdif()` writes a `tracks` table as `1TRC` in AudioSculpt's layout (float32 values, index = `track_id + 1`), filling in points where tracks do not share one time grid, or with `hop=` resampling every track onto frames `k * hop` (a Loris RBEP flute analysis: 62 MB filled, 1.5 MB at `hop=0.005`). `read_chord_seq()` reads AudioSculpt chord sequences into an `events` table. The reader and writer need only numpy (no pysdif3, whose 1.0 release cannot open AudioSculpt files); the reader matched pysdif3 0.9.1 on 36 test files, and pysdif3 0.9.1 and Partiels read the written files. A `1TRC` or `1HRM` index that skips a frame starts a new track; `RBEP` point times are frame time plus offset, and the Loris noise share is kept as `bandwidth`. See `docs/interop.md`.
- `model.convert.tracks_to_chords()` reduces a `tracks` table to one chord per time window for notation (a step in seconds or boundary times): each track that sounds in a window gives one note, at the amplitude-weighted median of its frequencies (which stays on the line through brief octave errors) and its time-averaged amplitude. In each window the loudest note is kept first and notes within `merge_cents` (50) of a louder kept note are dropped, which removes the copies of one partial split over several tracks; `min_db` and `max_notes` thin further. On the Saariaho flute PM2 export, 0.5 s windows give 15 chords of a median 21 notes (40 before merging). `tracks_at()` reads every track at one time as a snapshot, interpolated linearly. See `docs/interop.md`.
- Event notation on a real time axis: `events_to_mei()` stores each onset's time in the MEI document (`<performance><recording>` with one `<when>` per onset in ISO time, `measure@when`), and `render_svg()` / `render_svg_pages()` draw a ruler under each system from it: the system's time span at true scale, a dot per onset and a line from each chord to its dot, while Verovio's engraving stays as it is (`timeline=False` turns the ruler off). `events_to_mei(colors=True)` colours each onset's chords and their dot and line. In notebooks `nb.show_mei()` links a chord to its dot and line on hover. See `docs/notation.md`.
- `py.typed` and type hints on the public v1 API, with an `__init__.pyi` stub declaring the lazy top-level names.
- `synthesize(link_ramp=0.005)`: in `continuous` mode an amplitude step where one row continues another becomes a raised-cosine transition centred on the link (at most half of either row), so frame-by-frame tables no longer click at frame edges; `link_ramp=0` keeps hard steps. Modes and options are compared in `docs/synthesis.md`.
- `morph(duplicates=...)`: `"full"` (default for `equal`, `power`, `sigmoid`) keeps the 0.6 gains, every copy of a shared partial at full amplitude, and emits every pairing row; `"primary"` (default for `transport`) gives one row per partial the amplitude so both ends match the snapshots' levels. `attrs["morph"]` records the choice.
- MEI cents and onset labels are centred on their noteheads (`<dir><rend halign="center">`).
- In `snapshot_to_mei(layout="sequence")` the cents labels of consecutive notes on one staff spanning at most a third share one height (MEI `@vgrp`), so they no longer move up and down with each note's accidental.
- `nb.show_mei(highlight="#eb6834")`: hovering a notehead colours it and its cents label, and hovering a label colours its note, also inside dense chords. CSS only (`:has()`, no script), scoped to each output; `highlight=None` turns it off. Cents and onset labels carry MEI `@type` (`cents`, `onset`); onset labels are not paired.
- MEI chords (`snapshot_to_mei`, `events_to_mei`) show every note: Verovio draws at most two noteheads per staff step, so a staff with three or more notes on one step (D4 quarter-sharp, D♯4, D♯4, D4 three-quarter-sharp) continues the chord in further columns of the same measure, two notes per step each. Before, the third and fourth heads were drawn exactly on top of others (0.6 did the same). Chords without such a step are unchanged.

### Removed
- `audiospylt.mei_old`, `requirements.txt` and the matplotlib dependency.
- The hard dependencies on `ipython`, `ipykernel`, `ipywidgets`, `nbformat`, `tqdm`, `plotly`, `verovio` and `librosa`; they moved to the extras above.

## [0.6.3] - 2026-03-02

### Added
- `ipykernel` as a dependency so notebooks run in fresh environments without requiring a separate kernel install.

### Fixed
- `save_df_tsv(...)` now logs a normalized absolute output path instead of mixing hardcoded Windows-style separators with relative notebook paths.

### Changed
- Aligned plotting size parameters across the public plotting helpers by adding `plot_width` and `plot_height` support to `plot_waves(...)`, `plot_spectrogram(...)`, the `multiplotter.py` plotting functions, and `plot_waveforms(...)`, while keeping legacy size aliases where needed for backward compatibility.
- Added optional `plot_width` and `plot_height` to `merge_cdf_analysis(...)` and `merge_cdf(...)` so the CDF merge plots follow the same notebook plotting convention.
- Added optional `plot_width` and `plot_height` to `merge_sigmoid_analysis(...)` and `merge_sigmoid(...)` so sigmoid merge plots now follow the same sizing convention as the rest of the merge helpers.
- Added `show_plot`, `plot_width`, `plot_height`, and `plotly_layout` to `summarize_result(...)` so stored `search_f0(...)` figures can be displayed and resized from the summary helper without re-running the search plot.
- `plot_scatter_binned(...)` now uses a stepped count legend so low bin counts read more clearly than with a fully smooth colorbar.
- Expanded `tutorials_tech/symbolic_mei.ipynb` to explain `notebook_white_background`, dark-theme SVG contrast issues, and why saved SVG output remains transparent and may need a light background when viewed elsewhere.
- Updated `tutorials_tech/2df_merge.ipynb` so all merge-analysis and merge-plot examples pass explicit `plot_width` and `plot_height` values consistently.

## [0.6.2] - 2026-03-01

### Added
- New `search_f0(...)` utility for f0 detection from measured partial frequencies, with optional RMS-cents search-curve plotting.
- `search_f0(...)` now supports `top_k` candidate reporting, manual/amplitude-based weighting, robust outlier handling, dataframe input (`peaks_df`), and optional coarse-to-fine search.
- Added package-level `summarize_result(...)` helper for formatted `search_f0(...)` output with configurable precision settings.
- Added `notebook_white_background` to MEI notebook rendering helpers so Verovio previews can be shown on a white panel in dark notebook themes without changing saved transparent SVG output.

### Changed
- Enhanced `analyze_signal(...)` partial tracking overlays for multi-f0 workflows, with one color group per f0 and dotted center lines.
- Partial bandwidth highlights now toggle together with their matching partial center lines in the Plotly legend (per f0 group).
- Updated `case_studies/showcase_parm_v1.ipynb` to reflect current API and track ongoing case-study revisions.
- Added explicit `search_f0(...)` parameter reference and annotated usage example to the README.
- MEI notebook previews now default to a white canvas wrapper for better legibility in dark-themed Jupyter environments.

## [0.6.1] - 2026-02-21

### Added
- GitHub Actions workflow for publishing to PyPI.

### Fixed
- Removed the `typing` dependency from package requirements to avoid Google Colab kernel restart prompts after `pip install audiospylt`.

### Changed
- Raised the minimum supported Python version to 3.12.

## [0.6.0a2] - 2026-02-01

### Added
- First PyPI release (`pip install audiospylt`).
- New notebook organization:
  - `case_studies/`: narrative-driven notebooks demonstrating end-to-end workflows (analytic and creative).
  - `tutorials_tech/`: focused references explaining individual functions and parameters in isolation.
  - `tutorials_workflow/`: toolchain-based examples for audio data handling, spectral analysis, DataFrame manipulation, and sound synthesis.
- Several new notebooks added across these sections.
- Google Colab demo notebooks (should work similarly on the Jupyter4NFDI platform).

### Changed
- Major refactor and cleanup for the first official PyPI release.
- Integrated notebook-related dependencies (`ipython`, `ipywidgets`, `nbformat`) into core requirements for easier installation.
- Improved package structure for better distribution.
- Reorganized most existing notebooks into the new structure (legacy notebooks are kept in `notebooks_legacy/` for now).

## [0.5.0-alpha] - 2025-10-09

### Changed
- Major cleanup of the entire repository.
- Moved all conference-related resources (presentations, examples, posters) to the `conferences/` folder (GMTH23, INMUSIC24, MEC2025).
- Moved `exploration_of_timbre/` to `conferences/inmusic24/`.

## [0.4.0-alpha] - 2025-05-20

### Changed
- Improved and refactored MEI implementation (quarter-tone, eighth-tone, temporal structures).
- Added SVG export for MEI.
- Refactored Python scripts for plotting TSV frequency tables (`multiplotter.py`) and SSM representations (`ssm.py`).

## [0.3.0-alpha] - 2024-06-14

### Added
- Experimental ML-based sound generation (`exploration_of_timbre/`) using global optimization for FM/AM synthesis.
- New notebooks for spectral approximation and Ableton Operator preset editing.
- Optimization process visualizations and objective function distance demos.

## [0.2.0-alpha] - 2024-05-15

### Changed
- Complete code refactor for clarity and consistency.
- Centralized Python scripts in the `py_scripts/` folder.
- Rewrote `symbolic_mei.ipynb` for robust Verovio-based rendering with microtonal support.

### Added
- Comprehensive `tutorials/` folder covering MFCCs, peak detection, aliasing, and DFT resolution.
- `ssm.ipynb` for Plotly-based self-similarity matrix visualization.
