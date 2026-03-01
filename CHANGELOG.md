# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
