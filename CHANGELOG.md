# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0a1] - 2026-01-06

### Changed
- Major refactor and cleanup for the first official PyPI release.
- Integrated notebook-related dependencies (`ipython`, `ipywidgets`, `nbformat`) into core requirements for easier installation.
- Improved package structure for better distribution.
- new tutorial notebook structure:
- /case_studies - narrative-driven notebooks that demonstrate end-to-end workflows for both analytic and creative purposes
/tutorials_tech - explain individual functions and parameters in isolation, serving as a functional reference. 
/tutorials_workflow - focused examples demonstrating specific toolchains for audio data handling, spectral analysis, DataFrame manipulation, and sound synthesis.
- most of the old notebooks were sorted accordingly, several new notebooks added
- 

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
