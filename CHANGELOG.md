# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0-alpha] - YYYY-MM-DD

### Changed
- Major cleanup of the entire repository.
- All conference-related resources (presentations, examples, posters) have been moved to the `conferences/` folder. This includes dedicated folders for GMTH23, INMUSIC24, and MEC2025.
- The `exploration_of_timbre/` directory has been moved to `conferences/inmusic24/` for reference. Further development will continue at [egorpol/FFTimbre](https://github.com/egorpol/FFTimbre).

## [0.4.0-alpha] - 2025-05-20

### Changed
- Improved and refactored the MEI implementation, adding support for quarter-tone and eighth-tone representations, as well as basic temporal structures.
- Improved and refactored Python scripts for plotting TSV frequency tables (`multiplotter.py`) and SSM representations (`ssm.py`).

### Added
- SVG export functionality for MEI representations. See `symbolic_mei.ipynb` for details.

## [0.3.0-alpha] - 2024-06-14

### Added
- `exploration_of_timbre/` directory with experimental ML-based sound generation using global optimization algorithms, targeting restricted FM/AM synthesis for Ableton Operator.
- `spectral_fm3.ipynb`: Notebook for FM-based sound approximation from a single DFT frame.
- `spectral_am3.ipynb`: Notebook for AM-based sound approximation from a single DFT frame.
- `operator_fm.ipynb`: Notebook to adjust calculated FM values into Ableton Operator preset format.
- `operator_am.ipynb`: Notebook to adjust calculated AM values into Ableton Operator preset format.
- `operator_preset_editor_fm.ipynb`: Notebook for extracting and saving Ableton Operator presets (`.adv`).
- `optimization_gif.ipynb`: Notebook for visualizing the learning process of different optimization algorithms.
- `distances_demo.ipynb`: Notebook for visualizing distances of different objective functions.

## [0.2.0-alpha] - 2024-05-15

### Changed
- All code has been refactored for clarity and consistency.
- All Python scripts have been centralized in the `py_scripts/` folder.
- The `symbolic_mei.ipynb` notebook was completely rewritten for a more robust Verovio-based MEI rendering, now supporting MIDI cent deviation notation for microtonal analysis.

### Added
- **Tutorials:** A new `tutorials/` folder now contains all instructional notebooks:
  - `mfcc_bank.ipynb`: Introduction to MFCC-based sound representations.
  - `peaks_scipy_showcase.ipynb`: Guide to using `find_peaks` from `scipy.signal`.
  - `showcase_bayle.ipynb`, `showcase_noanoa.ipynb`, `showcase_parm.ipynb`: Examples of DFT-based peak detection and resynthesis.
  - `above_nyquist.ipynb`: Introduction to aliasing effects.
  - `dft_resolution.ipynb`: Introduction to the effects of sampling rate and sample length on DFT resolution.
- **Notebooks:**
  - `ssm.ipynb`: Plotly-based self-similarity matrix (SSM) visualization with 'chroma', 'mfcc', or 'chroma+mfcc' analysis methods.