![AudioSpylt Logo](./logo.png)

# AudioSpylt

> **Note**: This package is currently under development. The provided version should be treated as an alpha release.

**AudioSpylt** is a Python-based toolbox designed for sound analysis, resynthesis, and a variety of visual and symbolic sound representations. While it is primarily intended for instructional purposes, this toolbox seamlessly integrates with the Jupyter Notebook environment. Originally created for composition students, it places a special emphasis on diverse resynthesis techniques.

## Update alpha-0.4

- Improved and refactored MEI implementation, including quarter-tone and eighth-tonal representations, as well as basic support for temporal structure representation. Added SVG export. Check the `symbolic_mei.ipynb` notebook for more details.
- Improved and refactored Python scripts for plotting TSV frequency tables and SSM representations (`multiplotter.py` and `ssm.py`).

## Update alpha-0.3

- Added the `exploration_of_timbre/` directory with various experimental ML-based sound generation approaches using global optimization algorithms. The primary goal is to synthesize a DFT frame within a restricted FM/AM setup, optionally for use within Ableton Operator.
- `spectral_fm3.ipynb` - Notebook for FM-based sound approximation using a single DFT frame as the source.
- `spectral_am3.ipynb` - Notebook for AM-based sound approximation using a single DFT frame as the source.
- `operator_fm.ipynb` - Notebook for adjusting calculated FM values to the Ableton Operator preset format.
- `operator_am.ipynb` - Notebook for adjusting calculated AM values to the Ableton Operator preset format.
- `operator_preset_editor_fm.ipynb` - Notebook for extracting and saving Ableton Operator presets in `.adv` format (its native preset format).
- `optimization_gif.ipynb` - Notebook for creating learning process visualizations for different optimization algorithms.
- `distances_demo.ipynb` - Notebook for visualizing distances of different objective functions.

## Update alpha-0.2

### Refactoring

- All code has been refactored.
- All Python scripts have been moved to the `py_scripts/` folder.

### Tutorials Added

All tutorials can be found in the `tutorials/` folder:

- `mfcc_bank.ipynb` - Brief introduction to MFCC-based sound representations.
- `peaks_scipy_showcase.ipynb` - Quick introduction to the `find_peaks` function from `scipy.signal`, used for DFT-based peak filtering.
- `showcase_bayle.ipynb`, `showcase_noanoa.ipynb`, `showcase_parm.ipynb` - Various examples of DFT-based peak detection and resynthesis, aiming to extract symbolic representations from sounds and resynthesize DFT frames for aural judgment and exploration of the analyzed audio.
- `above_nyquist.ipynb` - Brief introduction to the effects of aliasing.
- `dft_resolution.ipynb` - Brief introduction to the effects of sampling rate and sample length on DFT resolution.

### Notebooks Added/Revised

- `symbolic_mei.ipynb` - Completely rewritten implementation of Verovio-based MEI rendering (see also `mei.py` in the `py_scripts/` folder). Now supports various rendering modes, including MIDI cent deviation notation above notes (useful for microtonal analysis).
- `ssm.ipynb` - Plotly-based self-similarity matrix (SSM) visualization of selected audio files; includes analysis methods such as 'chroma', 'mfcc', or 'chroma+mfcc'.

## Important Resources

- Slides from the 23rd GMTH Congress talk can be found in the `gmth_congress_slides/` folder.

## Toolbox Overview

The toolbox is organized into the following main categories:

### Instructional Notebooks
These notebooks provide comprehensive explanations and demonstrations of core audio concepts:

- **`wave_sampling_window`**:
  - Covers sampling rate, Nyquist frequency, and window functions.
  - Discusses the implications of sampled material length on frequency resolution.
- **`wave_vs_dft_3d`**:
  - Displays 2D and 3D representations of DFT spectra.
  - Emphasizes sine/cosine component visuals.

### Analysis Notebooks
- **`audio_load_dft`**:
  - Incorporates basic audio editing functions such as trim and fade.
  - Offers customizable peak detection methods.
  - Features thresholding functions and splits analyzed data into multiple DFTs.

### Visualizations and Symbolic Rendering
- **`visual_tsv`**:
  - Plotting scripts for data from TSV files and pandas DataFrames.
- **`symbolic_mei`**:
  - Symbolic visualizations tailored for data from TSV files or pandas DataFrames.

### TSV Manipulations and Resynthesis
- **`df_pitch_stretch`**:
  - Implements pitch/stretch alterations on time-domain data stored in TSV files.
- **`2df_copypaste`**, **`2df_merge`**:
  - Executes freeze effects and various kinds of spectral interpolation.
- **`resynth`**:
  - Resynthesizes audio based on time-domain data from TSV files.

## Getting Started

To get started, clone this repository and set up your Jupyter Notebook environment to run the notebooks.

## Dependencies

AudioSpylt requires the following Python libraries:

- `IPython`
- `librosa`
- `matplotlib`
- `numpy`
- `pandas`
- `plotly`
- `requests`
- `scipy`
- `soundfile`
- `tqdm`
- `verovio`

### Installation

To install the dependencies, navigate to the root directory of the project and run:

```bash
pip install -r requirements.txt
```

## Contributions

Your contributions are welcome! Feel free to enhance the project through pull requests or by opening issues.

## License

AudioSpylt is licensed under the MIT License.
