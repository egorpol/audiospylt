
![AudioSpylt Logo](./logo.png)

# AudioSpylt

> **Note**: This package is currently under development. The provided version should be treated as an alpha release.

**AudioSpylt** is a Python-based toolbox designed for sound analysis, re/synthesis, and a variety of visual and symbolic sound representations. While it is primarily intended for instructional purposes, this toolbox seamlessly integrates with the Jupyter Notebooks environment. Originally created for composition students, it places a special emphasis on diverse resynthesis techniques.

## Update alpha-0.2

### Refactoring

- All code has been refactored.
- All Python scripts are now moved to the `py_scripts` folder.

### Tutorials Added

All tutorials can be found in the `tutorials` folder:

- `mfcc_bank.ipynb` - Brief introduction to MFCC-based sound representations.
- `peaks_scipy_showcase.ipynb` - Quick introduction to `find_peaks` function from `scipy.signal` used for DFT-based peak filtering.
- `showcase_bayle.ipynb`, `showcase_noanoa.ipynb` - Various examples of DFT-based peak detection and resynthesis aimed to extract symbolic representations from various sounds and resynthesize the DFT frames for aural judgement and exploration of analyzed sounds.
- `above_nyquist.ipynb` - Brief introduction to the effects of aliasing.
- `dft_resolution.ipynb` - Brief introduction to the effects of sampling rate and sample length.

### Notebooks Added/Revised

- `symbolic_mei.ipynb` - Completely rewritten implementation of Verovio-based MEI rendering (check `mei.py` in `py_scripts` folder as well). Now supports various modes of rendering, including MIDI cent deviation notation above the note (useful for microtonal analysis).
- `ssm.ipynb` - Plotly-based self-similarity matrix visualization of selected audio files, includes 'chroma', 'mfcc', or 'chroma+mfcc' analysis methods.

## Important Resources

- Slides from the 23rd GMTH Congress talk can be found in the `gmth_congress_slides` folder.

## Toolbox Overview

### 1. **Instructional Notebooks**

These notebooks are designed to provide comprehensive explanations and demonstrations on core audio concepts:

- **wave_sampling_window**: 
  - Covers sampling rate, Nyquist Frequency, window functions.
  - Discusses implications of sampled material length on frequency resolution.

- **wave_vs_dft_3d**: 
  - Displays 2D and 3D representations of DFT spectra.
  - Emphasizes sine/cosine component visuals.

### 2. **Analysis Notebooks**

- **audio_load_dft**:
  - Incorporates basic audio editing functions such as trim and fade.
  - Offers customizable peak detection methods.
  - Features thresholding functions and splits analyzed data into multiple DFTs.

### 3. **Visualizations and Symbolic Rendering**

- **visual_tsv**:
  - Plotting scripts for TSV/data frames.

- **symbolic_mei**:
  - Symbolic visualizations tailored for data frames.

### 4. **TSV Manipulations and Resynthesis**

- **df_pitch_stretch**:
  - Implement pitch/stretch alterations on TSVs with time domain data.

- **2df_copypaste**, **2df_merge**:
  - Execute freeze effects and various kinds of spectral interpolation.

- **resynth**:
  - Resynthesize based on TSVs containing time domain data.

## Getting Started

To get started, clone the repository, and set up your Jupyter Notebook environment to run the notebooks.

## Dependencies

AudioSpylt requires the following libraries:

- requests
- numpy
- plotly
- librosa
- pandas
- scipy
- soundfile
- IPython
- verovio
- tqdm
- matplotlib

### Installation

To install the dependencies, navigate to the root directory of the project and run:

```bash
pip install -r requirements.txt
```

## Contributions

Your contributions are welcome! Feel free to enhance the project through pull requests or by opening issues.

## License

AudioSpylt is licensed under the MIT License.
