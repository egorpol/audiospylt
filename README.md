![AudioSpylt Logo](./logo.png)

# AudioSpylt

> **Note**: This package is currently under development. The provided version should be treated as an alpha release.

**AudioSpylt** is a Python-based toolbox designed for sound analysis, re/synthesis, and a variety of visual and symbolic sound representations. While it is primarily intended for instructional purposes, this toolbox seamlessly integrates with the Jupyter Notebooks environment. Originally created for composition students, it places a special emphasis on diverse resynthesis techniques.

## Toolbox Overview

### 1. **Instructional Notebooks**

These notebooks are designed to provide comprehensive explanations and demonstrations on core audio concepts:

- **wave_sampling_window**: 
  - Covers sampling rate, Nyquist Frequency, window functions
  - Discusses implications of sampled material length on frequency resolution

- **wave_vs_dft_3d**: 
  - Displays 2D and 3D representations of DFT spectra
  - Emphasizes sine/cosine component visuals

### 2. **Analysis Notebooks**

- **audio_load_dft**:
  - Incorporates basic audio editing functions such as trim and fade
  - Offers customizable peak detection methods
  - Features thresholding functions and splits analysed data into multiple DFTs

### 3. **Visualizations and Symbolic Rendering**

- **visual_tsv**:
  - Plotting scripts for TSV/data frames

- **symbolic_mei**:
  - Symbolic visualizations tailored for data frames

### 4. **TSV Manipulations and Resynthesis**

- **df_pitch_stretch**:
  - Implement pitch/stretch alterations on TSVs with time domain data

- **2df_copypaste, 2df_merge**:
  - Execute freeze effects and various kinds of spectral interpolation

- **resynth**:
  - Resynthesize based on TSVs containing time domain data

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
- verovio==4.0.1
- tqdm
- matplotlib

### Installation

To install the dependencies, navigate to the root directory of the project and run:

```
pip install -r requirements.txt
```

## Contributions

Your contributions are welcome! Feel free to enhance the project through pull requests or by opening issues.

## License

AudioSpylt is licensed under the MIT License.
