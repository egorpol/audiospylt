![AudioSpylt Logo](https://raw.githubusercontent.com/egorpol/audiospylt/main/logo.png)

# AudioSpylt

> **Note**: This package is currently under development. The provided version (`0.6.0a2`) should be treated as an alpha release. Although the package is already in development for quite some time, bugs and undocumennted features are still quite common. 

**AudioSpylt** is a Python-based toolbox designed for sound analysis, resynthesis, and a variety of visual and symbolic sound representations. While it is primarily intended for instructional purposes, this toolbox seamlessly integrates with the Jupyter Notebook environment. Originally created for composition students, it places a special emphasis on diverse resynthesis techniques.



For a detailed history of changes, see the [CHANGELOG.md](./CHANGELOG.md).

## Toolbox Overview

The toolbox is organized into the following main categories:

/case_studies - narrative-driven notebooks that demonstrate end-to-end workflows for both analytic and creative purposes
/conferences - various materials from past conferences where audiospylt package was presented
/mei - test file directory for mei output 
/samples - contains samples (audio and score sheets) used in case_studies
/tutorials_tech - explain individual functions and parameters in isolation, serving as a functional reference. 
/tutorials_workflow - focused examples demonstrating specific toolchains for audio data handling, spectral analysis, DataFrame manipulation, and sound synthesis. Showcasing mostly interaction between different modules.


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

### Installation (pip package)

```bash
pip install audiospylt
```
## Dependencies

AudioSpylt requires the following Python libraries:

- `IPython`, `ipywidgets`, `nbformat` (for notebook support)
- `librosa`, `soundfile` (audio processing)
- `numpy`, `scipy`, `pandas` (data science)
- `matplotlib`, `plotly` (visualization)
- `verovio` (symbolic rendering)
- `requests`, `tqdm` (utilities)

## Contributions

Your contributions are welcome! Feel free to enhance the project through pull requests or by opening issues.

## License

AudioSpylt is licensed under the MIT License.
