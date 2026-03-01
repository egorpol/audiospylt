<p align="center">
  <img src="https://raw.githubusercontent.com/egorpol/audiospylt/main/logo.png" alt="AudioSpylt Logo" width="240">
</p>

# AudioSpylt

> **Note**: This package is currently under development. The provided version (`0.6.2`) should be treated as a pre-stable release. Although the package has been in development for some time, bugs and undocumented features are still common.

**AudioSpylt** is a Python-based toolbox designed for sound analysis, resynthesis, and a variety of visual and symbolic sound representations. While it is primarily intended for instructional purposes, this toolbox seamlessly integrates with the Jupyter Notebook environment. Originally created for composition students, it places a special emphasis on diverse resynthesis techniques.

For a detailed history of changes, see the [CHANGELOG.md](https://github.com/egorpol/audiospylt/blob/main/CHANGELOG.md).

## Toolbox Overview

The toolbox is organized into the following main categories:

- `/case_studies` - narrative-driven notebooks that demonstrate end-to-end workflows for both analytic and creative purposes
- `/conferences` - various materials from past conferences where the AudioSpylt package was presented
- `/mei` - test file directory for MEI output
- `/samples` - contains samples (audio and score sheets) used in case_studies
- `/tutorials_tech` - explains individual functions and parameters in isolation, serving as a functional reference
- `/tutorials_workflow` - focused examples demonstrating specific toolchains for audio data handling, spectral analysis, DataFrame manipulation, and sound synthesis, showcasing interactions between different modules

## Google Colab Demos
You can find demo folder here: https://drive.google.com/drive/folders/157kHu95PW8tM25Bgyc4NzcVVcLfrqp0g?usp=sharing

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
