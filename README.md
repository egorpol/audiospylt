<p align="center">
  <img src="https://raw.githubusercontent.com/egorpol/audiospylt/main/logo.png" alt="AudioSpylt Logo" width="240">
</p>

# AudioSpylt

[![PyPI](https://img.shields.io/pypi/v/audiospylt)](https://pypi.org/project/audiospylt/)
[![Python](https://img.shields.io/pypi/pyversions/audiospylt)](https://pypi.org/project/audiospylt/)
[![License](https://img.shields.io/github/license/egorpol/audiospylt)](https://github.com/egorpol/audiospylt/blob/main/LICENSE)
[![Publish to PyPI](https://github.com/egorpol/audiospylt/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/egorpol/audiospylt/actions/workflows/publish-pypi.yml)
[![GitHub Release](https://github.com/egorpol/audiospylt/actions/workflows/release-from-changelog.yml/badge.svg)](https://github.com/egorpol/audiospylt/actions/workflows/release-from-changelog.yml)
![Jupyter](https://img.shields.io/badge/notebooks-Jupyter-F37626?logo=jupyter&logoColor=white)

> **Note**: This package is currently under development. The provided version (`0.6.3`) should be treated as a pre-stable release. Although the package has been in development for some time, bugs and undocumented features are still common.

**AudioSpylt** is a Python-based toolbox designed for sound analysis, resynthesis, and a variety of visual and symbolic sound representations. While it is primarily intended for instructional purposes, this toolbox seamlessly integrates with the Jupyter Notebook environment. Originally created for composition students, it places a special emphasis on diverse resynthesis techniques.

For a detailed history of changes, see the [CHANGELOG.md](https://github.com/egorpol/audiospylt/blob/main/CHANGELOG.md).

## Toolbox Overview

The toolbox is organized into the following main categories:

- `/case_studies` - narrative notebooks with end-to-end analytic and creative workflows
- `/conferences` - materials from conference presentations featuring AudioSpylt
- `/mei` - test output directory for generated MEI files
- `/samples` - audio files and score sheets used in the case studies
- `/tsv` - sample TSV files used in notebooks
- `/tutorials_tech` - function-focused references for individual parameters and utilities
- `/tutorials_workflow` - workflow examples that combine multiple tools for analysis, transformation, and synthesis

## Google Colab Demos

Demo notebooks are available in this [Google Drive folder](https://drive.google.com/drive/folders/157kHu95PW8tM25Bgyc4NzcVVcLfrqp0g?usp=sharing).

## Getting Started

### Installation

```bash
pip install audiospylt
```

AudioSpylt targets Python 3.12+. When installed from PyPI, `pip` will install the runtime dependencies automatically.

If you are working from a local clone for notebooks or development, you can also install the mirrored dependency list with:

```bash
pip install -r requirements.txt
```

## Dependencies

Current runtime dependencies:

- `ipython`, `ipykernel`, `ipywidgets`, `nbformat>=4.2.0` (notebook support)
- `librosa`, `soundfile` (audio processing)
- `numpy`, `pandas`, `scipy` (data science)
- `matplotlib`, `plotly` (visualization)
- `verovio` (symbolic rendering)
- `requests`, `tqdm` (utilities)

## Contributions

Contributions are welcome through pull requests and issues.

## License

AudioSpylt is released under the MIT License.
