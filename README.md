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

> **Note**: This package is currently under development. This branch is version `1.0.0.dev0`, the v1 redesign; the latest release on PyPI is `0.6.3`. Both should be treated as pre-stable releases: although the package has been in development for some time, bugs and undocumented features are still common.

**AudioSpylt** is a Python-based toolbox designed for sound analysis, resynthesis, and a variety of visual and symbolic sound representations. While it is primarily intended for instructional purposes, this toolbox seamlessly integrates with the Jupyter Notebook environment. Originally created for composition students, it places a special emphasis on diverse resynthesis techniques.

For a detailed history of changes, see the [CHANGELOG.md](https://github.com/egorpol/audiospylt/blob/main/CHANGELOG.md).

## Toolbox Overview

The toolbox is organized into the following main categories:

- `/audiospylt` - the Python package
- `/case_studies` - narrative notebooks with end-to-end analytic and creative workflows
- `/data/tables` - small example tables (TSV) read by the notebooks
- `/samples` - audio excerpts and a score page used in the case studies (sources and rights in `samples/README.md`)
- `/tests` - test suite
- `/tutorials_tech` - function-focused references for individual parameters and utilities

Older notebooks and conference material are kept on the `archive` branch.

## Google Colab Demos

Demo notebooks are available in this [Google Drive folder](https://drive.google.com/drive/folders/157kHu95PW8tM25Bgyc4NzcVVcLfrqp0g?usp=sharing).

## Getting Started

### Installation

```bash
pip install audiospylt
```

AudioSpylt targets Python 3.11+. `pip install audiospylt` installs the core: audio and table I/O, spectra,
peaks, spectrograms, synthesis, transforms and MEI export. Optional features are extras:

| Extra | Adds |
| --- | --- |
| `[plot]` | Plotly figures (`audiospylt.viz`) |
| `[notebook]` | display, playback and Jupyter support (`audiospylt.nb`) |
| `[notation]` | SVG rendering of MEI with Verovio |
| `[librosa]` | mel spectrograms and pyin f0 tracking (`asp.pyin`) |
| `[all]` | all of the above, and the 0.6 API |

The notebooks, on Colab as well, need all of them:

```bash
pip install "audiospylt[all]"
```

### Running the notebooks

Clone the repository and install the package into the environment that runs your Jupyter kernel:

```bash
git clone https://github.com/egorpol/audiospylt.git
cd audiospylt
pip install -e ".[all]"
```

The notebooks use paths relative to their own folder (Jupyter's default working directory). They read example
data from `data/tables/` and `samples/` and write their outputs (`tsv/`, `mei/`, `rendered_audio/`) next to the
notebook; these output folders are ignored by git. `tutorials_tech/audio_load_dft_single.ipynb` loads its example
sound from a URL and needs network access.

To check that every notebook runs from start to finish, install the development extra and run the headless
runner; `--offline` skips the notebooks that need network access:

```bash
pip install -e ".[dev]"
python scripts/run_notebooks.py --offline
```

The tests run from a git checkout only, with `pip install -e ".[dev]"` and `pytest`: the source distribution on
PyPI ships the package only, without tests or example data.

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `soundfile`, `soxr` (resampling) and `requests` (loading audio from URLs).

Extras:

- `[plot]`: `plotly`
- `[notebook]`: `ipython`, `ipykernel`, `ipywidgets`, `nbformat>=4.2.0`, `tqdm`
- `[notation]`: `verovio`
- `[librosa]`: `librosa`
- `[all]`: the four extras above
- `[dev]`: `[all]` plus the build, lint, type-check and test tools

## Contributions

Contributions are welcome through pull requests and issues.

## License

AudioSpylt is released under the MIT License.
