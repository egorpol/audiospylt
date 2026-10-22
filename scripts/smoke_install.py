"""Smoke-test an installed audiospylt wheel in a fresh venv: core-only or with [all].

Usage (from outside the checkout, so that the import finds the installed package):
    cd "$RUNNER_TEMP" && /path/to/venv/bin/python -I /path/to/scripts/smoke_install.py core
    cd "$RUNNER_TEMP" && /path/to/venv/bin/python -I /path/to/scripts/smoke_install.py all

Both modes run the compute pipeline on a synthetic stereo file: write_audio, load_audio with
resampling, spectrum, find_peaks, snapshot_to_events, synthesize, a table round trip,
pitch_table and snapshot_to_mei. "core" then checks that no optional package is installed or
imported and that each optional feature names its extra; "all" checks that those features
work and that the 0.6 API still resolves. Standard library and core dependencies only; no
repository files, no network. Prints one line per check and exits 1 on the first failure.
"""

import argparse
from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import pydoc
import sys
import sysconfig
import tempfile
import warnings

import numpy as np
import pandas as pd

import audiospylt as asp

# Not in the core dependency closure: the notebook stack, plotting, librosa and its numba.
OPTIONAL = (
    "librosa", "plotly", "IPython", "verovio", "matplotlib", "numba", "ipywidgets", "nbformat", "tqdm",
)


@contextmanager
def check(label):
    try:
        yield
    except Exception as error:
        print(f"FAIL  {label}: {type(error).__name__}: {error}", flush=True)
        sys.exit(1)
    print(f"ok    {label}", flush=True)


def raises_naming(call, extra):
    try:
        call()
    except ImportError as error:
        assert f"install audiospylt[{extra}]" in str(error), str(error)
    else:
        raise AssertionError("no ImportError")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["core", "all"])
    mode = parser.parse_args(argv).mode

    with check(f"audiospylt {asp.__version__} installed in the venv with py.typed and __init__.pyi"):
        package = Path(asp.__file__).resolve().parent
        site = Path(sysconfig.get_paths()["purelib"]).resolve()
        assert sys.prefix != sys.base_prefix, "not running in a venv"
        assert package.parent == site, f"{package} is not in {site}"
        for name in ("py.typed", "__init__.pyi"):
            assert (package / name).is_file(), f"{name} missing next to {package}"

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        with check("write_audio (stereo 44.1 kHz FLOAT) -> load_audio(sr=22050)"):
            times = np.arange(44100) / 44100
            tone = 0.5 * np.sin(2 * np.pi * 440 * times) + 0.25 * np.sin(2 * np.pi * 660 * times)
            path = asp.write_audio(folder / "tones.wav", np.stack([tone, tone]), 44100, subtype="FLOAT")
            audio, sr = asp.load_audio(path, sr=22050)
            assert sr == 22050 and audio.shape == (22050,), (audio.shape, sr)

        with check("spectrum -> find_peaks: 440 Hz at -6 dB and 660 Hz at -12 dB"):
            spec = asp.spectrum(audio, sr)
            peaks = asp.find_peaks(spec, min_db=-20)
            assert len(peaks) == 2, peaks
            for (_, row), freq, amp in zip(peaks.iterrows(), (440, 660), (0.5, 0.25)):
                assert abs(row["freq"] - freq) < 1, row
                assert abs(asp.amplitude_to_db(row["amp"]) - asp.amplitude_to_db(amp)) < 0.5, row

        with check("snapshot_to_events -> synthesize"):
            events = asp.snapshot_to_events(peaks, 0.0, 0.5)
            assert asp.synthesize(events, sr=sr).shape == (sr // 2,)

        with check("write_table / read_table round trip"):
            pd.testing.assert_frame_equal(asp.read_table(asp.write_table(events, folder / "events.tsv")), events)

    with check("pitch_table -> snapshot_to_mei"):
        assert list(asp.pitch_table(peaks["freq"])["written_midi"]) == [69, 76]
        mei = asp.snapshot_to_mei(peaks)
        assert "<mei" in mei

    if mode == "core":
        with check("no optional package installed or imported"):
            installed = [name for name in OPTIONAL if importlib.util.find_spec(name) is not None]
            loaded = [name for name in OPTIONAL if name in sys.modules]
            assert not installed and not loaded, f"installed {installed}, imported {loaded}"
        with check("mel_spectrogram names [librosa]"):
            raises_naming(lambda: asp.mel_spectrogram(asp.spectrogram(audio, sr)), "librosa")
        with check("render_svg names [notation]"):
            raises_naming(lambda: asp.render_svg(mei), "notation")
        with check("import audiospylt.viz names [plot]"):
            raises_naming(lambda: importlib.import_module("audiospylt.viz"), "plot")
        with check("help(audiospylt) renders; dir() leaves out viz"):
            pydoc.render_doc(asp)
            assert "viz" not in dir(asp) and "nb" in dir(asp), dir(asp)
        with check("nb.play names [notebook]"):
            raises_naming(lambda: asp.nb.play(audio, sr), "notebook")
        # render_audio needs core packages only; this 0.6 name fails on plotly first.
        with check("0.6 name load_audio_sample_and_preview names [all]"):
            raises_naming(lambda: asp.load_audio_sample_and_preview, "all")
    else:
        with check("mel_spectrogram"):
            assert isinstance(asp.mel_spectrogram(asp.spectrogram(audio, sr)), asp.MelSpectrogram)
        with check("viz.plot_spectrum returns a Plotly figure"):
            import plotly.graph_objects as go

            assert isinstance(asp.viz.plot_spectrum(spec, peaks), go.Figure)
        with check("render_svg returns SVG"):
            assert asp.render_svg(mei).startswith("<svg")
        with check("0.6 detect_dataframe_type on v1 events: multi_frame, one DeprecationWarning"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                kind = asp.detect_dataframe_type(events)["type"]
            deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert kind == "multi_frame" and len(deprecations) == 1, (kind, [str(w.message) for w in caught])
    return 0


if __name__ == "__main__":
    sys.exit(main())
