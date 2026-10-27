"""Smoke-test an installed audiospylt wheel in a fresh venv: core-only or with [all].

Usage (from outside the checkout, so that the import finds the installed package):
    cd "$RUNNER_TEMP" && /path/to/venv/bin/python -I /path/to/scripts/smoke_install.py core
    cd "$RUNNER_TEMP" && /path/to/venv/bin/python -I /path/to/scripts/smoke_install.py all

Both modes run the compute pipeline on a synthetic stereo file: write_audio, load_audio with
resampling, spectrum, find_peaks, snapshot_to_events, tracks_to_events, synthesize, a table round
trip, read_partiels, write_sdif / read_sdif, tracks_to_chords, tracks_at, pitch_table, snapshot_to_mei, events_to_mei and
harmonic_f0. "core" then checks that no optional package is installed or imported and that each
optional feature names its extra; "all" checks that those features work and that the 0.6 API
still resolves. Standard library and core dependencies only; no
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

        with check("tracks_to_events -> synthesize: one phase-continuous 440 Hz voice"):
            tracks = pd.DataFrame({"track_id": 0, "time": [0.0, 0.2, 0.5], "freq": 440.0, "amp": 0.5})
            voice = asp.synthesize(asp.tracks_to_events(tracks), sr=sr, fade=0)
            expected = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr)
            assert np.abs(voice - expected).max() < 1e-9

        with check("write_table / read_table round trip"):
            pd.testing.assert_frame_equal(asp.read_table(asp.write_table(events, folder / "events.tsv")), events)

        with check("read_partiels: a stereo PM2 CSV export, channel 1 split at its gap"):
            block = "0.01,0,440,0.5,0\n0.02,0,441,0.5,1\n0.03,0,\n0.04,0,442,0.25,2\n"
            (folder / "Group_1_Partial_1.csv").write_text(block + "\n" + block.replace("0.5", "0.4"))
            tracks = asp.read_partiels(folder / "Group_1_Partial_1.csv", channel=1)
            assert list(tracks["track_id"]) == [0, 0, 1] and list(tracks["amp"]) == [0.4, 0.4, 0.25], tracks

        with check("write_sdif / read_sdif: two tracks on different grids, the shorter one filled"):
            tracks = pd.DataFrame({"track_id": [0, 0, 1, 1, 1], "time": [0.0, 0.2, 0.05, 0.1, 0.15],
                                   "freq": [100.0, 300.0, 440.0, 440.0, 440.0], "amp": 0.5})
            back = asp.read_sdif(asp.write_sdif(tracks, folder / "tracks.sdif"))
            assert list(back.groupby("partial_id").size()) == [5, 3], back
            assert list(back.loc[back["partial_id"] == 1, "freq"]) == [100.0, 150.0, 200.0, 250.0, 300.0], back

        with check("tracks_to_chords: one chord per window, the louder of two tracks 20 cents apart"):
            tracks = pd.DataFrame({"track_id": [0, 0, 1, 1, 2, 2], "time": [0.0, 0.1] * 3,
                                   "freq": [440.0, 440.0, 445.0, 445.0, 660.0, 660.0],
                                   "amp": [0.5, 0.5, 0.2, 0.2, 0.25, 0.25]})
            chords = asp.tracks_to_chords(tracks, 0.5)
            assert list(chords["freq_start"]) == [440.0, 660.0] and list(chords["track_id"]) == [0, 2], chords
            assert list(asp.tracks_at(tracks, 0.05)["freq"]) == [440.0, 445.0, 660.0]

    with check("pitch_table -> snapshot_to_mei"):
        assert list(asp.pitch_table(peaks["freq"])["written_midi"]) == [69, 76]
        mei = asp.snapshot_to_mei(peaks)
        assert "<mei" in mei

    with check("events_to_mei writes the onset times (<when>, measure@when)"):
        events_mei = asp.events_to_mei(events)
        assert '<when xml:id="w1" absolute="00:00:00" />' in events_mei, events_mei
        assert 'when="#w1"' in events_mei

    with check("harmonic_f0: 440 + 660 Hz -> 220 Hz (harmonics 2, 3)"):
        result = asp.harmonic_f0(peaks, fmin=100, fmax=300)
        assert abs(result.f0 - 220) < 0.01 and list(result.harmonics) == [2, 3], result

    if mode == "core":
        with check("no optional package installed or imported"):
            installed = [name for name in OPTIONAL if importlib.util.find_spec(name) is not None]
            loaded = [name for name in OPTIONAL if name in sys.modules]
            assert not installed and not loaded, f"installed {installed}, imported {loaded}"
        with check("mel_spectrogram names [librosa]"):
            raises_naming(lambda: asp.mel_spectrogram(asp.spectrogram(audio, sr)), "librosa")
        with check("pyin names [librosa]"):
            raises_naming(lambda: asp.pyin(audio, sr, fmin=100, fmax=1000), "librosa")
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
        with check("pyin -> f0_to_tracks -> tracks_to_events: missing fundamental 220 Hz"):
            contour = asp.pyin(audio, sr, fmin=100, fmax=1000)
            assert contour.voiced.all() and abs(np.median(contour.f0) - 220) < 0.1, contour
            assert len(asp.tracks_to_events(asp.f0_to_tracks(contour))) == len(contour.f0) - 1
        with check("viz.plot_spectrum returns a Plotly figure"):
            import plotly.graph_objects as go

            assert isinstance(asp.viz.plot_spectrum(spec, peaks), go.Figure)
        with check("render_svg returns SVG; events get a time ruler"):
            assert asp.render_svg(mei).startswith("<svg") and "asp-ruler" not in asp.render_svg(mei)
            assert 'class="asp-ruler"' in asp.render_svg(events_mei)
        with check("0.6 detect_dataframe_type on v1 events: multi_frame, one DeprecationWarning"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                kind = asp.detect_dataframe_type(events)["type"]
            deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert kind == "multi_frame" and len(deprecations) == 1, (kind, [str(w.message) for w in caught])
    return 0


if __name__ == "__main__":
    sys.exit(main())
