"""Waveform, spectrum, and spectrogram figure contracts."""

from __future__ import annotations


import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from audiospylt.analysis.peaks import find_peaks
from audiospylt.analysis.spectrogram import mel_spectrogram, spectrogram
from audiospylt.analysis.spectrum import Spectrum, amplitude_to_db, spectrum
from audiospylt.io.tables import read_table, write_table
from audiospylt.model.convert import snapshot_to_events
from audiospylt.viz._common import (
    _minmax,
    colorway_colour,
    get_template,
    sequential_colorscale,
    theme_colors,
)
from audiospylt.viz.signal import plot_spectrogram, plot_spectrum, plot_waveform


def _sine(n=2048, sr=44100, f=440.0):
    return np.sin(2 * np.pi * f * np.arange(n) / sr), sr


def _assert_no_side_effects(func, *args, capsys, monkeypatch, **kwargs):
    shown = []

    def boom(*a, **k):
        shown.append(True)
        raise AssertionError("Figure.show must not be called")

    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", boom)
    before = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            before.append(arg.copy())
        elif isinstance(arg, pd.DataFrame):
            before.append(arg.copy(deep=True))
        elif isinstance(arg, Spectrum):
            before.append((arg.freqs.copy(), arg.amp.copy()))
        else:
            before.append(None)
    fig = func(*args, **kwargs)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert shown == []
    pio.to_json(fig)
    for arg, original in zip(args, before):
        if isinstance(arg, np.ndarray):
            np.testing.assert_array_equal(arg, original)
        elif isinstance(arg, pd.DataFrame):
            pd.testing.assert_frame_equal(arg, original)
            assert arg.attrs == original.attrs
        elif isinstance(arg, Spectrum):
            np.testing.assert_array_equal(arg.freqs, original[0])
            np.testing.assert_array_equal(arg.amp, original[1])
    return fig


def test_minmax_identity_and_extrema():
    x = np.arange(10, dtype=np.float64)
    y = np.array([3.0, 1.0, 4.0, 0.0, 5.0, 9.0, 2.0, 6.0, 8.0, 7.0])
    xs, ys = _minmax(x, y, 20)
    np.testing.assert_array_equal(xs, x)
    np.testing.assert_array_equal(ys, y)
    for max_points in (2, 3, 4, 5):
        xs, ys = _minmax(x, y, max_points)
        assert len(ys) <= max_points
        assert ys.min() == y.min() and ys.max() == y.max()
        assert np.all(np.diff(xs) >= 0)
    # odd and even lengths
    for n in (7, 8, 9):
        y2 = np.arange(n, dtype=np.float64)
        xs, ys = _minmax(np.arange(n, dtype=np.float64), y2, 4)
        assert len(ys) <= 4
        assert ys.min() == 0 and ys.max() == n - 1


def test_theme_palette_and_ninth_series():
    light = theme_colors("light")
    dark = theme_colors("dark")
    assert light["paper"] == "#fcfcfb" and dark["paper"] == "#1a1a19"
    assert light["ink"] == "#0b0b0b" and dark["ink"] == "#f0efec"
    assert light["secondary"] == "#52514e" and dark["secondary"] == "#c3c2b7"
    assert light["muted"] == dark["muted"] == "#898781"
    assert light["grid"] == "#e1e0d9" and dark["grid"] == "#2c2c2a"
    assert light["axis"] == "#c3c2b7" and dark["axis"] == "#383835"
    assert light["colorway"][0] == "#2a78d6"
    assert dark["colorway"][0] == "#3987e5"
    assert len(light["colorway"]) == 8
    seq_l = sequential_colorscale("light")
    seq_d = sequential_colorscale("dark")
    assert seq_l[0][1] == "#cde2fb" and seq_l[-1][1] == "#0d366b"
    assert seq_d[0][1] == "#0d366b" and seq_d[-1][1] == "#cde2fb"
    template = get_template("light")
    assert template.layout.paper_bgcolor == "#fcfcfb"
    assert template.layout.colorway[0] == "#2a78d6"
    y = np.zeros((9, 8))
    with pytest.raises(ValueError, match="facet instead"):
        plot_waveform(y, 100)


def test_plot_waveform_fidelity_and_gl(capsys, monkeypatch):
    audio, sr = _sine(100)
    fig = _assert_no_side_effects(plot_waveform, audio, sr, capsys=capsys, monkeypatch=monkeypatch)
    assert isinstance(fig.data[0], go.Scatter)
    np.testing.assert_allclose(fig.data[0].y, audio.astype(np.float32))
    assert fig.data[0].y.dtype == np.float32
    assert fig.layout.showlegend is False
    stereo = np.vstack([audio, -audio])
    fig = plot_waveform(stereo, sr, channel_labels=["L", "R"])
    assert fig.layout.showlegend is True
    assert [t.name for t in fig.data] == ["L", "R"]
    fig_gl = plot_waveform(audio, sr, gl=True)
    assert isinstance(fig_gl.data[0], go.Scattergl)
    with pytest.raises(ValueError, match="channel_labels"):
        plot_waveform(stereo, sr, channel_labels=["only"])
    with pytest.raises(ValueError, match="finite"):
        plot_waveform(np.array([1.0, np.nan]), sr)
    with pytest.raises(ValueError, match="boolean|integer|positive"):
        plot_waveform(audio, True)


def test_plot_waveform_window_overlay(capsys, monkeypatch):
    from scipy.signal import get_window

    from audiospylt.viz._common import colorway_colour

    n, sr = 4096, 44100
    audio = np.zeros(n, dtype=np.float64)
    audio[2] = 1.0
    audio[n // 2] = 0.3
    weights = get_window("hann", n, fftbins=True)
    expected = audio * weights
    # Decimate after multiply: the edge click is windowed away; the centre survives.
    fig = plot_waveform(audio, sr, window="hann", max_points=64)
    assert abs(float(np.max(np.abs(fig.data[0].y))) - 0.3) < 0.05
    np.testing.assert_allclose(
        fig.data[0].y,
        _minmax(np.arange(n) / sr, expected, 64)[1].astype(np.float32),
    )
    assert fig.data[0].y.dtype == fig.data[1].y.dtype == np.float32
    assert fig.data[1].name == "window"
    assert fig.data[1].yaxis == "y2"
    assert fig.data[1].hovertemplate == (
        "time: %{x:.4g} s<br>window: %{y:.4g}<extra></extra>"
    )
    win_colour = colorway_colour("light", 1)
    assert fig.data[1].line.color == win_colour
    assert fig.layout.showlegend is False
    assert fig.layout.yaxis.title.text == "windowed amplitude (re full scale)"
    assert fig.layout.yaxis2.title.text == "window"
    assert fig.layout.yaxis2.title.font.color == win_colour
    assert fig.layout.yaxis2.tickfont.color == win_colour
    assert fig.layout.yaxis2.overlaying == "y"
    assert fig.layout.yaxis2.side == "right"
    assert list(fig.layout.yaxis2.range) == [0, 1]
    assert fig.layout.yaxis2.showgrid is False
    assert fig.layout.yaxis2.zeroline is False
    offset = 1.25
    fig_off = plot_waveform(audio[:128], sr, window="hann", time_offset=offset, max_points=None)
    x_full = offset + np.arange(128, dtype=np.float64) / sr
    w128 = get_window("hann", 128, fftbins=True)
    np.testing.assert_allclose(fig_off.data[0].x, x_full)
    np.testing.assert_allclose(fig_off.data[1].x, x_full)
    np.testing.assert_allclose(fig_off.data[1].y, w128.astype(np.float32))
    assert isinstance(plot_waveform(audio[:64], sr, window="hann", gl=True).data[1], go.Scattergl)
    # Kaiser beta 0 is a real window; do not treat 0.0 as None.
    fig_k = plot_waveform(audio[:32], sr, window=0.0, max_points=None)
    np.testing.assert_allclose(
        fig_k.data[1].y, get_window(0.0, 32, fftbins=True).astype(np.float32),
    )
    with pytest.raises(ValueError, match="1-D"):
        plot_waveform(audio[np.newaxis, :], sr, window="hann")
    with pytest.raises(ValueError, match="1-D"):
        plot_waveform(np.vstack([audio, audio]), sr, window="hann")
    with pytest.raises(ValueError, match="finite, nonzero sum"):
        plot_waveform(audio[:16], sr, window=("gaussian", 0))
    # window=None: one trace, no second axis, the plain title.
    mono, mono_sr = _sine(100)
    plain = plot_waveform(mono, mono_sr, window=None)
    assert len(plain.data) == 1 and "yaxis2" not in plain.to_dict()["layout"]
    assert plain.layout.yaxis.title.text == "amplitude (re full scale)"
    stereo = np.vstack([mono, -mono])
    for kwargs in (
        {},
        {"theme": "dark"},
        {"gl": True},
        {"channel_labels": ["L", "R"]},
    ):
        args = (stereo if "channel_labels" in kwargs else mono, mono_sr)
        assert plot_waveform(*args, window=None, **kwargs).to_json() == (
            plot_waveform(*args, **kwargs).to_json()
        )
    _assert_no_side_effects(
        plot_waveform, audio, sr, window="hann", max_points=128,
        capsys=capsys, monkeypatch=monkeypatch,
    )
    np.testing.assert_array_equal(audio[2], 1.0)


def test_plot_spectrum_db_and_peaks(capsys, monkeypatch):
    audio, sr = _sine()
    spec = spectrum(audio, sr)
    peaks = find_peaks(spec, prominence_db=10, rel_db=-60)
    fig = _assert_no_side_effects(
        plot_spectrum, spec, peaks, fmax=5000, capsys=capsys, monkeypatch=monkeypatch,
    )
    assert fig.data[0].line.width == 1
    assert isinstance(fig.data[0], go.Scatter)
    fmin, fmax = 0.0, 5000.0
    mask = (spec.freqs >= fmin) & (spec.freqs <= fmax)
    expected = amplitude_to_db(spec.amp[mask])
    # below max_points: y equals full cropped dB (possibly float32)
    np.testing.assert_allclose(fig.data[0].y, expected.astype(np.float32), rtol=1e-5)
    assert "peaks" in fig.data[1].name
    assert plot_spectrum(spec, gl=True).data[0].__class__.__name__ == "Scattergl"
    with pytest.raises(TypeError, match="Spectrum"):
        plot_spectrum(spectrogram(audio, sr))
    with pytest.raises(ValueError, match="freq_scale"):
        plot_spectrum(spec, freq_scale="mel")
    with pytest.raises(ValueError, match="boolean|max_points"):
        plot_spectrum(spec, max_points=True)


def test_plot_spectrum_selection_false_and_non_bool(capsys, monkeypatch):
    audio, sr = _sine()
    spec = spectrum(audio, sr)
    peaks = find_peaks(spec, min_db=-45, fmin=100, fmax=3500, prominence_db=5)
    baseline = plot_spectrum(spec, peaks, fmax=5000)
    fig = _assert_no_side_effects(
        plot_spectrum, spec, peaks, fmax=5000, selection=False,
        capsys=capsys, monkeypatch=monkeypatch,
    )
    assert len(fig.data) == len(baseline.data) == 2
    assert [t.name for t in fig.data] == [t.name for t in baseline.data]
    with pytest.raises(ValueError, match="selection must be True or False"):
        plot_spectrum(spec, peaks, selection="yes")
    with pytest.raises(ValueError, match="selection must be True or False"):
        plot_spectrum(spec, peaks, selection=1)


def test_plot_spectrum_selection_lines(capsys, monkeypatch):
    audio, sr = _sine()
    spec = spectrum(audio, sr)
    peaks = find_peaks(spec, min_db=-45, fmin=100, fmax=3500, prominence_db=5)
    fig = _assert_no_side_effects(
        plot_spectrum, spec, peaks, fmax=5000, selection=True,
        capsys=capsys, monkeypatch=monkeypatch,
    )
    names = [t.name for t in fig.data]
    assert names[2:] == ["min_db -45 dB", "fmin 100 Hz", "fmax 3500 Hz"]
    colours = [colorway_colour("light", slot) for slot in (2, 3, 3)]
    assert colours[0] != colours[1]
    for trace, colour in zip(fig.data[2:], colours, strict=True):
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines"
        assert trace.line.dash == "dash"
        assert trace.line.width == 1.5
        assert trace.line.color == colour
        assert trace.hoverinfo == "skip"
        assert trace.legendgroup == "selection"
        assert trace.showlegend is True
        # Plotly may expose tuples; values must stay plain floats (not numpy).
        assert all(type(v) is float for v in list(trace.x) + list(trace.y))
    assert list(fig.data[2].y) == [-45.0, -45.0]
    assert list(fig.data[3].x) == [100.0, 100.0]
    assert list(fig.data[4].x) == [3500.0, 3500.0]
    # Frequency lines share one y span that covers the spectrum and the amp line.
    assert fig.data[3].y == fig.data[4].y
    assert fig.data[3].y[0] <= -45.0 <= fig.data[3].y[1]
    np.testing.assert_allclose(fig.data[3].y[0], min(fig.data[0].y), rtol=1e-5)
    np.testing.assert_allclose(fig.data[3].y[1], max(fig.data[0].y), rtol=1e-5)
    fig_lin = plot_spectrum(spec, peaks, db=False, fmax=5000, selection=True)
    amp = 10 ** (-45 / 20)
    assert fig_lin.data[2].name == "min_db -45 dB"
    assert list(fig_lin.data[2].y) == [amp, amp]
    fig_floor = plot_spectrum(
        spec, peaks, floor_db=-40, fmax=5000, selection=True,
    )
    assert list(fig_floor.data[2].y) == [-40.0, -40.0]


def test_plot_spectrum_selection_frequency_lines_reach_a_high_threshold():
    audio, sr = _sine()
    spec = spectrum(audio, sr)
    top = float(np.max(plot_spectrum(spec).data[0].y))
    peaks = find_peaks(spec, min_db=top + 5, fmin=100, fmax=2000)
    assert peaks.empty
    lines = [t for t in plot_spectrum(spec, peaks, selection=True).data if t.legendgroup == "selection"]
    assert lines[0].y[0] == pytest.approx(top + 5)
    assert max(lines[1].y) == pytest.approx(top + 5)


def test_plot_spectrum_selection_off_limits_and_empty_crop(capsys, monkeypatch):
    audio, sr = _sine()
    spec = spectrum(audio, sr)
    off = find_peaks(spec, min_db=None, fmin=0.0, fmax=None, prominence_db=5)
    fig = _assert_no_side_effects(
        plot_spectrum, spec, off, fmax=5000, selection=True,
        capsys=capsys, monkeypatch=monkeypatch,
    )
    assert len(fig.data) == 2
    assert all("min_db" not in t.name and "fmin" not in t.name and "fmax" not in t.name
               for t in fig.data)
    band = find_peaks(spec, min_db=-45, fmin=100, fmax=3500, prominence_db=5)
    cropped = plot_spectrum(spec, band, fmin=500, fmax=2000, selection=True)
    assert [t.name for t in cropped.data[2:]] == ["min_db -45 dB"]
    # Crop to DC only, then log drops it: empty spectrum, no selection lines.
    empty = plot_spectrum(
        spec, band, fmin=0.0, fmax=spec.bin_hz * 0.5, freq_scale="log", selection=True,
    )
    assert len(empty.data[0].x) == 0
    assert all(getattr(t, "legendgroup", None) != "selection" for t in empty.data)


def test_plot_spectrum_selection_roundtrip_meta(tmp_path, capsys, monkeypatch):
    audio, sr = _sine()
    spec = spectrum(audio, sr)
    peaks = find_peaks(spec, min_db=-45, fmin=100, fmax=3500, prominence_db=5)
    restored = read_table(write_table(peaks, tmp_path / "peaks.tsv"))
    assert restored.attrs["meta"]["min_db"] == "-45.0"
    assert restored.attrs["meta"]["fmin"] == "100.0"
    assert restored.attrs["meta"]["fmax"] == "3500.0"
    fig = _assert_no_side_effects(
        plot_spectrum, spec, restored, fmax=5000, selection=True,
        capsys=capsys, monkeypatch=monkeypatch,
    )
    assert [t.name for t in fig.data[2:]] == ["min_db -45 dB", "fmin 100 Hz", "fmax 3500 Hz"]
    none_peaks = find_peaks(spec, min_db=None, fmin=100, fmax=3500, prominence_db=5)
    none_restored = read_table(write_table(none_peaks, tmp_path / "none.tsv"))
    assert none_restored.attrs["meta"]["min_db"] == "None"
    fig_none = plot_spectrum(spec, none_restored, fmax=5000, selection=True)
    assert [t.name for t in fig_none.data[2:]] == ["fmin 100 Hz", "fmax 3500 Hz"]


def test_plot_spectrum_selection_errors():
    audio, sr = _sine()
    spec = spectrum(audio, sr)
    with pytest.raises(ValueError, match="selection=True requires a peaks table"):
        plot_spectrum(spec, selection=True)
    peaks = find_peaks(spec, min_db=-45, fmin=100, fmax=3500, prominence_db=5)
    stripped = peaks.copy()
    stripped.attrs["meta"] = {
        key: value for key, value in peaks.attrs["meta"].items()
        if key not in ("min_db", "fmin")
    }
    with pytest.raises(
        ValueError,
        match=r"missing min_db, fmin; find_peaks and spectrogram_peaks tables carry them",
    ):
        plot_spectrum(spec, stripped, selection=True)
    bad = peaks.copy()
    bad.attrs["meta"] = dict(peaks.attrs["meta"])
    bad.attrs["meta"]["min_db"] = "loud"
    with pytest.raises(ValueError, match=r"cannot be read as a number; got 'loud'"):
        plot_spectrum(spec, bad, selection=True)


def test_plot_spectrogram_pooling_mel_overlay(capsys, monkeypatch):
    audio, sr = _sine(8192)
    spec = spectrogram(audio, sr, frame_length=512, hop=128)
    fig = _assert_no_side_effects(
        plot_spectrogram, spec, max_frames=10, capsys=capsys, monkeypatch=monkeypatch,
    )
    assert fig.data[0].z.dtype == np.float32
    assert fig.data[0].z.shape[1] <= 10
    # after pooling, zmax/zmin limits
    assert fig.data[0].zmax - fig.data[0].zmin == pytest.approx(80.0)
    # linear mode
    fig_lin = plot_spectrogram(spec, db=False, max_frames=5)
    assert fig_lin.data[0].zmin == 0
    mel = mel_spectrogram(spec, n_mels=64)
    with pytest.raises(ValueError, match="log"):
        plot_spectrogram(mel, freq_scale="log")
    peaks = find_peaks(spectrum(audio[:2048], sr), prominence_db=5)
    events = snapshot_to_events(peaks, 0.0, 0.1)
    fig_o = plot_spectrogram(spec, overlay=events, audio=audio, max_frames=20)
    assert any(isinstance(t, go.Heatmap) for t in fig_o.data)
    assert any(isinstance(t, go.Scatter) for t in fig_o.data)
    with pytest.raises(ValueError, match="snapshot_to_events"):
        plot_spectrogram(spec, overlay=peaks)
    with pytest.raises(TypeError):
        plot_spectrogram(spectrum(audio, sr))
    fig_log = plot_spectrogram(spec, freq_scale="log", max_frames=15)
    assert fig_log.layout.yaxis.type == "log"
    n_frames = spec.amp.shape[1]
    fig_p = plot_spectrogram(spec, max_frames=max(1, n_frames // 3 + 1))
    assert fig_p.data[0].z.shape[1] <= max(1, n_frames // 3 + 1)


def test_plot_spectrogram_overlay_mode():
    audio, sr = _sine(8192)
    spec = spectrogram(audio, sr, frame_length=512, hop=128)
    events = snapshot_to_events(find_peaks(spectrum(audio[:2048], sr), prominence_db=5), 0.0, 0.1)
    fig = plot_spectrogram(spec, overlay=[events, events], overlay_mode=["markers", "lines"])
    assert [t.mode for t in fig.data[1:]] == ["markers", "lines"]
    # Dots go through the same Hz -> mel mapping as lines.
    mel = mel_spectrogram(spec, n_mels=64)
    fig_mel = plot_spectrogram(mel, overlay=[events, events], overlay_mode=["markers", "lines"])
    assert set(fig_mel.data[1].y) == {y for y in fig_mel.data[2].y if y is not None}
    with pytest.raises(ValueError, match=r"overlay_mode length \(2\) must match number of tables \(1\)"):
        plot_spectrogram(spec, overlay=events, overlay_mode=["markers", "lines"])
    with pytest.raises(ValueError, match="overlay_mode must be"):
        plot_spectrogram(spec, overlay_mode="dots")


def test_log_axis_range_is_log10_of_hz():
    audio, sr = _sine(4096)
    spec = spectrum(audio, sr)
    fmax = 5000.0
    fig_log = plot_spectrum(spec, freq_scale="log", fmax=fmax)
    mask = (spec.freqs > 0) & (spec.freqs <= fmax)
    first = float(spec.freqs[mask][0])
    np.testing.assert_allclose(
        fig_log.layout.xaxis.range, [np.log10(first), np.log10(fmax)],
    )
    fig_lin = plot_spectrum(spec, freq_scale="linear", fmax=fmax)
    np.testing.assert_allclose(fig_lin.layout.xaxis.range, [0.0, fmax])

    S = spectrogram(audio, sr, frame_length=512, hop=128)
    fig_sg = plot_spectrogram(S, freq_scale="log")
    shown = S.freqs[S.freqs > 0]
    np.testing.assert_allclose(
        fig_sg.layout.yaxis.range,
        [np.log10(float(shown[0])), np.log10(float(shown[-1]))],
    )
    fig_sg_lin = plot_spectrogram(S, freq_scale="linear")
    np.testing.assert_allclose(
        fig_sg_lin.layout.yaxis.range, [float(S.freqs[0]), float(S.freqs[-1])],
    )


def test_plot_spectrogram_overlay_holds_time_axis_to_data():
    audio, sr = _sine(n=8192)
    S = spectrogram(audio, sr, frame_length=512, hop=128)
    step = float(S.times[1] - S.times[0])
    edges = [float(S.times[0]) - step / 2, float(S.times[-1]) + step / 2]
    inside = pd.DataFrame({
        "track_id": [0, 0, 1], "time": [0.02, 0.1, 0.05],
        "freq": [440.0, 440.0, 880.0], "amp": [0.5, 0.5, 0.2],
    })
    for mode in ("markers", "lines"):
        fig = plot_spectrogram(S, overlay=inside, overlay_mode=mode)
        np.testing.assert_allclose(fig.layout.xaxis.range, edges)
    np.testing.assert_allclose(
        plot_spectrogram(S, overlay=inside, audio=audio).layout.xaxis.range, edges,
    )

    beyond = inside.assign(time=[0.02, edges[1] + 0.5, 0.05])
    fig = plot_spectrogram(S, overlay=beyond, overlay_mode="markers")
    np.testing.assert_allclose(fig.layout.xaxis.range, [edges[0], edges[1] + 0.5])

    assert plot_spectrogram(S).layout.xaxis.range is None


def test_spectrogram_customdata_hz_vs_mel_and_json_size():
    from audiospylt.analysis.spectrogram import Spectrogram

    rng = np.random.default_rng(0)
    n_bins, n_frames = 233, 733
    freqs = np.linspace(0.0, 22050.0, n_bins)
    times = np.linspace(0.0, 8.0, n_frames)
    amp = rng.random((n_bins, n_frames)) * 0.1
    synth = Spectrogram(
        freqs, times, amp, 44100, 512, 256, 512, "hann", True,
    )
    fig_hz = plot_spectrogram(synth, max_frames=None)
    assert fig_hz.data[0].customdata is None
    assert "%{y:.2f}" in fig_hz.data[0].hovertemplate
    payload = pio.to_json(fig_hz)
    assert len(payload.encode("utf-8")) < 1_300_000

    audio, sr = _sine(8192)
    S = spectrogram(audio, sr, frame_length=512, hop=128)
    mel = mel_spectrogram(S, n_mels=64)
    fig_mel = plot_spectrogram(mel)
    # Plain nested lists in the JSON: plotly.js 2.35.3, the VS Code and Cursor
    # notebook renderer, cannot read a typed array per cell.
    cd = json.loads(pio.to_json(fig_mel))["data"][0]["customdata"]
    assert isinstance(cd, list) and all(isinstance(row, list) for row in cd)
    assert np.shape(cd) == fig_mel.data[0].z.shape
    np.testing.assert_allclose(np.asarray(cd)[:, 0], mel.freqs, atol=0.005)
    assert "%{customdata:.2f}" in fig_mel.data[0].hovertemplate


def test_spectrogram_log_axis_labels_hz():
    audio, sr = _sine(8192)
    S = spectrogram(audio, sr, frame_length=512, hop=128)
    yaxis = plot_spectrogram(S, freq_scale="log", fmin=100, fmax=8000).layout.yaxis
    assert yaxis.type == "log"
    assert yaxis.tickvals == (200.0, 500.0, 1000.0, 2000.0, 5000.0)
    assert yaxis.ticktext == ("200", "500", "1k", "2k", "5k")
    assert plot_spectrogram(S).layout.yaxis.tickvals is None


@pytest.mark.parametrize("fmin,fmax,labelled", [(410, 490, False), (490, 510, False), (390, 510, True)])
def test_spectrogram_narrow_log_crop_keeps_tick_labels(fmin, fmax, labelled):
    # Under two 1-9 ticks in the crop: an array tick mode would leave the axis bare.
    audio, sr = _sine(44100)
    sg = spectrogram(audio, sr, frame_length=8192, hop=2048)
    axis = plot_spectrogram(sg, freq_scale="log", fmin=fmin, fmax=fmax).layout.yaxis
    assert axis.type == "log"
    assert (axis.tickmode == "array") is labelled
    assert labelled == (axis.tickvals is not None and len(axis.tickvals) >= 2)


def test_spectrogram_legend_colorbar_and_audio_colours():
    audio, sr = _sine(4096)
    stereo = np.vstack([audio, -audio])
    S = spectrogram(audio, sr, frame_length=512, hop=128)
    peaks = find_peaks(spectrum(audio, sr), prominence_db=5)
    ev1 = snapshot_to_events(peaks, 0.0, 0.05)
    ev2 = snapshot_to_events(peaks.iloc[:3], 0.05, 0.1)
    colours = theme_colors("light")["colorway"]

    fig_stereo = plot_spectrogram(S, audio=stereo, max_frames=20, title="t")
    assert fig_stereo.layout.showlegend is True
    assert fig_stereo.layout.legend.orientation == "h"
    assert fig_stereo.layout.legend.x == 0
    assert fig_stereo.layout.legend.xanchor == "left"
    assert fig_stereo.layout.legend.y == pytest.approx(1.02)
    assert fig_stereo.layout.legend.yanchor == "bottom"
    # Paper title.y is clamped to [0, 1] in Plotly 7; container + top margin
    # keeps the title above the horizontal legend.
    assert fig_stereo.layout.title.yref == "container"
    assert fig_stereo.layout.title.yanchor == "top"
    assert fig_stereo.layout.margin.t >= 80
    assert fig_stereo.data[0].colorbar.len == pytest.approx(0.75)
    assert fig_stereo.data[0].colorbar.lenmode == "fraction"
    assert fig_stereo.data[0].colorbar.y == pytest.approx(1)
    assert fig_stereo.data[0].colorbar.yanchor == "top"
    ch = [t for t in fig_stereo.data if t.name in ("ch 1", "ch 2")]
    assert [t.name for t in ch] == ["ch 1", "ch 2"]
    assert [t.line.color for t in ch] == [colours[1], colours[2]]

    fig_mono = plot_spectrogram(S, audio=audio, max_frames=20)
    wave = [t for t in fig_mono.data if t.name == "ch 1"]
    assert wave and wave[0].line.color == theme_colors("light")["secondary"]
    assert wave[0].showlegend is False
    assert fig_mono.layout.showlegend is False

    fig_ov = plot_spectrogram(S, audio=stereo, overlay=[ev1, ev2], max_frames=20)
    overlay_colours = [
        t.line.color for t in fig_ov.data
        if t.showlegend and t.name and t.name.startswith("table")
    ]
    waves = [t for t in fig_ov.data if t.name in ("ch 1", "ch 2")]
    assert overlay_colours == [colours[1], colours[2]]
    # overlays take slots 2,3 (indices 1,2); channels slots 4,5 (indices 3,4)
    assert [t.line.color for t in waves] == [colours[3], colours[4]]

    many = [ev1] * 6
    with pytest.raises(ValueError, match="facet instead"):
        plot_spectrogram(S, audio=stereo, overlay=many, max_frames=10)


def test_mel_ticks_thinned_five_percent():
    import librosa
    from audiospylt.viz.signal import _mel_ticks

    y_min, y_max = 0.0, float(librosa.hz_to_mel(22050.0, htk=False))
    vals, texts = _mel_ticks(y_min, y_max)
    assert texts == ["200", "500", "1k", "2k", "5k", "10k", "20k"]
    span = y_max - y_min
    for a, b in zip(vals, vals[1:]):
        assert (b - a) >= 0.05 * span - 1e-12


def test_spectrogram_z_equals_floored_db_and_pooling_values():
    audio, sr = _sine(8192)
    spec = spectrogram(audio, sr, frame_length=512, hop=128)
    fig = plot_spectrogram(spec, max_frames=None, range_db=60.0)
    expected = amplitude_to_db(spec.amp).astype(np.float32)
    np.testing.assert_allclose(fig.data[0].z, expected, rtol=1e-5)
    assert fig.data[0].zmax == pytest.approx(float(np.max(expected)))
    assert fig.data[0].zmin == pytest.approx(float(np.max(expected)) - 60.0)

    n_frames = spec.amp.shape[1]
    max_frames = n_frames // 3 + 1  # non-divisible
    from math import ceil
    group = int(ceil(n_frames / max_frames))
    fig_p = plot_spectrogram(spec, max_frames=max_frames)
    n_out = int(ceil(n_frames / group))
    assert fig_p.data[0].z.shape[1] == n_out
    for i in range(n_out):
        start, stop = i * group, min((i + 1) * group, n_frames)
        pooled = spec.amp[:, start:stop].max(axis=1)
        np.testing.assert_allclose(
            fig_p.data[0].z[:, i],
            amplitude_to_db(pooled).astype(np.float32),
            rtol=1e-5,
        )
        assert fig_p.data[0].x[i] == pytest.approx(spec.times[start:stop].mean())


@pytest.mark.parametrize("db", [True, False])
def test_peaks_overlay_one_marker_trace_per_table(db):
    from audiospylt.analysis.tracking import spectrogram_peaks
    from audiospylt.viz.tables import plot_events, plot_events_3d, plot_snapshot

    audio, sr = _sine()
    spec = spectrogram(audio, sr, frame_length=256, hop=128)
    peaks = spectrogram_peaks(spec, prominence_db=20)
    before = peaks.copy(deep=True)
    fig = plot_spectrogram(spec, overlay=[peaks, peaks], overlay_mode="markers", db=db,
                           overlay_labels=["one", "two"])
    assert len(fig.data) == 3
    for trace, name in zip(fig.data[1:], ["one", "two"]):
        assert trace.mode == "markers" and trace.name == name
        np.testing.assert_array_equal(trace.x, peaks.time)
        np.testing.assert_array_equal(trace.y, peaks.freq)
        assert "time:" in trace.text[0] and "frequency:" in trace.text[0]
        assert (" dB" in trace.text[0]) == db
    pd.testing.assert_frame_equal(peaks, before)
    for mode in ("lines", "lines+markers", ["markers", "lines"]):
        with pytest.raises(ValueError, match="peaks overlay_mode must be 'markers'"):
            plot_spectrogram(spec, overlay=[peaks, peaks], overlay_mode=mode)
    for call in (plot_snapshot, plot_events, plot_events_3d,
                 lambda df: plot_spectrum(spec.frame(0), peaks=df)):
        with pytest.raises(ValueError, match=r"plot_spectrogram\(overlay=\.\.\.\)"):
            call(peaks)
    empty = plot_spectrogram(spec, overlay=peaks.iloc[:0], overlay_mode="markers")
    assert len(empty.data) == 2


def test_peaks_and_mixed_overlay_modes_default_per_kind():
    from audiospylt.model.schema import validate

    audio, sr = _sine()
    spec = spectrogram(audio, sr, frame_length=256, hop=128)
    peaks = validate(pd.DataFrame({"frame": [0, 1], "time": spec.times[:2],
                                   "freq": [440., 441.], "amp": [.5, .4]}), "peaks")
    tracks = validate(peaks.drop(columns="frame").assign(track_id=0), "tracks")
    single = plot_spectrogram(spec, overlay=peaks)
    assert len(single.data) == 2
    assert single.data[1].mode == "markers"
    np.testing.assert_array_equal(single.data[1].x, peaks.time)
    np.testing.assert_array_equal(single.data[1].y, peaks.freq)
    mixed = plot_spectrogram(spec, overlay=[peaks, tracks])
    assert len(mixed.data) == 3
    assert mixed.data[1].mode == "markers"
    assert mixed.data[2].mode == "lines"
    for mode in ("lines", "lines+markers"):
        with pytest.raises(ValueError, match="overlay_mode='markers' or the default None"):
            plot_spectrogram(spec, overlay=peaks, overlay_mode=mode)
