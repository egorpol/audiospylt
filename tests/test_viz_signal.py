"""Waveform, spectrum, and spectrogram figure contracts."""

from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from audiospylt.analysis.peaks import find_peaks
from audiospylt.analysis.spectrogram import mel_spectrogram, spectrogram
from audiospylt.analysis.spectrum import Spectrum, amplitude_to_db, spectrum
from audiospylt.model.convert import snapshot_to_events
from audiospylt.viz._common import (
    _minmax,
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
    cd = np.asarray(fig_mel.data[0].customdata)
    assert cd.dtype == np.float32
    assert cd.shape == fig_mel.data[0].z.shape
    np.testing.assert_allclose(cd[:, 0], mel.freqs.astype(np.float32))
    assert "%{customdata:.2f}" in fig_mel.data[0].hovertemplate


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
