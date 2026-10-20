from pathlib import Path
import numpy as np
import pytest
import plotly.graph_objects as go
from audiospylt import audio_utils as audio
from tests import _cases as cases


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


@pytest.mark.parametrize("mono", [True, False], ids=["mono", "channel_first_stereo"])
def test_local_loading(tmp_path, golden, mono):
    path, _, sr = cases.local_wav(tmp_path)
    resolved = audio.load_wav_from_source(str(path), verbose=True)
    assert resolved == str(path.resolve())
    data, rate = audio.load_audio_data(resolved, None, mono)
    golden(f"audio/load/{mono}", cases.audio_load_case(tmp_path, mono), rtol=1e-5, atol=1e-7)
    loaded, loaded_rate, info = audio.load_audio_sample(str(path), desired_sample_rate=sr,
                                                       convert_to_mono=mono)
    np.testing.assert_array_equal(data, loaded)
    assert rate == loaded_rate == sr
    for key in ("wav_source", "wav_filename", "source_local_path"):
        assert Path(info[key]).resolve() == path.resolve()
    assert loaded.shape == ((256,) if mono else (2, 256))


@pytest.mark.parametrize("fades", [True, False])
def test_mono_trim(golden, fades):
    original = cases.tone()
    before = original.copy()
    golden(f"audio/trim/{fades}", cases.audio_trim_case(fades))
    cut, duration = audio.trim_and_fade_audio(original, 256, 1, 0.5, "sample.wav",
                         start_time=-1, end_time=1, add_fades=False)
    np.testing.assert_array_equal(cut, before)
    np.testing.assert_array_equal(original, before)
    assert duration == 0.5


@pytest.mark.parametrize("render", [True, False])
def test_trim_and_render(tmp_path, golden, render):
    y, duration, result = audio.trim_and_fade_and_render(audio_data=cases.tone(), sample_rate=256,
        audio_info={"num_channels": 1, "duration": 0.5, "wav_filename": "sample.wav"},
        start_time=0.0625, end_time=0.375, fade_in_duration=0.03125, fade_out_duration=0.046875,
        render=render, save_audio=True, player=True, bit_rate=16)
    golden("audio/trim/True", (y, duration))
    if render:
        assert isinstance(result, str) and Path(result).is_file()
        assert Path(result).resolve().is_relative_to(tmp_path)
    else:
        assert result is None


def test_waveform_return_and_display(inert_environment):
    result = audio.plot_waveform(cases.tone(), 256, "dir/sample.wav", title_mode="basename",
                                 plot_width=600, plot_height=300)
    assert result == 0.5
    fig = inert_environment[-1]
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Waveform of sample.wav"
    assert len(fig.data[0].y) == 128


@pytest.mark.parametrize("axis", ["linear", "log", "mixed", "mel"])
def test_spectrogram(golden, axis):
    fig, info = cases.spectrogram_case(axis)
    assert isinstance(fig, go.Figure) and len(fig.data) == 1
    golden(f"audio/spec/{axis}", info, rtol=1e-4, atol=1e-7)


def test_spectrogram_padded(golden):
    fig, info = cases.spectrogram_case(boundary="zeros_end", padded=True, time_reference="start")
    golden("audio/spec/padded", info, rtol=1e-4, atol=1e-7)
    assert info["times_start"][0] == 0
    assert fig.layout.xaxis.range[0] == 0


@pytest.mark.parametrize("mode,trace_count", [("integrated", 2), ("separate", 1), ("none", 1)])
def test_spectrogram_with_waveform(golden, mode, trace_count):
    fig, info = audio.plot_spectrogram_with_waveform(cases.tone(), 256, waveform_mode=mode,
        n_fft=32, overlap=0.5, waveform_max_points=32, show=True, print_info=True,
        audio_info={"wav_filename": "dir/sample.wav"}, plot_width=600, plot_height=400)
    assert isinstance(fig, go.Figure) and len(fig.data) == trace_count
    waveform = info.pop("waveform_fig", None)
    assert (waveform is not None) == (mode == "separate")
    golden("audio/spec/linear", info, rtol=1e-4, atol=1e-7)


def test_load_and_preview(tmp_path, golden, inert_environment):
    path, _, _ = cases.local_wav(tmp_path)
    data, sr, info = audio.load_audio_sample_and_preview(wav_source=str(path),
        plot_title_mode="basename", playback_save_audio=False)
    golden("audio/load/True", {"audio": data, "sr": sr, "info": {k: info[k] for k in (
        "num_channels", "duration", "playback_audio", "source_is_url", "source_is_tempfile",
        "source_tempfile_deleted")}}, atol=1e-7)
    assert info["waveform_duration_s"] == 256 / sr
    samples, rate = info["playback_result"]
    assert samples.shape == (256,) and samples.dtype == np.int32 and rate == sr
    assert len(inert_environment) >= 2
    assert not list(tmp_path.rglob("generated*.wav"))
