from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from audiospylt import generate_wave_file as wav
from tests import _cases as cases


@pytest.mark.parametrize("bits", [16, 24])
def test_scaling_and_native_export(tmp_path, golden, goldens, bits):
    actual = cases.export_case(tmp_path, bits)
    golden(f"export/{bits}", actual, rtol=0, atol=0)
    scaled = wav.scale_samples(cases.export_samples(), bits)
    np.testing.assert_array_equal(scaled, goldens[f"export/{bits}"][0])
    path = wav.generate_wave_file(cases.export_samples(), 8000, fs_target_name="source", bit_rate=bits,
                                  output_dir=tmp_path, custom_filename=f"native-{bits}")
    assert Path(path) == tmp_path / f"native-{bits}.wav"
    pcm, rate = sf.read(path, dtype="int16" if bits == 16 else "int32")
    np.testing.assert_array_equal(pcm, scaled)
    assert rate == 8000 and sf.info(path).subtype == f"PCM_{bits}"


@pytest.mark.parametrize("alias", [False, True], ids=["bandlimited", "intentional_aliasing"])
def test_resampling_export(tmp_path, golden, alias):
    # Quantized FFT results may differ by one LSB across scipy builds.
    golden(f"export/resample/{alias}", cases.export_case(tmp_path, 16, 6000, alias), rtol=0, atol=1)


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_mixing_trims_to_shortest_and_skips_missing(golden, mode):
    golden(f"export/mix/{mode}", cases.mix_case(mode))


@pytest.mark.parametrize("save", [True, False])
def test_render_audio(tmp_path, goldens, save):
    result = wav.render_audio(cases.export_samples(), 8000, fs_target_name="source", bit_rate=16,
        output_dir=tmp_path, custom_filename="rendered.wav", save_audio=save, player=True)
    if save:
        assert result == str(tmp_path / "rendered.wav")
        actual, rate = sf.read(result, dtype="int16")
    else:
        actual, rate = result
    np.testing.assert_array_equal(actual, goldens["export/16"][0])
    assert rate == 8000


@pytest.mark.parametrize("spacing_key", ["dt", "n"])
def test_render_selected_waveforms(tmp_path, goldens, spacing_key):
    result = wav.render_selected_waveforms({"a": cases.export_samples(), "b": np.zeros(12)},
        {"selected_waveforms": ["a", "b"], spacing_key: 1/8000}, fs_target_name="source", bit_rate=24,
        output_dir=tmp_path, custom_filename="selected.wav", save_audio=True, player=True)
    actual, rate = sf.read(result, dtype="int32")
    np.testing.assert_array_equal(actual, goldens["export/24"][0])
    assert rate == 8000


def test_silence_export_is_finite(tmp_path):
    data, rate = wav.generate_wave_file(np.zeros(12), 8000, fs_target_name="source",
                                       output_dir=tmp_path, save_to_file=False)
    np.testing.assert_array_equal(data, np.zeros(12, dtype=np.int16))
    assert rate == 8000
