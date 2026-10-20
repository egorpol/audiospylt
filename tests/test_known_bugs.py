"""Desired behaviour only: fixing a finding must produce a visible strict XPASS.

These assertions are deliberately independent of the current-code goldens.
"""
import numpy as np
import pandas as pd
import pytest
from audiospylt import audio_utils as audio, dft_analysis as dft, df_transforms as dfm
from audiospylt import generate_wave_file as wav, mei, plot_wave
from tests import _cases as cases


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-1: analysis peak amplitudes need one-sided and coherent-gain correction")
@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_analysis_reports_unit_sine_amplitude(window):
    sr = 2048
    y = np.sin(2 * np.pi * 125 * np.arange(sr) / sr)
    peaks = dft.analyze_signal(y, sr, "unit.wav", window_type=window,
        thresh_amp_low=0.01, thresh_amp_high=2., thresh_freq_low=120, thresh_freq_high=130, show_plot=False)
    assert len(peaks) == 1
    amplitude = float(peaks.iloc[0]["Amplitude"])
    if window == "boxcar":
        np.testing.assert_allclose(amplitude, 1., rtol=1e-5, atol=1e-8)
    else:
        assert abs(20 * np.log10(amplitude)) < 0.1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-2: off-bin Hann peaks need sub-bin interpolation below half a cent")
def test_off_bin_partial_frequency_accuracy():
    sr, frequency = 4096, 440.37
    y = np.sin(2 * np.pi * frequency * np.arange(sr) / sr)
    peaks = dft.analyze_signal(y, sr, "off-bin.wav", window_type="hann", thresh_amp_low=0.01,
        thresh_amp_high=2., thresh_freq_low=430, thresh_freq_high=450, show_plot=False)
    assert len(peaks) == 1
    measured = float(peaks.iloc[0]["Frequency (Hz)"])
    assert abs(1200 * np.log2(measured / frequency)) < 0.5


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-3: magnitude spectrogram dB must use 20 log10 for a 6.02 dB amplitude drop")
@pytest.mark.parametrize("combined", [False, True], ids=["spectrogram", "spectrogram_with_waveform"])
def test_spectrogram_amplitude_halving_is_six_db(combined):
    y = np.sin(2 * np.pi * 128 * np.arange(512) / 1024)
    function = audio.plot_spectrogram_with_waveform if combined else audio.plot_spectrogram
    _, full = function(y, 1024, n_fft=128, show=False)
    _, half = function(y / 2, 1024, n_fft=128, show=False)
    active = full["Sxx_raw"] > full["Sxx_raw"].max() * 0.1
    drop = full["Sxx_plot"][active] - half["Sxx_plot"][active]
    np.testing.assert_allclose(drop, 20 * np.log10(2), rtol=0, atol=0.1)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-4: safe synthesis sampling must retain a zero-phase top-frequency partial")
def test_safe_sampling_rate_preserves_top_partial():
    frame = pd.DataFrame({"freq_start": [100.], "freq_stop": [100.], "time_start": [0.],
                          "time_stop": [1.], "amp_min": [1.], "amp_max": [1.]})
    _, dt, _ = plot_wave.estimate_sampling_frequency_and_time_vector(frame)
    y = plot_wave.plot_waves(frame, k=dt, show_waveform=False)
    assert np.sqrt(np.mean(y**2)) > 0.5 / np.sqrt(2)


# MEI accidentals expressed in cents relative to the natural letter. Checking
# accid (written notation) rather than pnum avoids masking incorrect spelling.
ACCIDENTAL_CENTS = {"n": 0, "s": 100, "f": -100, "ss": 200, "ff": -200,
                    "1qs": 50, "3qs": 150, "1qf": -50, "3qf": -150,
                    "nu": 25, "nd": -25, "su": 125, "sd": 75, "fu": -75, "fd": -125}
NATURAL_SEMITONES = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-5: every written pitch must round within half the selected microtone step")
@pytest.mark.parametrize("resolution,tolerance", [("quarter_tone", 25), ("eighth_tone", 12.5)])
def test_all_pitch_bases_and_cent_deviations_are_accurate(resolution, tolerance):
    failures = []
    for letter, semitone in NATURAL_SEMITONES.items():
        for accidental, offset in (("", 0), ("#", 100), ("b", -100)):
            base = f"{letter}{accidental}4"
            for deviation in range(-50, 51):
                step, octave, alter = mei.pitch_name_and_deviation_to_step_octave_alter(base, deviation, resolution)
                written = 1200 * (octave + 1) + 100 * NATURAL_SEMITONES[step] + ACCIDENTAL_CENTS[alter]
                intended = 6000 + semitone * 100 + offset + deviation
                if abs(written - intended) > tolerance + 1e-8:
                    failures.append((base, deviation, alter, written - intended))
    assert not failures, f"{len(failures)} inaccurate written pitches; first examples: {failures[:8]}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-6: trim must slice the sample axis of channel-first stereo arrays")
def test_channel_first_stereo_trim_shape_and_samples():
    sr = 1000
    data = np.vstack([np.arange(1000, dtype=float), -np.arange(1000, dtype=float)])
    cut, duration = audio.trim_and_fade_audio(data, sr, 2, 1., "stereo.wav",
                                            start_time=0.1, end_time=0.3, add_fades=False)
    assert duration == pytest.approx(0.2)
    assert cut.shape == (2, 200)
    np.testing.assert_array_equal(cut, data[:, 100:300])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-7: adjacent non-overlapping events must not both contribute at their boundary by default")
def test_default_interval_contributes_once_at_shared_boundary():
    frame = pd.DataFrame({"freq_start": [1., 1.], "freq_stop": [1., 1.],
        "time_start": [0., 0.25], "time_stop": [0.25, 0.5], "amp_min": [1., 1.], "amp_max": [1., 1.]})
    y = plot_wave.plot_waves(frame, k=0.125, phase_mode="global", show_waveform=False)
    t = np.arange(y.size) * 0.125
    # The final stop lies on a zero crossing; only the internal shared boundary differs.
    np.testing.assert_allclose(y, np.sin(2 * np.pi * t), rtol=1e-5, atol=1e-8)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-8: non-antialiased resampling must use the declared-rate time grid")
def test_non_antialiased_resampling_uses_exact_sample_spacing():
    # A linear ramp in seconds exposes the interpolation coordinates without
    # depending on a tone fit or monkeypatching numpy's grid implementation.
    sr_in, sr_out = 1000, 600
    input_times = np.arange(12) / sr_in
    sampled_times = wav._resample_by_sampling(input_times, sr_in, sr_out)
    np.testing.assert_allclose(np.diff(sampled_times), 1 / sr_out, rtol=0, atol=1e-9)


# D13: each partial of the smaller snapshot carries its amplitude in exactly one
# row at its morph end; the other rows are births (0 at start) or deaths (0 at stop).
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-9 D13: each partial of the smaller snapshot carries its amplitude in exactly one row at its morph end")
@pytest.mark.parametrize("method", ["equal", "cdf", "sigmoid"])
@pytest.mark.parametrize("source_side", [1, 2])
@pytest.mark.parametrize("end", ["start", "stop"])
def test_morph_start_energy_is_preserved(method, source_side, end):
    small, large = cases.spectra()
    a, b = (small, large) if source_side == 1 else (large, small)
    merge = {"equal": dfm.merge_equal_split, "cdf": dfm.merge_cdf, "sigmoid": dfm.merge_sigmoid}[method]
    paired = merge(a, b)
    # Exercise each merge argument order at both the diverging and converging
    # morph end. Reverse the paired columns when needed to put the smaller side
    # at the end under test; its nonshared opposite end is already correct in 0.6.
    if (source_side == 1) != (end == "start"):
        paired = paired.rename(columns={"freq_data1": "freq_data2", "amp_data1": "amp_data2",
                                        "freq_data2": "freq_data1", "amp_data2": "amp_data1"})
    transition = dfm.create_time_transition_df(paired, 0.1, scale_amp=False)
    original_energy = float(np.square(small["Amplitude"]).sum())
    energy = float(np.square(transition["amp_min" if end == "start" else "amp_max"]).sum())
    assert energy == pytest.approx(original_energy, rel=0.01)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-9: pitch scaling must apply to every frequency column in wide snapshots")
def test_wide_snapshot_scales_all_frequency_columns():
    frame = pd.DataFrame({"freq1": [100.], "amp1": [0.5], "freq2": [300.], "amp2": [0.2]})
    result = dfm.process_spectral_dataframe(frame, pitch_factor=2.)
    np.testing.assert_allclose(result[["freq1", "freq2"]], [[200., 600.]], rtol=1e-5, atol=1e-8)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DSP-9: negative time stretch must reject reversed events")
@pytest.mark.parametrize("stretch", [-1., [1., -0.5]])
def test_negative_time_stretch_raises(stretch):
    try:
        dfm.process_spectral_dataframe(cases.events(), stretch_factor=stretch)
    except ValueError:
        return
    assert False, "Negative time stretch must raise ValueError"


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="ENG-4: audiospylt/__init__ still imports the 0.6 modules, which load "
           "IPython; flips in 1.7b",
)
def test_v1_modules_import_without_notebook_stack():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = r"""
import sys
modules = [
    "audiospylt.io", "audiospylt.io.audio", "audiospylt.io.tables",
    "audiospylt.analysis", "audiospylt.analysis.spectrum",
    "audiospylt.analysis.spectrogram", "audiospylt.analysis.peaks",
    "audiospylt.model", "audiospylt.model.schema", "audiospylt.model.convert",
    "audiospylt.synth", "audiospylt.synth.additive",
    "audiospylt.notation", "audiospylt.notation.pitch", "audiospylt.notation.mei",
    "audiospylt.notation.render",
    "audiospylt.transform", "audiospylt.transform.scale", "audiospylt.transform.morph",
    "audiospylt.viz", "audiospylt.viz._common", "audiospylt.viz.signal",
    "audiospylt.viz.tables", "audiospylt.viz.morph",
    "audiospylt.nb", "audiospylt.nb.display",
]
for name in modules:
    __import__(name)
forbidden = {"IPython", "ipywidgets", "nbformat", "matplotlib"}
found = sorted(forbidden.intersection(sys.modules))
assert not found, found
"""
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr or result.stdout
