"""Small deterministic inputs and snapshot producers shared with regeneration.

Expected results live ONLY in golden/current.json; these producers calculate actuals.
"""
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from audiospylt import audio_utils as audio, dft_analysis as dft, df_transforms as dfm
from audiospylt import f0_utils as f0, mei
from audiospylt import generate_wave_file as wav, io_utils
from audiospylt import plot_wave
from tests._support import normalize_xml


def tone(sr=256, size=128):
    t = np.arange(size) / sr
    return 0.8 * np.sin(2 * np.pi * 32 * t) + 0.3 * np.cos(2 * np.pi * 48 * t)


def peaks():
    return pd.DataFrame({"Frequency (Hz)": [100., 220., 350.], "Amplitude": [0.8, 0.4, 0.2]})


def spectra():
    return peaks(), pd.DataFrame({"Frequency (Hz)": [120., 180., 260., 390., 510.],
                                 "Amplitude": [0.5, 0.2, 0.6, 0.1, 0.3]})


def events():
    return pd.DataFrame({"freq_start": [13., 29., 13.], "freq_stop": [21., 29., 17.],
                         "time_start": [0., 0., 0.06], "time_stop": [0.06, 0.12, 0.12],
                         "amp_min": [0.8, 0.3, 0.5], "amp_max": [0.5, 0.3, 0.2]})


def adjacent_events():
    return pd.DataFrame({"freq_start": [13., 13.], "freq_stop": [13., 13.],
                         "time_start": [0., 0.06], "time_stop": [0.06, 0.12],
                         "amp_min": [0.8, 0.4], "amp_max": [0.8, 0.4]})


def local_wav(root):
    # FLOAT WAV avoids integer quantization and all librosa resampling internals.
    sr = 8000
    t = np.arange(256) / sr
    data = np.column_stack([0.6 * np.sin(2 * np.pi * 500 * t),
                            0.3 * np.cos(2 * np.pi * 1000 * t)]).astype(np.float32)
    path = Path(root) / "synthetic.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    return path, data, sr


def audio_load_case(root, mono):
    path, _, _ = local_wav(root)
    data, sr, info = audio.load_audio_sample(str(path), convert_to_mono=mono)
    # Paths are asserted separately against tmp_path, not embedded in portable goldens.
    return {"audio": data, "sr": sr, "info": {k: info[k] for k in (
        "num_channels", "duration", "playback_audio", "source_is_url",
        "source_is_tempfile", "source_tempfile_deleted")}}


def audio_trim_case(fades):
    return audio.trim_and_fade_audio(tone(), 256, 1, 0.5, "synthetic.wav",
                                    start_time=0.0625, end_time=0.375, add_fades=fades,
                                    fade_in_duration=0.03125, fade_out_duration=0.046875)


def spectrogram_case(axis="linear", **kwargs):
    fig, info = audio.plot_spectrogram(tone(), 256, n_fft=32, overlap=0.5,
                                      y_axis_mode=axis, mel_bins=8, show=True, **kwargs)
    return fig, info


def dft_case():
    y = tone()
    freqs, spec = dft.compute_fft(y, 256)
    selected = dft.filter_peaks(spec, freqs, 0.05, 0.6, 20, 60,
                               prominence_rel=0.1, width_hz=1, distance_hz=4)
    result = dft.analyze_signal(y, 256, "synthetic.wav", window_type="hann",
                               thresh_amp_low=0.05, thresh_amp_high=0.6,
                               thresh_freq_low=20, thresh_freq_high=60,
                               partial_tracking=True, f0=16, partial_bandwidth_hz=1,
                               show_plot=True, show_peaks=True, show_windowed_waveform=True)
    return {"frequencies": freqs, "spectrum": spec, "hann": dft.apply_window(y, "hann"),
            "selected_indices": selected, "analysis": result}


F0_CASES = ("harmonic", "inharmonic", "weighted", "trim_cents", "trim_fraction", "peaks_df", "coarse")


def f0_case(name, *, plot=False):
    kwargs = {"f0_min": 96., "f0_max": 104., "step": 0.125, "top_k": 3,
              "include_plot": plot, "show_plot": True}
    frequencies = [100.2, 201.4, 298.7, 403.2]
    if name == "harmonic":
        frequencies = [100., 200., 300., 400.]
    elif name == "weighted":
        kwargs["weights"] = [1., 0.5, 0.25, 0.1]
    elif name in {"trim_cents", "trim_fraction"}:
        frequencies = [100., 200., 300., 447.]
        kwargs["trim_outlier_cents" if name == "trim_cents" else "robust_trim_frac"] = (
            50. if name == "trim_cents" else 0.25)
    elif name == "peaks_df":
        kwargs.update(peaks_df=pd.DataFrame({"Frequency (Hz)": frequencies,
                                            "Amplitude": [1., 0.5, 0.25, 0.1]}), use_amp_weights=True)
        frequencies = None
    elif name == "coarse":
        kwargs.update(coarse_step=1., refine_half_width_hz=0.375)
    return f0.search_f0(frequencies, **kwargs)


def f0_numeric(result):
    return {k: v for k, v in result.items() if k != "fig"}


def f0_summary_case():
    result = f0_case("weighted", plot=True)
    output = io.StringIO()
    with redirect_stdout(output):
        returned = f0.summarize_result("weighted partials", result, top_n=2, precision=3,
                                      show_plot=True, plot_width=600, plot_height=300)
    assert returned is None
    return output.getvalue()


TRANSFORM_CASES = ("detect", "single", "multi", "clip", "threshold", "allow", "expand", "cues",
                   "equal", "cdf", "cdf_reverse", "sigmoid", "cdf_analysis", "sigmoid_analysis",
                   "transition", "transition_raw", "transition_override")


def transform_case(name):
    a, b = spectra()
    if name == "detect":
        return [dfm.detect_dataframe_type(x) for x in (a, events(),
            pd.DataFrame({"freq1": [100., 200.], "amp1": [0.2, 0.3],
                          "freq2": [300., np.nan], "amp2": [0.5, np.nan]}),
            pd.DataFrame({"other": [1]}))]
    if name == "single":
        return dfm.process_spectral_dataframe(a, pitch_factor=1.25, pitch_cents=1200)
    if name == "multi":
        return dfm.process_spectral_dataframe(events(), stretch_factor=[2., 0.5],
                                              pitch_factor=[1., 2.], pitch_cents=[1200., -1200.])
    if name in {"clip", "threshold", "allow"}:
        return dfm.process_spectral_dataframe(a, pitch_factor=2., pitch_min_hz=250,
                                              pitch_max_hz=500, pitch_overshoot=name)
    if name == "expand":
        return dfm.expand_to_multi(a, 0.12)
    if name == "cues":
        return dfm.add_time_cues(events(), [0., 0.03, 0.03, 0.06, 0.09, 0.12, 1.])
    if name == "equal":
        return dfm.merge_equal_split(a, b)
    if name in {"cdf", "cdf_reverse"}:
        return dfm.merge_cdf(a, b, power_factor=-1.5 if name.endswith("reverse") else None, plot=True)
    if name == "sigmoid":
        return dfm.merge_sigmoid(b, a, plot=True)
    if name == "cdf_analysis":
        return dfm.merge_cdf_analysis(a, b, plot=True)
    if name == "sigmoid_analysis":
        return dfm.merge_sigmoid_analysis(a, b, plot=True)
    merged = dfm.merge_equal_split(a, b)
    return dfm.create_time_transition_df(merged, 0.12,
        scale_amp=name != "transition_raw", override_amp=0.25 if name.endswith("override") else None)


WAVE_CASES = tuple(f"{phase}/{interval}/{fade}" for phase in ("chirp", "reset", "global")
                   for interval in ("closed", "half_open") for fade in ("none", "linear", "cosine", "sqrt_hann"))


def wave_case(name):
    phase, interval, fade = name.split("/")
    return plot_wave.plot_waves(events(), k=0.002, phase_mode=phase, interval_mode=interval,
                               edge_fade_s=0.012 if fade != "none" else 0.,
                               fade_shape="cosine" if fade == "none" else fade, show_waveform=True)


def overlap_case(phase):
    frame = plot_wave.crossfade_adjacent_events(adjacent_events(), 0.02)
    return {"events": frame, "wave": plot_wave.plot_waves(frame, k=0.002, phase_mode=phase,
                interval_mode="half_open", fade_shape="sqrt_hann", show_waveform=True)}


def row_fades_case():
    frame = events().assign(fade_in_s=[0.008, np.nan, 0.004], fade_out_s=[0.006, 0.012, np.nan])
    return plot_wave.plot_waves(frame, k=0.002, edge_fade_s=0.016,
                               fade_in_s=0.01, fade_out_s=0.02, fade_shape="linear")


def export_samples():
    return np.array([0., 0.125, 0.5, 1., 0.5, 0., -0.25, -0.5, -1., -0.5, 0., 0.25])


def export_case(root, bit_rate, target="source", alias=False):
    return wav.generate_wave_file(export_samples(), 8000, fs_target_name=target,
        bit_rate=bit_rate, output_dir=root, save_to_file=False, alias_above_nyquist=alias)


def mix_case(mode):
    return wav.mix_selected_waveforms({"a": export_samples(), "b": np.array([0.5, -0.5, 0.25])},
                                     ["a", "missing", "b"], mode=mode)


def io_events_case():
    return io_utils.events_df_from_peaks_by_interval([
        {"time_start": 0., "time_stop": 0.05, "peaks_df": peaks()},
        {"time_start": 0.05, "time_stop": 0.1, "peaks_df": None},
        {"time_start": 0.1, "time_stop": 0.2, "peaks_df": peaks().iloc[:1]},
        {"time_start": 0.2, "time_stop": 0.3, "peaks_df": peaks().iloc[:0]}])


def io_tsv_case(root):
    output = io_utils.save_df_tsv(io_events_case(), Path(root) / "nested" / "events.tsv", verbose=False)
    return Path(output).read_text(encoding="utf-8")


def mei_pitch_case():
    ratios = [0., 1., -1., 1/12, -9/12]
    return {"midi": [mei.midi_value(x) for x in ratios],
            "cents": [mei.midi_cent_value(x) for x in ratios],
            "names": [mei.acoustical_pitch_name(x) for x in [0, 48, 60, 61, 69, 127]],
            "convert": mei.convert_frequencies(pd.DataFrame({"Frequency (Hz)": [
                440., 220., 261.625565, 440 * 2**(31/1200), 0., -1., None, np.nan]})),
            "written": {resolution: [mei.pitch_name_and_deviation_to_step_octave_alter(pn, dev, resolution)
                for pn in ("C4", "C#4", "Db4") for dev in (-50, -30, -12, 0, 12, 30, 50)]
                for resolution in ("half_tone", "quarter_tone", "eighth_tone")},
            "accid": [[mei.map_alteration_to_mei_accid(x, natural) for x in (None, "", "n", "s", "1qf", "su")]
                      for natural in (False, True)]}


def mei_notes():
    return [("C", 3, "n"), ("A", 4, "n"), ("C", 5, "s")], [48, 69, 73], [-12, 0, 31]


def mei_string_case(mode):
    notes, midis, cents = mei_notes()
    return normalize_xml(mei.create_mei_string(notes, midis, cents, "GF", "half_tone_deviation", True, True, mode))


def mei_temporal_case(label_mode):
    notes, midis, cents = mei_notes()
    groups = [((0., 1.), notes, midis, cents), ((1., 1.5), notes[1:], midis[1:], cents[1:]),
              ((2., 2.), notes, midis, cents)]
    return normalize_xml(mei.create_temporal_mei_string(groups, "GF", "quarter_tone", True, 0.5,
                         display_time_start=True, time_label_mode=label_mode))


def mei_process_case(root, temporal=False):
    xml_path, svg_path = Path(root) / "notation.mei", Path(root) / "notation.svg"
    kwargs = {"save_mei": True, "mei_save_path": str(xml_path), "save_svg": True,
              "svg_save_path": str(svg_path), "display_df": True}
    if temporal:
        frame = pd.DataFrame({"freq_start": [440., 220., 660.],
                              "time_start": [0., 0., 1.], "time_stop": [1., 1., 1.5]})
        result = mei.process_temporal_chords(frame, display_time_start=True, **kwargs)
    else:
        frame = pd.DataFrame({"Frequency (Hz)": [440 * 2**(31/1200), 220., 440.]})
        result = mei.process_and_visualize(frame, note_order="ascending", **kwargs)
    assert result is None
    # Renderer versions/fonts alter SVG geometry; verify real rendering structurally.
    import xml.etree.ElementTree as ET
    svg = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    assert svg.tag.endswith("svg") and len(list(svg.iter())) > 10
    return normalize_xml(xml_path.read_text(encoding="utf-8"))
