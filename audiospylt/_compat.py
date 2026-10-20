"""Call-time deprecations for the unchanged 0.6 implementation."""

from __future__ import annotations

import contextvars
import functools
import warnings


class AudiospyltDeprecationWarning(DeprecationWarning):
    """A 0.6 name is used; it is removed in 1.1."""


REGISTRY: dict[str, dict[str, tuple[str | None, str | None]]] = {
    "audio_utils": {
        "ensure_finite_audio": (None, "numpy.isfinite / numpy.nan_to_num"),
        "load_wav_from_source": ("audiospylt.io.audio.load_audio", None),
        "load_audio_data": ("audiospylt.io.audio.load_audio (sr=, mono=)", None),
        "display_audio_properties": (None, "audio.shape and sr from audiospylt.io.audio.load_audio"),
        "plot_waveform": ("audiospylt.viz.signal.plot_waveform with audiospylt.nb.display.show", None),
        "trim_and_fade_audio": ("audiospylt.io.audio.trim and audiospylt.io.audio.fade", None),
        "trim_and_fade_and_render": (
            "audiospylt.io.audio.trim and audiospylt.io.audio.fade, then "
            "audiospylt.io.audio.write_audio or audiospylt.nb.display.play", None),
        "load_audio_sample": ("audiospylt.io.audio.load_audio", None),
        "load_audio_sample_and_preview": (
            "audiospylt.io.audio.load_audio, then audiospylt.viz.signal.plot_waveform "
            "and audiospylt.nb.display.play", None),
        "plot_spectrogram": (
            "audiospylt.analysis.spectrogram.spectrogram (or "
            "audiospylt.analysis.spectrogram.mel_spectrogram) with "
            "audiospylt.viz.signal.plot_spectrogram", None),
        "plot_spectrogram_with_waveform": (
            "audiospylt.analysis.spectrogram.spectrogram with "
            "audiospylt.viz.signal.plot_spectrogram(..., audio=...)", None),
    },
    "dft_analysis": {
        "apply_window": ("audiospylt.analysis.spectrum.spectrum(..., window=...)", None),
        "compute_fft": ("audiospylt.analysis.spectrum.spectrum", None),
        "filter_peaks": ("audiospylt.analysis.peaks.find_peaks", None),
        "plot_spectrum": ("audiospylt.viz.signal.plot_spectrum", None),
        "plot_windowed_waveform": (None, None),
        "analyze_signal": (
            "audiospylt.analysis.spectrum.spectrum, audiospylt.analysis.peaks.find_peaks "
            "and audiospylt.viz.signal.plot_spectrum", None),
    },
    "df_transforms": {
        "detect_dataframe_type": ("audiospylt.model.schema.infer_kind", None),
        "process_spectral_dataframe": (
            "audiospylt.transform.scale.transpose and audiospylt.transform.scale.stretch", None),
        "expand_to_multi": ("audiospylt.model.convert.snapshot_to_events", None),
        "add_time_cues": ("audiospylt.transform.scale.split_events", None),
        "merge_equal_split": ('audiospylt.transform.morph.pair(..., method="equal")', None),
        "create_time_transition_df": ("audiospylt.transform.morph.morph", None),
        "merge_cdf_analysis": (
            'audiospylt.transform.morph.parameter_range(..., method="power") '
            "with audiospylt.viz.morph.plot_morph_map", None),
        "merge_cdf": ('audiospylt.transform.morph.pair(..., method="power", power=...)', None),
        "merge_sigmoid_analysis": (
            'audiospylt.transform.morph.parameter_range(..., method="sigmoid") '
            "with audiospylt.viz.morph.plot_morph_map", None),
        "merge_sigmoid": (
            'audiospylt.transform.morph.pair(..., method="sigmoid", sigmoid_range=...)', None),
    },
    "f0_utils": {
        "search_f0": (None, None),
        "summarize_result": (None, None),
    },
    "generate_wave_file": {
        "ensure_finite_audio": (None, "numpy.isfinite / numpy.nan_to_num"),
        "scale_samples": (None, "audiospylt.io.audio.write_audio(..., subtype=...)"),
        "generate_wave_file": ("audiospylt.io.audio.write_audio", None),
        "render_audio": ("audiospylt.io.audio.write_audio and audiospylt.nb.display.play", None),
        "mix_selected_waveforms": (None, "sum the arrays with numpy"),
        "render_selected_waveforms": (
            None, "audiospylt.io.audio.write_audio and audiospylt.nb.display.play on a summed array"),
    },
    "io_utils": {
        "save_df_tsv": ("audiospylt.io.tables.write_table", None),
        "events_df_from_peaks_by_interval": (
            "audiospylt.model.convert.snapshot_to_events per interval, then pandas.concat", None),
    },
    "mei": {
        "acoustical_pitch_name": ("audiospylt.notation.pitch.pitch_table", None),
        "midi_value": ("audiospylt.notation.pitch.pitch_table", None),
        "midi_cent_value": ("audiospylt.notation.pitch.pitch_table", None),
        "convert_frequencies": ("audiospylt.notation.pitch.pitch_table", None),
        "pitch_name_and_deviation_to_step_octave_alter": ("audiospylt.notation.pitch.pitch_table", None),
        "map_alteration_to_mei_accid": ("audiospylt.notation.pitch.pitch_table (accid column)", None),
        "create_mei_string": ("audiospylt.notation.mei.snapshot_to_mei", None),
        "create_temporal_mei_string": ("audiospylt.notation.mei.events_to_mei", None),
        "process_and_visualize": (
            "audiospylt.notation.mei.snapshot_to_mei with audiospylt.nb.display.show_mei", None),
        "process_temporal_chords": (
            "audiospylt.notation.mei.events_to_mei with audiospylt.nb.display.show_mei", None),
    },
    "multiplotter": {
        "plot_combined": ("audiospylt.viz.tables.plot_events (snapshots: audiospylt.viz.tables.plot_snapshot)", None),
        "plot_combined_3d": ("audiospylt.viz.tables.plot_events_3d", None),
        "plot_scatter": ("audiospylt.viz.tables.plot_snapshot (events, tracks: audiospylt.viz.tables.plot_events)", None),
        "plot_scatter_binned": (None, None),
        "plot_equalizer_bars": (None, "audiospylt.viz.tables.plot_snapshot"),
    },
    "plotting": {
        "show_plotly": ("audiospylt.nb.display.show", None),
    },
    "plot_wave": {
        "plot_waves": ("audiospylt.synth.additive.synthesize with audiospylt.viz.signal.plot_waveform", None),
        "crossfade_adjacent_events": (
            None, 'audiospylt.synth.additive.synthesize(..., phase="continuous") joins '
            "abutting events without phase jumps"),
        "estimate_sampling_frequency_and_time_vector": (
            None, "audiospylt.synth.additive.synthesize(..., sr=...) renders at the rate you choose"),
    },
    "waveform_utils": {
        "generate_waveforms": (None, None),
        "apply_window_to_waveforms": (None, None),
        "plot_waveforms": (None, "audiospylt.viz.signal.plot_waveform and audiospylt.viz.signal.plot_spectrum"),
    },
}

_depth = contextvars.ContextVar("audiospylt_compat_depth", default=0)


def message(module: str, name: str) -> str:
    """Describe a deprecated name and its migration path."""
    use, hint = REGISTRY[module][name]
    prefix = f"audiospylt.{module}.{name} is deprecated (0.6 API, removed in 1.1)"
    if use is not None:
        return f"{prefix}; use {use} instead. See docs/migration.md."
    closest = f"; closest: {hint}" if hint is not None else ""
    return f"{prefix} and has no v1 replacement{closest}. See docs/migration.md."


def _wrapper(module, name, func):
    notice = message(module, name)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        depth = _depth.get()
        token = _depth.set(depth + 1)
        try:
            if depth == 0:
                warnings.warn(notice, AudiospyltDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        finally:
            _depth.reset(token)

    wrapped.__doc__ = ".. deprecated:: 1.0\n    " + notice + "\n\n" + (func.__doc__ or "")
    wrapped.__audiospylt_deprecated__ = (module, name)
    return wrapped


def deprecate_module(namespace: dict) -> None:
    """Wrap only registered functions defined by this 0.6 module."""
    qualified = namespace["__name__"]
    module = qualified.removeprefix("audiospylt.")
    for name in REGISTRY[module]:
        func = namespace.get(name)
        if type(func) is not type(_wrapper) or func.__module__ != qualified:
            raise RuntimeError(f"{qualified}.{name} must be a function defined in {qualified}")
    for name in REGISTRY[module]:
        namespace[name] = _wrapper(module, name, namespace[name])
