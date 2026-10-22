"""AudioSpylt: sound analysis, resynthesis, and visual/symbolic representations.

``import audiospylt as asp`` exposes the compute API at the top level
(``asp.load_audio``, ``asp.spectrum``, ``asp.find_peaks``, ``asp.morph``, …),
loaded lazily on first access. Plotly figures live in ``asp.viz`` (needs
``audiospylt[plot]``) and notebook display in ``asp.nb`` (needs
``audiospylt[notebook]``). The 0.6 names still resolve but are deprecated;
see ``docs/migration.md``.
"""

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from importlib.util import find_spec as _find_spec

try:
    __version__ = _version("audiospylt")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

# v1 name -> module (relative to the package) that defines it, in pipeline order.
_V1 = {
    "load_audio": "io.audio",
    "to_mono": "io.audio",
    "trim": "io.audio",
    "fade": "io.audio",
    "write_audio": "io.audio",
    "read_table": "io.tables",
    "write_table": "io.tables",
    "Spectrum": "analysis.spectrum",
    "spectrum": "analysis.spectrum",
    "amplitude_to_db": "analysis.spectrum",
    "db_to_amplitude": "analysis.spectrum",
    "find_peaks": "analysis.peaks",
    "Spectrogram": "analysis.spectrogram",
    "spectrogram": "analysis.spectrogram",
    "MelSpectrogram": "analysis.spectrogram",
    "mel_spectrogram": "analysis.spectrogram",
    "SchemaError": "model.schema",
    "LegacyFormatWarning": "model.schema",
    "validate": "model.schema",
    "infer_kind": "model.schema",
    "upgrade": "model.schema",
    "snapshot_to_events": "model.convert",
    "synthesize": "synth.additive",
    "transpose": "transform.scale",
    "stretch": "transform.scale",
    "split_events": "transform.scale",
    "parameter_range": "transform.morph",
    "pair": "transform.morph",
    "morph": "transform.morph",
    "pitch_table": "notation.pitch",
    "cents_label": "notation.pitch",
    "snapshot_to_mei": "notation.mei",
    "events_to_mei": "notation.mei",
    "render_svg": "notation.render",
    "render_svg_pages": "notation.render",
}

__all__ = [
    "__version__",
    "load_audio",
    "to_mono",
    "trim",
    "fade",
    "write_audio",
    "read_table",
    "write_table",
    "Spectrum",
    "spectrum",
    "amplitude_to_db",
    "db_to_amplitude",
    "find_peaks",
    "Spectrogram",
    "spectrogram",
    "MelSpectrogram",
    "mel_spectrogram",
    "SchemaError",
    "LegacyFormatWarning",
    "validate",
    "infer_kind",
    "upgrade",
    "snapshot_to_events",
    "synthesize",
    "transpose",
    "stretch",
    "split_events",
    "parameter_range",
    "pair",
    "morph",
    "pitch_table",
    "cents_label",
    "snapshot_to_mei",
    "events_to_mei",
    "render_svg",
    "render_svg_pages",
]

_SUBPACKAGES = frozenset({
    "io", "analysis", "model", "synth", "transform", "notation", "viz", "nb",
})

# 0.6 names and modules: still resolve, but stay out of __all__ and dir().
_LEGACY_EXPORTS = {
    "render_audio": "generate_wave_file",
    "load_audio_sample_and_preview": "audio_utils",
    "search_f0": "f0_utils",
    "summarize_result": "f0_utils",
    "process_spectral_dataframe": "df_transforms",
    "detect_dataframe_type": "df_transforms",
    "expand_to_multi": "df_transforms",
    "add_time_cues": "df_transforms",
    "merge_equal_split": "df_transforms",
    "create_time_transition_df": "df_transforms",
    "merge_cdf": "df_transforms",
    "merge_cdf_analysis": "df_transforms",
    "merge_sigmoid": "df_transforms",
    "merge_sigmoid_analysis": "df_transforms",
}
_LEGACY_MODULES = {
    "audio_utils", "dft_analysis", "df_transforms", "f0_utils",
    "generate_wave_file", "io_utils", "mei", "multiplotter", "plotting",
    "plot_wave", "waveform_utils",
}


def _import_legacy(name, module):
    try:
        return _import_module(f"{__name__}.{module}")
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if not missing or missing.startswith(__name__):
            raise
        missing = missing.partition(".")[0]
        raise ImportError(
            f"{name} is 0.6 API and needs {missing}; install audiospylt[all]"
        ) from error


def __getattr__(name):
    if name in _V1:
        value = getattr(_import_module(f"{__name__}.{_V1[name]}"), name)
    elif name in _SUBPACKAGES:
        value = _import_module(f"{__name__}.{name}")
    elif name in _LEGACY_EXPORTS:
        value = getattr(_import_legacy(name, _LEGACY_EXPORTS[name]), name)
    elif name in _LEGACY_MODULES:
        value = _import_legacy(name, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    # help() and inspect.getmembers() skip only AttributeError, so leave viz
    # out when plotly is missing; asp.viz still raises the install hint.
    names = set(__all__) | _SUBPACKAGES
    if _find_spec("plotly") is None:
        names.discard("viz")
    return sorted(names)
