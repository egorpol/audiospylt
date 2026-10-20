"""Public package exports for audiospylt."""

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("audiospylt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.6.3"

__all__ = [
    "__version__",
    "render_audio",
    "load_audio_sample_and_preview",
    "search_f0",
    "summarize_result",
    "process_spectral_dataframe",
    "detect_dataframe_type",
    "expand_to_multi",
    "add_time_cues",
    "merge_equal_split",
    "create_time_transition_df",
    "merge_cdf",
    "merge_cdf_analysis",
    "merge_sigmoid",
    "merge_sigmoid_analysis",
]


_LEGACY_EXPORTS = {
    "render_audio": "generate_wave_file",
    "load_audio_sample_and_preview": "audio_utils",
    "search_f0": "f0_utils",
    "summarize_result": "f0_utils",
    **{name: "df_transforms" for name in __all__[5:]},
}
_LEGACY_MODULES = {
    "audio_utils", "dft_analysis", "df_transforms", "f0_utils",
    "generate_wave_file", "io_utils", "mei", "multiplotter", "plotting",
    "plot_wave", "waveform_utils",
}


def __getattr__(name):
    if name in _LEGACY_EXPORTS:
        module = importlib.import_module(f"{__name__}.{_LEGACY_EXPORTS[name]}")
        value = getattr(module, name)
    elif name in _LEGACY_MODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _LEGACY_EXPORTS.keys() | _LEGACY_MODULES)
