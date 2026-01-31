from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("audiospylt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.6.0a1"

from .generate_wave_file import render_audio
from .audio_utils import load_audio_sample_and_preview
from .df_transforms import process_spectral_dataframe, detect_dataframe_type, expand_to_multi, add_time_cues, merge_equal_split, create_time_transition_df, merge_cdf, merge_cdf_analysis, merge_sigmoid, merge_sigmoid_analysis

__all__ = [
    "__version__",
    "render_audio",
    "load_audio_sample_and_preview",
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

