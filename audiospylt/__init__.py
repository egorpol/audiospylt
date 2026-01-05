from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("audiospylt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.6.0a0"

from .generate_wave_file import render_audio
from .audio_utils import load_audio_sample_and_preview

__all__ = [
    "__version__",
    "render_audio",
    "load_audio_sample_and_preview",
]

