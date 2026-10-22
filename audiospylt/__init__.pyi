"""Type stub for the lazy top level; 0.6 names are absent on purpose."""

from . import io as io
from . import analysis as analysis
from . import model as model
from . import synth as synth
from . import transform as transform
from . import notation as notation
from . import viz as viz
from . import nb as nb
from .io.audio import load_audio as load_audio
from .io.audio import to_mono as to_mono
from .io.audio import trim as trim
from .io.audio import fade as fade
from .io.audio import write_audio as write_audio
from .io.tables import read_table as read_table
from .io.tables import write_table as write_table
from .analysis.spectrum import Spectrum as Spectrum
from .analysis.spectrum import spectrum as spectrum
from .analysis.spectrum import amplitude_to_db as amplitude_to_db
from .analysis.spectrum import db_to_amplitude as db_to_amplitude
from .analysis.peaks import find_peaks as find_peaks
from .analysis.spectrogram import Spectrogram as Spectrogram
from .analysis.spectrogram import spectrogram as spectrogram
from .analysis.spectrogram import MelSpectrogram as MelSpectrogram
from .analysis.spectrogram import mel_spectrogram as mel_spectrogram
from .analysis.f0 import F0Contour as F0Contour
from .analysis.f0 import pyin as pyin
from .analysis.f0 import HarmonicF0 as HarmonicF0
from .analysis.f0 import harmonic_f0 as harmonic_f0
from .model.schema import SchemaError as SchemaError
from .model.schema import LegacyFormatWarning as LegacyFormatWarning
from .model.schema import validate as validate
from .model.schema import infer_kind as infer_kind
from .model.schema import upgrade as upgrade
from .model.convert import snapshot_to_events as snapshot_to_events
from .model.convert import f0_to_tracks as f0_to_tracks
from .model.convert import tracks_to_events as tracks_to_events
from .synth.additive import synthesize as synthesize
from .transform.scale import transpose as transpose
from .transform.scale import stretch as stretch
from .transform.scale import split_events as split_events
from .transform.morph import parameter_range as parameter_range
from .transform.morph import pair as pair
from .transform.morph import morph as morph
from .notation.pitch import pitch_table as pitch_table
from .notation.pitch import cents_label as cents_label
from .notation.mei import snapshot_to_mei as snapshot_to_mei
from .notation.mei import events_to_mei as events_to_mei
from .notation.render import render_svg as render_svg
from .notation.render import render_svg_pages as render_svg_pages

__version__: str

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
    "F0Contour",
    "pyin",
    "HarmonicF0",
    "harmonic_f0",
    "SchemaError",
    "LegacyFormatWarning",
    "validate",
    "infer_kind",
    "upgrade",
    "snapshot_to_events",
    "f0_to_tracks",
    "tracks_to_events",
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
