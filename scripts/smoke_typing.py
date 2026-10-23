"""Downstream type check of an installed audiospylt wheel; read by mypy, never executed.

Usage (from outside the checkout, in a venv with the wheel and mypy):
    cd "$RUNNER_TEMP" && /path/to/venv/bin/python -m mypy --warn-unused-ignores /path/to/scripts/smoke_typing.py

Without py.typed the import is untyped, and without __init__.pyi the lazy __getattr__ makes
every top-level name Any. Either way mypy exits non-zero: an import error, or the deliberate
errors below go unflagged and their ignores are reported as unused. Only numpy is imported
besides audiospylt, because pandas and plotly ship no py.typed.
"""

from __future__ import annotations

import numpy as np

import audiospylt as asp

audio: np.ndarray
sr: int
audio, sr = asp.load_audio("tones.wav", sr=22050)
spec: asp.Spectrum = asp.spectrum(audio, sr)
peaks = asp.find_peaks(spec, min_db=-20.0)
figure = asp.viz.plot_spectrum(spec, peaks, fmax=1000.0)
level: float = asp.amplitude_to_db(0.5)
levels: np.ndarray = asp.amplitude_to_db([0.5, 0.25])
contour: asp.F0Contour = asp.pyin(audio, sr, fmin=100.0, fmax=1000.0)
fundamental: float = asp.harmonic_f0(peaks, fmin=100.0, fmax=300.0).f0
voice: np.ndarray = asp.synthesize(asp.tracks_to_events(asp.f0_to_tracks(contour)), sr, fade=0.005)
partials = asp.tracks_to_events(asp.read_partiels(["Partial_1.json", "Partial_2.json"], channel=1))
chords = asp.tracks_to_chords(asp.read_partiels("json"), [0.0, 0.5, 1.0], merge_cents=None, max_notes=8)
chord: str = asp.snapshot_to_mei(asp.tracks_at(asp.read_partiels("json"), 0.5))
score: str = asp.render_svg(asp.events_to_mei(asp.snapshot_to_events(peaks, 0.0, 1.0), colors=True),
                            timeline=True)

bad: str = asp.amplitude_to_db(0.5)  # type: ignore[assignment]
asp.viz.plot_spectrum(audio)  # type: ignore[arg-type]
asp.pyin(audio, sr, fmin=100.0)  # type: ignore[call-arg]
asp.read_partiels("csv", dedupe="yes")  # type: ignore[arg-type]
asp.tracks_to_chords(chords, 0.5, max_notes="8")  # type: ignore[arg-type]
asp.render_svg("<mei/>", timeline="yes")  # type: ignore[arg-type]
