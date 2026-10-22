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

bad: str = asp.amplitude_to_db(0.5)  # type: ignore[assignment]
asp.viz.plot_spectrum(audio)  # type: ignore[arg-type]
