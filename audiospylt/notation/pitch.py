"""Absolute pitch quantisation, quarter-tone spelling, and cents labels."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_number

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "RESOLUTIONS",
    "SPELLINGS",
    "pitch_table",
    "cents_label",
]


RESOLUTIONS = ("semitone", "quarter", "eighth")
SPELLINGS = ("sharp", "flat")

# One entry per half-semitone, starting at C. Octave adjustment belongs to
# the written step: the final flat entry is C of the following octave.
_SHARP = (
    ("c", ""), ("c", "1qs"), ("c", "s"), ("c", "3qs"),
    ("d", ""), ("d", "1qs"), ("d", "s"), ("d", "3qs"),
    ("e", ""), ("e", "1qs"), ("f", ""), ("f", "1qs"),
    ("f", "s"), ("f", "3qs"), ("g", ""), ("g", "1qs"),
    ("g", "s"), ("g", "3qs"), ("a", ""), ("a", "1qs"),
    ("a", "s"), ("a", "3qs"), ("b", ""), ("b", "1qs"),
)
_FLAT = (
    ("c", ""), ("d", "3qf"), ("d", "f"), ("d", "1qf"),
    ("d", ""), ("e", "3qf"), ("e", "f"), ("e", "1qf"),
    ("e", ""), ("f", "1qf"), ("f", ""), ("g", "3qf"),
    ("g", "f"), ("g", "1qf"), ("g", ""), ("a", "3qf"),
    ("a", "f"), ("a", "1qf"), ("a", ""), ("b", "3qf"),
    ("b", "f"), ("b", "1qf"), ("b", ""), ("c", "1qf"),
)
_ALTER = {"": 0.0, "s": 100.0, "f": -100.0, "1qs": 50.0,
          "3qs": 150.0, "1qf": -50.0, "3qf": -150.0}


def _finite_scalar(value: float, name: str) -> float:
    """Check real scalars without accepting booleans or numeric strings."""
    try:
        if (not is_number(value)
                or isinstance(value, (bool, np.bool_, complex, np.complexfloating))):
            raise ValueError
        number = float(value)
        if not np.isfinite(number):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number; got {value!r}") from error
    return number


def _positive_scalar(value: float, name: str) -> float:
    number = _finite_scalar(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be > 0; got {value!r}")
    return number


def _decimals(decimals: int) -> int:
    if (isinstance(decimals, (bool, np.bool_)) or not isinstance(decimals, Integral)
            or decimals < 0):
        raise ValueError(f"decimals must be an integer >= 0; got {decimals!r}")
    return int(decimals)


def pitch_table(
    freq: ArrayLike, *, resolution: str = "quarter", spelling: str = "sharp", a4: float = 440.0,
) -> pd.DataFrame:
    """Quantise absolute pitch, then spell it using quarter-tone accidentals.

    Parameters
    ----------
    freq : real scalar or 1-D array-like
        Finite frequencies > 0 Hz. Input order is preserved; booleans rejected.
    resolution : {'semitone', 'quarter', 'eighth'}, default 'quarter'
        Semitone uses a 1-semitone grid; both other modes use a half-semitone
        grid. Eighth mode needs a cents label to represent the finer pitch.
        Half-grid ties round toward higher absolute pitch, with a tolerance
        of 1e-9 grid units for floating-point log2 noise.
    spelling : {'sharp', 'flat'}, default 'sharp'
        Choose the written step and Stein–Zimmermann accidental.
    a4 : float, default 440.0
        Finite positive reference frequency in Hz.

    Returns
    -------
    pandas.DataFrame
        Columns freq, midi, written_midi, step, octave, accid, alter_cents,
        residual. Residual is the unrounded input minus written pitch in cents.
        Octaves follow the written step (C4 = MIDI 60), without a range limit.

    Raises
    ------
    ValueError
        Invalid frequencies, resolution, spelling, or reference frequency.
        Invalid frequency positions are zero-based, independent of Series index.
    """
    if not isinstance(resolution, str) or resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}; got {resolution!r}")
    if not isinstance(spelling, str) or spelling not in SPELLINGS:
        raise ValueError(f"spelling must be one of {SPELLINGS}; got {spelling!r}")
    a4 = _positive_scalar(a4, "a4")
    try:
        # Object conversion preserves booleans in mixed numeric sequences.
        values = np.asarray(freq, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError("freq must be a scalar or a 1-D sequence of real numbers") from error
    if values.ndim > 1:
        raise ValueError("freq must be a scalar or a 1-D sequence of real numbers")
    values = values.reshape(-1)
    frequencies = np.empty(values.size, dtype=np.float64)
    bad = []
    for pos, value in enumerate(values):
        try:
            frequencies[pos] = _positive_scalar(value, "freq")
        except ValueError:
            bad.append(pos)
    if bad:
        raise ValueError(
            f"freq must contain finite real numbers > 0 (booleans rejected): "
            f"{len(bad)} value(s); first positions {bad[:5]}"
        )
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        logarithm = np.log2(frequencies / a4)
    # Finite positive inputs can still overflow/underflow their ratio.
    fallback = ~np.isfinite(logarithm)
    logarithm[fallback] = np.log2(frequencies[fallback]) - np.log2(a4)
    midi = 69 + 12 * logarithm
    grid = 1.0 if resolution == "semitone" else 0.5
    scaled = midi / grid
    lower = np.floor(scaled)
    ties = np.abs(scaled - (lower + 0.5)) <= 1e-9
    written = np.where(ties, lower + 1, np.floor(scaled + 0.5)) * grid
    units = (written * 2).astype(np.int64)
    pcs = units % 24
    table = _SHARP if spelling == "sharp" else _FLAT
    steps = [table[pc][0] for pc in pcs]
    accidentals = [table[pc][1] for pc in pcs]
    octaves = units // 24 - 1
    if spelling == "flat":
        octaves = octaves + (pcs == 23)
    return pd.DataFrame({
        "freq": frequencies, "midi": midi, "written_midi": written,
        "step": pd.Series(steps, dtype=object), "octave": octaves,
        "accid": pd.Series(accidentals, dtype=object),
        "alter_cents": np.array([_ALTER[a] for a in accidentals], dtype=np.float64),
        "residual": (midi - written) * 100,
    })


def cents_label(residual: float, decimals: int = 0) -> str:
    """Format a signed cents remainder, with unsigned zero and ASCII minus.

    Parameters
    ----------
    residual : float
        Finite real cents remainder relative to the written note.
    decimals : int, default 0
        Nonnegative number of decimal places; booleans rejected.

    Returns
    -------
    str
        For example '+12', '-3', '0', or with decimals=1, '+12.3', '0.0'.

    Raises
    ------
    ValueError
        Invalid decimals or a nonfinite/nonreal residual.
    """
    decimals = _decimals(decimals)
    residual = _finite_scalar(residual, "residual")
    text = f"{residual:+.{decimals}f}"
    return text[1:] if float(text) == 0 else text
