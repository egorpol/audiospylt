"""Synthetic suite with analytic truth: partial frequencies and amplitudes at any time.

Only numpy is needed, so every runner venv can import this file. Amplitudes are
sine peak amplitudes re full scale (AudioSpylt D4).
"""

from __future__ import annotations

import numpy as np

SR = 44100
DURATION = 2.0
FADE = 0.01    # raised-cosine fade in/out (s)
GUARD = 0.1    # evaluation skips this much at both ends (s)
INHARMONIC_FREQS = (223.37, 457.81, 911.29, 1843.43, 3299.61)
INHARMONIC_AMPS = (0.12, 0.10, 0.08, 0.06, 0.04)

# name -> spec; "f0" kinds hold k * f0(t) harmonics with amps base / k
SIGNALS: dict[str, dict] = {
    "stat_harm": {"family": "stationary", "kind": "f0", "f0": ("const", 220.0), "count": 12, "base": 0.25},
    "stat_inharm": {"family": "stationary", "kind": "fixed", "freqs": INHARMONIC_FREQS, "amps": INHARMONIC_AMPS},
    "gliss_600": {"family": "glissando", "kind": "f0", "f0": ("exp", 220.0, 440.0), "count": 6, "base": 0.25},
    "gliss_1200": {"family": "glissando", "kind": "f0", "f0": ("exp", 220.0, 880.0), "count": 6, "base": 0.25},
    "vib_50": {"family": "vibrato", "kind": "f0", "f0": ("vib", 440.0, 50.0, 5.5), "count": 8, "base": 0.25},
    "vib_100": {"family": "vibrato", "kind": "f0", "f0": ("vib", 440.0, 100.0, 5.5), "count": 8, "base": 0.25},
    **{f"pair_{d}": {"family": "close pair", "kind": "fixed", "freqs": (440.0, 440.0 + d), "amps": (0.4, 0.4)}
       for d in (5, 10, 20, 40)},
    **{f"noise_{s}": {"family": "noise", "kind": "f0", "f0": ("const", 220.0), "count": 12, "base": 0.1,
                      "snr_db": float(s)} for s in (20, 10, 0)},
}


def _f0(spec: tuple, t: np.ndarray) -> np.ndarray:
    if spec[0] == "const":
        return np.full_like(t, spec[1])
    if spec[0] == "exp":
        return spec[1] * (spec[2] / spec[1]) ** (t / DURATION)
    if spec[0] == "vib":
        return spec[1] * 2.0 ** (spec[2] / 1200 * np.sin(2 * np.pi * spec[3] * t))
    raise ValueError(spec)


def envelope(t: np.ndarray) -> np.ndarray:
    """Raised-cosine fades of FADE seconds at both ends, 1 in between."""
    rise = np.clip(t / FADE, 0, 1)
    fall = np.clip((DURATION - t) / FADE, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * np.minimum(rise, fall))


def partials(name: str, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True (freq, amp) arrays of shape (partials, len(t)), sorted by frequency."""
    spec = SIGNALS[name]
    t = np.asarray(t, dtype=float)
    if spec["kind"] == "f0":
        k = np.arange(1, spec["count"] + 1)[:, None]
        freq = k * _f0(spec["f0"], t)[None, :]
        amp = (spec["base"] / k) * np.ones_like(t)[None, :]
    else:
        freq = np.array(spec["freqs"], dtype=float)[:, None] * np.ones_like(t)[None, :]
        amp = np.array(spec["amps"], dtype=float)[:, None] * np.ones_like(t)[None, :]
    return freq, amp * envelope(t)[None, :]


def lowest_freq(name: str) -> float:
    """Lowest true frequency over the whole file (Loris resolution rule of thumb)."""
    t = np.arange(int(SR * DURATION)) / SR
    return float(partials(name, t)[0].min())


def tolerance_cents(freq: np.ndarray) -> np.ndarray:
    """Match tolerance per true partial: min(50 c, half the gap to a neighbour), same shape as freq."""
    cents = 1200 * np.log2(freq)
    gap = np.full_like(cents, np.inf)
    if len(cents) > 1:
        d = np.diff(cents, axis=0)
        gap[:-1] = np.minimum(gap[:-1], d)
        gap[1:] = np.minimum(gap[1:], d)
    return np.minimum(50.0, gap / 2)


def render(name: str) -> np.ndarray:
    """The file's samples (float64); phases from a fixed per-file seed."""
    n = np.arange(int(SR * DURATION))
    t = n / SR
    freq, amp = partials(name, t)
    rng = np.random.default_rng(sorted(SIGNALS).index(name))
    phase0 = rng.uniform(0, 2 * np.pi, size=(len(freq), 1))
    # trapezoidal integral of the instantaneous frequency
    inc = np.concatenate([np.zeros((len(freq), 1)), 0.5 * (freq[:, 1:] + freq[:, :-1]) / SR], axis=1)
    x = np.sum(amp * np.sin(phase0 + 2 * np.pi * np.cumsum(inc, axis=1)), axis=0)
    snr = SIGNALS[name].get("snr_db")
    if snr is not None:
        power = np.sum(amp[:, len(t) // 2] ** 2 / 2)
        x = x + rng.standard_normal(len(t)) * np.sqrt(power / 10 ** (snr / 10))
    return x
