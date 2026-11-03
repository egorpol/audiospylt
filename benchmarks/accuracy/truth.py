"""Synthetic suite with analytic truth: partial frequencies and amplitudes at any time.

Only numpy is needed, so every runner venv can import this file. Amplitudes are
sine peak amplitudes re full scale (AudioSpylt D4).
"""

from __future__ import annotations

import numpy as np

SR = 44100
DURATION = 2.0
FADE = 0.01    # raised-cosine fade in/out (s)
EDGE = 0.005   # total raised-cosine transition width (s)
NEUTRAL = 0.05  # unscored half-width around transitions (s)
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


ORIGINAL = tuple(SIGNALS)
SIGNALS.update({
    "cross_fast": {"family": "crossing", "kind": "sweeps", "sweeps": [(300, 900), (900, 300)],
                   "amps": (0.2, 0.1)},
    "cross_slow": {"family": "crossing", "kind": "sweeps", "sweeps": [(400, 600), (600, 400)],
                   "amps": (0.2, 0.1)},
    "stagger_harm": {"family": "onset/offset", "kind": "f0", "f0": ("const", 220.0), "count": 6, "base": 0.25,
                     "on": [0.2 + 0.1 * k for k in range(6)], "off": [1.8 - 0.1 * k for k in range(6)]},
    "stagger_inharm": {"family": "onset/offset", "kind": "fixed", "freqs": INHARMONIC_FREQS,
                       "amps": INHARMONIC_AMPS, "on": (0.25, 0.65, 0.40, 0.90, 0.55),
                       "off": (1.75, 1.20, 1.55, 1.40, 1.85)},
    **{f"drop_{ms}": {"family": "dropout", "kind": "f0", "f0": ("const", 220.0), "count": 6, "base": 0.25,
                      "drop": [[0.35 + 0.25 * k, ms / 1000] for k in range(6)]} for ms in (30, 80)},
    **{f"trem_{db}": {"family": "tremolo", "kind": "f0", "f0": ("const", 220.0), "count": 8, "base": 0.25,
                      "trem": (6, depth)} for db, depth in ((20, 0.9), (40, 0.99))},
})

# Tuning uses only these perturbed signals, never the published suite.
DEV_SIGNALS: dict[str, dict] = {
    "dev_stat_harm": {"family": "stationary", "kind": "f0", "f0": ("const", 196.0), "count": 12, "base": 0.25},
    "dev_stat_inharm": {"family": "stationary", "kind": "fixed",
                        "freqs": (187.31, 503.93, 1019.87, 2087.21, 3611.47), "amps": INHARMONIC_AMPS},
    "dev_gliss_900": {"family": "glissando", "kind": "f0", "f0": ("exp", 196.0, 196 * 2**1.5),
                      "count": 6, "base": 0.25},
    "dev_gliss_down": {"family": "glissando", "kind": "f0", "f0": ("exp", 440.0, 220.0),
                       "count": 6, "base": 0.25},
    "dev_vib_75": {"family": "vibrato", "kind": "f0", "f0": ("vib", 392.0, 75.0, 6.0), "count": 8, "base": 0.25},
    "dev_vib_150": {"family": "vibrato", "kind": "f0", "f0": ("vib", 392.0, 150.0, 4.5), "count": 8, "base": 0.25},
    **{f"dev_pair_{d}": {"family": "close pair", "kind": "fixed", "freqs": (392.0, 392.0 + d), "amps": (0.4, 0.4)}
       for d in (15, 30)},
    **{f"dev_noise_{s}": {"family": "noise", "kind": "f0", "f0": ("const", 196.0), "count": 12, "base": 0.1,
                          "snr_db": float(s)} for s in (15, 5)},
    "dev_cross": {"family": "crossing", "kind": "sweeps", "sweeps": [(350, 700), (700, 350)], "amps": (0.15, 0.2)},
    "dev_stagger": {"family": "onset/offset", "kind": "f0", "f0": ("const", 196.0), "count": 6, "base": 0.25,
                    "on": [0.7 - 0.1 * k for k in range(6)], "off": [1.8 - 0.1 * k for k in range(6)]},
    **{f"dev_drop_{ms}": {"family": "dropout", "kind": "f0", "f0": ("const", 196.0), "count": 6, "base": 0.25,
                          "drop": [[1.6 - 0.25 * k, ms / 1000] for k in range(6)]} for ms in (40, 60)},
    "dev_trem_30": {"family": "tremolo", "kind": "f0", "f0": ("const", 196.0), "count": 8, "base": 0.25,
                    "trem": (5, 0.968)},
}


def _spec(name: str) -> dict:
    return SIGNALS[name] if name in SIGNALS else DEV_SIGNALS[name]


def seed(name: str) -> int:
    """Stable seeds, with separate ranges for original, added and development files."""
    if name in ORIGINAL:
        return sorted(ORIGINAL).index(name)
    if name in SIGNALS:
        return 100 + sorted(set(SIGNALS) - set(ORIGINAL)).index(name)
    return 1000 + sorted(DEV_SIGNALS).index(name)


def transitions(name: str) -> list[list[float]]:
    """In-file transition times per fixed partial row (file fades excluded)."""
    spec = _spec(name)
    count = spec["count"] if spec["kind"] == "f0" else len(spec["amps"])
    times = [[] for _ in range(count)]
    for key in ("on", "off"):
        for k, tau in enumerate(spec.get(key, [])):
            times[k].append(tau)
    for k, (centre, width) in enumerate(spec.get("drop", [])):
        times[k].extend([centre - width / 2, centre + width / 2])
    return [[tau for tau in row if 0 < tau < DURATION] for row in times]


def gates(name: str, t: np.ndarray) -> np.ndarray:
    """Raised-cosine gates, one fixed row per partial, without tremolo or file fades."""
    spec = _spec(name)
    t = np.asarray(t, dtype=float)
    count = spec["count"] if spec["kind"] == "f0" else len(spec["amps"])
    gate = np.ones((count, len(t)))

    def rise(tau):
        return 0.5 - 0.5 * np.cos(np.pi * np.clip((t - tau + EDGE / 2) / EDGE, 0, 1))

    for k, tau in enumerate(spec.get("on", [])):
        gate[k] = np.minimum(gate[k], rise(tau))
    for k, tau in enumerate(spec.get("off", [])):
        gate[k] = np.minimum(gate[k], 1 - rise(tau))
    for k, (centre, width) in enumerate(spec.get("drop", [])):
        dropout = np.maximum(1 - rise(centre - width / 2), rise(centre + width / 2))
        gate[k] = np.minimum(gate[k], dropout)
    return gate


def states(name: str, t: np.ndarray) -> np.ndarray:
    """int8 partial states: absent (0), present (1), or neutral near a transition (-1)."""
    t = np.asarray(t, dtype=float)
    state = gates(name, t).astype(np.int8)
    for k, times in enumerate(transitions(name)):
        for tau in times:
            state[k, np.abs(t - tau) <= NEUTRAL] = -1
    return state


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
    """True (freq, amp) arrays of shape (partials, len(t)), in fixed partial identity order."""
    spec = _spec(name)
    t = np.asarray(t, dtype=float)
    if spec["kind"] == "f0":
        k = np.arange(1, spec["count"] + 1)[:, None]
        freq = k * _f0(spec["f0"], t)[None, :]
        amp = (spec["base"] / k) * np.ones_like(t)[None, :]
    elif spec["kind"] == "sweeps":
        ends = np.asarray(spec["sweeps"], dtype=float)
        freq = ends[:, :1] * (ends[:, 1:] / ends[:, :1]) ** (t[None, :] / DURATION)
        amp = np.array(spec["amps"], dtype=float)[:, None] * np.ones_like(t)[None, :]
    else:
        freq = np.array(spec["freqs"], dtype=float)[:, None] * np.ones_like(t)[None, :]
        amp = np.array(spec["amps"], dtype=float)[:, None] * np.ones_like(t)[None, :]
    amp = amp * gates(name, t)
    if "trem" in spec:
        rate, depth = spec["trem"]
        amp = amp * (1 - depth * (1 - np.cos(2 * np.pi * rate * t)) / 2)[None, :]
    return freq, amp * envelope(t)[None, :]


def lowest_freq(name: str) -> float:
    """Lowest true frequency over the whole file (Loris resolution rule of thumb)."""
    t = np.arange(int(SR * DURATION)) / SR
    return float(partials(name, t)[0].min())


def tolerance_cents(freq: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
    """Half the nearest non-absent frequency gap, capped at 50 cents; absent rows get NaN."""
    cents = 1200 * np.log2(freq)
    active = np.ones(freq.shape, bool) if state is None else state != 0
    order = np.argsort(np.where(active, cents, np.inf), axis=0, kind="stable")
    ordered = np.take_along_axis(cents, order, axis=0)
    valid = np.take_along_axis(active, order, axis=0)
    gap = np.full_like(cents, np.inf)
    if len(cents) > 1:
        d = np.where(valid[:-1] & valid[1:], np.diff(ordered, axis=0), np.inf)
        gap[:-1] = np.minimum(gap[:-1], d)
        gap[1:] = np.minimum(gap[1:], d)
    tol = np.empty_like(cents)
    np.put_along_axis(tol, order, np.minimum(50.0, gap / 2), axis=0)
    return np.where(active, tol, np.nan)


def render(name: str) -> np.ndarray:
    """The file's samples (float64); phases from a fixed per-file seed."""
    n = np.arange(int(SR * DURATION))
    t = n / SR
    freq, amp = partials(name, t)
    rng = np.random.default_rng(seed(name))
    phase0 = rng.uniform(0, 2 * np.pi, size=(len(freq), 1))
    # trapezoidal integral of the instantaneous frequency
    inc = np.concatenate([np.zeros((len(freq), 1)), 0.5 * (freq[:, 1:] + freq[:, :-1]) / SR], axis=1)
    x = np.sum(amp * np.sin(phase0 + 2 * np.pi * np.cumsum(inc, axis=1)), axis=0)
    snr = _spec(name).get("snr_db")
    if snr is not None:
        power = np.sum(amp[:, len(t) // 2] ** 2 / 2)
        x = x + rng.standard_normal(len(t)) * np.sqrt(power / 10 ** (snr / 10))
    return x
