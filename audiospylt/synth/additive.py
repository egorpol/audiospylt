"""Additive synthesis of events tables at the output sample rate (D9 phase modes)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from audiospylt.analysis.spectrum import _finite_scalar
from audiospylt.io.audio import _sample_rate
from audiospylt.model.schema import SchemaError, validate

PHASE_MODES = ("continuous", "reset", "absolute")
_OLD_PHASE_NAMES = {"chirp": "reset", "global": "absolute"}
# Event times this close (in samples) to a sample instant start or stop on it.
_SNAP_SAMPLES = 1e-6


def _sample_index(times: np.ndarray, sr: int) -> np.ndarray:
    """Return the first sample index n with n / sr >= t, absorbing float noise."""
    position = np.asarray(times, dtype=np.float64) * sr
    nearest = np.round(position)
    snapped = np.where(np.abs(position - nearest) <= _SNAP_SAMPLES, nearest, np.ceil(position))
    return snapped.astype(np.int64)


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = _finite_scalar(value, name)
    if number < 0:
        raise ValueError(f"{name} must be >= 0; got {value!r}")
    return number


def _track_ids(table: pd.DataFrame) -> np.ndarray | None:
    if "track_id" not in table.columns:
        return None
    column = table["track_id"]
    if is_integer_dtype(column.dtype) and not column.isna().any():
        return column.to_numpy(dtype=np.int64)
    # Compare in Python numbers, not float64: read_table keeps large ids as exact Decimals.
    ids = []
    for value in column.to_list():
        try:
            number = int(value)
            exact = not isinstance(value, (str, bytes)) and number == value
        except (TypeError, ValueError, OverflowError):
            exact = False
        if not exact or not -(2**63) <= number < 2**63:
            raise ValueError("track_id must hold int64 integers, with no missing values")
        ids.append(number)
    return np.array(ids, dtype=np.int64)


def _link(
    n0: np.ndarray, n1: np.ndarray, freq_start: np.ndarray, freq_stop: np.ndarray,
    track: np.ndarray | None, link_cents: float | None,
) -> np.ndarray:
    """Return each event's predecessor row, or -1 for the first event of a voice."""
    pred = np.full(n0.size, -1, dtype=np.int64)
    covered = n1 > n0
    # A link never crosses 0 Hz: predecessors must stop, successors start, above it.
    rows = np.flatnonzero(covered & (freq_stop > 0))
    by_end = rows[np.argsort(n1[rows], kind="stable")]
    rows = np.flatnonzero(covered & (freq_start > 0))
    by_start = rows[np.argsort(n0[rows], kind="stable")]
    ends, starts = n1[by_end], n0[by_start]
    for boundary in np.intersect1d(ends, starts):
        prev = by_end[np.searchsorted(ends, boundary):np.searchsorted(ends, boundary, "right")]
        succ = by_start[np.searchsorted(starts, boundary):np.searchsorted(starts, boundary, "right")]
        cents = np.abs(1200 * np.log2(freq_start[succ][None, :] / freq_stop[prev][:, None]))
        allowed = np.ones(cents.shape, dtype=bool) if link_cents is None else cents <= link_cents
        if track is not None:
            allowed &= track[prev][:, None] == track[succ][None, :]
        i, j = np.nonzero(allowed)
        # Nearest frequency first; ties go to the earlier predecessor, then successor row.
        taken_prev: set[int] = set()
        taken_succ: set[int] = set()
        for k in np.lexsort((succ[j], prev[i], cents[i, j])):
            a, b = int(prev[i[k]]), int(succ[j[k]])
            if a not in taken_prev and b not in taken_succ:
                pred[b] = a
                taken_prev.add(a)
                taken_succ.add(b)
    return pred


def synthesize(
    events: pd.DataFrame,
    sr: int = 44100,
    *,
    phase: str = "continuous",
    link_cents: float | None = 100.0,
    fade: float = 0.0,
    nyquist: str = "drop",
    duration: float | None = None,
) -> np.ndarray:
    """Render an events table as a sum of sinusoids, sampled directly at ``sr``.

    Each row is one sinusoid over the half-open interval ``[time_start,
    time_stop)``. Frequency and amplitude move linearly from their ``_start``
    to their ``_stop`` values, and the phase is the exact integral of that
    frequency ramp: ``amp(t) * sin(phi0 + 2*pi*integral(freq))``. Sample n
    sits at ``n / sr`` and belongs to a row when ``time_start <= n / sr <
    time_stop``, so rows that meet at a boundary never both contribute there.
    Times within 1e-6 of a sample period from a sample instant count as on
    it, so float noise in table times does not shift a boundary.

    Parameters
    ----------
    events : pandas.DataFrame
        A v1 ``events`` table (validated; 0.6 ``amp_min``/``amp_max`` names
        must first go through ``audiospylt.model.schema.upgrade``). Amplitudes
        are D4 peak amplitudes re full scale, as ``find_peaks`` reports them.
        An optional integer ``track_id`` column restricts continuous links.
    sr : int, default 44100
        Output sample rate in Hz; synthesis evaluates the oscillators at this
        rate, with no resampling stage.
    phase : {'continuous', 'reset', 'absolute'}, default 'continuous'
        Start phase ``phi0`` of each row:

        - ``'continuous'``: rows are linked into voices. A row continues
          another when it starts on the sample where the other stops and its
          ``freq_start`` lies within ``link_cents`` of the other's
          ``freq_stop``; links are one-to-one, nearest frequency first. A
          linked row starts with its predecessor's end phase, so the waveform
          stays phase-continuous across frequency changes. The first row of a
          voice starts at phase 0. When the two rows' times differ by less
          than a sample, the handover follows the predecessor's ramp to the
          successor's ``time_start``.
        - ``'reset'``: every row starts at phase 0 (0.6 ``reset`` and
          ``chirp``, which compute the same thing).
        - ``'absolute'``: ``phi0 = 2*pi*freq_start*time_start``, as if the
          oscillator had run at ``freq_start`` since time 0 (0.6 ``global``).
          Constant-frequency rows at the same frequency join seamlessly;
          frequency changes between rows leave phase jumps.
    link_cents : float or None, default 100.0
        Largest frequency step, in cents, that ``'continuous'`` bridges.
        None links regardless of frequency. Links never cross 0 Hz (a 0 Hz row
        stays silent rather than holding an inherited phase as DC), and rows
        that cover no sample are never linked.
    fade : float, default 0.0
        Raised-cosine ramp, in seconds, at the start and end of each voice
        (each row is its own voice in ``'reset'`` and ``'absolute'``).
        Links inside a voice get no fade. When a voice is shorter than two
        ramps, the ramps overlap and multiply.
    nyquist : {'drop', 'alias'}, default 'drop'
        ``'drop'`` silences every sample whose instantaneous frequency is at
        or above ``sr / 2`` and warns once; ``'alias'`` keeps them, so they
        fold exactly as sampling predicts (30 kHz at 44.1 kHz sounds at
        14.1 kHz). A partial at exactly ``sr / 2`` in sine phase samples to
        zero either way.
    duration : float or None, default None
        Output length in seconds (rounded up to whole samples). None ends at
        the latest ``time_stop``; a shorter duration cuts rows off, and the
        cut gets no fade (``fade`` ramps follow the table's voice ends).

    Returns
    -------
    numpy.ndarray
        Float64 mono samples ``(samples,)``. Nothing is normalised or
        clipped: ``audiospylt.io.audio.write_audio`` refuses to write PCM
        that would clip, and its ``normalize`` option scales on request.

    Notes
    -----
    Measured 2026-10-19 (``revision/step-1.4``), resynthesising 5 stationary
    partials from 20 frames of 50 ms (v1 ``find_peaks``, 100 ms Hann
    windows): log-spectral distance 0.004 dB continuous, 0.024 dB absolute,
    16 dB reset (phase resets click at every frame). With a 7.4 Hz/s
    glissando in the top partial: 0.056, 10.5 and 18 dB.

    Raises
    ------
    SchemaError
        The table is not a valid ``events`` table.
    ValueError
        Invalid rate, mode, ``link_cents``, ``fade``, ``duration`` or
        ``track_id`` values.
    """
    columns = getattr(events, "columns", ())
    legacy = [name for name in ("amp_min", "amp_max") if name in columns]
    if legacy:
        raise SchemaError(
            f"events use 0.6 column names {legacy}; convert them with "
            "audiospylt.model.schema.upgrade(events) first"
        )
    table = validate(events, "events")
    sr = _sample_rate(sr)
    if isinstance(phase, str) and phase in _OLD_PHASE_NAMES:
        raise ValueError(f"phase {phase!r} is the 0.6 name; use phase={_OLD_PHASE_NAMES[phase]!r}")
    if phase not in PHASE_MODES:
        raise ValueError(f"phase must be one of {PHASE_MODES}; got {phase!r}")
    if nyquist not in ("drop", "alias"):
        raise ValueError(f"nyquist must be 'drop' or 'alias'; got {nyquist!r}")
    link_cents = _optional_nonnegative(link_cents, "link_cents")
    fade = _optional_nonnegative(fade, "fade")
    duration = _optional_nonnegative(duration, "duration")

    t0, t1 = table["time_start"].to_numpy(), table["time_stop"].to_numpy()
    f0, f1 = table["freq_start"].to_numpy(), table["freq_stop"].to_numpy()
    a0, a1 = table["amp_start"].to_numpy(), table["amp_stop"].to_numpy()
    n0, n1 = _sample_index(t0, sr), _sample_index(t1, sr)
    if duration is not None:
        n_samples = int(_sample_index(duration, sr))
    else:
        n_samples = int(n1.max()) if len(table) else 0
    output = np.zeros(n_samples)
    if not len(table):
        return output

    order = np.argsort(n0, kind="stable")
    pred = np.full(len(table), -1, dtype=np.int64)
    if phase == "continuous":
        pred = _link(n0, n1, f0, f1, _track_ids(table), link_cents)
    # Phases are tracked in cycles (mod 1) so long voices keep their precision.
    start_cycles = np.zeros(len(table))
    if phase == "absolute":
        start_cycles = np.mod(f0 * t0, 1.0)
    voice_start = t0.copy()
    for row in order:
        p = pred[row]
        if p >= 0:
            # Hand over at this row's own start time: rows that meet on one sample from
            # slightly different float times then neither drop nor repeat the gap's phase.
            tau = t0[row] - t0[p]
            ramp = f0[p] + 0.5 * (f1[p] - f0[p]) * tau / (t1[p] - t0[p])
            start_cycles[row] = np.mod(start_cycles[p] + tau * ramp, 1.0)
            voice_start[row] = voice_start[p]
    voice_stop = t1.copy()
    succ = np.full(len(table), -1, dtype=np.int64)
    succ[pred[pred >= 0]] = np.flatnonzero(pred >= 0)
    for row in order[::-1]:
        if succ[row] >= 0:
            voice_stop[row] = voice_stop[succ[row]]

    dropped = 0
    for row in order:
        first, stop = n0[row], min(n1[row], n_samples)
        if stop <= first:
            continue
        span = t1[row] - t0[row]
        times = np.arange(first, stop) / sr
        tau = times - t0[row]
        frac = tau / span
        freq = f0[row] + (f1[row] - f0[row]) * frac
        cycles = start_cycles[row] + tau * (f0[row] + 0.5 * (f1[row] - f0[row]) * frac)
        amp = a0[row] + (a1[row] - a0[row]) * frac
        if fade > 0:
            rise = np.clip((times - voice_start[row]) / fade, 0.0, 1.0)
            fall = np.clip((voice_stop[row] - times) / fade, 0.0, 1.0)
            amp = amp * (0.5 - 0.5 * np.cos(np.pi * rise)) * (0.5 - 0.5 * np.cos(np.pi * fall))
        keep = slice(None)
        if nyquist == "drop":
            keep = freq < sr / 2
            if not keep.all():
                dropped += 1
        # Gated samples are never evaluated, so an extreme frequency cannot leave NaN behind.
        output[first:stop][keep] += amp[keep] * np.sin(2 * np.pi * cycles[keep])
    if dropped:
        warnings.warn(
            f"{dropped} event(s) reach the Nyquist frequency {sr / 2:g} Hz (sr={sr}); "
            "their samples at or above it are silent. Use a higher sr, or "
            "nyquist='alias' to let them fold.",
            UserWarning,
            stacklevel=2,
        )
    return output
