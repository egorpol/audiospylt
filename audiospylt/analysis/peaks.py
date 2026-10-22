"""QIFFT-dB partial estimates as validated v1 snapshot tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from audiospylt.analysis.spectrum import (
    Spectrum, _finite_scalar, amplitude_to_db, db_to_amplitude,
)
from audiospylt.model.schema import validate

__all__ = ["find_peaks"]


def find_peaks(
    spec: Spectrum,
    *,
    min_db: float | None = -60.0,
    rel_db: float | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
    min_distance_hz: float | None = None,
    prominence_db: float | None = None,
    interpolate: bool = True,
    floor_db: float = -120.0,
) -> pd.DataFrame:
    """Estimate partial frequencies and D4 peak amplitudes with QIFFT on dB.

    Parameters
    ----------
    spec : Spectrum
        One-sided D4 spectrum. Endpoints (DC and Nyquist) are never peaks.
    min_db : float or None, default -60.0
        Absolute amplitude threshold in dB re full scale; None disables it.
        The 0.6 linear thresholds on unscaled magnitudes do not carry over.
    rel_db : float or None
        Nonpositive dB offset from the loudest frequency/absolute survivor.
    fmin, fmax : float, optional
        Inclusive interpolated-frequency bounds in Hz; fmax defaults to sr/2.
        Require 0 <= fmin < fmax.
    min_distance_hz : float or None
        Positive minimum spacing. Greedily keep loudest survivors, breaking
        ties by lower frequency. Peaks exactly this distance apart survive.
    prominence_db : float or None
        Nonnegative minimum whole-spectrum prominence in dB, measured before
        interpolation on the same floored dB array used for detection.
    interpolate : bool, default True
        Use QIFFT-dB (Abe & Smith, 2004) for every window; False keeps raw bins.
    floor_db : float, default -120.0
        Finite dB floor shared by detection, interpolation, and prominence.

    Returns
    -------
    pandas.DataFrame
        Validated snapshot, sorted by freq with a RangeIndex: float64 freq
        (Hz), amp (D4 peak units), nullable Int64 bin, float64 prominence (dB).
        Empty results retain these columns/dtypes. attrs['meta'] records sr,
        frame_length, n_fft, window (string), and interpolation; no phase is
        estimated. Filters act in frequency, absolute, relative, prominence,
        then distance order, so excluded peaks cannot suppress survivors.

    Notes
    -----
    Worst measured isolated-sine errors (200 random offsets/phases), in cents
    at 4, 5, 6, 8, and 22 cycles per window respectively:

    * Hann, zero_pad=4: 1.02, 0.41, 0.20, 0.09, 0.016; amplitude errors
      0.007 dB at 4 cycles and 0.001 dB at 22 cycles.
    * Hann, zero_pad=1: 7.8 at 4, 4.9 at 6, 1.3 at 22 cycles; up to 0.32 dB.
    * Boxcar, zero_pad=4: 21, 11, 8.4, 4.5, 0.57; 0.36/0.06 dB at 4/22.
    * Hamming, zero_pad=4: 5.0, 3.4, 2.4, 1.3, 0.17; 0.05 dB at 4 cycles.
    * Blackman-Harris, zero_pad=4: 0.06, 0.04, 0.02, 0.01, 0.004 cents.

    Boxcar and Hamming are biased under QIFFT. Peaks on zero_pad=1 spectra
    (including default spectrogram frames) can have 7.8 cents / 0.32 dB error
    at 4 cycles; use zero_pad=4 for peak picking. Negative-frequency-image
    interference causes the roughly 1-cent Hann floor at 4 cycles; increasing
    zero-padding does not reduce it. D4 uses 20*log10(amp) re 1.0.

    Raises
    ------
    ValueError
        Nonfinite thresholds, invalid bounds/signs, or invalid amplitudes.
    """
    fmin = _finite_scalar(fmin, "fmin")
    fmax = spec.sr / 2 if fmax is None else _finite_scalar(fmax, "fmax")
    if fmin < 0 or fmax <= fmin:
        raise ValueError("frequency bounds must satisfy 0 <= fmin < fmax")
    if min_db is not None:
        min_db = _finite_scalar(min_db, "min_db")
    if rel_db is not None:
        rel_db = _finite_scalar(rel_db, "rel_db")
        if rel_db > 0:
            raise ValueError("rel_db must be <= 0")
    if prominence_db is not None:
        prominence_db = _finite_scalar(prominence_db, "prominence_db")
        if prominence_db < 0:
            raise ValueError("prominence_db must be >= 0")
    if min_distance_hz is not None:
        min_distance_hz = _finite_scalar(min_distance_hz, "min_distance_hz")
        if min_distance_hz <= 0:
            raise ValueError("min_distance_hz must be > 0")

    db = amplitude_to_db(spec.amp, floor_db=floor_db)
    bins = signal.find_peaks(db)[0]
    prominence = signal.peak_prominences(db, bins)[0]
    peak_db = db[bins]
    if interpolate:
        a, b, c = db[bins - 1], peak_db, db[bins + 1]
        den = a - 2 * b + c
        p = np.zeros_like(b)
        np.divide(0.5 * (a - c), den, out=p, where=den < 0)
        p = np.clip(p, -0.5, 0.5)
        freq = (bins + p) * spec.bin_hz
        peak_db = b - 0.25 * (a - c) * p
        amp = db_to_amplitude(peak_db)
    else:
        freq = bins * spec.bin_hz
        amp = spec.amp[bins]

    survivors = np.flatnonzero((freq >= fmin) & (freq <= fmax))
    if min_db is not None:
        survivors = survivors[peak_db[survivors] >= min_db]
    if rel_db is not None and survivors.size:
        survivors = survivors[peak_db[survivors] >= peak_db[survivors].max() + rel_db]
    if prominence_db is not None:
        survivors = survivors[prominence[survivors] >= prominence_db]
    if min_distance_hz is not None:
        ranked = survivors[np.lexsort((freq[survivors], -amp[survivors]))]
        kept: list[int] = []
        for candidate in ranked:
            if all(abs(freq[candidate] - freq[other]) >= min_distance_hz for other in kept):
                kept.append(candidate)
        survivors = np.asarray(kept, dtype=np.int64)
    survivors = survivors[np.argsort(freq[survivors])]
    result = validate(pd.DataFrame({
        "freq": freq[survivors],
        "amp": amp[survivors],
        "bin": pd.array(bins[survivors], dtype="Int64"),
        "prominence": prominence[survivors],
    }), "snapshot")
    result.attrs["meta"] = {
        "sr": spec.sr, "frame_length": spec.frame_length, "n_fft": spec.n_fft,
        "window": str(spec.window), "interpolation": "qifft-db" if interpolate else "none",
    }
    return result
