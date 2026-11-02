"""Legacy-compatible and transport pairing; morph gains of shared partials (0.6 full or primary)."""

from __future__ import annotations

import heapq
import warnings
from functools import partial

import numpy as np
import pandas as pd
from scipy.special import expit

from audiospylt.model.schema import validate
from audiospylt.transform.scale import _finite_scalar

__all__ = [
    "METHODS",
    "parameter_range",
    "pair",
    "morph",
]

METHODS = ("equal", "power", "sigmoid", "transport")
FREQ_SPACES = ("hz", "pitch")
_MASS_TOL = 1e-12
_MAX_SEGMENTS = 256
_SERIES_CURVE = 1e-6
_CURVE_BOUND = 1e4
_FINE = np.linspace(0.0, 1.0, 2049)
_PROBES = np.linspace(0.0, 1.0, 33)[1:-1]
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
_REFINE_WIDTH = 1e-9
# Fixed golden-section steps shrink a unit bracket under _REFINE_WIDTH.
_REFINE_ITERS = int(np.ceil(np.log(_REFINE_WIDTH) / np.log(_GOLDEN))) + 2
# Bisection on [0, 1] to double precision (~2^-60).
_BISECT_ITERS = 60
_MID_PROBE = np.array([0.5])
_CHEAP_PROBES = np.array([0.25, 0.5, 0.75])


def _snapshots(source, target):
    tables = []
    for name, table in (("source", source), ("target", target)):
        table = validate(table, "snapshot")
        if table.empty:
            raise ValueError(f"{name} snapshot must not be empty")
        dc = np.flatnonzero(table.freq.to_numpy() == 0)
        if dc.size:
            raise ValueError(
                f"{name}: {dc.size} partial(s) at 0 Hz; first row positions "
                f"{dc[:5].tolist()}; filter DC first (0 Hz has no pitch distance)"
            )
        # The normalized RangeIndex records validated input positions after sorting.
        tables.append(table.sort_values("freq", kind="stable"))
    return tables


def _cdf_mapping_indices(n_src, n_dst, power):
    # Operation order and forced endpoints are deliberately identical to 0.6.
    bins = np.linspace(0, 1, n_src) ** abs(power)
    bins[0], bins[-1] = 0, 1
    thresholds = (bins[:-1] + bins[1:]) / 2
    targets = np.linspace(0, 1, n_dst, endpoint=False) + 0.5 / n_dst
    indices = np.clip(np.digitize(targets, thresholds), 0, n_src - 1)
    indices[0], indices[-1] = 0, n_src - 1
    if power < 0:
        indices = (n_src - 1) - indices
        indices[0], indices[-1] = n_src - 1, 0
    return indices


def _sigmoid_mapping_indices(n_src, n_dst, sigmoid_range):
    raw = expit(np.linspace(-sigmoid_range, sigmoid_range, n_src))
    low, high = raw.min(), raw.max()
    sources = (np.linspace(0, 1, n_src) if high - low < 1e-9
               else (raw - low) / (high - low))
    thresholds = (sources[:-1] + sources[1:]) / 2
    return np.clip(np.digitize(np.linspace(0, 1, n_dst), thresholds), 0, n_src - 1)


def _parameter_range(n, m, method):
    if n == m:
        return (1.0, 1.0)
    if method == "power":
        candidates = np.concatenate([np.linspace(0.01, 2, 1000), np.linspace(2.01, 100, 1000)])
        mapping, fallback = _cdf_mapping_indices, 1.0
    else:
        candidates = np.concatenate([np.linspace(0.01, 10, 1000), np.linspace(10.01, 100, 1000)])
        mapping, fallback = _sigmoid_mapping_indices, 6.0
    valid = [float(p) for p in candidates if np.unique(mapping(n, m, p)).size == n]
    if not valid:
        warnings.warn(
            f"No fully covering {method} parameter found. Using default "
            f"({fallback}, {fallback}).", UserWarning, stacklevel=3,
        )
        return (fallback, fallback)
    return (valid[0], valid[-1])


def parameter_range(source: pd.DataFrame, target: pd.DataFrame, method: str) -> tuple[float, float]:
    """Find the 0.6 parameter range that covers every smaller-side partial.

    Parameters
    ----------
    source, target : pandas.DataFrame
        Nonempty v1 snapshots with strictly positive frequencies.
    method : {'power', 'sigmoid'}
        Power-law or normalized sigmoid warping. Candidate grids match 0.6.

    Returns
    -------
    tuple of float
        Minimum and maximum covering parameters; equal sizes give (1, 1).

    Warns
    -----
    UserWarning
        No covering candidate; fallback is (1, 1) for power, (6, 6) for sigmoid.

    Raises
    ------
    SchemaError
        Either input violates the snapshot contract.
    ValueError
        Unsupported method, empty snapshot, or DC partials.
    """
    if not isinstance(method, str) or method not in ("power", "sigmoid"):
        raise ValueError("parameter_range method must be 'power' or 'sigmoid'")
    source, target = _snapshots(source, target)
    return _parameter_range(min(len(source), len(target)), max(len(source), len(target)), method)


def _transport(source, target, weight):
    masses = []
    for name, table in (("source", source), ("target", target)):
        amp = table.amp.to_numpy()
        maximum = amp.max()
        if maximum == 0:
            raise ValueError(f"{name} has zero total {weight} mass")
        # Scaling before squaring/summing also handles very large peak amplitudes.
        mass = amp / maximum
        if weight == "power":
            mass = mass**2
        masses.append(mass / mass.sum())
    a, b = masses
    i = j = 0
    ra, rb = a[0], b[0]
    src, dst, cells = [], [], []
    while i < len(a) and j < len(b):
        mass = min(ra, rb)
        if mass > _MASS_TOL:
            src.append(i)
            dst.append(j)
            cells.append(mass)
        ra -= mass
        rb -= mass
        # Advance both on simultaneous exhaustion; retain tiny residuals as in
        # the reference probe rather than discarding cumulative marginal mass.
        if ra <= _MASS_TOL:
            i += 1
            if i < len(a):
                ra += a[i]
        if rb <= _MASS_TOL:
            j += 1
            if j < len(b):
                rb += b[j]
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), np.array(cells, dtype=np.float64)


def _primaries(table, own, other, identity):
    primary = np.zeros(len(table), dtype=bool)
    for rows in table.groupby(identity, sort=False).indices.values():
        group = table.iloc[rows]
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            distance = np.abs(np.log2(group[f"freq_{other}"] / group[f"freq_{own}"]))
        # Fall back to log differences for extreme positive Hz ratios.
        distance = np.where(np.isfinite(distance), distance,
                            np.abs(np.log2(group[f"freq_{other}"]) - np.log2(group[f"freq_{own}"])))
        amplitude = group["amp_target" if other == "stop" else "amp_source"]
        order = np.lexsort((rows, -amplitude, distance, -group.mass))
        primary[rows[order[0]]] = True
    return primary


def pair(
    source: pd.DataFrame,
    target: pd.DataFrame,
    method: str = "equal",
    *,
    power: float | None = None,
    sigmoid_range: float | None = None,
    weight: str = "amp",
) -> pd.DataFrame:
    """Pair partials by a 0.6 frequency map or monotone 1-D optimal transport.

    Parameters
    ----------
    source, target : pandas.DataFrame
        Nonempty v1 snapshots; amplitudes are used as given (D4 peak units).
        Frequencies must be positive. Both inputs are stably frequency-sorted.
    method : {'equal', 'power', 'sigmoid', 'transport'}
        Legacy methods remap the smaller side (target at equal sizes), keeping
        the larger side's row order. Negative parameters reverse the mapping.
    power, sigmoid_range : float or None
        Only the matching method accepts its parameter. None uses the mean
        covering range. Extreme parameters may leave smaller partials unused.
    weight : {'amp', 'power'}
        Transport mass is normalized amp or amp**2; only transport accepts a
        nondefault weight. Zero-amplitude partials have no transport rows.

    Returns
    -------
    pandas.DataFrame
        Plain pairing table, not a registered table kind. src/dst are int64
        positions in the validated inputs. Frequencies and original amplitudes
        precede mass, primary flags, and D13 gains. Each used partial has one
        primary: greatest mass, then nearest cents, then greatest other-end
        amplitude, then earliest row. Only its primary carries its amplitude.
        attrs['morph'] records method, parameter, and weight.

    Notes
    -----
    The north-west-corner monotone plan is optimal for every convex cost of
    frequency distance in 1-D; Hz versus cents does not change this coupling.
    Cells of mass <= 1e-12 are omitted; rows follow staircase order. Other
    methods assign mass 1/N per row. Silent 0 -> 0 rows remain in this table.

    Warns
    -----
    UserWarning
        A parameter leaves smaller-side partials unused or has no covering range.

    Raises
    ------
    SchemaError
        Either input violates the snapshot contract.
    ValueError
        Invalid method/parameters/weight, empty input, DC, or zero transport mass.
    """
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    if power is not None and method != "power":
        raise ValueError("power is only supported for method='power'")
    if sigmoid_range is not None and method != "sigmoid":
        raise ValueError("sigmoid_range is only supported for method='sigmoid'")
    if not isinstance(weight, str) or weight not in ("amp", "power"):
        raise ValueError("weight must be 'amp' or 'power'")
    if method != "transport" and weight != "amp":
        raise ValueError("weight is only supported for method='transport'")
    source, target = _snapshots(source, target)
    parameter = None
    if method == "transport":
        i, j, mass = _transport(source, target, weight)
    else:
        n, m = min(len(source), len(target)), max(len(source), len(target))
        if method == "equal":
            mapped = np.linspace(0, n - 1, m).round().astype(np.int64)
        else:
            value = power if method == "power" else sigmoid_range
            parameter = (float(np.mean(_parameter_range(n, m, method))) if value is None
                         else _finite_scalar(value, "power" if method == "power" else "sigmoid_range"))
            mapping = _cdf_mapping_indices if method == "power" else _sigmoid_mapping_indices
            mapped = mapping(n, m, parameter)
            unused = n - np.unique(mapped).size
            if unused:
                warnings.warn(f"{unused} smaller-side partial(s) unused by {method} pairing",
                              UserWarning, stacklevel=2)
        i, j = (mapped, np.arange(m)) if len(source) < len(target) else (np.arange(m), mapped)
        mass = np.full(m, 1 / m)
    left, right = source.iloc[i], target.iloc[j]
    result = pd.DataFrame({
        "src": left.index.to_numpy(dtype=np.int64), "dst": right.index.to_numpy(dtype=np.int64),
        "freq_start": left.freq.to_numpy(), "freq_stop": right.freq.to_numpy(),
        "amp_source": left.amp.to_numpy(), "amp_target": right.amp.to_numpy(), "mass": mass,
    })
    # Group identities use partial positions, never equal frequency values.
    for own, other, identity in (("start", "stop", "src"), ("stop", "start", "dst")):
        result[f"primary_{own}"] = _primaries(result, own, other, identity)
    result["amp_start"] = np.where(result.primary_start, result.amp_source, 0.0)
    result["amp_stop"] = np.where(result.primary_stop, result.amp_target, 0.0)
    # Legacy methods give every row mass 1/N, so no weight applies to them.
    result.attrs["morph"] = {"method": method, "parameter": parameter,
                             "weight": weight if method == "transport" else None}
    return result


def _warp(x, c):
    """Curve w(x) = expm1(c x) / expm1(c) and its first two derivatives, stable for any finite c."""
    if c == 0:
        return x, np.ones_like(x), np.zeros_like(x)
    if abs(c) < _SERIES_CURVE:
        # expm1 ratios lose x for subnormal c; first order in c is exact to ~c**2.
        slope = 1 + c * (x - 0.5)
        return x + 0.5 * c * x * (x - 1), slope, np.full_like(x, c)
    if c > 0:
        # e^{c(x-1)} keeps large positive curves from overflowing.
        scale = np.exp(c * (x - 1)) / -np.expm1(-c)
        value = np.exp(c * (x - 1)) * np.expm1(-c * x) / np.expm1(-c)
    else:
        scale = np.exp(c * x) / np.expm1(c)
        value = np.expm1(c * x) / np.expm1(c)
    return value, c * scale, c * c * scale


def _curve_grid(c, count):
    """Breakpoints equidistributing sqrt(w''), closed form; uniform for c = 0."""
    u = np.linspace(0.0, 1.0, count)
    if abs(c) < _SERIES_CURVE:
        return u
    if c > 0:
        with np.errstate(divide="ignore"):
            return 1 + 2 / c * np.log(u + (1 - u) * np.exp(-c / 2))
    with np.errstate(divide="ignore"):
        return 2 / c * np.log1p(u * np.expm1(c / 2))


def _freq_path(x, f0, f1, space, c):
    w, w1, w2 = _warp(x, c)
    if space == "hz" or f0 == f1:
        # f0 == f1 keeps f0 bit for bit (exp(log f0) can be off by a few ulp).
        return f0 + (f1 - f0) * w, (f1 - f0) * w2
    span = np.log(f1) - np.log(f0)
    value = np.exp(np.log(f0) + span * w)
    return value, value * (span * span * w1 * w1 + span * w2)


def _peak_brackets(sample, xs, left_x, right_x, left_v, right_v):
    """Bracket each local maximum of ``sample`` (segment ends count as 0).

    Plateaus keep only their left edge (``>`` left, ``>=`` right) so a flat
    sample does not emit one peak per probe. Ends count as 0, so a negative
    local max against a zero end is not a peak of the signed field.
    """
    n_seg, n_probe = sample.shape
    left = np.concatenate([np.zeros((n_seg, 1)), sample[:, :-1]], axis=1)
    right = np.concatenate([sample[:, 1:], np.zeros((n_seg, 1))], axis=1)
    seg_i, probe_i = np.nonzero((sample > left) & (sample >= right) & (sample > 0))
    if seg_i.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    lo = np.where(probe_i == 0, left_x[seg_i], xs[seg_i, np.maximum(probe_i - 1, 0)])
    hi = np.where(probe_i == n_probe - 1, right_x[seg_i], xs[seg_i, np.minimum(probe_i + 1, n_probe - 1)])
    return lo, hi, left_x[seg_i], right_x[seg_i], left_v[seg_i], right_v[seg_i]


def _refine_brackets(err_at, lo, hi, lx, rx, lv, rv):
    """Vectorized golden section maximizing ``err_at``; every bracket under ``_REFINE_WIDTH``."""
    if lo.size == 0:
        return 0.0
    a = np.asarray(lo, dtype=np.float64).copy()
    b = np.asarray(hi, dtype=np.float64).copy()
    flat = b <= a
    best = np.zeros(a.size, dtype=np.float64)
    if np.any(flat):
        best[flat] = err_at(a[flat], lx[flat], rx[flat], lv[flat], rv[flat])
    active = ~flat
    if not np.any(active):
        return float(np.max(best))
    a, b = a[active], b[active]
    lx, rx, lv, rv = lx[active], rx[active], lv[active], rv[active]
    width = float(np.max(b - a))
    iters = _REFINE_ITERS if width <= 0 else min(
        _REFINE_ITERS, int(np.ceil(np.log(_REFINE_WIDTH / width) / np.log(_GOLDEN))) + 2,
    )
    for _ in range(iters):
        c = b - _GOLDEN * (b - a)
        d = a + _GOLDEN * (b - a)
        fc = err_at(c, lx, rx, lv, rv)
        fd = err_at(d, lx, rx, lv, rv)
        take_left = fc >= fd
        b = np.where(take_left, d, b)
        a = np.where(take_left, a, c)
    mid = 0.5 * (a + b)
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    peak = np.maximum(np.maximum(err_at(c, lx, rx, lv, rv), err_at(d, lx, rx, lv, rv)),
                      err_at(mid, lx, rx, lv, rv))
    best[active] = peak
    return float(np.max(best))


def _refine_abs_extrema(signed_err, signed_sample, xs, left_x, right_x, left_v, right_v):
    """Max |signed error| from refined local maxima of ``e`` and of ``-e``."""
    peak = float(np.max(np.abs(signed_sample))) if signed_sample.size else 0.0
    lo, hi, lx, rx, lv, rv = _peak_brackets(signed_sample, xs, left_x, right_x, left_v, right_v)
    peak = max(peak, abs(_refine_brackets(signed_err, lo, hi, lx, rx, lv, rv)))
    lo, hi, lx, rx, lv, rv = _peak_brackets(-signed_sample, xs, left_x, right_x, left_v, right_v)

    def neg_err(x, lxx, rxx, lvv, rvv):
        return -signed_err(x, lxx, rxx, lvv, rvv)

    peak = max(peak, abs(_refine_brackets(neg_err, lo, hi, lx, rx, lv, rv)))
    return peak


def _log_abs_w2(x, c):
    """``ln|w''(x)|`` in log form; safe for ``|c|`` up to the D1 bound."""
    x = np.asarray(x, dtype=np.float64)
    ac = abs(c)
    if ac < _SERIES_CURVE:
        return np.full(np.shape(x), np.log(ac) if ac > 0 else -np.inf)
    if c > 0:
        return 2.0 * np.log(ac) + c * (x - 1.0) - np.log(-np.expm1(-c))
    return 2.0 * np.log(ac) + c * x - np.log(-np.expm1(c))


def _bisect_root(eval_at, lo, hi, iters=_BISECT_ITERS):
    """Vectorized bisection of a sign change on ``[lo, hi]`` (``eval_at(lo)*eval_at(hi) < 0``)."""
    a = np.asarray(lo, dtype=np.float64).copy()
    b = np.asarray(hi, dtype=np.float64).copy()
    fa = eval_at(a)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        fm = eval_at(mid)
        left = fa * fm <= 0.0
        b = np.where(left, mid, b)
        a = np.where(left, a, mid)
        fa = np.where(left, fa, fm)
    return 0.5 * (a + b)


def _pitch_inflection_roots(xa, xb, la, lb, span, c):
    """Roots of ``g`` inside each segment, shape (n, 2), NaN where absent; ``g < 0`` iff ``e'' > 0``.

    ``g(x) = 2 ln|m| - 2 ln L(x) - ln|span w''(x)|`` with ``L`` the line through
    ``(xa, la)`` and ``(xb, lb)`` (``la, lb > 0``, ``la != lb``). With
    ``span * c < 0`` it is convex: ``ln|w''|`` is linear and ``-2 ln L`` convex.
    ``g' = 0`` at ``L = -2m / c`` splits it into two monotone parts with at most
    one root each: column 0 left of that point, column 1 right of it.
    """
    n = xa.size
    m = (lb - la) / (xb - xa)
    log_m = np.log(np.abs(m))
    log_span = np.log(abs(span))
    mid = xb
    if abs(c) >= _SERIES_CURVE:  # on the series branch ln|w''| is constant: g is monotone
        l_star = -2.0 * m / c
        x_star = xa + (l_star - la) / m
        mid = np.where((l_star > 0.0) & (x_star > xa) & (x_star < xb), x_star, xb)
    seg = np.concatenate([np.arange(n), np.arange(n)])
    lo, hi = np.concatenate([xa, mid]), np.concatenate([mid, xb])

    def g(x, i):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * log_m[i] - 2.0 * np.log(la[i] + m[i] * (x - xa[i])) - log_span - _log_abs_w2(x, c)

    glo, ghi = g(lo, seg), g(hi, seg)
    pick = np.flatnonzero(np.isfinite(glo) & np.isfinite(ghi) & (glo * ghi < 0.0))
    roots = np.full((n, 2), np.nan)
    if pick.size:
        root = _bisect_root(partial(g, i=seg[pick]), lo[pick], hi[pick])
        inside = (root > lo[pick]) & (root < hi[pick])
        roots[seg[pick][inside], pick[inside] // n] = root[inside]
    return roots


def _pitch_opposite_abs_cents(left_x, right_x, left_f, right_f, f0, f1, c):
    """Exact max ``|1200 log2(L/F)|`` on pitch segments with ``span * c < 0``.

    The roots of ``g`` (where ``e''`` changes sign) cut each segment into at most
    three pieces on which ``e'`` is monotone, so ``|e|`` peaks at the root of
    ``e'`` inside a piece or at a piece boundary (``e = 0`` at the segment ends).
    """
    span = np.log(f1) - np.log(f0)
    keep = (right_x > left_x) & (left_f > 0) & (right_f > 0) & (right_f != left_f)
    if span * c >= 0 or not np.any(keep):
        return 0.0
    xa, xb, la, lb = left_x[keep], right_x[keep], left_f[keep], right_f[keep]
    m = (lb - la) / (xb - xa)
    log_f0 = np.log(f0)

    def abs_cents(x, i):
        line = la[i] + m[i] * (x - xa[i])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(1200.0 * np.log2(line / np.exp(log_f0 + span * _warp(x, c)[0])))

    def e_prime(x, i):
        # A per-segment ratio near 1e15 can cancel the line to 0 at xb; the root is dropped as non-finite.
        with np.errstate(divide="ignore", invalid="ignore"):
            return m[i] / (la[i] + m[i] * (x - xa[i])) - span * _warp(x, c)[1]

    roots = _pitch_inflection_roots(xa, xb, la, lb, span, c)
    found = np.isfinite(roots)
    peak = float(np.max(abs_cents(roots[found], np.nonzero(found)[0]), initial=0.0))
    # Pieces [xa, r0], [r0, r1], [r1, xb]; a missing root collapses its piece to a point.
    bounds = np.column_stack([xa, np.where(found[:, 0], roots[:, 0], xa),
                              np.where(found[:, 1], roots[:, 1], xb), xb])
    seg = np.repeat(np.arange(xa.size), 3)
    lo, hi = bounds[:, :3].ravel(), bounds[:, 1:].ravel()
    elo, ehi = e_prime(lo, seg), e_prime(hi, seg)
    pick = np.flatnonzero((hi > lo) & np.isfinite(elo) & np.isfinite(ehi) & (elo * ehi < 0.0))
    if pick.size:
        root = _bisect_root(partial(e_prime, i=seg[pick]), lo[pick], hi[pick])
        peak = max(peak, float(np.max(abs_cents(root, seg[pick]))))
    return peak


def _glide_plan(f0, f1, a0, a1, space, freq_curve, amp_curve, max_cents):
    """(grid, errors, total) of one curved pair, or None when it is straight.

    grid(n) gives n + 1 breakpoint fractions equidistributing the error density;
    errors(points) gives (worst error / tolerance, cents, absolute amp error),
    accepting on the refined absolute maximum of signed-error extrema.
    """
    freq_straight = f0 == f1 or (space == "hz" and freq_curve == 0)
    amp_straight = a0 == a1 or amp_curve == 0
    if freq_straight and amp_straight:
        return None
    relative = 2 ** (max_cents / 1200) - 1
    amp_tol = 1e-3 * max_cents * max(a0, a1)
    grids = [_FINE]
    for straight, c in ((freq_straight, freq_curve), (amp_straight, amp_curve)):
        if not straight and c != 0:
            grids.append(_curve_grid(c, 1025))
    fine = np.unique(np.clip(np.concatenate(grids), 0.0, 1.0))
    density = np.zeros_like(fine)
    if not freq_straight:
        value, second = _freq_path(fine, f0, f1, space, freq_curve)
        density = np.maximum(density, np.abs(second) / (8 * relative * value))
    if not amp_straight:
        second = (a1 - a0) * _warp(fine, amp_curve)[2]
        density = np.maximum(density, np.abs(second) / (8 * amp_tol))
    density = np.sqrt(density)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))])
    total = cumulative[-1]

    def grid(n):
        if n == 1 or total <= 0:
            return np.linspace(0.0, 1.0, n + 1)
        points = np.interp(np.linspace(0.0, total, n + 1), cumulative, fine)
        points[0], points[-1] = 0.0, 1.0
        return np.maximum.accumulate(points)

    def _segment_errors(points, frac):
        """Worst |cents| / |amp| on probes at fractions ``frac`` of every segment."""
        left = np.repeat(np.arange(points.size - 1), frac.size)
        share = np.tile(frac, points.size - 1)
        probes = (points[:-1, None] + np.diff(points)[:, None] * frac).ravel()
        cents = amp = 0.0
        freq_at = amp_at = None
        if not freq_straight:
            freq_at = _freq_path(points, f0, f1, space, freq_curve)[0]
            exact = _freq_path(probes, f0, f1, space, freq_curve)[0]
            line = freq_at[left] + (freq_at[left + 1] - freq_at[left]) * share
            cents = float(np.max(np.abs(1200 * np.log2(line / exact))))
            if cents > max_cents:
                # Already over tolerance; amp cannot un-fail the ratio.
                return cents, amp, freq_at, amp_at
        if not amp_straight:
            amp_at = a0 + (a1 - a0) * _warp(points, amp_curve)[0]
            exact = a0 + (a1 - a0) * _warp(probes, amp_curve)[0]
            line = amp_at[left] + (amp_at[left + 1] - amp_at[left]) * share
            amp = float(np.max(np.abs(line - exact)))
        return cents, amp, freq_at, amp_at

    def _refine(points, freq_at, amp_at):
        if not amp_straight and amp_at is None:
            # At the cap a failing cents check returns before the amp values exist.
            amp_at = a0 + (a1 - a0) * _warp(points, amp_curve)[0]
        spans = np.diff(points)
        left_x, right_x = points[:-1], points[1:]
        xs = left_x[:, None] + spans[:, None] * _PROBES
        cents = amp = 0.0
        if not freq_straight:
            left_f, right_f = freq_at[:-1], freq_at[1:]
            line = left_f[:, None] + (right_f - left_f)[:, None] * _PROBES
            exact = _freq_path(xs.ravel(), f0, f1, space, freq_curve)[0].reshape(xs.shape)
            signed = 1200 * np.log2(line / exact)

            def freq_signed(x, lxx, rxx, lvv, rvv):
                exact = _freq_path(x, f0, f1, space, freq_curve)[0]
                return 1200 * np.log2((lvv + (rvv - lvv) * (x - lxx) / (rxx - lxx)) / exact)

            # Bracket maxima of e and of -e, refine each, take the absolute max.
            cents = _refine_abs_extrema(freq_signed, signed, xs, left_x, right_x, left_f, right_f)
            # Pitch + opposite-sign curvature: e can have up to 3 interior extrema; a
            # hump between the last probe and the end needs the g-root / e' path.
            if space == "pitch" and f0 != f1 and (np.log(f1) - np.log(f0)) * freq_curve < 0:
                cents = max(
                    cents,
                    _pitch_opposite_abs_cents(left_x, right_x, left_f, right_f, f0, f1, freq_curve),
                )
        if not amp_straight:
            left_a, right_a = amp_at[:-1], amp_at[1:]
            line = left_a[:, None] + (right_a - left_a)[:, None] * _PROBES
            exact = (a0 + (a1 - a0) * _warp(xs.ravel(), amp_curve)[0]).reshape(xs.shape)
            signed = line - exact

            def amp_signed(x, lxx, rxx, lvv, rvv):
                exact = a0 + (a1 - a0) * _warp(x, amp_curve)[0]
                return lvv + (rvv - lvv) * (x - lxx) / (rxx - lxx) - exact

            amp = _refine_abs_extrema(amp_signed, signed, xs, left_x, right_x, left_a, right_a)
        return cents, amp

    def errors(points, *, refine=False):
        # Mid / 3-probe / 31-probe reject (refined max cannot undershoot a sample).
        cents = amp = 0.0
        freq_at = amp_at = None
        for frac in (_MID_PROBE, _CHEAP_PROBES, _PROBES):
            cents, amp, freq_at, amp_at = _segment_errors(points, frac)
            ratio = max(cents / max_cents, amp / amp_tol if not amp_straight else 0.0)
            if ratio > 1 and not refine:
                return ratio, cents, amp
        # Refine when the cheap check passed, or always at the cap (for the warning).
        cents, amp = _refine(points, freq_at, amp_at)
        return max(cents / max_cents, amp / amp_tol if not amp_straight else 0.0), cents, amp

    return grid, errors, total


def _glide_breakpoints(f0, f1, a0, a1, space, freq_curve, amp_curve, max_cents):
    """Breakpoint fractions of one pair, and its errors (cents, re peak) if the cap was hit."""
    plan = _glide_plan(f0, f1, a0, a1, space, freq_curve, amp_curve, max_cents)
    if plan is None:
        return np.array([0.0, 1.0]), None
    grid, errors, _total = plan
    # Fewest segments: smallest m whose refined error passes (exact by upward scan).
    cents = amp = 0.0
    for n in range(1, _MAX_SEGMENTS + 1):
        ratio, cents, amp = errors(grid(n), refine=(n == _MAX_SEGMENTS))
        if ratio <= 1:
            return grid(n), None
    # amp is 0 when only the frequency is curved (the pair may be silent).
    return grid(_MAX_SEGMENTS), (cents, amp / max(a0, a1) if amp else 0.0)


def _segment_rows(table, time_start, duration, space, freq_curve, amp_curve, max_cents):
    """Split each row into linear rows on its curve; straight rows stay whole."""
    columns: dict[str, list[np.ndarray]] = {name: [] for name in
                                            ("freq_start", "freq_stop", "amp_start", "amp_stop", "time_start", "time_stop")}
    capped = []
    for f0, f1, a0, a1 in table[["freq_start", "freq_stop", "amp_start", "amp_stop"]].itertuples(index=False):
        points, worst = _glide_breakpoints(f0, f1, a0, a1, space, freq_curve, amp_curve, max_cents)
        if worst is not None:
            capped.append(worst)
        freq = _freq_path(points, f0, f1, space, freq_curve)[0]
        amp = a0 + (a1 - a0) * _warp(points, amp_curve)[0]
        times = time_start + duration * points
        # Ends are the inputs bit for bit, so chained morphs meet exactly.
        freq[0], freq[-1], amp[0], amp[-1] = f0, f1, a0, a1
        times[0], times[-1] = time_start, time_start + duration
        for name, values in (("freq", freq), ("amp", amp), ("time", times)):
            columns[f"{name}_start"].append(values[:-1])
            columns[f"{name}_stop"].append(values[1:])
    if capped:
        cents = max(item[0] for item in capped)
        amp = max(item[1] for item in capped)
        amp_db = 20 * np.log10(amp) if amp > 0 else -np.inf
        warnings.warn(
            f"{len(capped)} pair(s) exceed max_cents={max_cents:g} at {_MAX_SEGMENTS} segments; "
            f"worst error {cents:.3g} c, {amp_db:.1f} dB re the pair's peak",
            UserWarning, stacklevel=3,
        )
    return pd.DataFrame({name: np.concatenate(parts) for name, parts in columns.items()})


def _emission_order(table):
    # Pairing edges connect source and target partial vertices. Legacy maps are
    # disjoint stars; a monotone staircase adds an edge only when at least one
    # endpoint is new (both on a degenerate step). Thus this graph is a forest.
    # Primary-before-peer constraints cannot cycle: a shortest directed cycle
    # using different groups would give an undirected cycle in that forest;
    # within one group all arrows leave its single primary. Kahn's heap selects
    # the earliest available pairing row, giving a stable topological order.
    successors: list[set[int]] = [set() for _ in range(len(table))]
    degree = np.zeros(len(table), dtype=np.int64)
    for identity, flag in (("src", "primary_start"), ("dst", "primary_stop")):
        for rows in table.groupby(identity, sort=False).indices.values():
            primary = rows[table.iloc[rows][flag].to_numpy()][0]
            for row in rows:
                if row != primary and row not in successors[primary]:
                    successors[primary].add(row)
                    degree[row] += 1
    available = np.flatnonzero(degree == 0).tolist()
    heapq.heapify(available)
    order = []
    while available:
        row = heapq.heappop(available)
        order.append(row)
        for successor in successors[row]:
            degree[successor] -= 1
            if degree[successor] == 0:
                heapq.heappush(available, successor)
    if len(order) != len(table):
        raise RuntimeError("pairing groups are cyclic; no primary-first emission order exists")
    return order


def morph(
    source: pd.DataFrame,
    target: pd.DataFrame,
    duration: float,
    method: str = "equal",
    *,
    time_start: float = 0.0,
    power: float | None = None,
    sigmoid_range: float | None = None,
    weight: str = "amp",
    duplicates: str | None = None,
    freq_space: str = "hz",
    freq_curve: float = 0.0,
    amp_curve: float = 0.0,
    max_cents: float = 1.0,
) -> pd.DataFrame:
    """Emit a morph as v1 events, with primaries first for continuous linking.

    Parameters
    ----------
    source, target : pandas.DataFrame
        Nonempty v1 snapshots with positive frequencies and D4 peak amplitudes.
    duration : float
        Finite, strictly positive morph length in seconds.
    method : {'equal', 'power', 'sigmoid', 'transport'}
        Pairing strategy; see pair for parameter and mass semantics.
    time_start : float, default 0
        Finite, nonnegative onset in seconds.
    power, sigmoid_range : float or None
        Method-specific warping parameter; None uses the mean covering range.
    weight : {'amp', 'power'}
        Transport mass choice; only transport accepts a nondefault weight.
    duplicates : {'full', 'primary'} or None, default None
        Gain of a partial that several rows share at one end. ``'full'``: every
        row carries its partials' full amplitudes at both ends, as in 0.6, and
        every pairing row is emitted. A partial used by k rows sounds k times
        at that end, so that end is louder than its snapshot: copies leaving
        one partial start in phase (k times its amplitude), copies arriving
        at one partial have different phase histories, so they add about k
        times its power and beat slowly against each other. Measured with
        'equal' over 15 s: speech1 (41 peaks) → conga3 (10) ends +5.7 dB
        above the static conga3; conga3 → speech1 starts +11.2 dB above the
        static conga3, the same levels as 0.6. ``'primary'``: only the
        group's primary row carries the amplitude there; the other rows start
        or end at 0, so both ends match the snapshots' levels, and rows primary
        at neither end are dropped. None means ``'full'`` for the 0.6 methods
        ('equal', 'power', 'sigmoid') and ``'primary'`` for 'transport'.
    freq_space : {'hz', 'pitch'}, default 'hz'
        Space in which ``freq_curve`` shapes each glide: ``'hz'`` moves
        f0 + (f1 - f0) w(x), ``'pitch'`` moves log frequency, f0 (f1/f0)^w(x),
        so ``freq_curve=0`` there glides at constant cents per second.
    freq_curve, amp_curve : float, default 0
        Curve c of w(x) = expm1(c x) / expm1(c), x the glide's time fraction
        (the Csound ``transeg`` / SuperCollider ``Env`` float curve). 0 is
        linear; c > 0 moves late (slow start, fast arrival), c < 0 early, for
        rising and falling glides alike. ``amp_curve`` acts on linear
        amplitude, so births and deaths still reach 0. Absolute value must not
        exceed 1e4.
    max_cents : float, default 1
        Tolerance of curved glides, > 0: freq error in cents and amp error
        1e-3 * max_cents re the pair's peak (-60 dB at 1 c, -54 dB at 2 c).

    Returns
    -------
    pandas.DataFrame
        Validated events with attrs['morph'] copied from pair, plus the
        resolved ``duplicates``, ``freq_space``, ``freq_curve``, ``amp_curve``
        and ``max_cents``. Stable topological pairing order puts every
        group's primary before its other rows. Amplitudes are not normalized.
        A straight pair (linear in Hz and amp, as by default) is one row. A
        curved pair becomes consecutive linear rows whose breakpoints lie on
        the curve: the fewest (at most 256) within ``max_cents`` on a grid that
        is denser where the curve bends. Its first row starts and its last row
        stops exactly at the pair's values; interior joins share one value, so
        ``synthesize`` links them in ``phase='continuous'`` (including across
        sub-sample rows that cover no sample).

    Warns
    -----
    UserWarning
        Some curved pair still exceeds ``max_cents`` at 256 segments.

    Raises
    ------
    SchemaError
        Invalid input snapshots or resulting events.
    ValueError
        Invalid duration/onset/duplicates/curve options, or any invalid
        pairing argument (see pair).

    Notes
    -----
    Render curved morphs with ``phase='continuous'`` (the default):
    ``'reset'`` and ``'absolute'`` restart the phase at every join and click.
    ``events_to_mei`` notates every segment as a note.
    """
    duration = _finite_scalar(duration, "duration")
    time_start = _finite_scalar(time_start, "time_start")
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if time_start < 0:
        raise ValueError("time_start must be >= 0")
    if duplicates is not None and (not isinstance(duplicates, str) or duplicates not in ("full", "primary")):
        raise ValueError(f"duplicates must be 'full', 'primary' or None; got {duplicates!r}")
    if not isinstance(freq_space, str) or freq_space not in FREQ_SPACES:
        raise ValueError(f"freq_space must be one of {FREQ_SPACES}; got {freq_space!r}")
    freq_curve = _finite_scalar(freq_curve, "freq_curve")
    amp_curve = _finite_scalar(amp_curve, "amp_curve")
    if abs(freq_curve) > _CURVE_BOUND:
        raise ValueError(f"freq_curve must satisfy |freq_curve| <= {_CURVE_BOUND:g}; got {freq_curve!r}")
    if abs(amp_curve) > _CURVE_BOUND:
        raise ValueError(f"amp_curve must satisfy |amp_curve| <= {_CURVE_BOUND:g}; got {amp_curve!r}")
    max_cents = _finite_scalar(max_cents, "max_cents")
    if max_cents <= 0:
        raise ValueError("max_cents must be > 0")
    pairing = pair(source, target, method, power=power, sigmoid_range=sigmoid_range, weight=weight)
    if duplicates is None:
        duplicates = "primary" if method == "transport" else "full"
    if duplicates == "full":
        sounding = pairing.assign(amp_start=pairing.amp_source, amp_stop=pairing.amp_target)
    else:
        sounding = pairing.loc[pairing.primary_start | pairing.primary_stop].reset_index(drop=True)
    sounding = sounding.iloc[_emission_order(sounding)]
    result = sounding[["freq_start", "freq_stop", "amp_start", "amp_stop"]].copy()
    if freq_space == "hz" and freq_curve == 0 and amp_curve == 0:
        # Straight glides keep the one-row-per-pair table exactly as before.
        result["time_start"] = time_start
        result["time_stop"] = time_start + duration
    else:
        result = _segment_rows(result, time_start, duration, freq_space, freq_curve, amp_curve, max_cents)
    result.attrs["morph"] = {**pairing.attrs["morph"], "duplicates": duplicates, "freq_space": freq_space,
                             "freq_curve": freq_curve, "amp_curve": amp_curve, "max_cents": max_cents}
    return validate(result, "events")
