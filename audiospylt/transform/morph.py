"""Legacy-compatible and transport pairing; morph gains of shared partials (0.6 full or primary)."""

from __future__ import annotations

import heapq
import warnings

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
_MASS_TOL = 1e-12


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

    Returns
    -------
    pandas.DataFrame
        Validated events with attrs['morph'] copied from pair, plus the
        resolved ``duplicates``. Stable topological pairing order puts every
        group's primary before its other rows. Amplitudes are not normalized.

    Raises
    ------
    SchemaError
        Invalid input snapshots or resulting events.
    ValueError
        Invalid duration/onset/duplicates, or any invalid pairing argument
        (see pair).
    """
    duration = _finite_scalar(duration, "duration")
    time_start = _finite_scalar(time_start, "time_start")
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if time_start < 0:
        raise ValueError("time_start must be >= 0")
    if duplicates is not None and (not isinstance(duplicates, str) or duplicates not in ("full", "primary")):
        raise ValueError(f"duplicates must be 'full', 'primary' or None; got {duplicates!r}")
    pairing = pair(source, target, method, power=power, sigmoid_range=sigmoid_range, weight=weight)
    if duplicates is None:
        duplicates = "primary" if method == "transport" else "full"
    if duplicates == "full":
        sounding = pairing.assign(amp_start=pairing.amp_source, amp_stop=pairing.amp_target)
    else:
        sounding = pairing.loc[pairing.primary_start | pairing.primary_stop].reset_index(drop=True)
    sounding = sounding.iloc[_emission_order(sounding)]
    result = sounding[["freq_start", "freq_stop", "amp_start", "amp_stop"]].copy()
    result["time_start"] = time_start
    result["time_stop"] = time_start + duration
    result.attrs["morph"] = {**pairing.attrs["morph"], "duplicates": duplicates}
    return validate(result, "events")
