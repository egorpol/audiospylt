"""Excess/deficit split of the fidelity LSD (D30)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATE = ROOT / "benchmarks" / "fidelity" / "evaluate.py"
SR = 44100


@pytest.fixture
def evaluate_mod():
    """Load benchmarks/fidelity/evaluate.py by path; restore sys.path and sys.modules after."""
    saved_path = list(sys.path)
    saved = {name: sys.modules.get(name) for name in ("evaluate", "excerpts", "points")}
    for name in ("evaluate", "excerpts", "points"):
        sys.modules.pop(name, None)
    try:
        spec = importlib.util.spec_from_file_location("fidelity_evaluate", EVALUATE)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fidelity_evaluate"] = mod
        # excerpts defines a dataclass; register evaluate under a unique name before exec.
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = saved_path
        for name in ("evaluate", "excerpts", "points", "fidelity_evaluate"):
            sys.modules.pop(name, None)
        for name, prev in saved.items():
            if prev is not None:
                sys.modules[name] = prev


def _lsd_unmodified(ref, y, sr, evaluate_mod):
    """Copy of the D26 LSD formulas (no excess/deficit)."""
    y = y[: ref.size] if y.size >= ref.size else np.pad(y, (0, ref.size - y.size))
    f, sx = evaluate_mod.magnitudes(ref, sr)
    _, sy = evaluate_mod.magnitudes(y, sr)
    floor = sx.max() * evaluate_mod.BIN_FLOOR
    d_bin = (20 * np.log10(np.maximum(sx, floor)) - 20 * np.log10(np.maximum(sy, floor))) ** 2
    edges = evaluate_mod.band_edges(sr)
    sel = [(f >= lo) & (f < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    px = np.array([(sx[s] ** 2).sum(axis=0) for s in sel])
    py = np.array([(sy[s] ** 2).sum(axis=0) for s in sel])
    bfloor = px.max() * 1e-8
    d_band = (10 * np.log10(np.maximum(px, bfloor)) - 10 * np.log10(np.maximum(py, bfloor))) ** 2
    return float(np.sqrt(d_bin.mean())), float(np.sqrt(d_band.mean()))


def test_score_excess_deficit_on_white_noise(evaluate_mod):
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(SR)
    sc = evaluate_mod.score(ref, ref, SR)
    for key in ("bin_excess", "bin_deficit", "band_excess", "band_deficit"):
        assert sc[key] == pytest.approx(0.0, abs=1e-12)

    loud = evaluate_mod.score(ref, 2 * ref, SR)
    assert loud["bin_deficit"] == pytest.approx(0.0, abs=1e-12)
    assert loud["band_deficit"] == pytest.approx(0.0, abs=1e-12)
    assert loud["band_excess"] == pytest.approx(6.02, abs=0.05)

    quiet = evaluate_mod.score(ref, ref / 2, SR)
    assert quiet["bin_excess"] == pytest.approx(0.0, abs=1e-12)
    assert quiet["band_excess"] == pytest.approx(0.0, abs=1e-12)
    assert quiet["band_deficit"] == pytest.approx(6.02, abs=0.05)

    other = rng.standard_normal(SR)
    sc = evaluate_mod.score(ref, other, SR)
    assert sc["bin"] ** 2 == pytest.approx(sc["bin_excess"] ** 2 + sc["bin_deficit"] ** 2, rel=1e-9)
    assert sc["band"] ** 2 == pytest.approx(sc["band_excess"] ** 2 + sc["band_deficit"] ** 2, rel=1e-9)

    bin_lsd, band_lsd = _lsd_unmodified(ref, other, SR, evaluate_mod)
    assert sc["bin"] == pytest.approx(bin_lsd, rel=0, abs=0)
    assert sc["band"] == pytest.approx(band_lsd, rel=0, abs=0)
