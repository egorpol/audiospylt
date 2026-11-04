"""Tests for fidelity tracked rows, ORDER, match_rms, silent-render scoring and listening."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FIDELITY = ROOT / "benchmarks" / "fidelity"
EVALUATE = FIDELITY / "evaluate.py"
RUN_ASP = FIDELITY / "run_audiospylt.py"


def _load_fidelity(path: Path, name: str):
    """Load a fidelity module by path; register before exec; restore path and modules after."""
    clash = ("evaluate", "excerpts", "points", "run_audiospylt", name)
    saved_path = list(sys.path)
    saved = {n: sys.modules.get(n) for n in clash}
    for n in clash:
        sys.modules.pop(n, None)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = saved_path
        for n in clash:
            sys.modules.pop(n, None)
        for n, prev in saved.items():
            if prev is not None:
                sys.modules[n] = prev


@pytest.fixture
def evaluate_mod():
    yield from _load_fidelity(EVALUATE, "fidelity_evaluate_tracked")


@pytest.fixture
def run_asp_mod():
    yield from _load_fidelity(RUN_ASP, "fidelity_run_audiospylt")


def test_tracked_rows_calls_and_meta(run_asp_mod, tmp_path, monkeypatch):
    mod = run_asp_mod
    excerpts = sys.modules["excerpts"]
    t = np.arange(2048) / 44100
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    calls: list[tuple] = []
    metas: list[tuple] = []
    renders: list[tuple] = []

    def fake_load(e):
        return signal.copy()

    def fake_write_render(out, row, e, y):
        renders.append((row, e.name, e.sr, np.asarray(y)))

    def fake_write_meta(out, row, **meta):
        metas.append((row, meta))

    @functools.wraps(mod.asp.track_partials)
    def spy_track(x, sr, **kwargs):
        calls.append((sr, kwargs))
        return pd.DataFrame({
            "track_id": [0, 0],
            "time": [0.0, 0.05],
            "freq": [440.0, 440.0],
            "amp": [0.1, 0.1],
        })

    monkeypatch.setattr(excerpts, "load", fake_load)
    monkeypatch.setattr(excerpts, "write_render", fake_write_render)
    monkeypatch.setattr(excerpts, "write_meta", fake_write_meta)
    monkeypatch.setattr(mod.asp, "track_partials", spy_track)

    chosen = [
        SimpleNamespace(name="a441", set="case", sr=44100),
        SimpleNamespace(name="a220", set="case", sr=22050),
    ]
    mod.tracked_rows(tmp_path, chosen)

    expected = [
        ("audiospylt-v1/tracked", {}),
        ("audiospylt-v1/tracked-noprom", {"prominence_db": None}),
        ("audiospylt-v1/tracked-wide", {"prominence_db": None, "max_cents": 400.0}),
    ]
    assert [m[0] for m in metas] == [r for r, _ in expected]
    assert len(calls) == 6  # 3 rows × 2 rates

    for i, (row, overrides) in enumerate(expected):
        for j, sr in enumerate((44100, 22050)):
            call_sr, kwargs = calls[i * 2 + j]
            assert call_sr == sr
            frame, hop = mod.frame_hop(mod.MATCHED, sr)
            assert kwargs["frame_length"] == frame
            assert kwargs["hop"] == hop
            for key, value in overrides.items():
                assert kwargs[key] == value
            # no unexpected override keys beyond frame/hop and the row's own
            assert set(kwargs) - {"frame_length", "hop"} == set(overrides)

        meta = metas[i][1]
        params = meta["params"]
        assert params["min_duration"] == 0.1
        if row == "audiospylt-v1/tracked":
            assert params["prominence_db"] == 20.0
            assert params["max_cents"] == 50.0
        elif row == "audiospylt-v1/tracked-noprom":
            assert params["prominence_db"] is None
            assert params["max_cents"] == 50.0
        else:
            assert params["prominence_db"] is None
            assert params["max_cents"] == 400.0

        for name, sr in (("a441", 44100), ("a220", 22050)):
            done = meta["excerpts"][name]
            frame, hop = mod.frame_hop(mod.MATCHED, sr)
            assert done["frame"] == frame
            assert done["hop"] == hop
            assert done["params"]["prominence_db"] == params["prominence_db"]
            assert done["params"]["max_cents"] == params["max_cents"]
            assert done["params"]["min_duration"] == 0.1


def test_tracked_keys_in_order(evaluate_mod, run_asp_mod):
    tracked = list(run_asp_mod.TRACKED)
    order = evaluate_mod.ORDER
    assert all(k in order for k in tracked)
    i = order.index("audiospylt-v1/matched")
    assert order[i + 1: i + 1 + len(tracked)] == tracked


def test_match_rms(evaluate_mod):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(1000)
    original = y.copy()
    out = evaluate_mod.match_rms(y, 0.25)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(0.25)
    np.testing.assert_array_equal(y, original)

    quiet = np.full(1000, 1e-6)
    original_q = quiet.copy()
    out_q = evaluate_mod.match_rms(quiet, 0.1)
    assert np.sqrt(np.mean(out_q ** 2)) == pytest.approx(0.1)
    np.testing.assert_array_equal(quiet, original_q)

    zeros = np.zeros(1000)
    original_z = zeros.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_z = evaluate_mod.match_rms(zeros, 0.5)
    assert caught == []
    assert np.all(out_z == 0)
    assert np.all(np.isfinite(out_z))
    np.testing.assert_array_equal(zeros, original_z)


def test_score_silent_render_no_warning(evaluate_mod):
    sr = 44100
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(sr)
    silent = np.zeros_like(ref)
    power = float(np.mean(ref ** 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        _ = 10 * np.log10(0.0 / power)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        sc = evaluate_mod.score(ref, silent, sr)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert sc["gain"] == -np.inf
    assert np.isfinite(sc["bin"])
    assert np.isfinite(sc["band"])


def test_write_listening_silent_and_normal(evaluate_mod, tmp_path):
    sr = 44100
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(sr)
    ref /= np.sqrt(np.mean(ref ** 2))  # RMS 1
    normal = 0.3 * rng.standard_normal(sr)
    silent = np.zeros(sr)

    e = SimpleNamespace(name="clip", set="case", sr=sr)
    # ORDER sorts matched before tracked; silent is the tracked render (the case the fix exists for)
    silent_row, normal_row = "audiospylt-v1/tracked", "audiospylt-v1/matched"
    for row, y in ((silent_row, silent), (normal_row, normal)):
        folder = tmp_path / "raw" / row
        folder.mkdir(parents=True)
        sf.write(folder / "clip.wav", y.astype(np.float32), sr, subtype="FLOAT")
        (folder / "meta.json").write_text(json.dumps({"excerpts": {"clip": {"set": "case", "sr": sr}}}))

    evaluate_mod.write_listening(tmp_path, [e], [silent_row, normal_row], {"clip": ref})

    listen = tmp_path / "listening" / "clip"
    files = sorted(listen.glob("*.wav"))
    assert [p.name for p in files] == [
        "00_original.wav",
        "01_audiospylt-v1_matched.wav",
        "02_audiospylt-v1_tracked.wav",
    ]
    original, normal_wav, silent_wav = (sf.read(p)[0] for p in files)
    for w in (original, normal_wav, silent_wav):
        assert np.all(np.isfinite(w))
    assert np.all(silent_wav == 0)

    target = float(np.sqrt(np.mean(ref ** 2)))
    peak = max(np.abs(c).max() for c in (ref, np.zeros_like(ref), evaluate_mod.match_rms(normal, target)))
    scale = min(1.0, 10 ** (evaluate_mod.LISTEN_PEAK_DB / 20) / peak)
    assert np.sqrt(np.mean(normal_wav ** 2)) == pytest.approx(target * scale, rel=1e-5)
    # silent clip must not change the shared scale (same as without it)
    peak_without = max(np.abs(c).max() for c in (ref, evaluate_mod.match_rms(normal, target)))
    assert peak == pytest.approx(peak_without)
