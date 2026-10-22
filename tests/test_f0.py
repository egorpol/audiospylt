"""Temporal pYIN f0 with its harmonic refit, snapshot harmonic f0, and f0 tracks (D20)."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pytest

from audiospylt.analysis.f0 import F0Contour, HarmonicF0, harmonic_f0, pyin
from audiospylt.analysis.peaks import find_peaks
from audiospylt.analysis.spectrogram import spectrogram
from audiospylt.analysis.spectrum import spectrum
from audiospylt.io.audio import load_audio, trim
from audiospylt.io.tables import read_table, write_table
from audiospylt.model.convert import f0_to_tracks
from audiospylt.model.schema import SchemaError, validate


SAMPLES = Path(__file__).resolve().parents[1] / "samples"
SR = 44100
C2, C7 = 65.41, 2093.0

pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def cents(a, b):
    return 1200 * np.log2(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def harmonics_tone(f0, harmonics, amps, seconds=1.0):
    t = np.arange(int(seconds * SR)) / SR
    return sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in zip(harmonics, amps))


def inner(contour, start=0.05, stop=0.95):
    return (contour.times >= start) & (contour.times <= stop)


@pytest.fixture(scope="module")
def cello():
    return load_audio(SAMPLES / "cello_arco_d_sul_c_01.wav", mono=True)


@pytest.fixture(scope="module")
def cello_strong_peaks(cello):
    x, sr = cello
    return find_peaks(spectrum(trim(x, sr, 1.5, 2.5), sr), min_db=None, prominence_db=10, rel_db=-40)


# pyin: accuracy of the refit against pYIN's 10-cent grid (revision/step-2.4/probe_refine.out)

@pytest.mark.parametrize("label,audio,truth,max_cents", [
    ("sine 73.8 Hz", harmonics_tone(73.8, [1], [0.5]), 73.8, 1.5),
    ("sine 223.3 Hz", harmonics_tone(223.3, [1], [0.5]), 223.3, 0.05),
    ("sawtooth 73.8 Hz", harmonics_tone(73.8, range(1, 30), 0.3 / np.arange(1, 30)), 73.8, 0.5),
    ("missing fundamental 110 Hz", harmonics_tone(110.0, range(2, 7), [0.2] * 5), 110.0, 0.3),
], ids=lambda value: value if isinstance(value, str) else "")
def test_refit_accuracy_on_stationary_tones(label, audio, truth, max_cents):
    contour = pyin(audio, SR, fmin=C2, fmax=C7)
    frames = inner(contour)
    assert contour.voiced[frames].all() and contour.refined[frames].all()
    assert np.max(np.abs(cents(contour.f0[frames], truth))) < max_cents
    # pYIN itself stays on its 0.1-semitone grid above fmin.
    steps = cents(contour.f0_pyin[contour.voiced], C2) / 10
    np.testing.assert_allclose(steps, np.rint(steps), atol=1e-6)


def test_pyin_low_f0_bias_is_what_the_refit_removes():
    contour = pyin(harmonics_tone(73.8, [1], [0.5]), SR, fmin=C2, fmax=C7)
    frames = inner(contour)
    assert np.median(cents(contour.f0_pyin[frames], 73.8)) > 15
    assert abs(np.median(cents(contour.f0[frames], 73.8))) < 0.1


def test_refit_follows_vibrato():
    t = np.arange(2 * SR) / SR

    def instantaneous(times):
        return 220 * 2 ** (50 / 1200 * np.sin(2 * np.pi * 5.5 * times))

    phase = 2 * np.pi * np.cumsum(instantaneous(t)) / SR
    audio = 0.3 * sum(np.sin(k * phase) / k for k in range(1, 15))
    contour = pyin(audio, SR, fmin=C2, fmax=C7)
    frames = inner(contour, 0.05, 1.95)
    error = np.abs(cents(contour.f0[frames], instantaneous(contour.times[frames])))
    assert np.percentile(error, 95) < 2.0


def test_fundamental_amplitude_is_d4_and_zero_when_missing():
    sine = pyin(harmonics_tone(223.3, [1], [0.5]), SR, fmin=C2, fmax=C7)
    assert np.max(np.abs(20 * np.log10(sine.amp[inner(sine)] / 0.5))) < 0.1
    missing = pyin(harmonics_tone(110.0, range(2, 7), [0.2] * 5), SR, fmin=C2, fmax=C7)
    assert np.all(missing.amp[inner(missing)] == 0)


def test_refine_false_keeps_pyin_values_and_still_reports_amplitude():
    audio = harmonics_tone(223.3, [1], [0.5])
    raw = pyin(audio, SR, fmin=C2, fmax=C7, refine=False)
    refined = pyin(audio, SR, fmin=C2, fmax=C7)
    np.testing.assert_array_equal(raw.f0, raw.f0_pyin)
    np.testing.assert_array_equal(raw.f0_pyin, refined.f0_pyin)
    assert not raw.refined.any()
    assert np.max(np.abs(20 * np.log10(raw.amp[inner(raw)] / 0.5))) < 0.1


@pytest.mark.parametrize("sr,frame_length,hop,seconds", [(SR, 2048, 512, 0.5), (8000, 1001, 300, 0.5)])
def test_frames_share_the_spectrogram_grid(sr, frame_length, hop, seconds):
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)
    contour = pyin(audio, sr, fmin=100, fmax=2000, frame_length=frame_length, hop=hop)
    grid = spectrogram(audio, sr, frame_length=frame_length, hop=hop)
    np.testing.assert_array_equal(contour.times, grid.times)
    shapes = {getattr(contour, name).shape for name in
              ("f0", "voiced", "voiced_prob", "f0_pyin", "refined", "amp")}
    assert shapes == {grid.times.shape}
    assert contour.voiced.dtype == bool and contour.refined.dtype == bool
    assert (contour.sr, contour.frame_length, contour.hop, contour.fmin, contour.fmax) == (
        sr, frame_length, hop, 100.0, 2000.0)


def test_silence_is_unvoiced_everywhere():
    contour = pyin(np.zeros(SR // 2), SR, fmin=C2, fmax=C7)
    assert not contour.voiced.any() and not contour.refined.any()
    assert np.isnan(contour.f0).all() and np.isnan(contour.f0_pyin).all() and np.isnan(contour.amp).all()
    tracks = f0_to_tracks(contour)
    assert tracks.empty
    assert list(tracks.columns) == ["track_id", "time", "freq", "amp", "voiced_prob", "refined"]


def test_cello_contour_is_steady_d2(cello):
    x, sr = cello
    before = x.copy()
    contour = pyin(x, sr, fmin=C2, fmax=C7)
    np.testing.assert_array_equal(x, before)
    steady = contour.voiced & (contour.times >= 1) & (contour.times <= 7)
    assert steady.sum() > 0.95 * ((contour.times >= 1) & (contour.times <= 7)).sum()
    assert contour.refined[contour.voiced].all()
    median = np.median(contour.f0[steady])
    assert median == pytest.approx(74.3, abs=0.1)  # D2 + 21 cents (pYIN alone: 74.27 on its grid)
    assert np.std(cents(contour.f0[steady], median)) < 4.5
    with pytest.raises(FrozenInstanceError):
        contour.f0 = contour.f0_pyin  # type: ignore[misc]


@pytest.mark.parametrize("kwargs,match", [
    ({"fmin": 0, "fmax": 500}, "0 < fmin < fmax"),
    ({"fmin": 500, "fmax": 500}, "0 < fmin < fmax"),
    ({"fmin": 100, "fmax": 30000}, r"sr/2 = 22050"),
    ({"fmin": float("nan"), "fmax": 500}, "fmin must be finite"),
    ({"fmin": 100, "fmax": 500, "refine": "yes"}, "refine must be True or False"),
    ({"fmin": 100, "fmax": 500, "frame_length": 2}, "frame_length must be an integer >= 3"),
    ({"fmin": 100, "fmax": 500, "hop": 0}, "hop must be an integer >= 1"),
])
def test_pyin_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pyin(np.zeros(4096), SR, **kwargs)


def test_pyin_rejects_multichannel_audio():
    with pytest.raises(ValueError, match="to_mono"):
        pyin(np.zeros((2, 4096)), SR, fmin=100, fmax=500)


def test_pyin_without_librosa_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "librosa", None)
    message = "pyin f0 tracking requires librosa; install audiospylt[librosa]"
    with pytest.raises(ImportError, match=f"^{re.escape(message)}$"):
        pyin(np.zeros(4096), SR, fmin=100, fmax=500)


# f0_to_tracks

def contour(voiced):
    voiced = np.asarray(voiced, dtype=bool)
    count = len(voiced)
    f0 = np.where(voiced, 100.0 + np.arange(count), np.nan)
    amp = np.where(voiced, 0.1 * np.arange(count), np.nan)
    return F0Contour(
        np.arange(count) * 0.01, f0, voiced, np.linspace(0, 1, count), f0.copy(),
        voiced & (np.arange(count) % 2 == 0), amp, 44100, 2048, 441, 50.0, 1000.0,
    )


def test_voiced_runs_become_tracks_with_extras_and_meta():
    tracks = f0_to_tracks(contour([False, True, True, False, True, False, False, True]))
    assert tracks.attrs["audiospylt_table"] == "tracks"
    assert tracks["track_id"].tolist() == [0, 0, 1, 2]
    np.testing.assert_allclose(tracks["time"], [0.01, 0.02, 0.04, 0.07])
    np.testing.assert_allclose(tracks["freq"], [101, 102, 104, 107])
    np.testing.assert_allclose(tracks["amp"], [0.1, 0.2, 0.4, 0.7])
    assert tracks["refined"].tolist() == [False, True, True, False]
    np.testing.assert_allclose(tracks["voiced_prob"], np.linspace(0, 1, 8)[[1, 2, 4, 7]])
    assert tracks.attrs["meta"] == {"method": "pyin", "sr": 44100, "frame_length": 2048, "hop": 441,
                                    "fmin": 50.0, "fmax": 1000.0}
    validate(tracks, "tracks")


def test_tracks_round_trip_through_tsv(tmp_path):
    tracks = f0_to_tracks(contour([True, True, False, True]))
    write_table(tracks, tmp_path / "f0.tsv")
    loaded = read_table(tmp_path / "f0.tsv")
    pd.testing.assert_frame_equal(loaded, tracks, check_dtype=False)
    assert loaded.attrs["meta"]["method"] == "pyin"


def test_missing_fundamental_frames_keep_zero_amplitude():
    base = contour([True, True, True])
    tracks = f0_to_tracks(replace(base, amp=np.array([0.0, 0.5, 0.0])))
    assert tracks["amp"].tolist() == [0.0, 0.5, 0.0]


# harmonic_f0 (replaces 0.6 search_f0)

def snapshot(freq, amp=None):
    freq = np.asarray(freq, dtype=float)
    return validate(pd.DataFrame({"freq": freq, "amp": np.ones_like(freq) if amp is None else amp}),
                    "snapshot")


def test_exact_optimum_for_a_detuned_harmonic_series():
    rng = np.random.default_rng(7)
    offsets = rng.uniform(-5, 5, 6)
    freq = 100 * np.arange(1, 7) * 2 ** (offsets / 1200)
    amp = rng.uniform(0.1, 1, 6)
    result = harmonic_f0(snapshot(freq, amp), fmin=60, fmax=150)
    expected = 2 ** (np.sum(amp * (np.log2(freq) - np.log2(np.arange(1, 7)))) / np.sum(amp))
    assert isinstance(result, HarmonicF0)
    assert result.f0 == pytest.approx(expected, rel=1e-12)
    assert result.harmonics.tolist() == [1, 2, 3, 4, 5, 6] and result.harmonics.dtype == np.int64
    residual = cents(freq, np.arange(1, 7) * expected)
    np.testing.assert_allclose(result.cents, residual, atol=1e-9)
    assert result.rms_cents == pytest.approx(np.sqrt(np.sum(amp * residual**2) / np.sum(amp)))
    assert (result.fmin, result.fmax, result.weight, result.tol_cents) == (60.0, 150.0, "amp", 3.0)


def test_subharmonic_tie_prefers_the_higher_candidate():
    freq = 100 * np.arange(1, 7) * 2 ** (np.array([1, -2, 0.5, 3, -1, 2]) / 1200)
    result = harmonic_f0(snapshot(freq), fmin=40, fmax=150)
    assert result.f0 == pytest.approx(100, abs=0.2)
    assert np.any(np.abs(result.candidate_f0 - 50) < 0.2)  # the subharmonic fits as well


def test_cello_matches_0_6_optimum_and_tolerance_picks_the_octave_above(cello_strong_peaks):
    from audiospylt.f0_utils import search_f0

    freq, amp = cello_strong_peaks["freq"].to_numpy(), cello_strong_peaks["amp"].to_numpy()
    old = search_f0(freq, 35.0, 150.0, 0.001, weights=amp)
    lowest = harmonic_f0(cello_strong_peaks, fmin=35, fmax=150, tol_cents=0)
    assert lowest.f0 == pytest.approx(old["best_f0_hz"], abs=0.001)
    assert lowest.rms_cents == pytest.approx(old["best_rms_cents"], abs=1e-3)
    chosen = harmonic_f0(cello_strong_peaks, fmin=35, fmax=150)
    assert chosen.f0 == pytest.approx(74.40, abs=0.01)
    assert chosen.rms_cents == pytest.approx(17.79, abs=0.01)
    equal = harmonic_f0(cello_strong_peaks, fmin=35, fmax=150, weight="equal")
    assert equal.f0 == pytest.approx(74.39, abs=0.01)
    np.testing.assert_array_equal(np.diff(chosen.candidate_f0) > 0, True)
    assert chosen.candidate_f0[0] == 35 and chosen.candidate_f0[-1] == 150
    assert chosen.f0 in chosen.candidate_f0


def test_zero_hz_rows_are_skipped_and_low_rows_get_harmonic_one():
    result = harmonic_f0(snapshot([0.0, 20.0, 100.0, 200.0, 300.0], [1, 0.01, 1, 1, 1]), fmin=60, fmax=150)
    assert result.harmonics.tolist() == [0, 1, 1, 2, 3]
    assert np.isnan(result.cents[0]) and np.isfinite(result.cents[1:]).all()


@pytest.mark.parametrize("table,kwargs,error,match", [
    (snapshot([0.0]), {}, ValueError, "at least one row with freq > 0"),
    (snapshot([100.0], [0.0]), {}, ValueError, "positive sum"),
    (snapshot([100.0]), {"weight": "power"}, ValueError, "weight must be one of"),
    (snapshot([100.0]), {"tol_cents": -1}, ValueError, "tol_cents must be >= 0"),
    (snapshot([100.0]), {"fmin": 150}, ValueError, "0 < fmin < fmax"),
    (pd.DataFrame({"freq": [-1.0], "amp": [1.0]}), {}, SchemaError, "must be >= 0"),
])
def test_harmonic_f0_rejects_invalid_input(table, kwargs, error, match):
    with pytest.raises(error, match=match):
        harmonic_f0(table, **{"fmin": 50, "fmax": 150, **kwargs})
