"""Measured-frame invariants and exact weighted partial linking."""

from dataclasses import replace
from itertools import combinations, permutations
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from audiospylt.analysis.peaks import find_peaks
from audiospylt.analysis.spectrogram import spectrogram
from audiospylt.analysis.tracking import (
    _assign_frame, link_peaks, spectrogram_peaks, track_partials,
)
from audiospylt.model.schema import SchemaError, validate


def peaks(rows):
    return validate(pd.DataFrame(rows, columns=["frame", "time", "freq", "amp"]), "peaks")


def assert_tracks(actual, points):
    expected = validate(pd.DataFrame([
        (i, time, freq, amp) for i, track in enumerate(points) for time, freq, amp in track
    ], columns=["track_id", "time", "freq", "amp"]), "tracks")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert isinstance(actual.index, pd.RangeIndex)


def hz(cents):
    return 1000 * 2 ** (np.asarray(cents, dtype=float) / 1200)


@pytest.fixture
def multi_spec():
    sr = 8000
    t = np.arange(sr) / sr
    audio = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(330, .5), (700, .2), (1100, .1)])
    audio[2400:5600] = 0
    return spectrogram(audio, sr, frame_length=512, hop=256, zero_pad=4)


@pytest.mark.parametrize("kwargs", [{}, {"prominence_db": 20, "min_db": -80}, {"rel_db": -30},
                                    {"min_distance_hz": 100}, {"interpolate": False},
                                    {"fmin": 500, "fmax": 900},
                                    {"min_db": None, "floor_db": -100.},
                                    {"fmin": True, "min_db": "-60"}])
def test_spectrogram_peaks_equals_frame_loop_bit_for_bit(multi_spec, kwargs):
    result = spectrogram_peaks(multi_spec, **kwargs)
    assert list(result) == ["frame", "time", "freq", "amp", "bin", "prominence"]
    assert [str(d) for d in result.dtypes] == ["int64", "float64", "float64", "float64", "Int64", "float64"]
    assert isinstance(result.index, pd.RangeIndex)
    silent = 0
    for i, time in enumerate(multi_spec.times):
        expected = find_peaks(multi_spec.frame(i), **kwargs)
        frame = result[result.frame == i].reset_index(drop=True)
        if expected.empty:
            silent += 1
        np.testing.assert_array_equal(frame.time, np.full(len(expected), time))
        for column in ["freq", "amp", "prominence"]:
            np.testing.assert_array_equal(frame[column].to_numpy().view("uint64"),
                                          expected[column].to_numpy().view("uint64"))
        pd.testing.assert_series_equal(frame.bin, expected.bin)
    assert silent > 0
    assert result.sort_values(["frame", "freq"]).index.equals(result.index)
    meta = result.attrs["meta"]
    assert meta == {
        "sr": 8000, "frame_length": 512, "n_fft": 2048, "window": "hann",
        "interpolation": "none" if kwargs.get("interpolate") is False else "qifft-db",
        "hop": 256, "center": True, "n_frames": len(multi_spec.times),
        "min_db": (None if kwargs.get("min_db", -60.) is None
                   else float(kwargs.get("min_db", -60.))), "rel_db": kwargs.get("rel_db"),
        "fmin": float(kwargs.get("fmin", 0.)), "fmax": kwargs.get("fmax", 4000.),
        "min_distance_hz": kwargs.get("min_distance_hz"),
        "prominence_db": kwargs.get("prominence_db"), "floor_db": kwargs.get("floor_db", -120.),
    }


def test_spectrogram_peaks_constructs_and_validates_only_one_table(multi_spec):
    import audiospylt.analysis.tracking as tracking
    construct = Mock(wraps=pd.DataFrame)
    pandas_proxy = SimpleNamespace(DataFrame=construct, array=pd.array)
    with patch.object(tracking, "validate", wraps=validate) as check:
        with patch.object(tracking, "pd", pandas_proxy):
            spectrogram_peaks(multi_spec)
    assert check.call_count == construct.call_count == 1


@pytest.mark.parametrize("zero_frames", [False, True])
def test_spectrogram_peaks_empty_and_smallest_spectrogram(zero_frames):
    # Empty audio, minimum length 3, center=True: 0 frames (2 padded samples).
    # A single zero sample gives 1 frame at the same minimum window length.
    spec = spectrogram(np.zeros(0 if zero_frames else 1), 8000, frame_length=3)
    assert len(spec.times) == (0 if zero_frames else 1)
    result = spectrogram_peaks(spec)
    assert result.empty
    assert [str(d) for d in result.dtypes] == ["int64", "float64", "float64", "float64", "Int64", "float64"]
    assert spectrogram_peaks(spectrogram(np.zeros(8000), 8000)).empty


@pytest.mark.parametrize("kwargs", [{"fmin": -1}, {"fmin": 2, "fmax": 1}, {"fmin": "bad"},
    {"fmax": np.inf}, {"min_db": np.nan}, {"rel_db": 1}, {"rel_db": "bad"},
    {"min_distance_hz": 0}, {"prominence_db": -1}, {"floor_db": -np.inf}])
def test_peak_settings_errors_identical_to_find_peaks(multi_spec, kwargs):
    with pytest.raises(ValueError) as expected:
        find_peaks(multi_spec.frame(0), **kwargs)
    for spec in (multi_spec, replace(multi_spec, times=np.empty(0), amp=multi_spec.amp[:, :0])):
        with pytest.raises(ValueError) as actual:
            spectrogram_peaks(spec, **kwargs)
        assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("bad", [-1., np.nan])
def test_spectrogram_bad_amplitudes_same_error(multi_spec, bad):
    amp = multi_spec.amp.copy()
    amp[1, 0] = bad
    invalid = replace(multi_spec, amp=amp)
    with pytest.raises(ValueError) as expected:
        find_peaks(invalid.frame(0))
    with pytest.raises(ValueError) as actual:
        spectrogram_peaks(invalid)
    assert str(actual.value) == str(expected.value)


def test_two_steady_partials_and_every_output_point_is_measured():
    source = peaks([(i, i / 8, f, a) for i in range(4) for f, a in [(440., .5), (880., .2)]])
    before = source.copy(deep=True)
    result = link_peaks(source)
    assert_tracks(result, [[(i / 8, f, a) for i in range(4)] for f, a in [(440., .5), (880., .2)]])
    for seed in range(5):
        pd.testing.assert_frame_equal(link_peaks(source.sample(frac=1, random_state=seed)), result,
                                      check_exact=True)
    values = source[["time", "freq", "amp"]].to_numpy()
    assert all(np.any(np.all(values == row, axis=1)) for row in result[["time", "freq", "amp"]].to_numpy())
    pd.testing.assert_frame_equal(source, before, check_exact=True)


@pytest.mark.parametrize("frequency", [2000., 2100.])
def test_at_or_beyond_gate_starts_new_track(frequency):
    source = peaks([(0, 0., 1000., .1), (1, .125, frequency, .2)])
    assert_tracks(link_peaks(source, max_cents=1200, min_duration=0),
                  [[(0., 1000., .1)], [(.125, frequency, .2)]])


def test_cascade_prefers_two_small_deviations_to_three_larger_links():
    f = hz([0, 40, 80, -40, 5, 45])
    source = peaks([(0, 0., f[i], 1.) for i in range(3)] +
                   [(1, .125, f[i], 1.) for i in range(3, 6)])
    assert_tracks(link_peaks(source, min_duration=0), [
        [(0., f[0], 1.), (.125, f[4], 1.)], [(0., f[1], 1.), (.125, f[5], 1.)],
        [(0., f[2], 1.)], [(.125, f[3], 1.)],
    ])


@pytest.mark.parametrize("gap", [1, 2, 3])
@pytest.mark.parametrize("other_partial", [False, True])
def test_gaps_include_absent_frame_numbers_and_have_no_synthetic_rows(gap, other_partial):
    rows = [(0, 0., 440., .5), (gap + 1, (gap + 1) / 8, 440., .4)]
    if other_partial:
        rows += [(i, i / 8, 880., .1) for i in range(gap + 2)]
    source = peaks(rows)
    for allowed in (gap - 1, gap, gap + 1):
        result = link_peaks(source, max_gap=allowed, min_duration=0)
        low = result[result.freq == 440.]
        assert low.track_id.nunique() == (1 if allowed >= gap else 2)
        assert low.time.tolist() == [0., (gap + 1) / 8]
        assert len(result) == len(source)


def test_waiting_candidate_competes_at_last_frequency_and_closed_track_cannot_return():
    f = hz([0, 30, 2, 1])
    # Both candidates survive frame 1; the waiting 0-cent candidate wins at 2.
    source = peaks([(0, 0., f[0], .1), (0, 0., f[1], .2),
                    (1, .125, f[1], .3), (2, .25, f[2], .4)])
    assert_tracks(link_peaks(source, max_gap=1, min_duration=0), [
        [(0., f[0], .1), (.25, f[2], .4)], [(0., f[1], .2), (.125, f[1], .3)],
    ])
    source = peaks([(0, 0., f[0], .1), (1, .125, 4000., .2), (2, .25, f[3], .3)])
    assert_tracks(link_peaks(source, min_duration=0), [
        [(0., f[0], .1)], [(.125, 4000., .2)], [(.25, f[3], .3)],
    ])


def test_duration_boundary_renumbering_metadata_and_empty_outputs():
    source = peaks([(0, 0., 100., 1.), (0, 0., 500., 1.), (0, 0., 300., 1.),
                    (1, .125, 500., .5), (1, .125, 300., .5), (2, .25, 500., .3),
                    (2, .25, 800., .1), (3, .375, 800., .2)])
    source.attrs["meta"] = {"source": "measured", "method": "old"}
    result = link_peaks(source, min_duration=.125)
    assert_tracks(result, [[(0., 300., 1.), (.125, 300., .5)],
                           [(0., 500., 1.), (.125, 500., .5), (.25, 500., .3)],
                           [(.25, 800., .1), (.375, 800., .2)]])
    assert result.attrs["meta"] == {"source": "measured", "method": "link_peaks",
                                    "objective": "weighted", "max_cents": 50.,
                                    "max_gap": 0, "min_duration": .125}
    assert source.attrs["meta"] == {"source": "measured", "method": "old"}
    for empty in (link_peaks(source, min_duration=1.), link_peaks(source.iloc[:0])):
        assert_tracks(empty, [])
        assert [str(d) for d in empty.dtypes] == ["int64", "float64", "float64", "float64"]


@pytest.mark.parametrize("kwargs", [
    {"max_cents": 0}, {"max_cents": -1}, {"max_cents": np.inf}, {"max_cents": np.nan},
    {"max_cents": True}, {"max_cents": "50"}, {"max_cents": [50]},
    {"max_gap": -1}, {"max_gap": True}, {"max_gap": np.bool_(False)}, {"max_gap": 1.},
    {"max_gap": np.nan}, {"max_gap": "1"}, {"min_duration": -1}, {"min_duration": np.inf},
    {"min_duration": np.nan}, {"min_duration": False}, {"min_duration": "0"},
])
def test_link_parameter_errors(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        link_peaks(peaks([]), **kwargs)


def test_link_input_errors_and_positive_frequency_positions():
    source = peaks([(0, 0., 0., .1), (1, 1., 440., .2), (2, 2., 0., .3)])
    source.index = [10, 20, 30]
    with pytest.raises(ValueError, match=r"freq must be > 0; row positions \[0, 2\]"):
        link_peaks(source)
    for invalid in (source.drop(columns="frame"), source.assign(frame=-1),
                    source.assign(time=[0, 0, 2]), source.assign(amp=-1),
                    source.assign(freq=-1), source.assign(frame=[0, np.nan, 2]),
                    source.assign(frame=[0, .5, 2]), source.assign(time=[0, np.inf, 2]),
                    source.assign(frame=[0, 0, 2])):
        with pytest.raises(SchemaError):
            link_peaks(invalid)
    link_peaks(peaks([]), max_gap=np.int64(1))


@pytest.mark.parametrize("n,m", [(0, 3), (3, 0)] + [(n, m) for n in range(1, 7) for m in range(1, 7)])
def test_assignment_matches_brute_force_all_partial_matchings(n, m, record_property):
    rng = np.random.default_rng(102 + 7 * n + m)
    tied_draws = 0
    for _ in range(100):
        predicted, frequencies = hz(rng.uniform(-120, 120, n)), hz(rng.uniform(-120, 120, m))
        distance = np.abs(1200 * np.log2(frequencies[None, :] / predicted[:, None]))
        gains = np.maximum(50 - distance, 0)
        matchings = [(0., set())]
        for size in range(1, min(n, m) + 1):
            for candidates in combinations(range(n), size):
                for selected in permutations(range(m), size):
                    if any(distance[i, j] >= 50 for i, j in zip(candidates, selected)):
                        continue
                    gain = sum(gains[i, j] for i, j in zip(candidates, selected))
                    matchings.append((gain, set(zip(candidates, selected))))
        best_gain = max(gain for gain, _ in matchings)
        best_links = [links for gain, links in matchings if abs(gain - best_gain) <= 1e-10]
        tied_draws += len(best_links) > 1
        ordered_optima = []
        for links in best_links:
            by_peak = sorted(links, key=lambda pair: frequencies[pair[1]])
            if all(predicted[a[0]] < predicted[b[0]] for a, b in zip(by_peak, by_peak[1:])):
                ordered_optima.append(by_peak)
        assert len(ordered_optima) == 1
        candidates, selected = _assign_frame(predicted, frequencies, 50.)
        assert sum(gains[i, j] for i, j in zip(candidates, selected)) == pytest.approx(
            best_gain, rel=0, abs=1e-10,
        )
        assert list(zip(candidates, selected)) == ordered_optima[0]
    record_property("tied_optima_draws", tied_draws)


@pytest.mark.parametrize("options", [{}, {"frame_length": 512, "hop": 128, "zero_pad": 2,
    "window": "blackmanharris", "center": False, "min_db": -70., "rel_db": -30.,
    "fmin": 100., "fmax": 2000., "min_distance_hz": 80., "prominence_db": 10.,
    "max_cents": 80., "max_gap": 1, "min_duration": .025}])
def test_track_partials_equals_composed_calls(options):
    sr = 8000
    audio = .5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    stft = {"frame_length": 2048, "hop": 256, "zero_pad": 4, "window": "hann", "center": True}
    pick = {"min_db": -80., "rel_db": None, "fmin": 0., "fmax": None,
            "min_distance_hz": None, "prominence_db": 20.}
    link = {"max_cents": 50., "max_gap": 0, "min_duration": .05}
    for group in (stft, pick, link):
        group.update({key: value for key, value in options.items() if key in group})
    expected = link_peaks(spectrogram_peaks(spectrogram(audio, sr, **stft), **pick), **link)
    expected.attrs["meta"]["zero_pad"] = stft["zero_pad"]
    actual = track_partials(audio, sr, **options)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs == expected.attrs


def test_stationary_three_partial_tone():
    sr = 44100
    t = np.arange(sr) / sr
    audio = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(440, .5), (880, .25), (1320, .125)])
    tracks = track_partials(audio, sr)
    assert tracks.track_id.nunique() == 3  # No extra tracks at these defaults.
    errors = []
    for frequency in (440, 880, 1320):
        choices = [(i, frame) for i, frame in tracks.groupby("track_id")
                   if abs(1200 * np.log2(frame.freq.mean() / frequency)) < 1]
        assert len(choices) == 1
        _, frame = choices[0]
        assert len(frame) == 173
        assert frame.time.iloc[0] == 0.
        assert frame.time.iloc[-1] == 44032 / sr
        errors.append(np.mean(np.abs(1200 * np.log2(frame.freq / frequency))))
    np.testing.assert_allclose(errors, [.1642453146, .0289298151, .0919770577], rtol=0, atol=1e-6)


def test_exponential_glissando_silence_and_audio_shape_errors():
    sr = 44100
    t = np.arange(sr) / sr
    audio = .5 * np.sin(2 * np.pi * 300 * np.expm1(np.log(2) * t) / np.log(2))
    tracks = track_partials(audio, sr)
    assert tracks.track_id.nunique() == 1
    assert tracks.time.iloc[0] <= .025 and tracks.time.iloc[-1] >= .975
    assert_tracks(track_partials(np.zeros(sr), sr), [])
    with pytest.raises(ValueError) as expected:
        spectrogram(np.zeros((2, 100)), sr)
    with pytest.raises(ValueError) as actual:
        track_partials(np.zeros((2, 100)), sr)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("initial,shifted,gate", [
    ([0, 15], [30, 45], 50.),
    ([0, 10, 20], [35, 45, 55], 60.),
])
def test_parallel_shifts_keep_frequency_order(direction, initial, shifted, gate):
    predicted = np.sort(hz(direction * np.array(initial)))
    frequencies = np.sort(hz(direction * np.array(shifted)))
    # The three-partial gate is 60 cents so all six permutations are in-gate.
    for pairing in permutations(frequencies):
        assert np.all(np.abs(1200 * np.log2(pairing / predicted)) < gate)
    source = peaks([(0, 0., f, 1.) for f in predicted] +
                   [(1, .125, f, .5) for f in frequencies])
    assert_tracks(link_peaks(source, max_cents=gate), [
        [(0., first, 1.), (.125, second, .5)]
        for first, second in zip(predicted, frequencies)
    ])


def test_equal_frequencies_keep_candidate_creation_and_peak_row_order():
    candidates, selected = _assign_frame(np.array([1010., 1000., 1000.]),
                                          np.array([1020., 1015., 1015.]), 50.)
    np.testing.assert_array_equal(candidates, [1, 2, 0])
    np.testing.assert_array_equal(selected, [1, 2, 0])


def test_spectrogram_peaks_rejects_a_single_spectrum_with_route(multi_spec):
    with pytest.raises(TypeError, match=r"spec must be a Spectrogram; got Spectrum; use find_peaks"):
        spectrogram_peaks(multi_spec.frame(0))
