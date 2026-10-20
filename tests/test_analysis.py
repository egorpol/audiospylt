"""D4 analysis regressions, filter ordering, and independent real-audio checks."""

import builtins
from dataclasses import FrozenInstanceError
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from audiospylt.analysis.peaks import find_peaks
from audiospylt.analysis.spectrogram import MelSpectrogram, Spectrogram, mel_spectrogram, spectrogram
from audiospylt.analysis.spectrum import Spectrum, amplitude_to_db, db_to_amplitude, spectrum
from audiospylt.dft_analysis import compute_fft
from audiospylt.io.audio import load_audio, trim
from audiospylt.io.tables import read_table, write_table
from audiospylt.model.schema import validate


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
# Fixed values supplied by revision/step-1.3/expect_1_3.py (plain numpy/scipy).
CELLO_PARTIALS = [
    (298, 74.405809, -36.614177, 75.677468),
    (593, 148.323924, -29.951086, 86.614474),
    (894, 223.531482, -39.509243, 66.565470),
    (1186, 296.454975, -43.087103, 61.612167),
    (1488, 371.951680, -35.750771, 68.665128),
    (1786, 446.435833, -38.628361, 65.444440),
    (2083, 520.814507, -27.568420, 92.408213),
    (2381, 595.220897, -39.103963, 67.304463),
    (2679, 669.633326, -47.290497, 56.279573),
    (2976, 743.935747, -53.091520, 46.241964),
]


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def _sine(freq, amp=0.37, *, sr=1000, length=1000, phase=0.0):
    return amp * np.sin(2 * np.pi * freq * np.arange(length) / sr + phase)


def _db_spectrum(db):
    """Hand-authored bins, 1 Hz apart, independent of the FFT implementation."""
    amp = 10.0 ** (np.asarray(db, dtype=np.float64) / 20)
    n_fft = 2 * (len(amp) - 1)
    return Spectrum(np.arange(len(amp), dtype=np.float64), amp, n_fft, n_fft, n_fft, "boxcar")


@pytest.fixture(scope="module")
def cello():
    return load_audio(SAMPLES / "cello_arco_d_sul_c_01.wav", mono=True)


@pytest.fixture(scope="module")
def cello_partials(cello):
    x, sr = cello
    segment = trim(x, sr, 1.5, 2.5)
    assert len(segment) == 44100
    spec = spectrum(segment, sr)
    assert spec.n_fft == 176400
    return find_peaks(spec, min_db=None, prominence_db=10)


@pytest.mark.parametrize("window", ["boxcar", "hann", "blackmanharris"])
@pytest.mark.parametrize("zero_pad", [1, 4])
def test_on_bin_d4_sine_scaling(window, zero_pad):
    x = _sine(10)
    original = x.copy()
    spec = spectrum(x, 1000, window=window, zero_pad=zero_pad)
    assert spec.amp[10 * zero_pad] == pytest.approx(0.37, abs=1e-12)
    assert spec.amp.dtype == spec.freqs.dtype == np.float64
    assert (spec.sr, spec.frame_length, spec.n_fft, spec.window) == (
        1000, 1000, 1000 * zero_pad, window,
    )
    assert spec.bin_hz == 1 / zero_pad
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("window", ["boxcar", "hann", "blackmanharris"])
@pytest.mark.parametrize("zero_pad", [1, 4])
def test_dc_and_even_nyquist_are_not_doubled(window, zero_pad):
    assert spectrum(np.full(1000, 0.37), 1000, window=window, zero_pad=zero_pad).amp[0] == (
        pytest.approx(0.37, abs=1e-12)
    )
    alternating = 0.37 * (-1.0) ** np.arange(1000)
    assert spectrum(alternating, 1000, window=window, zero_pad=zero_pad).amp[-1] == (
        pytest.approx(0.37, abs=1e-12)
    )


@pytest.mark.parametrize("length, zero_pad", [(31, 1), (32, 1), (31, 4), (32, 4)])
@pytest.mark.parametrize("window", ["boxcar", "hann", ("kaiser", 6.0)])
def test_scaled_rfft_reference_including_odd_last_bin(length, zero_pad, window):
    x = np.random.default_rng(831).normal(size=length)
    w = signal.get_window(window, length, fftbins=True)
    n_fft = length * zero_pad
    expected = np.abs(np.fft.rfft(x * w, n_fft)) * 2 / w.sum()
    expected[0] /= 2
    if n_fft % 2 == 0:
        expected[-1] /= 2
    spec = spectrum(x, 8000, window=window, zero_pad=zero_pad)
    np.testing.assert_array_equal(spec.amp, expected)
    np.testing.assert_array_equal(spec.freqs, np.fft.rfftfreq(n_fft, 1 / 8000))
    assert spec.window is window


def test_dsp1_link_to_legacy_interior_bins():
    x = np.random.default_rng(33).normal(size=1000)
    old_freqs, old_amp = compute_fft(x, 1000)
    spec = spectrum(x, 1000, window="boxcar", zero_pad=1)
    np.testing.assert_array_equal(spec.freqs, old_freqs)
    np.testing.assert_allclose(spec.amp[1:-1], 2 * old_amp[1:-1], rtol=1e-14)


@pytest.mark.parametrize("audio", [[], [1], [1, 2]])
def test_spectrum_rejects_short_frames(audio):
    with pytest.raises(ValueError, match="frame_length.*3"):
        spectrum(audio, 1000)


@pytest.mark.parametrize("function", [spectrum, spectrogram])
@pytest.mark.parametrize("shape", [(2, 10), (10, 2), (1, 10)])
def test_analysis_rejects_2d_with_explicit_layout_and_remedies(function, shape):
    with pytest.raises(ValueError) as caught:
        function(np.zeros(shape), 1000)
    message = str(caught.value)
    assert str(shape) in message
    assert "2-D is read as (channels, samples)" in message
    assert "to_mono(x)" in message and "x[0]" in message


@pytest.mark.parametrize("function", [spectrum, spectrogram])
@pytest.mark.parametrize("audio", [np.array(1.0), np.zeros((1, 2, 3))])
def test_analysis_rejects_other_dimensions(function, audio):
    with pytest.raises(ValueError, match="shape.*1-D"):
        function(audio, 1000)


@pytest.mark.parametrize("function", [spectrum, spectrogram])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1j])
def test_analysis_rejects_nonfinite_or_complex_audio(function, bad):
    with pytest.raises(ValueError, match="finite"):
        function(np.array([0, bad, 0]), 1000)


@pytest.mark.parametrize("function", [spectrum, spectrogram])
@pytest.mark.parametrize("sr", [0, -1, True, np.bool_(True), 1000.0, np.inf, np.nan, "1000"])
def test_analysis_rejects_invalid_rates(function, sr):
    with pytest.raises(ValueError, match="sr.*positive integer"):
        function(np.zeros(16), sr)


@pytest.mark.parametrize("function", [spectrum, spectrogram])
@pytest.mark.parametrize("zero_pad", [0, -1, True, np.bool_(True), 1.0, np.inf, np.nan])
def test_analysis_rejects_invalid_padding(function, zero_pad):
    with pytest.raises(ValueError, match="zero_pad"):
        function(np.zeros(16), 1000, zero_pad=zero_pad)


def test_numpy_integer_parameters_and_frozen_results():
    spec = spectrum(np.ones(16), np.int64(1000), zero_pad=np.int64(2))
    assert type(spec.sr) is type(spec.n_fft) is int
    with pytest.raises(FrozenInstanceError):
        spec.sr = 10
    stft = spectrogram(np.ones(16), 1000, frame_length=np.int64(8), hop=np.int64(2))
    with pytest.raises(FrozenInstanceError):
        stft.hop = 1


@pytest.mark.parametrize("function", [spectrum, spectrogram])
def test_zero_gain_windows_raise_clear_error(function):
    with pytest.raises(ValueError, match="window.*nonzero sum"):
        function(np.ones(16), 1000, window=("general_cosine", [0.0]))


@pytest.mark.parametrize("amp, expected", [(1.0, 0.0), (0.5, -6.020599913279624), (0, -120.0)])
def test_amplitude_to_db_scalar(amp, expected):
    result = amplitude_to_db(amp)
    assert type(result) is float
    assert result == pytest.approx(expected, abs=1e-12)


def test_db_conversions_shape_dtype_floor_and_no_mutation():
    amp = np.array([[1, 0.5], [0, 0.001]], dtype=np.float32)
    original = amp.copy()
    db = amplitude_to_db(amp, floor_db=-40)
    assert db.shape == amp.shape and db.dtype == np.float64
    np.testing.assert_allclose(db, [[0, -6.020599913279624], [-40, -40]], atol=1e-12)
    np.testing.assert_array_equal(amp, original)
    db_original = db.copy()
    restored = db_to_amplitude(db)
    assert restored.dtype == np.float64 and restored.shape == db.shape
    np.testing.assert_allclose(restored, [[1, 0.5], [0.01, 0.01]], atol=1e-15)
    np.testing.assert_array_equal(db, db_original)
    assert type(db_to_amplitude(0)) is float
    assert db_to_amplitude(-np.inf) == 0.0
    assert db_to_amplitude(-6.020599913279624) == pytest.approx(0.5)
    assert amplitude_to_db(np.float64(1)) == 0.0
    assert type(amplitude_to_db(np.array(1))) is float
    assert amplitude_to_db(np.inf) == np.inf
    assert amplitude_to_db(0, floor_db=-1e6) == -1e6
    assert amplitude_to_db(1, floor_db=1e6) == 1e6


@pytest.mark.parametrize("amp", [-0.1, np.nan, [0, -1], [0, np.nan]])
def test_amplitude_to_db_rejects_invalid_amplitudes(amp):
    with pytest.raises(ValueError, match="amp"):
        amplitude_to_db(amp)


@pytest.mark.parametrize("floor", [np.inf, -np.inf, np.nan, None, [0, 1]])
def test_amplitude_to_db_rejects_nonfinite_floor(floor):
    with pytest.raises(ValueError, match="floor_db"):
        amplitude_to_db(1, floor_db=floor)


@pytest.mark.parametrize("cycles, max_cents", [(4, 1.5), (5, 0.5), (6, 0.5), (8, 0.5), (22, 0.5)])
def test_qifft_random_offset_and_phase_accuracy(cycles, max_cents):
    rng = np.random.default_rng(20261019)
    cents_errors, db_errors = [], []
    for _ in range(200):
        freq = (cycles + rng.uniform(-0.5, 0.5)) * 44100 / 4410
        x = _sine(freq, 0.5, sr=44100, length=4410, phase=rng.uniform(0, 2 * np.pi))
        peaks = find_peaks(spectrum(x, 44100, window="hann", zero_pad=4))
        peak = peaks.iloc[np.argmin(np.abs(peaks.freq.to_numpy() - freq))]
        cents_errors.append(abs(1200 * np.log2(peak.freq / freq)))
        db_errors.append(abs(20 * np.log10(peak.amp / 0.5)))
    assert max(cents_errors) < max_cents
    assert max(db_errors) < 0.1


@pytest.mark.parametrize("length", [4410, 44100])
def test_five_partial_static_accuracy(length):
    frequencies = np.array([223.37, 457.81, 911.29, 1843.43, 3299.61])
    amplitudes = np.array([0.12, 0.10, 0.08, 0.06, 0.04])
    t = np.arange(length) / 44100
    x = np.sum(amplitudes[:, None] * np.sin(2 * np.pi * frequencies[:, None] * t), axis=0)
    peaks = find_peaks(spectrum(x, 44100))
    for freq, amp in zip(frequencies, amplitudes, strict=True):
        row = peaks.iloc[np.argmin(np.abs(peaks.freq.to_numpy() - freq))]
        assert abs(1200 * np.log2(row.freq / freq)) < 0.1
        assert abs(20 * np.log10(row.amp / amp)) < 0.1


def test_qifft_formula_filters_use_interpolated_frequency_and_amplitude():
    spec = _db_spectrum([-80, -40, -10, -20, -80])
    peaks = find_peaks(spec, min_db=-9, fmin=2.2, fmax=2.3)
    assert peaks.bin.tolist() == [2]
    assert peaks.freq.iloc[0] == pytest.approx(2.25)
    assert amplitude_to_db(peaks.amp.iloc[0]) == pytest.approx(-8.75)
    assert peaks.prominence.iloc[0] == pytest.approx(70)
    assert find_peaks(spec, min_db=-9, interpolate=False).empty
    assert find_peaks(spec, min_db=None, fmin=2.2, fmax=2.3, interpolate=False).empty


@pytest.mark.parametrize("kwargs, bins", [
    ({}, [1, 3, 5, 7]),
    ({"min_db": -25}, [3, 5]),
    ({"min_db": None}, [1, 3, 5, 7]),
    ({"fmin": 3, "fmax": 5}, [3, 5]),
    ({"rel_db": -15}, [3, 5]),
    ({"rel_db": 0}, [3]),
    ({"prominence_db": 55}, [3, 5]),
    ({"prominence_db": 0}, [1, 3, 5, 7]),
    ({"min_distance_hz": 3}, [3, 7]),
    ({"fmin": 4, "min_distance_hz": 3}, [5]),
    ({"fmin": 4, "rel_db": -15}, [5]),
])
def test_peak_filters_and_frequency_before_distance_or_relative(kwargs, bins):
    spec = _db_spectrum([-80, -30, -80, -10, -80, -20, -80, -50, -80])
    result = find_peaks(spec, **kwargs)
    assert result.bin.tolist() == bins


def test_relative_precedes_prominence_and_prominence_precedes_distance():
    # The loudest peak has a shallow left base; the quieter peak is prominent.
    spec = _db_spectrum([-10, 0, -1, -2, -3, -4, -5, -60, -10, -60])
    assert find_peaks(spec, rel_db=-5, prominence_db=20).empty
    result = find_peaks(spec, prominence_db=20, min_distance_hz=10)
    assert result.bin.tolist() == [8]
    assert result.prominence.iloc[0] == pytest.approx(50)


def test_distance_ties_prefer_lower_frequency_and_exact_spacing_survives():
    spec = _db_spectrum([-80, -10, -80, -10, -80, -20, -80])
    assert find_peaks(spec, min_distance_hz=3).bin.tolist() == [1, 5]
    assert find_peaks(spec, min_distance_hz=2).bin.tolist() == [1, 3, 5]


def test_scipy_plateau_handling_endpoints_and_nonconcave_interpolation():
    spec = _db_spectrum([0, -80, -10, -10, -10, -80, 0])
    result = find_peaks(spec)
    assert result.bin.tolist() == [3]
    assert result.freq.tolist() == [3.0]
    assert amplitude_to_db(result.amp.iloc[0]) == pytest.approx(-10)
    assert find_peaks(_db_spectrum([0, -20, -40]), min_db=None).empty


def test_detection_interpolation_and_prominence_share_one_floor():
    spec = _db_spectrum([-100, -40, -10, -20, -100])
    result = find_peaks(spec, floor_db=-20, min_db=None)
    assert result.bin.tolist() == [2]
    assert result.freq.tolist() == [2.0]
    assert amplitude_to_db(result.amp.iloc[0]) == pytest.approx(-10)
    assert result.prominence.iloc[0] == pytest.approx(10)
    assert find_peaks(spec, floor_db=0, min_db=None).empty


@pytest.mark.parametrize("interpolate", [True, False])
def test_snapshot_schema_metadata_provenance_roundtrip_and_input_purity(interpolate, tmp_path):
    spec = spectrum(_sine(10.2), 1000, window=("kaiser", 6.0))
    original = spec.amp.copy()
    result = find_peaks(spec, interpolate=interpolate)
    assert list(result.columns) == ["freq", "amp", "bin", "prominence"]
    assert [str(dtype) for dtype in result.dtypes] == ["float64", "float64", "Int64", "float64"]
    assert isinstance(result.index, pd.RangeIndex)
    assert result.freq.is_monotonic_increasing
    pd.testing.assert_frame_equal(validate(result, "snapshot"), result)
    meta = {
        "sr": 1000, "frame_length": 1000, "n_fft": 4000,
        "window": "('kaiser', 6.0)", "interpolation": "qifft-db" if interpolate else "none",
    }
    assert result.attrs == {"audiospylt_table": "snapshot", "meta": meta}
    np.testing.assert_array_equal(spec.amp, original)
    raw_bins = result.bin.to_numpy(dtype=int)
    db = amplitude_to_db(spec.amp)
    np.testing.assert_array_equal(result.prominence, signal.peak_prominences(db, raw_bins)[0])
    if not interpolate:
        np.testing.assert_array_equal(result.freq, raw_bins * spec.bin_hz)
        np.testing.assert_array_equal(result.amp, spec.amp[raw_bins])
    path = write_table(result, tmp_path / "peaks.tsv")
    restored = read_table(path)
    pd.testing.assert_frame_equal(restored, result, check_exact=True)
    assert restored.attrs["meta"] == {key: str(value) for key, value in meta.items()}


@pytest.mark.parametrize("kind", ["silence", "filtered", "endpoints"])
def test_empty_snapshots_keep_schema_and_provenance(kind):
    spec = (spectrum(np.zeros(32), 1000) if kind == "silence" else
            _db_spectrum([0, -20, -40]) if kind == "endpoints" else
            _db_spectrum([-80, -10, -80]))
    result = find_peaks(spec, min_db=1)
    assert result.empty
    assert list(result.columns) == ["freq", "amp", "bin", "prominence"]
    assert [str(dtype) for dtype in result.dtypes] == ["float64", "float64", "Int64", "float64"]
    assert isinstance(result.index, pd.RangeIndex)
    assert result.attrs["meta"]["interpolation"] == "qifft-db"
    pd.testing.assert_frame_equal(validate(result, "snapshot"), result)


@pytest.mark.parametrize("name", [
    "min_db", "rel_db", "fmin", "fmax", "min_distance_hz", "prominence_db", "floor_db",
])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_find_peaks_rejects_each_nonfinite_parameter(name, value):
    with pytest.raises(ValueError, match=name):
        find_peaks(_db_spectrum([-80, -10, -80]), **{name: value})


@pytest.mark.parametrize("kwargs, match", [
    ({"rel_db": 0.01}, "rel_db"),
    ({"fmin": -1}, "fmin"),
    ({"fmin": 1, "fmax": 1}, "fmax"),
    ({"fmin": 1, "fmax": 0}, "fmax"),
    ({"fmin": 2}, "fmax"),
    ({"prominence_db": -1}, "prominence_db"),
    ({"min_distance_hz": 0}, "min_distance_hz"),
    ({"min_distance_hz": -1}, "min_distance_hz"),
])
def test_find_peaks_rejects_parameter_signs_and_empty_frequency_ranges(kwargs, match):
    with pytest.raises(ValueError, match=match):
        find_peaks(_db_spectrum([-80, -10, -80]), **kwargs)


@pytest.mark.parametrize("frame_length", [3, 4, 5, 6])
@pytest.mark.parametrize("hop", [1, 3, 5])
@pytest.mark.parametrize("center", [True, False])
def test_small_spectrogram_frame_counts_and_times(frame_length, hop, center):
    x = np.arange(13, dtype=np.float64)
    original = x.copy()
    result = spectrogram(x, 1000, frame_length=frame_length, hop=hop, center=center)
    pad = frame_length // 2 if center else 0
    n_frames = 1 + (len(x) + 2 * pad - frame_length) // hop
    offset = 0 if center else frame_length // 2
    expected_times = (np.arange(n_frames) * hop + offset) / 1000
    assert result.amp.shape == (frame_length // 2 + 1, n_frames)
    np.testing.assert_array_equal(result.times, expected_times)
    np.testing.assert_array_equal(result.freqs, np.fft.rfftfreq(frame_length, 1 / 1000))
    assert result.amp.dtype == result.freqs.dtype == result.times.dtype == np.float64
    assert result.center is center
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("length, frame_length", [(1, 5), (2, 4), (0, 4), (0, 5)])
def test_centered_short_audio_and_zero_padding(length, frame_length):
    x = np.ones(length)
    result = spectrogram(x, 1000, frame_length=frame_length, hop=2)
    padded = np.pad(x, (frame_length // 2, frame_length // 2))
    n_frames = 1 + (len(padded) - frame_length) // 2
    assert result.amp.shape[1] == n_frames
    for k in range(n_frames):
        np.testing.assert_array_equal(
            result.amp[:, k], spectrum(padded[2*k:2*k+frame_length], 1000, zero_pad=1).amp,
        )


@pytest.mark.parametrize("frame_length", [31, 32])
@pytest.mark.parametrize("zero_pad", [1, 4])
@pytest.mark.parametrize("window", ["boxcar", "hann", ("kaiser", 6)])
def test_spectrogram_columns_equal_standalone_spectra_exactly(frame_length, zero_pad, window):
    x = np.random.default_rng(20261019).normal(size=257)
    result = spectrogram(
        x, 44100, frame_length=frame_length, hop=7, window=window,
        zero_pad=zero_pad, center=False,
    )
    for k in range(result.amp.shape[1]):
        expected = spectrum(x[k*7:k*7+frame_length], 44100, window=window, zero_pad=zero_pad)
        np.testing.assert_array_equal(result.amp[:, k], expected.amp)
        np.testing.assert_array_equal(result.freqs, expected.freqs)


def test_spectrogram_frame_metadata_negative_index_and_independent_copies():
    window = ("kaiser", 6)
    result = spectrogram(np.arange(32), 1000, frame_length=9, hop=3, window=window, zero_pad=4)
    frame = result.frame(-1)
    assert isinstance(frame, Spectrum)
    assert (frame.sr, frame.frame_length, frame.n_fft, frame.window) == (1000, 9, 36, window)
    assert frame.bin_hz == 1000 / 36
    np.testing.assert_array_equal(frame.amp, result.amp[:, -1])
    np.testing.assert_array_equal(frame.freqs, result.freqs)
    frame.amp[:] = 0
    frame.freqs[:] = -1
    assert np.any(result.amp[:, -1] != 0) and np.all(result.freqs >= 0)
    with pytest.raises(IndexError):
        result.frame(result.amp.shape[1])
    with pytest.raises(IndexError):
        result.frame(-result.amp.shape[1] - 1)


@pytest.mark.parametrize("frame_length", [0, -1, 2, True, 4.0])
def test_spectrogram_rejects_invalid_frame_length(frame_length):
    with pytest.raises(ValueError, match="frame_length"):
        spectrogram(np.zeros(16), 1000, frame_length=frame_length)


@pytest.mark.parametrize("hop", [0, -1, True, 1.0, np.inf])
def test_spectrogram_rejects_invalid_hop(hop):
    with pytest.raises(ValueError, match="hop"):
        spectrogram(np.zeros(16), 1000, frame_length=8, hop=hop)


def test_uncentered_spectrogram_requires_full_frame():
    with pytest.raises(ValueError, match="15 samples.*center=False.*frame_length=16"):
        spectrogram(np.zeros(15), 1000, frame_length=16, center=False)


def test_dsp3_amplitude_halving_is_six_db_and_centered_edges_under_read():
    # Integer cycles in every window, with stationary middle frames each side.
    x = _sine(100, 1, sr=1000, length=2000)
    x[:1000] *= 0.4
    x[1000:] *= 0.2
    result = spectrogram(x, 1000, frame_length=100, hop=25)
    sine_bin = 10
    left = np.argmin(np.abs(result.times - 0.5))
    right = np.argmin(np.abs(result.times - 1.5))
    db = amplitude_to_db(result.amp)
    assert db[sine_bin, right] - db[sine_bin, left] == pytest.approx(-6.0206, abs=0.001)
    assert result.amp[sine_bin, left] == pytest.approx(0.4, abs=1e-12)
    assert result.amp[sine_bin, right] == pytest.approx(0.2, abs=1e-12)
    assert result.amp[sine_bin, 0] < result.amp[sine_bin, left]
    assert result.amp[sine_bin, -1] < result.amp[sine_bin, right]


def test_cello_stft_matches_independent_librosa_reference(cello):
    import librosa

    x, sr = cello
    x = trim(x, sr, 0, 2)
    result = spectrogram(x, sr, frame_length=2048, hop=512, window="hann", center=True)
    reference = np.abs(librosa.stft(
        x, n_fft=2048, hop_length=512, window="hann", center=True, pad_mode="constant",
    )) * 2 / signal.get_window("hann", 2048, fftbins=True).sum()
    reference[0] /= 2
    reference[-1] /= 2
    assert result.amp.shape == reference.shape
    np.testing.assert_allclose(result.amp, reference, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(
        result.times, librosa.frames_to_time(range(reference.shape[1]), sr=sr, hop_length=512),
    )


@pytest.mark.parametrize("band", [40, 60])
def test_mel_band_centre_amplitude_and_zero_padding_agreement(band):
    import librosa

    sr = 44100
    centres = librosa.mel_frequencies(66, fmin=0, fmax=sr/2)[1:-1]
    x = _sine(centres[band], 0.3, sr=sr, length=sr)
    values = []
    for zero_pad in (1, 4):
        stft = spectrogram(x, sr, frame_length=2048, hop=512, zero_pad=zero_pad)
        original = stft.amp.copy()
        mel = mel_spectrogram(stft, n_mels=64)
        assert isinstance(mel, MelSpectrogram)
        assert mel.amp.shape == (64, stft.amp.shape[1])
        assert mel.amp.dtype == mel.freqs.dtype == mel.times.dtype == np.float64
        assert (mel.sr, mel.n_mels, mel.fmin, mel.fmax) == (sr, 64, 0, sr/2)
        np.testing.assert_array_equal(mel.freqs, centres)
        np.testing.assert_array_equal(mel.times, stft.times)
        assert not np.shares_memory(mel.times, stft.times)
        np.testing.assert_array_equal(stft.amp, original)
        amp = mel.amp[band, np.argmin(np.abs(mel.times - 0.5))]
        assert abs(20 * np.log10(amp / 0.3)) < 0.25
        values.append(amp)
    assert abs(20 * np.log10(values[1] / values[0])) < 0.05


@pytest.mark.parametrize("window", ["boxcar", "hann", "blackmanharris", ("kaiser", 6)])
@pytest.mark.parametrize("zero_pad", [1, 4])
def test_mel_formula_with_explicit_edges_and_parseval(window, zero_pad):
    import librosa

    x = _sine(100, 0.3, sr=1024, length=2048)
    stft = spectrogram(x, 1024, frame_length=256, hop=128, window=window,
                       zero_pad=zero_pad, center=False)
    mel = mel_spectrogram(stft, n_mels=16, fmin=10, fmax=400)
    weights = signal.get_window(window, 256, fftbins=True)
    enbw = 256 * np.sum(weights**2) / np.sum(weights)**2
    bands = librosa.filters.mel(sr=1024, n_fft=256*zero_pad, n_mels=16,
                                fmin=10, fmax=400, norm=None)
    expected = np.sqrt(bands @ stft.amp**2 / (zero_pad * enbw))
    np.testing.assert_array_equal(mel.amp, expected)
    np.testing.assert_array_equal(mel.freqs, librosa.mel_frequencies(18, fmin=10, fmax=400)[1:-1])
    # For windows with compact Fourier support, this on-bin sine has exact energy.
    if window in ("boxcar", "hann", "blackmanharris"):
        np.testing.assert_allclose(np.sum(stft.amp**2, axis=0), 0.3**2 * zero_pad * enbw,
                                   rtol=1e-12, atol=1e-15)
    assert (mel.fmin, mel.fmax) == (10, 400)
    with pytest.raises(FrozenInstanceError):
        mel.n_mels = 8


def test_mel_does_not_hide_empty_filter_warnings():
    stft = spectrogram(np.ones(32), 44100, frame_length=8, hop=4)
    with pytest.warns(UserWarning, match="Empty filters"):
        mel_spectrogram(stft, n_mels=64)


@pytest.mark.parametrize("kwargs, match", [
    ({"n_mels": 0}, "n_mels"), ({"n_mels": -1}, "n_mels"),
    ({"n_mels": True}, "n_mels"), ({"n_mels": 4.0}, "n_mels"),
    ({"fmin": -1}, "fmin"), ({"fmin": np.nan}, "fmin"),
    ({"fmax": np.inf}, "fmax"), ({"fmax": np.nan}, "fmax"),
    ({"fmax": 0}, "fmax"), ({"fmin": 10, "fmax": 10}, "fmax"),
    ({"fmin": 500}, "fmax"),
])
def test_mel_rejects_invalid_parameters(kwargs, match):
    stft = spectrogram(np.ones(32), 1000, frame_length=8)
    with pytest.raises(ValueError, match=match):
        mel_spectrogram(stft, **kwargs)


@pytest.mark.parametrize("name", ["spectrum", "peaks", "spectrogram"])
def test_analysis_modules_do_not_eagerly_import_librosa(name, monkeypatch):
    # The unchanged 0.6 package initializer imports its own dependencies. Load
    # each v1 source separately after that initializer to isolate its imports.
    original_import = builtins.__import__

    def guarded_import(module_name, *args, **kwargs):
        if module_name.split(".")[0] == "librosa":
            raise AssertionError("librosa must be imported only inside mel_spectrogram")
        return original_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module_name = f"_isolated_analysis_{name}"
    definition = spec_from_file_location(module_name, ROOT / "audiospylt" / "analysis" / f"{name}.py")
    module = module_from_spec(definition)
    monkeypatch.setitem(sys.modules, module_name, module)
    definition.loader.exec_module(module)


def test_compute_functions_have_no_display_print_or_file_effects(tmp_path, capsys, inert_environment):
    stft = spectrogram(_sine(10), 1000, frame_length=100)
    assert isinstance(stft, Spectrogram)
    find_peaks(stft.frame(0))
    mel_spectrogram(stft, n_mels=8)
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert not inert_environment
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("h, expected", list(enumerate(CELLO_PARTIALS, start=1)))
def test_real_cello_partial_bins_frequencies_and_amplitudes(cello_partials, h, expected):
    bin_index, freq, amp_db, _ = expected
    row = cello_partials.iloc[np.argmin(np.abs(cello_partials.freq.to_numpy() - h * 74.4))]
    assert row.bin == bin_index
    np.testing.assert_allclose([row.freq, 20*np.log10(row.amp)], [freq, amp_db],
                               rtol=0, atol=2e-6, err_msg=f"harmonic {h}")


@pytest.mark.parametrize("h, expected", list(enumerate(CELLO_PARTIALS, start=1)))
def test_real_cello_partial_prominences(cello_partials, h, expected):
    row = cello_partials.iloc[np.argmin(np.abs(cello_partials.freq.to_numpy() - h * 74.4))]
    assert row.prominence == pytest.approx(expected[3], abs=2e-6)


def test_cello_prominences_match_whole_spectrum_scipy_including_corrected_dc(cello, cello_partials):
    x, sr = cello
    segment = trim(x, sr, 1.5, 2.5)
    weights = signal.get_window("hann", len(segment), fftbins=True)
    amp = np.abs(np.fft.rfft(segment * weights, 4 * len(segment))) * 2 / weights.sum()
    amp[0] /= 2
    amp[-1] /= 2
    db = 20 * np.log10(np.maximum(amp, 1e-6))
    bins = cello_partials.bin.to_numpy(dtype=int)
    reference, left, right = signal.peak_prominences(db, bins)
    np.testing.assert_array_equal(cello_partials.prominence, reference)

    h2 = np.flatnonzero(bins == 593)[0]
    assert (left[h2], right[h2]) == (0, 810)
    assert db[0] == pytest.approx(-116.588218914, abs=1e-9)
    assert reference[h2] == pytest.approx(86.614473642, abs=1e-9)
