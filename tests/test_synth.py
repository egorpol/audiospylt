"""Additive synthesis (D9 phase modes, half-open sample grid, Nyquist), snapshot and tracks conversion."""

import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.signal import stft

from audiospylt.analysis.peaks import find_peaks
from audiospylt.analysis.spectrum import spectrum
from audiospylt.model.convert import snapshot_to_events, tracks_to_events
from audiospylt.model.schema import SchemaError, validate
from audiospylt.plot_wave import plot_waves
from audiospylt.synth.additive import synthesize

SR = 44100
COLUMNS = ["time_start", "time_stop", "freq_start", "freq_stop", "amp_start", "amp_stop"]
# One voice: constant, then a glide, then a 68-cent step down. No boundary falls on a sample.
VOICE = [
    (0.0, 0.037, 441.3, 441.3, 0.5, 0.5),
    (0.037, 0.078, 441.3, 612.7, 0.5, 0.4),
    (0.078, 0.107, 589.1, 589.1, 0.4, 0.4),
]


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def _events(rows, **extra):
    table = pd.DataFrame(rows, columns=COLUMNS, dtype=float)
    for name, values in extra.items():
        table[name] = values
    return table


def _render(voices, n_samples, starts=None):
    """Sum voices (lists of rows) with the phase integrated across each voice's rows."""
    t = np.arange(n_samples) / SR
    out = np.zeros(n_samples)
    for v, voice in enumerate(voices):
        cycles = 0.0 if starts is None else starts[v]
        for ts, te, f0, f1, a0, a1 in voice:
            inside = (t >= ts) & (t < te)
            tau, span = t[inside] - ts, te - ts
            amp = a0 + (a1 - a0) * tau / span
            out[inside] += amp * np.sin(2 * np.pi * (cycles + f0 * tau + 0.5 * (f1 - f0) * tau**2 / span))
            cycles += span * (f0 + f1) / 2
    return out


def _legacy(rows):
    return _events(rows).rename(columns={"amp_start": "amp_min", "amp_stop": "amp_max"})


def _lsd(ref, y):
    """RMS STFT dB distance with a floor 80 dB below the reference peak (2026-10-10 audit)."""
    _, _, sx = stft(ref, fs=SR, nperseg=2048, noverlap=1536, boundary=None, padded=False)
    _, _, sy = stft(y, fs=SR, nperseg=2048, noverlap=1536, boundary=None, padded=False)
    floor = np.abs(sx).max() * 1e-4
    dx = 20 * np.log10(np.maximum(np.abs(sx), floor))
    dy = 20 * np.log10(np.maximum(np.abs(sy), floor))
    return float(np.sqrt(np.mean((dx - dy) ** 2)))


# ---------------------------------------------------------------- sample grid


def test_abutting_events_cover_each_sample_once():
    one = _events([(0.0, 0.1, 1001.3, 1001.3, 0.5, 0.5)])
    two = _events([(0.0, 0.05, 1001.3, 1001.3, 0.5, 0.5), (0.05, 0.1, 1001.3, 1001.3, 0.5, 0.5)])
    for phase in ("absolute", "continuous"):
        np.testing.assert_allclose(synthesize(two, SR, phase=phase), synthesize(one, SR, phase=phase),
                                   rtol=0, atol=1e-12)
    assert synthesize(one, SR).shape == (4410,)
    # 1001.3 Hz * 50 ms is not a whole number of cycles, so a reset is audible.
    assert np.abs(synthesize(two, SR, phase="reset") - synthesize(one, SR, phase="reset")).max() > 0.1


def test_membership_is_half_open_and_absorbs_float_noise():
    assert 1.1 * SR > 48510  # 48510.00000000001: a plain ceil would skip sample 48510
    y = synthesize(_events([(0.7, 1.1, 441.0, 441.0, 1.0, 1.0)]), SR, phase="absolute")
    assert y.shape == (48510,)
    n = np.arange(30870, 48510)
    assert not y[:30870].any()
    np.testing.assert_allclose(y[30870:], np.sin(2 * np.pi * 441.0 * n / SR), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- phase modes (D9)


def test_continuous_integrates_phase_across_a_voice():
    n = 4719  # ceil(0.107 * 44100)
    y = synthesize(_events(VOICE), SR)
    np.testing.assert_allclose(y, _render([VOICE], n), rtol=0, atol=1e-9)
    np.testing.assert_allclose(synthesize(_events(VOICE), SR, link_cents=None), y, rtol=0, atol=0)


def test_link_cents_limits_continuous_links():
    y = synthesize(_events(VOICE), SR, link_cents=50)  # the 68-cent step is not linked
    expected = _render([VOICE[:2], VOICE[2:]], 4719)
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-9)


def test_reset_starts_every_row_at_zero_and_matches_06_reset_and_chirp():
    y = synthesize(_events(VOICE), SR, phase="reset")
    np.testing.assert_allclose(y, _render([[row] for row in VOICE], 4719), rtol=0, atol=1e-9)
    for mode in ("reset", "chirp"):
        old = plot_waves(_legacy(VOICE), k=1 / SR, interval_mode="half_open", phase_mode=mode,
                         show_waveform=False)
        np.testing.assert_allclose(y, old[: y.size], rtol=0, atol=1e-9)


def test_absolute_uses_global_time_and_matches_06_global():
    y = synthesize(_events(VOICE), SR, phase="absolute")
    starts = [f0 * ts for ts, _, f0, *_ in VOICE]
    np.testing.assert_allclose(y, _render([[row] for row in VOICE], 4719, starts), rtol=0, atol=1e-9)
    t = np.arange(1632) / SR  # constant first row: plain sin(2 pi f t)
    np.testing.assert_allclose(y[:1632], 0.5 * np.sin(2 * np.pi * 441.3 * t), rtol=0, atol=1e-9)
    old = plot_waves(_legacy(VOICE), k=1 / SR, interval_mode="half_open", phase_mode="global",
                     show_waveform=False)
    np.testing.assert_allclose(y, old[: y.size], rtol=0, atol=1e-9)


def test_the_three_modes_differ():
    ys = {mode: synthesize(_events(VOICE), SR, phase=mode) for mode in ("continuous", "reset", "absolute")}
    for a, b in (("continuous", "reset"), ("continuous", "absolute"), ("reset", "absolute")):
        assert np.abs(ys[a] - ys[b]).max() > 0.01


def test_links_are_one_to_one_nearest_first():
    rows = [(0.0, 0.05, 440.0, 440.0, 0.3, 0.3), (0.0, 0.05, 452.0, 452.0, 0.3, 0.3),
            (0.05, 0.1, 450.0, 450.0, 0.3, 0.3), (0.05, 0.1, 445.0, 445.0, 0.3, 0.3)]
    # cents: 452->450 (7.7) first, then 440->445 (19.6)
    expected = _render([[rows[0], rows[3]], [rows[1], rows[2]]], 4410)
    np.testing.assert_allclose(synthesize(_events(rows), SR), expected, rtol=0, atol=1e-9)
    crossed = _render([[rows[0], rows[2]], [rows[1], rows[3]]], 4410)
    assert np.abs(expected - crossed).max() > 0.01


def test_track_id_restricts_links():
    rows = [(0.0, 0.05, 440.0, 440.0, 0.3, 0.3), (0.0, 0.05, 452.0, 452.0, 0.3, 0.3),
            (0.05, 0.1, 450.0, 450.0, 0.3, 0.3), (0.05, 0.1, 445.0, 445.0, 0.3, 0.3)]
    expected = _render([[rows[0], rows[2]], [rows[1], rows[3]]], 4410)
    for ids in ([1, 2, 1, 2], [1.0, 2.0, 1.0, 2.0], pd.array([1, 2, 1, 2], dtype="Int64")):
        y = synthesize(_events(rows, track_id=ids), SR)
        np.testing.assert_allclose(y, expected, rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="track_id"):
        synthesize(_events(rows, track_id=[1, 2, np.nan, 2]), SR)
    with pytest.raises(ValueError, match="track_id"):
        synthesize(_events(rows, track_id=[1, 2, 1.5, 2]), SR)
    with pytest.raises(ValueError, match="track_id"):
        synthesize(_events(rows, track_id=["1", "2", "1", "2"]), SR)
    # Only continuous mode reads track_id.
    synthesize(_events(rows, track_id=[1, 2, np.nan, 2]), SR, phase="reset")


def test_a_gap_of_one_sample_breaks_the_link():
    # The first row covers samples 0..2204; the second starts at 2206, leaving 2205 empty.
    rows = [(0.0, 0.05, 441.3, 441.3, 0.5, 0.5), (0.05 + 1 / SR, 0.1, 441.3, 441.3, 0.5, 0.5)]
    expected = _render([[rows[0]], [rows[1]]], 4410)
    np.testing.assert_allclose(synthesize(_events(rows), SR), expected, rtol=0, atol=1e-9)


def test_track_ids_above_2_53_stay_distinct():
    # read_table keeps such ids as exact Decimals; through float64 they would collapse into one.
    rows = [(0.0, 0.05, 441.3, 441.3, 0.5, 0.5), (0.05, 0.1, 441.3, 441.3, 0.5, 0.5)]
    ids = [Decimal(2**53), Decimal(2**53 + 1)]
    expected = _render([[rows[0]], [rows[1]]], 4410)
    np.testing.assert_allclose(synthesize(_events(rows, track_id=ids), SR), expected, rtol=0, atol=1e-9)


def test_off_grid_rows_meeting_on_one_sample_hand_over_at_the_successor_start():
    # Both times map to boundary sample 2205; the join must equal one unbroken row.
    rows = [(0.0, 2204.5 / SR, 440.0, 440.0, 0.5, 0.5), (2205 / SR, 0.1, 440.0, 440.0, 0.5, 0.5)]
    whole = synthesize(_events([(0.0, 0.1, 440.0, 440.0, 0.5, 0.5)]), SR)
    np.testing.assert_allclose(synthesize(_events(rows), SR), whole, rtol=0, atol=1e-9)


def test_zero_hz_and_zero_length_rows_are_silent_and_never_linked():
    rows = [(0.0, 0.05, 441.3, 441.3, 0.5, 0.5), (0.05, 0.1, 0.0, 0.0, 1.0, 1.0),
            (0.05, 0.05, 441.3, 441.3, 1.0, 1.0), (0.1, 0.15, 441.3, 441.3, 0.5, 0.5)]
    y = synthesize(_events(rows), SR)
    assert not y[2205:4410].any()  # 0 Hz after a 441 Hz row: no DC from an inherited phase
    expected = _render([[rows[0]], [rows[3]]], 6615)
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-9)
    # Only the side touching the boundary must be above 0 Hz: a glide up from 0 Hz still links.
    glide = [(0.0, 0.05, 0.0, 441.3, 0.5, 0.5), (0.05, 0.1, 441.3, 441.3, 0.5, 0.5)]
    np.testing.assert_allclose(synthesize(_events(glide), SR), _render([glide], 4410), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- fades, Nyquist, duration


def _raised_cosine(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def test_fade_ramps_voice_ends_only():
    t = np.arange(4719) / SR
    envelope = _raised_cosine(t / 0.01) * _raised_cosine((0.107 - t) / 0.01)
    y = synthesize(_events(VOICE), SR, fade=0.01)
    np.testing.assert_allclose(y, _render([VOICE], 4719) * envelope, rtol=0, atol=1e-9)
    assert y[0] == 0.0
    # In reset mode every row is its own voice, so every row fades.
    expected = sum(
        _render([[row]], 4719) * _raised_cosine((t - row[0]) / 0.01) * _raised_cosine((row[1] - t) / 0.01)
        for row in VOICE
    )
    np.testing.assert_allclose(synthesize(_events(VOICE), SR, phase="reset", fade=0.01), expected,
                               rtol=0, atol=1e-9)


def test_fades_on_a_short_voice_overlap_and_multiply():
    rows = [(0.0, 0.004, 1000.0, 1000.0, 1.0, 1.0)]
    t = np.arange(177) / SR
    envelope = _raised_cosine(t / 0.005) * _raised_cosine((0.004 - t) / 0.005)
    y = synthesize(_events(rows), SR, fade=0.005)
    np.testing.assert_allclose(y, np.sin(2 * np.pi * 1000 * t) * envelope, rtol=0, atol=1e-9)
    assert np.abs(y).max() < 0.7


def _carrier(rows, n_samples):
    """The voice of `rows` at unit amplitude (continuous phase)."""
    return _render([[(*row[:4], 1.0, 1.0) for row in rows]], n_samples)


@pytest.mark.parametrize("rows, half, left, right", [
    # Constant rows with a 14 dB step: the ramp runs from 0.5 to 0.1 over 0.0475..0.0525 s.
    ([(0.0, 0.05, 1000.0, 1000.0, 0.5, 0.5), (0.05, 0.1, 1003.0, 1003.0, 0.1, 0.1)], 0.0025, 0.5, 0.1),
    # Sloped rows join the ramp at its edges: 0.2 + 0.4 * 0.95 and 0.1 + 0.2 * 0.05.
    ([(0.0, 0.05, 1000.0, 1000.0, 0.2, 0.6), (0.05, 0.1, 1000.0, 990.0, 0.1, 0.3)], 0.0025, 0.58, 0.11),
    # A 4 ms row clamps the ramp to half its length on both sides of the link.
    ([(0.0, 0.004, 1000.0, 1000.0, 0.5, 0.5), (0.004, 0.1, 1000.0, 1000.0, 0.1, 0.1)], 0.002, 0.5, 0.1),
])
def test_link_ramp_replaces_amplitude_steps_with_a_raised_cosine(rows, half, left, right):
    n = 4410
    t = np.arange(n) / SR
    centre = rows[1][0]
    lines = np.where(t < centre, rows[0][4] + (rows[0][5] - rows[0][4]) * t / centre,
                     rows[1][4] + (rows[1][5] - rows[1][4]) * (t - centre) / (0.1 - centre))
    inside = (t >= centre - half) & (t < centre + half)
    ramp = left + (right - left) * _raised_cosine((t - centre + half) / (2 * half))
    envelope = np.where(inside, ramp, lines)
    y = synthesize(_events(rows), SR, link_ramp=0.005)
    np.testing.assert_allclose(y, _carrier(rows, n) * envelope, rtol=0, atol=1e-9)
    np.testing.assert_allclose(synthesize(_events(rows), SR), y, rtol=0, atol=0)  # 5 ms is the default
    # The envelope has no jump: neighbouring samples differ by far less than the 0.4 step.
    assert np.abs(np.diff(envelope)).max() < 0.02
    # link_ramp=0 keeps the hard step.
    np.testing.assert_allclose(synthesize(_events(rows), SR, link_ramp=0), _carrier(rows, n) * lines,
                               rtol=0, atol=1e-9)


def test_link_ramp_needs_a_link_and_a_step():
    step = [(0.0, 0.05, 1000.0, 1000.0, 0.5, 0.5), (0.05, 0.1, 1000.0, 1000.0, 0.1, 0.1)]
    for kwargs in ({"phase": "reset"}, {"phase": "absolute"}, {"link_cents": 0.0, "track_id": [1, 2]}):
        track = kwargs.pop("track_id", None)
        events = _events(step) if track is None else _events(step, track_id=track)
        np.testing.assert_array_equal(synthesize(events, SR, link_ramp=0.01, **kwargs),
                                      synthesize(events, SR, link_ramp=0, **kwargs))
    # Without a step the amplitude lines stay (VOICE has none at its links).
    np.testing.assert_array_equal(synthesize(_events(VOICE), SR, link_ramp=0.01),
                                  synthesize(_events(VOICE), SR, link_ramp=0))


def test_link_ramp_and_voice_fades_multiply():
    rows = [(0.0, 0.05, 1000.0, 1000.0, 0.5, 0.5), (0.05, 0.1, 1000.0, 1000.0, 0.1, 0.1)]
    t = np.arange(4410) / SR
    fades = _raised_cosine(t / 0.03) * _raised_cosine((0.1 - t) / 0.03)
    np.testing.assert_allclose(synthesize(_events(rows), SR, fade=0.03),
                               synthesize(_events(rows), SR) * fades, rtol=0, atol=1e-12)


def test_nyquist_drop_warns_and_alias_folds():
    rows = [(0.0, 0.1, 30000.0, 30000.0, 0.5, 0.5)]
    with pytest.warns(UserWarning, match=r"1 event\(s\) reach the Nyquist frequency 22050 Hz"):
        dropped = synthesize(_events(rows), SR)
    assert dropped.shape == (4410,) and not dropped.any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        folded = synthesize(_events(rows), SR, nyquist="alias")
        synthesize(_events([(0.0, 0.1, 22000.0, 22000.0, 0.5, 0.5)]), SR)
    peaks = find_peaks(spectrum(folded, SR), min_db=-20)
    assert len(peaks) == 1
    assert peaks.freq[0] == pytest.approx(14100.0, abs=0.01)
    assert 20 * np.log10(peaks.amp[0] / 0.5) == pytest.approx(0.0, abs=0.01)


def test_nyquist_drop_gates_samples_not_rows():
    with pytest.warns(UserWarning, match="Nyquist"):
        top = synthesize(_events([(0.0, 0.01, SR / 2, SR / 2, 1.0, 1.0)]), SR, phase="absolute")
    assert not top.any()
    rows = [(0.0, 0.1, 21000.0, 23000.0, 0.5, 0.5)]  # crosses 22050 Hz at 52.5 ms
    with pytest.warns(UserWarning, match="Nyquist"):
        y = synthesize(_events(rows), SR)
    alias = synthesize(_events(rows), SR, nyquist="alias")
    crossing = int(np.ceil(0.0525 * SR))
    np.testing.assert_array_equal(y[:crossing], alias[:crossing])
    assert not y[crossing:].any() and np.abs(y[:crossing]).max() > 0.4


def test_duration_pads_or_cuts():
    events = _events(VOICE)
    full = synthesize(events, SR)
    longer = synthesize(events, SR, duration=0.2)
    assert longer.shape == (8820,)
    np.testing.assert_array_equal(longer[: full.size], full)
    assert not longer[full.size:].any()
    np.testing.assert_array_equal(synthesize(events, SR, duration=0.05), full[:2205])


def test_empty_table():
    empty = _events([])
    assert synthesize(empty, SR).shape == (0,)
    np.testing.assert_array_equal(synthesize(empty, SR, duration=0.01), np.zeros(441))


# ---------------------------------------------------------------- validation


def test_invalid_arguments():
    events = _events(VOICE)
    with pytest.raises(SchemaError, match="upgrade"):
        synthesize(_legacy(VOICE), SR)
    with pytest.raises(SchemaError, match="time_stop"):
        synthesize(_events([(0.1, 0.05, 440.0, 440.0, 1.0, 1.0)]), SR)
    with pytest.raises(ValueError, match="phase='absolute'"):
        synthesize(events, SR, phase="global")
    with pytest.raises(ValueError, match="phase='reset'"):
        synthesize(events, SR, phase="chirp")
    for kwargs in ({"phase": "random"}, {"phase": ["reset"]}, {"nyquist": "fold"}, {"link_cents": -1.0},
                   {"link_cents": np.nan}, {"link_ramp": -0.001}, {"link_ramp": np.nan}, {"link_ramp": "5 ms"},
                   {"fade": -0.1}, {"fade": np.inf}, {"duration": -1.0}):
        with pytest.raises(ValueError):
            synthesize(events, SR, **kwargs)
    for sr in (0, 44100.0, True):
        with pytest.raises(ValueError, match="sr"):
            synthesize(events, sr)


def test_input_table_is_not_modified():
    events = _events(VOICE, track_id=[1, 1, 1])
    before = events.copy()
    synthesize(events, SR, fade=0.01)
    pd.testing.assert_frame_equal(events, before)


# ---------------------------------------------------------------- analysis -> synthesis (acceptance)

F = np.array([223.37, 457.81, 911.29, 1843.43, 3299.61])
A = np.array([0.12, 0.10, 0.08, 0.06, 0.04])


def _test_signal(gliss):
    t = np.arange(SR) / SR
    slopes = np.zeros_like(F)
    slopes[-1] = 7.4 if gliss else 0.0
    return np.sum(A[:, None] * np.sin(2 * np.pi * (F[:, None] * t + 0.5 * slopes[:, None] * t * t)), axis=0)


def _frame_events(x, hop=0.05, frame=0.1):
    frames = []
    for i in range(20):
        start = round(min(max((i + 0.5) * hop - frame / 2, 0.0), 1.0 - frame) * SR)
        peaks = find_peaks(spectrum(x[start:start + round(frame * SR)], SR),
                           min_db=-40, fmin=100, fmax=4000, prominence_db=20)
        frames.append(snapshot_to_events(peaks, i * hop, (i + 1) * hop))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("gliss, phase, limit", [
    (False, "continuous", 0.02), (False, "absolute", 0.1), (True, "continuous", 0.2)])
def test_frame_based_resynthesis_log_spectral_distance(gliss, phase, limit):
    # Measured 2026-10-19: static 0.004 (continuous) / 0.024 dB (absolute), gliss 0.056 dB; reset 16-18 dB.
    x = _test_signal(gliss)
    events = _frame_events(x)
    assert len(events) == 100
    assert _lsd(x, synthesize(events, SR, phase=phase)[: x.size]) < limit


@pytest.mark.parametrize("phase", ["continuous", "reset", "absolute"])
def test_stationary_snapshot_resynthesis(phase):
    x = _test_signal(False)
    events = snapshot_to_events(find_peaks(spectrum(x, SR), min_db=-40, prominence_db=20), 0.0, 1.0)
    assert len(events) == 5
    assert _lsd(x, synthesize(events, SR, phase=phase)) < 0.01  # measured 0.0003 dB



def test_nyquist_drop_never_evaluates_gated_samples():
    events = _events([(0.0, 1.0, 1e308, 1e308, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0, 0.5, 0.5)])
    with pytest.warns(UserWarning, match="1 event"):
        y = synthesize(events, 8)
    np.testing.assert_allclose(y, 0.5 * np.sin(2 * np.pi * np.arange(8) / 8), rtol=0, atol=1e-12)

# ---------------------------------------------------------------- snapshot_to_events


def test_snapshot_to_events_holds_partials_constant():
    snapshot = validate(pd.DataFrame({"freq": [220.0, 440.5], "amp": [0.25, 0.125],
                                      "prominence": [30.0, 20.0], "note": ["a", "b"]}), "snapshot")
    snapshot.attrs["meta"] = {"sr": 44100}
    events = snapshot_to_events(snapshot, 0.5, 1.25)
    assert list(events.columns) == COLUMNS
    expected = _events([(0.5, 1.25, 220.0, 220.0, 0.25, 0.25), (0.5, 1.25, 440.5, 440.5, 0.125, 0.125)])
    pd.testing.assert_frame_equal(events, expected, check_like=False)
    assert events.attrs["audiospylt_table"] == "events"
    assert events.attrs["meta"] == {"sr": 44100}
    events.attrs["meta"]["sr"] = 1
    assert snapshot.attrs["meta"] == {"sr": 44100}


def test_snapshot_to_events_edge_cases():
    empty = snapshot_to_events(pd.DataFrame({"freq": [], "amp": []}), 0.0, 1.0)
    assert list(empty.columns) == COLUMNS and len(empty) == 0
    snapshot = pd.DataFrame({"freq": [100.0], "amp": [0.5]})
    for start, stop in ((1.0, 0.5), (-0.1, 0.5), (0.0, np.inf), (np.nan, 1.0)):
        with pytest.raises(ValueError, match="time_start"):
            snapshot_to_events(snapshot, start, stop)
    with pytest.raises(SchemaError):
        snapshot_to_events(pd.DataFrame({"freq": [100.0]}), 0.0, 1.0)


# ---------------------------------------------------------------- tracks_to_events


def _tracks(rows, **extra):
    table = pd.DataFrame(rows, columns=["track_id", "time", "freq", "amp"])
    for name, values in extra.items():
        table[name] = values
    return table


def test_tracks_to_events_pairs_consecutive_points_of_each_track():
    # Interleaved tracks 3 and 7, single-point track 9; phase and extras are dropped.
    tracks = validate(_tracks([(3, 0.0, 100.0, 0.1), (7, 0.0, 200.0, 0.2), (3, 0.1, 110.0, 0.3),
                               (9, 0.05, 300.0, 0.6), (7, 0.1, 210.0, 0.4), (3, 0.2, 120.0, 0.5)],
                              phase=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], note=list("abcdef")), "tracks")
    tracks.attrs["meta"] = {"method": "pyin"}
    events = tracks_to_events(tracks)
    expected = _events([(0.0, 0.1, 100.0, 110.0, 0.1, 0.3), (0.0, 0.1, 200.0, 210.0, 0.2, 0.4),
                        (0.1, 0.2, 110.0, 120.0, 0.3, 0.5)], track_id=np.array([3, 7, 3], dtype=np.int64))
    pd.testing.assert_frame_equal(events, expected)
    assert events.attrs["audiospylt_table"] == "events"
    assert events.attrs["meta"] == {"method": "pyin"}
    events.attrs["meta"]["method"] = "x"
    assert tracks.attrs["meta"] == {"method": "pyin"}


def test_tracks_to_events_edge_cases():
    empty = tracks_to_events(_tracks([]))
    assert list(empty.columns) == COLUMNS + ["track_id"] and len(empty) == 0
    assert empty["track_id"].dtype == np.int64
    assert tracks_to_events(_tracks([(0, 0.0, 100.0, 0.5), (1, 0.0, 200.0, 0.5)])).empty
    with pytest.raises(SchemaError, match="strictly increasing"):
        tracks_to_events(_tracks([(0, 0.1, 100.0, 0.5), (0, 0.1, 200.0, 0.5)]))
    with pytest.raises(SchemaError, match="track_id"):
        tracks_to_events(pd.DataFrame({"time": [0.0], "freq": [1.0], "amp": [1.0]}))


def test_tracks_resynthesise_as_one_phase_continuous_voice_per_track():
    # Track 0 runs through VOICE's points (off-grid times, a glide, an amplitude change); track 1
    # starts where track 0 stops, on its frequency, and must still begin a new voice at phase 0.
    a = [(0, t, f, amp) for t, f, amp in ((0.0, 441.3, 0.5), (0.037, 441.3, 0.5), (0.078, 612.7, 0.4),
                                          (0.107, 589.1, 0.4))]
    b = [(1, 0.107, 589.1, 0.3), (1, 0.15, 520.0, 0.2)]
    events = tracks_to_events(_tracks(a + b))
    voice_a = [(a[i][1], a[i + 1][1], a[i][2], a[i + 1][2], a[i][3], a[i + 1][3]) for i in range(3)]
    voice_b = [(0.107, 0.15, 589.1, 520.0, 0.3, 0.2)]
    y = synthesize(events, SR)
    np.testing.assert_allclose(y, _render([voice_a, voice_b], y.size), rtol=0, atol=1e-9)
    # Without track_id the abutting tracks would link into one voice.
    assert np.abs(synthesize(events.drop(columns="track_id"), SR, link_ramp=0) - y).max() > 0.1
