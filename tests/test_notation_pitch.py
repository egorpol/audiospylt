"""Absolute quantisation and exact-remainder acceptance tests for v1 notation."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from audiospylt.notation.pitch import RESOLUTIONS, SPELLINGS, cents_label, pitch_table


PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
SHARP = [
    ("c", ""), ("c", "1qs"), ("c", "s"), ("c", "3qs"),
    ("d", ""), ("d", "1qs"), ("d", "s"), ("d", "3qs"),
    ("e", ""), ("e", "1qs"), ("f", ""), ("f", "1qs"),
    ("f", "s"), ("f", "3qs"), ("g", ""), ("g", "1qs"),
    ("g", "s"), ("g", "3qs"), ("a", ""), ("a", "1qs"),
    ("a", "s"), ("a", "3qs"), ("b", ""), ("b", "1qs"),
]
FLAT = [
    ("c", ""), ("d", "3qf"), ("d", "f"), ("d", "1qf"),
    ("d", ""), ("e", "3qf"), ("e", "f"), ("e", "1qf"),
    ("e", ""), ("f", "1qf"), ("f", ""), ("g", "3qf"),
    ("g", "f"), ("g", "1qf"), ("g", ""), ("a", "3qf"),
    ("a", "f"), ("a", "1qf"), ("a", ""), ("b", "3qf"),
    ("b", "f"), ("b", "1qf"), ("b", ""), ("c", "1qf"),
]


def hz(midi, a4=440.0):
    return a4 * 2 ** ((np.asarray(midi) - 69) / 12)


@pytest.fixture(scope="module")
def sweep():
    bases = np.array([12 * (octave + 1) + pc for octave in range(1, 9) for pc in PC.values()])
    offsets = np.concatenate((np.arange(-150, 150.5, 0.5),
                              np.array([-150, -125, -100, -75, -50, -25, 25, 50, 75, 100, 125, 150])))
    # Construct as f_base * 2**(cents/1200), deliberately including log2 noise.
    frequencies = (hz(bases)[:, None] * 2 ** (offsets[None, :] / 1200)).ravel()
    half_cents = (bases[:, None] * 200 + offsets[None, :] * 2).astype(np.int64).ravel()
    return frequencies, half_cents


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_exhaustive_naturals_c1_to_b8_and_offsets(sweep, spelling, resolution):
    frequencies, half_cents = sweep
    grid_half_cents = 200 if resolution == "semitone" else 100
    # Integer arithmetic makes the independent half-up oracle exact.
    expected = ((half_cents + grid_half_cents // 2) // grid_half_cents) * grid_half_cents / 200
    result = pitch_table(frequencies, resolution=resolution, spelling=spelling)
    np.testing.assert_array_equal(result.freq, frequencies)
    np.testing.assert_allclose(result.midi, half_cents / 200, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.written_midi, expected)
    np.testing.assert_array_equal(
        result.written_midi, 12 * (result.octave + 1) + result.step.map(PC) + result.alter_cents / 100)
    np.testing.assert_array_equal(result.residual, (result.midi - result.written_midi) * 100)
    limit = 50 if resolution == "semitone" else 25
    assert np.max(np.abs(result.residual)) <= limit + 1e-6
    # Exact ties always choose the higher written pitch: a negative remainder.
    ties = half_cents % grid_half_cents == grid_half_cents // 2
    np.testing.assert_allclose(result.residual[ties], -limit, rtol=0, atol=1e-9)
    ideal_residual = (half_cents / 200 - expected) * 100
    np.testing.assert_array_equal(np.sign(result.residual[abs(ideal_residual) > 1e-8]),
                                  np.sign(ideal_residual[abs(ideal_residual) > 1e-8]))
    allowed = {"", "s", "1qs", "3qs"} if spelling == "sharp" else {"", "f", "1qf", "3qf"}
    if resolution == "semitone":
        allowed &= {"", "s", "f"}
    assert set(result.accid) <= allowed


@pytest.mark.parametrize("spelling, expected", [("sharp", SHARP), ("flat", FLAT)])
def test_literal_24_pitch_class_spelling_table(spelling, expected):
    result = pitch_table(hz(np.arange(60, 72, 0.5)), spelling=spelling)
    assert list(zip(result.step, result.accid, strict=True)) == expected
    assert result.octave.tolist() == [4] * 23 + [5 if spelling == "flat" else 4]
    alter = {"": 0, "s": 100, "f": -100, "1qs": 50, "3qs": 150, "1qf": -50, "3qf": -150}
    assert result.alter_cents.tolist() == [alter[a] for _, a in expected]


@pytest.mark.parametrize("spelling, expected", [
    ("sharp", [("b", 4, "1qs"), ("b", 3, "1qs"), ("c", 4, "1qs"), ("c", 5, "1qs")]),
    ("flat", [("c", 5, "1qf"), ("c", 4, "1qf"), ("d", 4, "3qf"), ("d", 5, "3qf")]),
])
def test_octave_crossings_and_mirrors(spelling, expected):
    result = pitch_table(hz([71.5, 59.5, 60.5, 72.5]), spelling=spelling)
    assert list(zip(result.step, result.octave, result.accid, strict=True)) == expected


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_eighth_mode_accuracy_for_entire_sweep(sweep, spelling, decimals):
    result = pitch_table(sweep[0], resolution="eighth", spelling=spelling)
    labels = [cents_label(r, decimals) for r in result.residual]
    reconstituted = result.written_midi * 100 + np.array(labels, dtype=float)
    assert np.max(np.abs(reconstituted - result.midi * 100)) <= 0.5 * 10 ** -decimals + 1e-9
    assert all(label == f"{0:.{decimals}f}" for label in labels if float(label) == 0)


@pytest.mark.parametrize("resolution, grid", [("semitone", 1), ("quarter", 0.5), ("eighth", 0.5)])
def test_tie_tolerance_and_nearby_values(resolution, grid):
    # Include negative MIDI and offsets away from A, above and below the tie.
    ties = np.array([-12, 0, 60, 69, 120]) + grid / 2
    offsets = np.array([-2e-9, -0.5e-9, 0, 0.5e-9, 2e-9]) * grid
    values = (ties[:, None] + offsets).ravel()
    result = pitch_table(hz(values), resolution=resolution)
    expect = np.repeat(ties + grid / 2, 5)
    expect[::5] -= grid
    np.testing.assert_array_equal(result.written_midi, expect)


def test_residual_does_not_wrap_after_integer_rounding():
    result = pitch_table(hz([69.495, 69.5]), resolution="semitone")
    np.testing.assert_allclose(result.residual, [49.5, -50], atol=1e-9)
    result = pitch_table([440 * 2 ** (25 / 1200), 440 * 2 ** (-25 / 1200)])
    assert result.written_midi.tolist() == [69.5, 69]
    np.testing.assert_allclose(result.residual, [-25, -25], atol=1e-9)


@pytest.mark.parametrize("freq", [440, [440, 220], np.array([440, 220]),
                                  pd.Series([440, 220], index=["x", "y"])])
def test_input_shapes_order_dtypes_and_reference(freq):
    result = pitch_table(freq, a4=442)
    assert list(result) == ["freq", "midi", "written_midi", "step", "octave", "accid", "alter_cents", "residual"]
    assert [str(d) for d in result.dtypes] == [
        "float64", "float64", "float64", "object", "int64", "object", "float64", "float64"]
    np.testing.assert_allclose(result.midi, 69 + 12 * np.log2(result.freq / 442))
    assert pitch_table(442, a4=Decimal(442)).midi.tolist() == [69]


def test_empty_and_extreme_positive_finite_inputs():
    assert pitch_table([]).empty
    for a4 in (np.nextafter(0.0, 1.0), 1e308, 440):
        result = pitch_table([np.nextafter(0.0, 1.0), 1e308], a4=a4)
        assert np.isfinite(result.midi).all()
        assert np.isfinite(result.residual).all()


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(False),
                                "440", None, 440 + 0j, pd.NA, 10**400])
def test_bad_frequencies_report_count_and_positions(bad):
    with pytest.raises(ValueError, match=r"2 value\(s\); first positions \[1, 3\]"):
        pitch_table(pd.Series([440, bad, 220, bad], index=[8, 7, 6, 5], dtype=object))
    with pytest.raises(ValueError, match=r"1 value\(s\); first positions \[0\]"):
        pitch_table(bad)


def test_multidimensional_frequency_input_rejected():
    with pytest.raises(ValueError, match="1-D"):
        pitch_table([[440]])
    with pytest.raises(ValueError, match=r"2 value\(s\); first positions \[0, 1\]"):
        pitch_table([[440], [220, 330]])


@pytest.mark.parametrize("name, bad, allowed", [
    ("resolution", "tone", RESOLUTIONS), ("resolution", ["quarter"], RESOLUTIONS),
    ("spelling", "mixed", SPELLINGS), ("spelling", None, SPELLINGS),
])
def test_invalid_tokens_list_allowed_values(name, bad, allowed):
    with pytest.raises(ValueError) as caught:
        pitch_table(440, **{name: bad})
    assert all(token in str(caught.value) for token in allowed)


@pytest.mark.parametrize("a4", [True, np.bool_(True), 0, -440, np.nan, np.inf, "440", [440], 440j])
def test_invalid_reference(a4):
    with pytest.raises(ValueError, match="a4"):
        pitch_table(440, a4=a4)


@pytest.mark.parametrize("residual, decimals, expected", [
    (12.3, 0, "+12"), (-3.2, 0, "-3"), (0, 0, "0"), (-0.0, 0, "0"),
    (-0.49, 0, "0"), (0.49, 0, "0"), (12.3, 1, "+12.3"), (-0.4, 1, "-0.4"),
    (-0.01, 1, "0.0"), (0.01, 1, "0.0"), (-0.0, 2, "0.00"),
])
def test_cents_label_format(residual, decimals, expected):
    assert cents_label(residual, decimals) == expected
    assert cents_label(residual, np.int64(decimals)) == expected


@pytest.mark.parametrize("decimals", [-1, 1.0, True, np.bool_(False), "1", None])
def test_bad_label_decimals(decimals):
    with pytest.raises(ValueError, match="decimals"):
        cents_label(1, decimals)


@pytest.mark.parametrize("residual", [np.nan, np.inf, -np.inf, True, "1", 1j])
def test_bad_label_residual(residual):
    with pytest.raises(ValueError, match="residual"):
        cents_label(residual)
