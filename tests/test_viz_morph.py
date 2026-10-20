"""Morph-map figure contracts against speech1 ↔ conga3 pairings."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from audiospylt.io.tables import read_table
from audiospylt.transform.morph import pair
from audiospylt.viz.morph import plot_morph_map

DATA = Path(__file__).resolve().parents[1] / "data" / "tables"

# Expected category counts checked 2026-10-20 (speech1 → conga3):
# kept / birth / death / silent / nrows
CASES = [
    ("equal", {}, (10, 0, 31, 0, 41)),
    ("power", {"power": 1.5}, (10, 0, 31, 0, 41)),
    ("sigmoid", {"sigmoid_range": 3}, (10, 0, 31, 0, 41)),
    ("transport", {"weight": "amp"}, (10, 0, 31, 9, 50)),
    ("transport", {"weight": "power"}, (4, 6, 37, 3, 50)),
]


def _load():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_table(DATA / "speech1.tsv"), read_table(DATA / "conga3.tsv")


def _counts(pairing):
    kept = birth = death = silent = 0
    for _, row in pairing.iterrows():
        start, stop = bool(row.primary_start), bool(row.primary_stop)
        if start and stop:
            kept += 1
        elif stop:
            birth += 1
        elif start:
            death += 1
        else:
            silent += 1
    return kept, birth, death, silent


@pytest.mark.parametrize("method,kwargs,expected", CASES)
@pytest.mark.parametrize("reverse", [False, True], ids=["fwd", "rev"])
def test_morph_map_categories_and_widths(method, kwargs, expected, reverse, capsys, monkeypatch):
    monkeypatch.setattr(
        "plotly.basedatatypes.BaseFigure.show",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("show")),
    )
    speech, conga = _load()
    source, target = (conga, speech) if reverse else (speech, conga)
    before = source.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairing = pair(source, target, method, **kwargs)
    kept, birth, death, silent, nrows = expected
    if reverse:
        birth, death = death, birth
    assert len(pairing) == nrows
    assert _counts(pairing) == (kept, birth, death, silent)

    fig = plot_morph_map(pairing)
    assert capsys.readouterr() == ("", "")
    pio.to_json(fig)
    pd.testing.assert_frame_equal(source, before)

    line_traces = [t for t in fig.data if t.mode == "lines"]
    assert len(line_traces) == nrows
    present = {t.legendgroup for t in line_traces if t.showlegend}
    expected_present = {
        name for name, count in zip(
            ("kept", "birth", "death", "silent"), (kept, birth, death, silent),
        ) if count
    }
    assert present == expected_present

    masses = pairing["mass"].to_numpy()
    equal = float(np.max(np.abs(masses - masses[0]))) <= 1e-12 * max(1.0, abs(float(masses[0])))
    if equal:
        assert all(t.line.width == 2 for t in line_traces)
    else:
        silent_widths = [t.line.width for t in line_traces if t.legendgroup == "silent"]
        assert silent_widths and all(w == 1 for w in silent_widths)
        other = [t.line.width for t in line_traces if t.legendgroup != "silent"]
        assert max(other) >= min(other)

    marker_traces = [t for t in fig.data if t.mode == "markers"]
    assert len(marker_traces) == 2
    assert len(marker_traces[0].x) == pairing["src"].nunique()
    assert len(marker_traces[1].x) == pairing["dst"].nunique()


def test_morph_map_validation():
    speech, conga = _load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairing = pair(speech, conga, "equal")
    bad = pairing.drop(columns=["mass"])
    with pytest.raises(ValueError, match="mass"):
        plot_morph_map(bad)
    with pytest.raises(ValueError, match="freq_scale"):
        plot_morph_map(pairing, freq_scale="mel")


def test_morph_map_draw_order_and_legendrank(capsys, monkeypatch):
    monkeypatch.setattr(
        "plotly.basedatatypes.BaseFigure.show",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("show")),
    )
    speech, conga = _load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairing = pair(speech, conga, "transport", weight="power")
    before = pairing.copy(deep=True)
    fig = plot_morph_map(pairing)
    assert capsys.readouterr() == ("", "")
    pio.to_json(fig)
    pd.testing.assert_frame_equal(pairing, before)

    line_groups = [t.legendgroup for t in fig.data if t.mode == "lines"]
    # Draw order: silent, death, birth, kept (kept on top); markers last.
    assert line_groups == sorted(
        line_groups, key=lambda g: {"silent": 0, "death": 1, "birth": 2, "kept": 3}[g],
    )
    assert all(t.mode == "markers" for t in fig.data if t.mode != "lines")
    assert fig.data[-1].mode == "markers" and fig.data[-2].mode == "markers"

    ranks = {
        t.legendgroup: t.legendrank
        for t in fig.data if t.mode == "lines" and t.showlegend
    }
    assert ranks["kept"] < ranks["birth"] < ranks["death"] < ranks["silent"]
