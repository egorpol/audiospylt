"""Snapshot, events, and tracks figure contracts."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from audiospylt.analysis.spectrum import amplitude_to_db
from audiospylt.io.tables import read_table
from audiospylt.model.schema import validate
from audiospylt.viz._common import theme_colors
from audiospylt.viz.tables import plot_events, plot_events_3d, plot_snapshot

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "tables"


def snapshot(freq, amp=None):
    amp = np.ones(len(freq)) if amp is None else amp
    return validate(pd.DataFrame({"freq": freq, "amp": amp}), "snapshot")


def events_table(rows):
    return validate(pd.DataFrame(rows), "events")


def tracks_table(rows):
    return validate(pd.DataFrame(rows), "tracks")


def test_plot_snapshot_stems_and_legend(capsys, monkeypatch):
    shown = []
    monkeypatch.setattr(
        "plotly.basedatatypes.BaseFigure.show",
        lambda *a, **k: shown.append(1) or (_ for _ in ()).throw(AssertionError("show")),
    )
    a = snapshot([100.0, 200.0], [0.5, 0.25])
    b = snapshot([150.0], [0.1])
    before = a.copy(deep=True)
    fig = plot_snapshot(a, b, labels=["A", "B"], floor_db=-120.0)
    assert capsys.readouterr() == ("", "")
    assert shown == []
    pio.to_json(fig)
    pd.testing.assert_frame_equal(a, before)
    # two traces per snapshot (stems + markers)
    assert len(fig.data) == 4
    colours = theme_colors("light")["colorway"]
    assert fig.data[0].line.color == colours[0]
    assert fig.data[2].line.color == colours[1]
    # stems from baseline
    assert fig.data[0].y[0] == pytest.approx(-120.0)
    assert fig.data[0].y[1] == pytest.approx(float(amplitude_to_db(0.5)))
    # empty snapshot keeps legend
    empty = snapshot([], [])
    # empty validate may fail — use empty validated via columns
    empty = validate(pd.DataFrame({"freq": pd.Series(dtype=np.float64),
                                   "amp": pd.Series(dtype=np.float64)}), "snapshot")
    fig_e = plot_snapshot(empty, labels=["empty"])
    assert any(t.showlegend for t in fig_e.data)
    with pytest.raises(ValueError, match="at least one"):
        plot_snapshot()
    with pytest.raises(ValueError, match="labels"):
        plot_snapshot(a, labels=["x", "y"])
    with pytest.raises(ValueError, match="plot_events|snapshot"):
        plot_snapshot(events_table({
            "time_start": [0.0], "time_stop": [1.0],
            "freq_start": [100.0], "freq_stop": [100.0],
            "amp_start": [1.0], "amp_stop": [1.0],
        }))
    with pytest.raises(ValueError, match="facet instead"):
        plot_snapshot(*[snapshot([100.0 + i], [0.1]) for i in range(9)])


def test_plot_events_vertices_tracks_opacity(capsys, monkeypatch):
    monkeypatch.setattr(
        "plotly.basedatatypes.BaseFigure.show",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("show")),
    )
    table = events_table({
        "time_start": [0.0, 1.0], "time_stop": [0.5, 1.5],
        "freq_start": [100.0, 200.0], "freq_stop": [150.0, 250.0],
        "amp_start": [1.0, 0.01], "amp_stop": [0.5, 0.01],
    })
    before = table.copy(deep=True)
    fig = plot_events(table, amp=None)
    assert capsys.readouterr() == ("", "")
    pio.to_json(fig)
    pd.testing.assert_frame_equal(table, before)
    # vertices equal row values (None-separated)
    x = [v for v in fig.data[0].x if v is not None]
    y = [v for v in fig.data[0].y if v is not None]
    assert x == [0.0, 0.5, 1.0, 1.5]
    assert y == [100.0, 150.0, 200.0, 250.0]

    tracks = tracks_table({
        "track_id": [2, 2, 1, 1],
        "time": [0.1, 0.2, 0.0, 1.0],
        "freq": [100.0, 200.0, 300.0, 400.0],
        "amp": [0.1, 0.2, 0.3, 0.4],
    })
    fig_t = plot_events(tracks, amp=None)
    # groupby sort=False keeps first appearance order: track 2 then 1
    y = [v for v in fig_t.data[0].y if v is not None]
    assert y[:2] == [100.0, 200.0]
    assert y[2:] == [300.0, 400.0]

    # Exact opacity bins: full scale → bin 7 (1.0); below window → 0.15; interior k.
    loud = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [200.0], "freq_stop": [200.0],
        "amp_start": [1.0], "amp_stop": [1.0],
    })
    quiet = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [100.0], "freq_stop": [100.0],
        "amp_start": [1e-10], "amp_stop": [1e-10],
    })
    # Interior bin k=3: level = L_max - range_db + (3.5/8)*range_db
    range_db = 40.0
    l_max = 0.0  # 20*log10(1)
    level_k = l_max - range_db + (3.5 / 8.0) * range_db
    amp_k = 10 ** (level_k / 20)
    mid = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [150.0], "freq_stop": [150.0],
        "amp_start": [amp_k], "amp_stop": [amp_k],
    })
    fig_bins = plot_events(quiet, mid, loud, amp="opacity", range_db=range_db)
    by_name = {}
    for t in fig_bins.data:
        if t.showlegend:
            by_name[t.name] = t
        elif t.name not in by_name:
            by_name[t.name] = t
    # Collect opacity per legendgroup (one segment per occupied bin)
    groups = {}
    for t in fig_bins.data:
        groups.setdefault(t.legendgroup, []).append(t.opacity)
    assert 1.0 in groups["table 3"]  # full-scale → bin 7
    assert 0.15 in groups["table 1"]  # below window → bin 0
    assert any(
        o == pytest.approx(0.15 + 0.85 * 3 / 7) for o in groups["table 2"]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        zero = events_table({
            "time_start": [0.0], "time_stop": [1.0],
            "freq_start": [0.0], "freq_stop": [100.0],
            "amp_start": [1.0], "amp_stop": [1.0],
        })
        plot_events(zero, amp=None, freq_scale="log")
    assert len(caught) == 1
    assert "1 row" in str(caught[0].message)

    with pytest.raises(ValueError, match="snapshot_to_events"):
        plot_events(snapshot([100.0], [1.0]))
    with pytest.raises(ValueError, match="amp"):
        plot_events(table, amp="width")
    with pytest.raises(ValueError, match="boolean|range_db"):
        plot_events(table, range_db=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real = read_table(DATA / "conga_multidft.tsv")
    fig3 = plot_events_3d(real)
    assert fig3.data[0].type == "scatter3d"
    assert fig3.layout.scene.zaxis.title.text.startswith("amplitude")


def test_plot_events_3d_linear_z_from_zero():
    table = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [100.0], "freq_stop": [200.0],
        "amp_start": [0.5], "amp_stop": [0.25],
    })
    fig = plot_events_3d(table, db=False)
    assert fig.layout.scene.zaxis.range[0] == 0


def test_snapshot_stems_linear_baseline_and_hover_pitch(capsys, monkeypatch):
    monkeypatch.setattr(
        "plotly.basedatatypes.BaseFigure.show",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("show")),
    )
    snap = snapshot([440.0, 0.0], [0.5, 0.25])
    before = snap.copy(deep=True)
    fig = plot_snapshot(snap, db=False, floor_db=-120.0)
    assert capsys.readouterr() == ("", "")
    pio.to_json(fig)
    pd.testing.assert_frame_equal(snap, before)
    # stems from linear baseline 0
    assert fig.data[0].y[0] == pytest.approx(0.0)
    assert fig.data[0].y[1] == pytest.approx(0.5)
    texts = list(fig.data[1].text)
    assert "pitch: A4" in texts[0] and "c" in texts[0]
    assert "pitch:" not in texts[1]


def test_plot_events_and_3d_no_side_effects(capsys, monkeypatch):
    monkeypatch.setattr(
        "plotly.basedatatypes.BaseFigure.show",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("show")),
    )
    table = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [440.0], "freq_stop": [440.0],
        "amp_start": [1.0], "amp_stop": [0.5],
    })
    before = table.copy(deep=True)
    fig = plot_events(table, amp=None)
    assert capsys.readouterr() == ("", "")
    pio.to_json(fig)
    pd.testing.assert_frame_equal(table, before)
    hover = fig.data[0].text[0]
    assert "pitch: A4" in hover and "c" in hover
    fig3 = plot_events_3d(table)
    assert capsys.readouterr() == ("", "")
    pio.to_json(fig3)
    pd.testing.assert_frame_equal(table, before)
