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
from audiospylt.viz._common import pitch_hover_label, pitch_hover_labels, theme_colors
from audiospylt.viz.tables import _nice_ticks, plot_events, plot_events_3d, plot_snapshot

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


def _vertices(trace):
    return [(x, y, z) for x, y, z in zip(trace.x, trace.y, trace.z) if x is not None]


def _db_of(amp, floor):
    return max(float(amplitude_to_db(amp, floor_db=-120.0)), floor)


def test_plot_events_3d_db_default_range_floor_and_layout():
    table = events_table({
        "time_start": [0.0, 0.0, 0.25], "time_stop": [1.0, 1.0, 0.75],
        "freq_start": [440.0, 880.0, 63.0], "freq_stop": [440.0, 880.0, 63.0],
        "amp_start": [0.5, 0.001, 1e-7], "amp_stop": [0.5, 0.001, 1e-7],
    })
    fig = plot_events_3d(table)
    scene = fig.layout.scene
    l_max = float(amplitude_to_db(0.5))
    assert scene.zaxis.title.text == "amplitude (dB re full scale)"
    assert scene.zaxis.range[0] == pytest.approx(l_max - 80.0)
    assert scene.zaxis.range[1] == pytest.approx(l_max + 0.03 * 80.0)
    # The floor value is not labelled: no tick shares the corner with another axis.
    assert min(scene.zaxis.tickvals) > scene.zaxis.range[0]
    # The -140 dB row lies on the floor, the others at their levels.
    levels = sorted({round(z, 9) for _, _, z in _vertices(fig.data[0])})
    assert levels == pytest.approx([l_max - 80.0, -60.0, l_max])
    # Tight time range; frequency range starts above 0 Hz, never at a "0" tick.
    assert list(scene.xaxis.range) == [0.0, 1.0]
    assert 0 < scene.yaxis.range[0] < 63.0
    assert scene.yaxis.range[1] > 880.0
    assert scene.aspectmode == "manual"
    assert scene.camera.eye.x < 0 and scene.camera.eye.y < 0
    assert fig.layout.height == 520
    assert fig.layout.margin.l == 0 and fig.layout.margin.b == 0
    hover = fig.data[0].text[0]
    assert hover.startswith("time: 0 s") and "pitch: A4" in hover and " dB" in hover


def test_plot_events_3d_shadow_on_the_floor():
    table = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [440.0], "freq_stop": [660.0],
        "amp_start": [0.5], "amp_stop": [0.5],
    })
    a, b = table, table.assign(freq_start=220.0, freq_stop=220.0)
    fig = plot_events_3d(a, b, labels=["a", "b"])
    lines = [t for t in fig.data if t.showlegend]
    shadows = [t for t in fig.data if not t.showlegend]
    assert [t.name for t in lines] == ["a", "b"]
    assert [t.legendgroup for t in shadows] == ["a", "b"]
    floor = fig.layout.scene.zaxis.range[0]
    for line, shade in zip(lines, shadows):
        assert shade.hoverinfo == "skip"
        assert shade.opacity < 1
        assert list(shade.x) == list(line.x) and list(shade.y) == list(line.y)
        assert all(z == floor for z in shade.z if z is not None)
    assert fig.layout.legend.xanchor == "left"
    assert len(plot_events_3d(a, shadow=False).data) == 1
    linear = plot_events_3d(a, db=False)
    assert all(z == 0.0 for z in linear.data[1].z if z is not None)


def test_plot_events_3d_draws_db_fades_along_the_synthesized_ramp():
    fade = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [440.0], "freq_stop": [440.0],
        "amp_start": [0.5], "amp_stop": [0.0],
    })
    fig = plot_events_3d(fade, shadow=False)
    floor = fig.layout.scene.zaxis.range[0]
    vertices = _vertices(fig.data[0])
    assert len(vertices) > 20
    assert vertices[0] == (0.0, 440.0, pytest.approx(float(amplitude_to_db(0.5))))
    assert vertices[-1] == (1.0, 440.0, floor)
    for t, _, z in vertices:
        # Each vertex lies on the linear amplitude ramp synthesize renders.
        assert z == pytest.approx(_db_of(0.5 * (1.0 - t), floor), abs=1e-9)
    steps = np.diff([z for _, _, z in vertices])
    assert (steps <= 1e-12).all() and (steps >= -3.0 - 1e-9).all()
    # The ramp meets the floor once, then runs along it.
    on_floor = [t for t, _, z in vertices if z == pytest.approx(floor)]
    assert on_floor[0] == pytest.approx(1.0 - 10 ** (floor / 20) / 0.5)
    # Linear amplitude draws a straight row: its two ends only.
    assert len(_vertices(plot_events_3d(fade, db=False, shadow=False).data[0])) == 2


def test_plot_events_3d_log_frequency_axis():
    glide = events_table({
        "time_start": [0.0, 0.0], "time_stop": [2.0, 2.0],
        "freq_start": [220.0, 0.0], "freq_stop": [880.0, 100.0],
        "amp_start": [0.5, 0.5], "amp_stop": [0.5, 0.5],
    })
    with pytest.warns(UserWarning, match="1 row.*frequency <= 0"):
        fig = plot_events_3d(glide, freq_scale="log", shadow=False)
    yaxis = fig.layout.scene.yaxis
    assert yaxis.type == "log"
    assert yaxis.range[0] < np.log10(220.0) and yaxis.range[1] > np.log10(880.0)
    assert list(yaxis.tickvals) == [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]
    assert list(yaxis.ticktext) == ["300", "400", "500", "600", "700", "800", "900"]
    vertices = _vertices(fig.data[0])
    assert len(vertices) >= 25  # 2400 cents in 100-cent steps
    for t, f, _ in vertices:
        assert f == pytest.approx(220.0 + 330.0 * t)  # the linear-Hz path, a curve on the axis
    wide = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [40.0], "freq_stop": [12000.0],
        "amp_start": [0.5], "amp_stop": [0.5],
    })
    ticks = plot_events_3d(wide, freq_scale="log").layout.scene.yaxis
    assert list(ticks.ticktext) == ["50", "100", "200", "500", "1k", "2k", "5k", "10k"]
    with pytest.raises(ValueError, match="freq_scale"):
        plot_events_3d(wide, freq_scale="mel")
    with pytest.raises(ValueError, match="range_db"):
        plot_events_3d(wide, range_db=0)


def test_plot_events_3d_edge_tables_and_ticks():
    silent = events_table({
        "time_start": [0.0], "time_stop": [1.0],
        "freq_start": [440.0], "freq_stop": [440.0],
        "amp_start": [0.0], "amp_stop": [0.0],
    })
    zaxis = plot_events_3d(silent).layout.scene.zaxis
    assert zaxis.range[0] == -120.0
    assert zaxis.range[1] - zaxis.range[0] == pytest.approx(80.0 * 1.03)
    tiny = silent.assign(amp_start=3e-14, amp_stop=1e-14)
    assert list(plot_events_3d(tiny, db=False).layout.scene.zaxis.tickvals) == [1e-14, 2e-14, 3e-14]
    # Ticks are integer multiples of the step: a tick at zero is exactly 0.
    assert _nice_ticks(-0.31, 0.42) == [-0.2, 0.0, 0.2, 0.4]
    assert _nice_ticks(-80.0, 2.4) == [-80.0, -60.0, -40.0, -20.0, 0.0]


def test_plot_events_3d_one_point_tracks_as_dots():
    tracks = tracks_table({
        "track_id": [1, 1, 2], "time": [0.0, 0.5, 0.2],
        "freq": [200.0, 210.0, 300.0], "amp": [0.01, 0.01, 1.0],
    })
    fig = plot_events_3d(tracks)
    dots = [t for t in fig.data if t.mode == "markers"]
    assert len(dots) == 1 and dots[0].legendgroup == "table 1" and not dots[0].showlegend
    assert list(zip(dots[0].x, dots[0].y, dots[0].z)) == [(0.2, 300.0, 0.0)]
    assert [(t, f) for t, f, _ in _vertices(fig.data[0])] == [(0.0, 200.0), (0.5, 210.0)]


def test_pitch_hover_labels_match_the_single_value_label():
    freqs = np.concatenate([
        [0.0, -5.0, np.nan, np.inf, 1e-300, 1e300, 440.0, 466.1637615, 27.5],
        np.geomspace(20.0, 20000.0, 500),
    ])
    assert pitch_hover_labels(freqs) == [pitch_hover_label(f) for f in freqs]
    assert pitch_hover_labels([]) == []


def test_plot_events_log_axis_draws_rows_along_their_hz_path():
    glide = events_table({
        "time_start": [0.0], "time_stop": [2.0],
        "freq_start": [220.0], "freq_stop": [880.0],
        "amp_start": [0.5], "amp_stop": [0.05],
    })
    for amp in (None, "opacity"):
        fig = plot_events(glide, amp=amp, freq_scale="log")
        points = [(x, y) for x, y in zip(fig.data[0].x, fig.data[0].y) if x is not None]
        assert len(points) >= 25
        assert points[0] == (0.0, 220.0) and points[-1] == (2.0, 880.0)
        for t, f in points:
            assert f == pytest.approx(220.0 + 330.0 * t)
        assert len(fig.data[0].text) == len(fig.data[0].x)
        # Linear axes keep the two row ends.
        assert list(plot_events(glide, amp=amp).data[0].x) == [0.0, 2.0, None]
    tracks = tracks_table({
        "track_id": [1, 1, 1], "time": [0.0, 1.0, 1.01],
        "freq": [220.0, 880.0, 881.0], "amp": [0.5, 0.5, 0.5],
    })
    ys = [y for y in plot_events(tracks, amp=None, freq_scale="log").data[0].y if y is not None]
    assert len(ys) >= 26 and ys[-2:] == [880.0, 881.0]


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


def _dots(fig, group):
    return [t for t in fig.data if t.mode == "markers" and t.legendgroup == group]


def test_lines_mode_draws_one_point_tracks_as_dots():
    tracks = tracks_table({
        "track_id": [1, 1, 2], "time": [0.0, 0.5, 0.2],
        "freq": [200.0, 210.0, 300.0], "amp": [0.01, 0.01, 1.0],
    })
    for amp in (None, "opacity"):
        fig = plot_events(tracks, amp=amp, range_db=80.0)
        lines = [t for t in fig.data if t.mode == "lines"]
        dots = _dots(fig, "table 1")
        assert [(x, y) for t in dots for x, y in zip(t.x, t.y)] == [(0.2, 300.0)]
        assert dots[0].text[0].startswith("time: 0.2 s")
        assert [t.showlegend for t in fig.data] == [True, False]
    # The lone dot is the loudest level drawn: the -40 dB segment falls to bin 4 of 80 dB.
    assert lines[0].opacity == pytest.approx(0.15 + 0.85 * 4 / 7)
    assert dots[0].opacity == 1.0


def test_markers_mode_draws_each_point_once():
    events = events_table({
        "time_start": [0.0, 0.5], "time_stop": [0.5, 1.0],
        "freq_start": [100.0, 110.0], "freq_stop": [110.0, 120.0],
        "amp_start": [1.0, 0.5], "amp_stop": [0.5, 1e-10],
    })
    tracks = tracks_table({
        "track_id": [1, 1, 1, 2], "time": [0.0, 0.5, 1.0, 0.2],
        "freq": [100.0, 110.0, 120.0, 300.0], "amp": [1.0, 0.5, 1e-10, 0.1],
    })
    fig = plot_events(events, tracks, mode="markers", range_db=40.0)
    assert {t.mode for t in fig.data} == {"markers"}
    for group, count in (("table 1", 3), ("table 2", 4)):
        dots = _dots(fig, group)
        assert sum(len(t.x) for t in dots) == count  # the shared row end is one dot
        assert [t.showlegend for t in dots].count(True) == 1
    # Opacity from each point's own level: full scale, -6 dB (bin 6 of 40 dB), below the window.
    opacity = {(x, y): t.opacity for t in _dots(fig, "table 1") for x, y in zip(t.x, t.y)}
    assert opacity == {
        (0.0, 100.0): 1.0,
        (0.5, 110.0): pytest.approx(0.15 + 0.85 * 6 / 7),
        (1.0, 120.0): 0.15,
    }

    mixed = plot_events(tracks, events, mode=["markers", "lines"], amp=None)
    assert [(t.legendgroup, t.mode) for t in mixed.data] == [("table 1", "markers"), ("table 2", "lines")]
    both = plot_events(tracks, mode="lines+markers", amp=None)
    assert [(t.mode, t.showlegend) for t in both.data] == [("lines", True), ("markers", False)]
    assert len(both.data[1].x) == 4  # the one-point track is not drawn twice
    empty = plot_events(tracks.iloc[:0], mode="markers")
    assert [(t.mode, t.showlegend, t.x) for t in empty.data] == [("markers", True, (None,))]

    with pytest.raises(ValueError, match="mode must be 'lines', 'markers', or 'lines\\+markers'"):
        plot_events(tracks, mode="dots")
    with pytest.raises(ValueError, match=r"mode length \(1\) must match number of tables \(2\)"):
        plot_events(tracks, events, mode=["markers"])
