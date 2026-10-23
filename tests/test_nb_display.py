"""Notebook display helpers: dispatch, SVG wrapping, play, lazy IPython."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from audiospylt.nb import display as nb
from audiospylt.viz.signal import plot_waveform


class FakeDisplay:
    def __init__(self):
        self.calls = []

    def __call__(self, obj):
        self.calls.append(obj)


def test_show_figure_svg_mei_dataframe(monkeypatch):
    shown = FakeDisplay()
    monkeypatch.setattr("IPython.display.display", shown)
    fig = plot_waveform(np.zeros(8), 8)
    calls = []

    def fake_show(*, renderer=None):
        calls.append(renderer)

    monkeypatch.setattr(fig, "show", fake_show)
    assert nb.show(fig) is None
    assert calls == [None]
    assert nb.show(fig, renderer="json") is None
    assert calls == [None, "json"]

    svg_calls = []
    monkeypatch.setattr(nb, "show_svg", lambda svg, **k: svg_calls.append(svg) or None)
    xml_svg = '<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    nb.show(xml_svg)
    assert svg_calls == [xml_svg]

    with pytest.raises(ValueError, match="show_mei"):
        nb.show("<mei xmlns='http://www.music-encoding.org/ns/mei'></mei>")

    shown.calls.clear()
    frame = pd.DataFrame({"a": [1]})
    # Patch the lazy import path used inside show()
    import IPython.display as ipd
    monkeypatch.setattr(ipd, "display", shown)
    nb.show(frame)
    assert shown.calls == [frame]


def test_show_svg_background(monkeypatch):
    shown = FakeDisplay()
    import IPython.display as ipd
    monkeypatch.setattr(ipd, "display", shown)
    svg = '<?xml version="1.0"?><!DOCTYPE svg PUBLIC "x"><svg></svg>'
    nb.show_svg(svg, background="#ffffff")
    html = shown.calls[0]
    assert "background:#ffffff" in html.data
    assert "<?xml" not in html.data
    assert html.data.endswith("</div>")

    shown.calls.clear()
    nb.show_svg("<svg></svg>", background=None)
    assert shown.calls[0].data == "<svg></svg>" or "SVG" in type(shown.calls[0]).__name__

    with pytest.raises(ValueError, match="background"):
        nb.show_svg("<svg></svg>", background="not a colour!!")


def test_show_mei_pages(monkeypatch):
    pytest.importorskip("verovio")
    pages = ["<svg>1</svg>", "<svg>2</svg>"]
    timelines = []
    monkeypatch.setattr(
        "audiospylt.notation.render.render_svg_pages",
        lambda mei, options=None, timeline=None: timelines.append(timeline) or pages,
    )
    monkeypatch.setattr(
        "audiospylt.notation.render.render_svg",
        lambda mei, page=1, options=None, timeline=None: timelines.append(timeline) or pages[page - 1],
    )
    seen = []
    monkeypatch.setattr(nb, "show_svg", lambda svg, **k: seen.append(svg) or None)
    nb.show_mei("<mei/>")
    assert seen == pages
    seen.clear()
    nb.show_mei("<mei/>", page=2, timeline=False)
    assert seen == ["<svg>2</svg>"]
    assert timelines == [None, False]


def _hover_outputs(monkeypatch, mei, **kwargs):
    shown = FakeDisplay()
    import IPython.display as ipd
    monkeypatch.setattr(ipd, "display", shown)
    nb.show_mei(mei, **kwargs)
    return [call.data for call in shown.calls]


def test_show_mei_hover_pairs_cents_labels_not_onsets(monkeypatch):
    pytest.importorskip("verovio")
    import re

    from audiospylt.notation.mei import events_to_mei

    # Two onsets, three notes; each onset label points at the top note, which
    # also has a cents label.
    table = pd.DataFrame({"time_start": [0.0, 0.0, 1.0], "time_stop": [0.5, 0.5, 2.0],
                          "freq_start": [130.0, 445.0, 262.0], "freq_stop": [130.0, 445.0, 262.0],
                          "amp_start": [0.5] * 3, "amp_stop": [0.25] * 3})
    mei = events_to_mei(table, labels=True, time_labels=True)
    (html,) = _hover_outputs(monkeypatch, mei)
    scope = re.match(r'<div id="(asp-mei-[0-9a-f]{12})" style="background:#ffffff;'
                     r'display:inline-block;padding:8px"><style>', html).group(1)
    css = html[html.index("<style>"):html.index("</style>")]
    pairs = re.findall(rf'#{scope}:has\(:is\(g\.note\[id="(n\d+)"\], '
                       rf'g\.dir:not\(\.onset\)\[data-startid="#\1"\]\):hover\)', css)
    assert pairs == ["n1", "n2", "n3"]
    assert css.endswith("{ fill: #eb6834; }")
    # The SVG keeps the pairing attributes the CSS selects on.
    assert html.count('class="dir cents" data-startid="#n') == 3
    assert html.count('class="dir onset" data-startid="#n') == 2
    # A second call gets its own scope: notebooks repeat ids n1… across scores.
    (again,) = _hover_outputs(monkeypatch, mei, highlight="rgb(0, 90, 200)", background=None)
    assert f'id="{scope}"' not in again
    assert re.match(r'<div id="asp-mei-[0-9a-f]{12}" style="display:inline-block"><style>', again)
    assert again[:again.index("</style>")].endswith("{ fill: rgb(0, 90, 200); }")


def test_show_mei_without_pairs_or_highlight_is_plain(monkeypatch):
    pytest.importorskip("verovio")
    from audiospylt.notation import render
    from audiospylt.notation.mei import snapshot_to_mei

    seen = []
    real = render.render_svg_pages
    monkeypatch.setattr(render, "render_svg_pages",
                        lambda mei, options=None, timeline=None:
                        seen.append(options) or real(mei, options=options, timeline=timeline))
    labelled = snapshot_to_mei(pd.DataFrame({"freq": [261.6, 330.0], "amp": [1.0, 0.5]}), labels=True)
    unlabelled = snapshot_to_mei(pd.DataFrame({"freq": [261.6, 330.0], "amp": [1.0, 0.5]}))
    options = {"scale": 40, "svgAdditionalAttribute": ["note@pname"]}
    before = {"scale": 40, "svgAdditionalAttribute": ["note@pname"]}
    for html in (_hover_outputs(monkeypatch, labelled, highlight=None)
                 + _hover_outputs(monkeypatch, unlabelled, options=options)):
        assert html.startswith('<div style="background:#ffffff;display:inline-block;padding:8px"><svg')
        assert "<style>" not in html
    assert options == before
    assert seen == [None, {"scale": 40, "svgAdditionalAttribute": ["note@pname", "dir@startid"]}]
    with pytest.raises(ValueError, match="highlight"):
        nb.show_mei(labelled, highlight="red; } body { x")
    with pytest.raises(ValueError, match="background"):
        nb.show_mei(labelled, background="url(x)")
    assert len(seen) == 2  # colours are checked before rendering


def test_hover_notes_skip_onsets_external_and_missing_targets():
    svg = """<svg><g id="n1" class="note"><g class="notehead"/></g>
    <g id="n2" class="note"/><g id="x" class="dir cents" data-startid="#n2">
    <g id="y" class="dir onset" data-startid="#n1"/><g id="z" class="dir" data-startid="#n9"/>
    <g id="w" class="dir" data-startid="other.mei#n1"/><g id="v" class="dir" data-startid="#n2"/>
    <g id="n3" class="note"/><g id="u" class="dir" data-startid="#n3"/></svg>"""
    # n1 has only an onset label and an external reference, so it is not a pair.
    assert nb._hover_notes(svg) == ["n2", "n3"]
    assert nb._hover_css("<svg/>", "s", "red") == ""


def test_show_mei_timeline_hover_links_onsets(monkeypatch):
    pytest.importorskip("verovio")
    import re

    from audiospylt.notation.mei import events_to_mei

    table = pd.DataFrame({"time_start": [0.0, 0.0, 1.0], "time_stop": [0.5, 0.5, 2.0],
                          "freq_start": [130.0, 445.0, 262.0], "freq_stop": [130.0, 445.0, 262.0],
                          "amp_start": [0.5] * 3, "amp_stop": [0.25] * 3})
    mei = events_to_mei(table, labels=True, time_labels=True)
    (html,) = _hover_outputs(monkeypatch, mei)
    scope = re.match(r'<div id="(asp-mei-[0-9a-f]{12})"', html).group(1)
    css = html[html.index("<style>"):html.index("</style>")]
    assert nb._hover_measures(html) == ["m1", "m3"]
    for m in ("m1", "m3"):
        notes, when = f'g.measure[id="{m}"]', f'g.asp-when[data-measure="{m}"]'
        anything = f":is({notes} g.note, {when}, {notes} g.dir.onset)"
        assert f"#{scope}:has({anything}:hover) {when} .asp-connector" in css
        assert f"#{scope}:has({anything}:hover) :is({when} .asp-dot, {notes} g.dir.onset)" in css
        assert (f"#{scope}:has(:is({when}, {notes} g.dir.onset):hover) "
                f":is({notes} g.note, {notes} g.dir.cents)") in css
    # Connectors are strokes, everything else fills; the cents pairs stay as before.
    strokes, fills = css.removeprefix("<style>").split("; }\n")
    assert strokes.endswith("{ stroke: #eb6834") and fills.endswith("{ fill: #eb6834; }")
    assert ".asp-connector" in strokes and ".asp-connector" not in fills
    assert strokes.count(":hover)") == 2 and fills.count(":hover)") == 2 * 2 + 3
    # The timeline alone (no cents labels) still gets hover; timeline=False drops it.
    (unlabelled,) = _hover_outputs(monkeypatch, events_to_mei(table))
    assert "<style>" in unlabelled and "g.dir:not(.onset)" not in unlabelled
    (plain,) = _hover_outputs(monkeypatch, events_to_mei(table), timeline=False)
    assert "asp-when" not in plain and "<style>" not in plain


def test_hover_measures_skip_duplicates_and_missing_measures():
    svg = """<svg><g id="m1" class="measure"/><g id="m3" class="measure"/>
    <g class="asp-when" data-when="#w1" data-measure="m1"/><g class="asp-when" data-measure="m1"/>
    <g class="asp-when" data-measure="m9"/><g class="asp-when"/><g class="asp-ruler" data-measure="m3"/>
    <g class="asp-when" data-measure="m3"/></svg>"""
    assert nb._hover_measures(svg) == ["m1", "m3"]


def test_play_mono_stereo_and_peak(monkeypatch):
    shown = FakeDisplay()
    import IPython.display as ipd
    monkeypatch.setattr(ipd, "display", shown)

    class FakeAudio:
        def __init__(self, data=None, rate=None, normalize=False):
            self.data = data
            self.rate = rate
            self.normalize = normalize

    monkeypatch.setattr(ipd, "Audio", FakeAudio)
    mono = np.zeros(10)
    assert nb.play(mono, 100) is None
    assert isinstance(shown.calls[0], FakeAudio)
    assert shown.calls[0].data.shape == (10,)
    stereo = np.zeros((2, 10))
    nb.play(stereo, 100)
    assert shown.calls[1].data.shape == (2, 10)
    hot = np.array([2.0])
    with pytest.raises(ValueError, match="normalize=True"):
        nb.play(hot, 100, normalize=False)
    nb.play(hot, 100, normalize=True)


def test_ipython_missing_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    # Force re-import failure inside the helper
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("IPython"):
            raise ImportError("hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"audiospylt\[notebook\]"):
        nb.show("hello")
    with pytest.raises(ImportError, match=r"audiospylt\[notebook\]"):
        nb.play(np.zeros(4), 8)
