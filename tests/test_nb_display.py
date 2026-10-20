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
    monkeypatch.setattr(
        "audiospylt.notation.render.render_svg_pages", lambda mei, options=None: pages,
    )
    monkeypatch.setattr(
        "audiospylt.notation.render.render_svg",
        lambda mei, page=1, options=None: pages[page - 1],
    )
    seen = []
    monkeypatch.setattr(nb, "show_svg", lambda svg, **k: seen.append(svg) or None)
    nb.show_mei("<mei/>")
    assert seen == pages
    seen.clear()
    nb.show_mei("<mei/>", page=2)
    assert seen == ["<svg>2</svg>"]


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
