"""Snapshot serialization and an offline, inert display environment."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import json
import math
import re
import socket
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

GOLDEN_DIR = Path(__file__).parent / "golden"


def snapshot(value):
    if isinstance(value, pd.DataFrame):
        return {"columns": list(value.columns), "data": snapshot(value.to_numpy()),
                "attrs": snapshot(value.attrs)}
    if isinstance(value, np.ndarray):
        return snapshot(value.tolist())
    if isinstance(value, np.generic):
        return snapshot(value.item())
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def assert_snapshot(actual, expected, *, rtol=1e-5, atol=1e-8, path="root"):
    """Compare schema/strings exactly and each numeric leaf with version-safe tolerances."""
    actual = snapshot(actual)
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_snapshot(actual[key], expected[key], rtol=rtol, atol=atol,
                            path=f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_snapshot(a, e, rtol=rtol, atol=atol, path=f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=path)
    else:
        assert actual == expected, path


def normalize_xml(xml):
    """Canonicalize IDs AND references, retaining topology and musical tstamp values."""
    root = ET.fromstring(xml)
    id_key = "{http://www.w3.org/XML/1998/namespace}id"
    ids = {node.attrib[id_key]: f"id{i}" for i, node in enumerate(root.iter())
           if id_key in node.attrib}
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if key == id_key:
                node.set(key, ids[value])
            elif key in {"startid", "endid", "plist", "sameas", "corresp"}:
                node.set(key, " ".join("#" + ids[v[1:]] if v.startswith("#") and v[1:] in ids
                                      else v for v in value.split()))
            elif key.lower() in {"created", "modified", "timestamp"}:
                node.set(key, "TIMESTAMP")
        if node.tag.rsplit("}", 1)[-1] == "date" and node.text:
            node.text = re.sub(r"\d{4}-\d{2}-\d{2}(?:T[^ ]+)?", "TIMESTAMP", node.text)
    return ET.canonicalize(ET.tostring(root, encoding="unicode"))


def no_network(*args, **kwargs):
    raise RuntimeError("Network access is forbidden in the characterization suite")


@contextmanager
def offline_display():
    """Patch display sinks only: figures, audio widgets, and Verovio still get built."""
    import IPython.display
    import plotly.basedatatypes
    import plotly.io
    import requests.sessions
    import urllib.request
    import webbrowser
    from contextlib import ExitStack
    from audiospylt import audio_utils, dft_analysis, mei

    shown = []
    def display(*objects, **kwargs):
        shown.extend(objects)
    def show(fig, *args, **kwargs):
        shown.append(fig)
    with ExitStack() as stack:
        for target, replacement in [
            ("IPython.display.display", display),
            ("audiospylt.audio_utils.ipy_display", display),
            ("audiospylt.dft_analysis.ipy_display", display),
            ("audiospylt.mei.display", display),
            ("plotly.basedatatypes.BaseFigure.show", show),
            ("plotly.io.show", show),
            ("webbrowser.open", no_network),
            ("requests.sessions.Session.request", no_network),
            ("urllib.request.urlopen", no_network),
            ("socket.socket.connect", no_network),
            ("socket.socket.connect_ex", no_network),
            ("socket.create_connection", no_network),
        ]:
            stack.enter_context(patch(target, replacement))
        yield shown


def read_goldens():
    return json.loads((GOLDEN_DIR / "current.json").read_text(encoding="utf-8"))
