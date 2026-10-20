"""Migration rows agree with the warning registry; examples run on real data."""

from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path
import re
import warnings

import numpy as np
import pytest
import soundfile as sf

from audiospylt._compat import REGISTRY
from audiospylt.model.schema import LegacyFormatWarning

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "migration.md"
TEXT = DOC.read_text()
NAME_SECTION = TEXT.split("## Name by name\n", 1)[1].split("## Known 0.6 issues\n", 1)[0]
ROWS = []
for line in NAME_SECTION.splitlines():
    if line.startswith("| `"):
        old, new, notes = [cell.strip() for cell in line.strip("|").split("|")]
        assert re.fullmatch(r"`\w+\.\w+`", old), old
        ROWS.append((old.strip("`"), new, notes))
BLOCKS = re.findall(r"^```python\n(.*?)^```\s*$", TEXT, re.MULTILINE | re.DOTALL)


def resolve(relative):
    parts = ("audiospylt." + relative).split(".")
    for length in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:length]))
        except ModuleNotFoundError as error:
            if error.name != ".".join(parts[:length]):
                raise
            continue
        for attr in parts[length:]:
            obj = getattr(obj, attr)
        return obj
    raise AssertionError(relative)


def test_table_inventory_exactly_matches_registry_and_constants():
    expected = {f"{module}.{name}" for module, entries in REGISTRY.items() for name in entries}
    expected.update(f"mei.{name}" for name in ("A4_FREQUENCY", "MIDI_BASE", "CENT_FACTOR", "XML_NS"))
    assert Counter(old for old, _, _ in ROWS) == Counter({name: 1 for name in expected})
    assert NAME_SECTION.count("| 0.6 | v1 | notes |") == len(REGISTRY)


@pytest.mark.parametrize("old,new,notes", ROWS, ids=[row[0] for row in ROWS])
def test_replacements_include_all_registry_names_and_resolve(old, new, notes):
    module, name = old.split(".")
    assert notes
    if name in REGISTRY[module]:
        use, _ = REGISTRY[module][name]
        if use is None:
            assert new == "None"
        else:
            for relative in re.findall(r"audiospylt\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)", use):
                assert relative in new
    for span in re.findall(r"`([^`]+)`", new):
        assert callable(resolve(span)), span


def test_document_has_three_examples_and_26_known_issue_cases():
    assert len(BLOCKS) == 3
    issues = TEXT.split("## Known 0.6 issues\n", 1)[1].split("## Examples\n", 1)[0]
    assert re.findall(r"^(\d+)\. ", issues, re.MULTILINE) == [str(n) for n in range(1, 27)]


@pytest.mark.parametrize("source", BLOCKS, ids=["cello-preview", "cello-spectrum-notation", "transport-morph"])
def test_python_examples_run_in_isolated_directory(source, tmp_path, monkeypatch):
    if "show_mei(" in source:
        pytest.importorskip("verovio")
    import IPython.display
    import plotly.basedatatypes

    for name in ("samples", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    displayed = []
    monkeypatch.setattr(IPython.display, "display", lambda *objects, **kwargs: displayed.extend(objects))
    monkeypatch.setattr(plotly.basedatatypes.BaseFigure, "show", lambda fig, *args, **kwargs: displayed.append(fig))
    namespace = {"__name__": "__migration_example__"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LegacyFormatWarning)
        exec(compile(source, str(DOC), "exec"), namespace)
    assert all(w.category is LegacyFormatWarning for w in caught)
    if "morph.wav" in source:
        assert len(caught) == 2
        assert not displayed
        output, sr = sf.read(tmp_path / "morph.wav")
        assert sr == 44100 and output.shape == (88200,)
        assert len(namespace["events"]) == 41
        assert np.max(np.abs(output)) == pytest.approx(0.0587308078, abs=2e-7)
        assert {p.name for p in tmp_path.iterdir()} == {"samples", "data", "morph.wav"}
    else:
        assert not caught
        assert len(displayed) == 2
        assert namespace["sr"] == 44100
        assert namespace["excerpt"].shape == (22050,)
        if "peaks" in namespace:
            assert len(namespace["peaks"]) == 40
            assert '<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1"' in namespace["mei"]
        assert {p.name for p in tmp_path.iterdir()} == {"samples", "data"}
