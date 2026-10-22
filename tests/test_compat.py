"""Lazy 0.6 exports and call-time deprecations preserve legacy behavior."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import importlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Barrier
import warnings

import numpy as np
import pandas as pd
import pytest

import audiospylt
from audiospylt._compat import (
    AudiospyltDeprecationWarning, REGISTRY, deprecate_module, message,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = [(module, name) for module, entries in REGISTRY.items() for name in entries]
# The 0.6 top-level exports: they still resolve, but stay out of __all__ and dir().
EXPORTS = {
    "render_audio": "generate_wave_file",
    "load_audio_sample_and_preview": "audio_utils",
    "search_f0": "f0_utils",
    "summarize_result": "f0_utils",
    "process_spectral_dataframe": "df_transforms",
    "detect_dataframe_type": "df_transforms",
    "expand_to_multi": "df_transforms",
    "add_time_cues": "df_transforms",
    "merge_equal_split": "df_transforms",
    "create_time_transition_df": "df_transforms",
    "merge_cdf": "df_transforms",
    "merge_cdf_analysis": "df_transforms",
    "merge_sigmoid": "df_transforms",
    "merge_sigmoid_analysis": "df_transforms",
}
LEGACY = set(EXPORTS) | set(REGISTRY)


def resolve(name):
    """Import the longest module prefix, then resolve any attributes."""
    parts = name.split(".")
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
    raise AssertionError(f"Unresolvable name: {name}")


@pytest.mark.parametrize("module", REGISTRY)
def test_registry_matches_public_definitions_and_footer(module):
    source = (ROOT / "audiospylt" / f"{module}.py").read_text()
    tree = ast.parse(source)
    public = {node.name for node in tree.body
              if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")}
    assert public == REGISTRY[module].keys()
    footer = ast.parse("from ._compat import deprecate_module\ndeprecate_module(globals())")
    assert [ast.dump(node) for node in tree.body[-2:]] == [ast.dump(n) for n in footer.body]
    assert source.count("deprecate_module") == 2


def test_registry_inventory():
    assert len(REGISTRY) == 11
    assert len(NAMES) == 59
    assert sum(use is None for entries in REGISTRY.values() for use, _ in entries.values()) == 16
    assert "mei_old" not in REGISTRY
    assert not {"A4_FREQUENCY", "MIDI_BASE", "CENT_FACTOR", "XML_NS"} & REGISTRY["mei"].keys()
    assert issubclass(AudiospyltDeprecationWarning, DeprecationWarning)
    assert AudiospyltDeprecationWarning.__doc__ == "A 0.6 name is used; it is removed in 1.1."


@pytest.mark.parametrize("module,name", NAMES)
def test_wrapper_metadata(module, name):
    wrapped = getattr(importlib.import_module(f"audiospylt.{module}"), name)
    original = wrapped.__wrapped__
    assert wrapped.__audiospylt_deprecated__ == (module, name)
    assert inspect.isfunction(original)
    assert original.__module__ == f"audiospylt.{module}"
    assert inspect.signature(wrapped) == inspect.signature(original)
    assert wrapped.__name__ == original.__name__ == name
    assert wrapped.__qualname__ == original.__qualname__
    assert wrapped.__doc__ == ".. deprecated:: 1.0\n    " + message(module, name) + "\n\n" + (original.__doc__ or "")


@pytest.mark.parametrize("module", REGISTRY)
def test_only_registry_objects_are_replaced(module):
    namespace = vars(importlib.import_module(f"audiospylt.{module}")).copy()
    for name in REGISTRY[module]:
        namespace[name] = namespace[name].__wrapped__
    before = namespace.copy()
    deprecate_module(namespace)
    for name, obj in before.items():
        if name in REGISTRY[module]:
            assert namespace[name].__wrapped__ is obj
        else:
            assert namespace[name] is obj
    if module == "mei":
        assert {"display", "HTML", "A4_FREQUENCY", "MIDI_BASE", "CENT_FACTOR", "XML_NS"} <= namespace.keys()
        assert any(name.startswith("_") and inspect.isfunction(obj) for name, obj in before.items())


def toy(monkeypatch, source):
    namespace = {"__name__": "audiospylt.toy"}
    exec(source, namespace)
    names = {name: ("audiospylt.io.audio.load_audio", None)
             for name, obj in namespace.items() if inspect.isfunction(obj)}
    monkeypatch.setitem(REGISTRY, "toy", names)
    deprecate_module(namespace)
    return namespace


@pytest.mark.parametrize("invalid", ["missing", "imported", "callable", "constant"])
def test_invalid_registry_function_is_a_programming_error(monkeypatch, invalid):
    monkeypatch.setitem(REGISTRY, "toy", {"bad": (None, None)})
    namespace = {"__name__": "audiospylt.toy"}
    if invalid == "imported":
        namespace["bad"] = test_registry_inventory
    elif invalid == "callable":
        namespace["bad"] = type("Callable", (), {"__call__": lambda self: None})()
    elif invalid == "constant":
        namespace["bad"] = 3
    before = namespace.copy()
    with pytest.raises(RuntimeError, match=r"audiospylt.toy.bad must be a function defined in audiospylt.toy"):
        deprecate_module(namespace)
    assert namespace == before


def test_exact_messages():
    assert message("dft_analysis", "compute_fft") == (
        "audiospylt.dft_analysis.compute_fft is deprecated (0.6 API, removed in 1.1); "
        "use audiospylt.analysis.spectrum.spectrum instead. See docs/migration.md.")
    assert message("f0_utils", "search_f0") == (
        "audiospylt.f0_utils.search_f0 is deprecated (0.6 API, removed in 1.1) "
        "and has no v1 replacement. See docs/migration.md.")
    assert message("generate_wave_file", "scale_samples") == (
        "audiospylt.generate_wave_file.scale_samples is deprecated (0.6 API, removed in 1.1) "
        "and has no v1 replacement; closest: audiospylt.io.audio.write_audio(..., subtype=...). "
        "See docs/migration.md.")


@pytest.mark.parametrize("module,name", NAMES)
def test_message_names_resolve(module, name):
    notice = message(module, name)
    assert f"audiospylt.{module}.{name}" in notice
    assert "1.1" in notice
    for dotted in re.findall(r"audiospylt(?:\.[A-Za-z_]\w*)+", notice):
        assert callable(resolve(dotted)), dotted


def test_nested_calls_warn_once_and_preserve_arguments_return_and_effects(monkeypatch):
    ns = toy(monkeypatch, '''
effects = []
def inner(value, *, label="x"):
    """Original docstring."""
    effects.append(label)
    return value
def outer(value, *, label="x"):
    return inner(value, label=label)
''')
    value = object()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AudiospyltDeprecationWarning)
        assert ns["outer"](value, label="wrapped") is value
        assert ns["inner"](value, label="direct") is value
    assert ns["effects"] == ["wrapped", "direct"]
    assert [str(w.message) for w in caught] == [message("toy", "outer"), message("toy", "inner")]
    assert all(w.category is AudiospyltDeprecationWarning for w in caught)
    assert all(w.filename == __file__ for w in caught)
    assert ns["inner"].__wrapped__(value, label="original") is value
    assert ns["effects"][-1] == "original"


def test_original_exception_and_warning_error_both_reset_depth(monkeypatch):
    ns = toy(monkeypatch, '''
def fail():
    raise problem
def succeed():
    return 42
''')
    problem = ValueError("original exception")
    ns["problem"] = problem
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AudiospyltDeprecationWarning)
        with pytest.raises(ValueError) as raised:
            ns["fail"]()
        assert raised.value is problem
        assert ns["succeed"]() == 42
    assert len(caught) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", AudiospyltDeprecationWarning)
        with pytest.raises(AudiospyltDeprecationWarning, match="toy.succeed"):
            ns["succeed"]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", AudiospyltDeprecationWarning)
        assert ns["succeed"]() == 42
    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_depth_is_isolated_between_threads(monkeypatch):
    ns = toy(monkeypatch, '''
def inner():
    return 7
def outer():
    barrier.wait(timeout=10)
    return inner()
''')
    ns["barrier"] = Barrier(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AudiospyltDeprecationWarning)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(ns["outer"]) for _ in range(2)]
            assert [f.result(timeout=15) for f in futures] == [7, 7]
    assert [str(w.message) for w in caught] == [message("toy", "outer")] * 2


def test_real_mei_nested_calls_warn_only_at_entry(monkeypatch):
    pytest.importorskip("verovio")
    from audiospylt import mei

    monkeypatch.setattr(mei, "display", lambda *args, **kwargs: None)
    snapshot = pd.DataFrame({"Frequency (Hz)": [440.0], "Amplitude": [0.5]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AudiospyltDeprecationWarning)
        mei.process_and_visualize(snapshot, display_df=False)
    assert len(caught) == 1
    assert caught[0].category is AudiospyltDeprecationWarning
    assert caught[0].filename == __file__
    assert str(caught[0].message) == message("mei", "process_and_visualize")


def test_real_merges_and_transition_warn_per_user_call():
    from audiospylt import df_transforms as legacy

    a = pd.DataFrame({"Frequency (Hz)": [110., 220.], "Amplitude": [.5, .25]})
    b = pd.DataFrame({"Frequency (Hz)": [330., 440., 550.], "Amplitude": [.4, .3, .1]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AudiospyltDeprecationWarning)
        merged = legacy.merge_equal_split(a, b)
        transition = legacy.create_time_transition_df(merged, .5)
    assert [str(w.message) for w in caught] == [message("df_transforms", name) for name in
                                              ("merge_equal_split", "create_time_transition_df")]
    assert all(w.category is AudiospyltDeprecationWarning and w.filename == __file__ for w in caught)
    pd.testing.assert_frame_equal(merged, legacy.merge_equal_split.__wrapped__(a, b))
    pd.testing.assert_frame_equal(transition, legacy.create_time_transition_df.__wrapped__(merged, .5))


def test_real_compute_return_matches_original():
    from audiospylt.dft_analysis import compute_fft

    x = np.arange(16, dtype=float)
    with pytest.warns(AudiospyltDeprecationWarning) as caught:
        actual = compute_fft(x, 16)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    for result, expected in zip(actual, compute_fft.__wrapped__(x, 16)):
        np.testing.assert_array_equal(result, expected)


def run_script(source):
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_bare_import_is_lazy():
    run_script(f'''
import sys
import audiospylt
assert not {LEGACY!r} & set(audiospylt.__all__)
assert not {set(REGISTRY)!r} & {{n.removeprefix("audiospylt.") for n in sys.modules if n.startswith("audiospylt.")}}
assert not {{"IPython", "ipywidgets", "nbformat", "matplotlib", "plotly"}} & sys.modules.keys()
assert "audiospylt._compat" not in sys.modules
assert not {LEGACY!r} & set(dir(audiospylt))
try:
    audiospylt.nope
except AttributeError as error:
    assert str(error) == "module 'audiospylt' has no attribute 'nope'"
else:
    raise AssertionError("missing AttributeError")
from audiospylt import io
assert io.__name__ == "audiospylt.io"
''')


def test_top_level_access_imports_only_the_requested_module_dependencies():
    run_script('''
import sys
import warnings
import audiospylt
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    value = audiospylt.search_f0
assert not caught, caught
assert "audiospylt.f0_utils" in sys.modules
assert not {"audiospylt.audio_utils", "audiospylt.generate_wave_file", "audiospylt.df_transforms"} & sys.modules.keys()
assert value is audiospylt.f0_utils.search_f0
assert audiospylt.__dict__["search_f0"] is value
''')


def test_all_lazy_exports_star_import_and_module_access():
    run_script(f'''
import importlib
import warnings
import audiospylt
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert audiospylt.mei is importlib.import_module("audiospylt.mei")
    for name, module in {EXPORTS!r}.items():
        assert getattr(audiospylt, name) is getattr(importlib.import_module("audiospylt." + module), name)
        assert name in audiospylt.__dict__
    for name in {list(REGISTRY)!r}:
        assert getattr(audiospylt, name) is importlib.import_module("audiospylt." + name)
    namespace = {{}}
    exec("from audiospylt import *", namespace)
    assert not {LEGACY!r} & namespace.keys()
    assert {{n for n in namespace if n != "__builtins__"}} == set(audiospylt.__all__)
assert not caught, caught
''')


def test_legacy_names_stay_out_of_all():
    assert not LEGACY & set(audiospylt.__all__)
