"""AST contracts: compute layers stay display-free; viz/nb stay narrowly coupled."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "audiospylt"

_CORE = ("io", "analysis", "model", "synth", "notation", "transform")
_FORBIDDEN_TOP = {"IPython", "ipywidgets", "nbformat", "matplotlib", "plotly"}
_LEGACY = {
    "audio_utils", "dft_analysis", "multiplotter", "plotting", "plot_wave",
    "waveform_utils", "mei", "mei_old", "df_transforms", "f0_utils",
    "generate_wave_file", "io_utils",
}


def _iter_imports(path: Path):
    tree = ast.parse(path.read_text())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, parents, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            yield node, parents, module
            for alias in node.names:
                if node.level:
                    yield node, parents, alias.name


def _inside_function(node, parents) -> bool:
    ancestor = parents.get(node)
    while ancestor is not None:
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        ancestor = parents.get(ancestor)
    return False


def _top(name: str) -> str:
    return name.split(".")[0] if name else ""


def test_core_has_no_display_or_legacy_imports():
    for package in _CORE:
        for path in sorted((ROOT / package).rglob("*.py")):
            for node, parents, name in _iter_imports(path):
                top = _top(name)
                assert top not in _FORBIDDEN_TOP, f"{path}: imports {name}"
                assert top not in {"viz", "nb"}, f"{path}: imports {name}"
                if name.startswith("audiospylt."):
                    rest = name.split(".", 1)[1]
                    head = rest.split(".", 1)[0]
                    assert head not in {"viz", "nb"} | _LEGACY, f"{path}: imports {name}"
                if top in _LEGACY:
                    raise AssertionError(f"{path}: imports legacy {name}")


def test_viz_has_no_ipython_matplotlib_or_nb_and_librosa_is_lazy():
    forbidden = _FORBIDDEN_TOP - {"plotly"} | {"nb"}
    for path in sorted((ROOT / "viz").rglob("*.py")):
        for node, parents, name in _iter_imports(path):
            top = _top(name)
            assert top not in forbidden, f"{path}: imports {name}"
            if name.startswith("audiospylt.nb") or top == "nb":
                raise AssertionError(f"{path}: imports {name}")
            if name.startswith("audiospylt.") and _top(name.split(".", 1)[1]) in _LEGACY:
                raise AssertionError(f"{path}: imports {name}")
            if top in _LEGACY:
                raise AssertionError(f"{path}: imports {name}")
            if top == "librosa":
                assert _inside_function(node, parents), f"{path}: librosa must be lazy"


def test_nb_imports_ipython_only_inside_functions():
    for path in sorted((ROOT / "nb").rglob("*.py")):
        for node, parents, name in _iter_imports(path):
            if _top(name) == "IPython" or name.startswith("IPython"):
                assert _inside_function(node, parents), f"{path}: IPython must be lazy"


def test_viz_and_nb_init_are_single_docstring():
    for package in ("viz", "nb"):
        init = ast.parse((ROOT / package / "__init__.py").read_text())
        assert len(init.body) == 1 and isinstance(init.body[0], ast.Expr)
        assert isinstance(init.body[0].value, ast.Constant)
        assert isinstance(init.body[0].value.value, str)


def test_compat_imports_only_its_four_stdlib_dependencies():
    imports = {name for _, _, name in _iter_imports(ROOT / "_compat.py")}
    assert imports == {"__future__", "contextvars", "functools", "warnings"}


def test_package_init_has_only_stdlib_imports_at_module_level():
    for node, parents, name in _iter_imports(ROOT / "__init__.py"):
        if not _inside_function(node, parents):
            assert not isinstance(node, ast.ImportFrom) or node.level == 0
            assert _top(name) in sys.stdlib_module_names, name


def test_v1_packages_never_import_compat():
    for package in (*_CORE, "viz", "nb"):
        for path in sorted((ROOT / package).rglob("*.py")):
            for _, _, name in _iter_imports(path):
                assert "_compat" not in name.split("."), f"{path}: imports {name}"
