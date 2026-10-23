"""The v1 top-level API, its stub, laziness, annotations and declared dependencies."""

from __future__ import annotations

import ast
import importlib
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import numpy as np
import pytest
import soundfile as sf

import audiospylt
from audiospylt._compat import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "audiospylt"

# The public API: a change here is an API change and must edit this list.
API = [
    "__version__",
    "load_audio", "to_mono", "trim", "fade", "write_audio",
    "read_table", "write_table", "read_partiels", "read_sdif", "write_sdif", "read_chord_seq",
    "Spectrum", "spectrum", "amplitude_to_db", "db_to_amplitude",
    "find_peaks",
    "Spectrogram", "spectrogram", "MelSpectrogram", "mel_spectrogram",
    "F0Contour", "pyin", "HarmonicF0", "harmonic_f0",
    "SchemaError", "LegacyFormatWarning", "validate", "infer_kind", "upgrade",
    "snapshot_to_events", "f0_to_tracks", "tracks_to_events", "tracks_to_chords", "tracks_at",
    "synthesize",
    "transpose", "stretch", "split_events",
    "parameter_range", "pair", "morph",
    "pitch_table", "cents_label",
    "snapshot_to_mei", "events_to_mei",
    "render_svg", "render_svg_pages",
]
SUBPACKAGES = ["analysis", "io", "model", "nb", "notation", "synth", "transform", "viz"]
VIZ = {
    "plot_waveform": "signal", "plot_spectrum": "signal", "plot_spectrogram": "signal",
    "plot_snapshot": "tables", "plot_events": "tables", "plot_events_3d": "tables",
    "plot_morph_map": "morph",
}
NB = {"show": "display", "show_svg": "display", "show_mei": "display", "play": "display"}
# Public module constants that belong in their module's __all__ (and not at the top level).
CONSTANTS = {
    "model.schema": {"TABLE_CONTRACTS"},
    "synth.additive": {"PHASE_MODES"},
    "transform.morph": {"METHODS"},
    "notation.pitch": {"RESOLUTIONS", "SPELLINGS"},
}
V1_FILES = sorted(path for package in SUBPACKAGES for path in (PACKAGE / package).rglob("*.py"))
V1_MODULES = {
    ".".join(path.relative_to(PACKAGE).with_suffix("").parts): path
    for path in V1_FILES if not path.name.startswith("_")
}
LEGACY_MODULES = {f"audiospylt.{name}" for name in REGISTRY}
MEL_MESSAGE = "mel spectrogram support requires librosa; install audiospylt[librosa]"


def run_script(source):
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def module_all(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ["__all__"]:
            return ast.literal_eval(node.value)
    return None


def test_public_api_is_pinned():
    assert audiospylt.__all__ == API
    assert list(audiospylt._V1) == API[1:]
    assert sorted(audiospylt._SUBPACKAGES) == SUBPACKAGES


def test_stub_declares_exactly_the_runtime_api():
    tree = ast.parse((PACKAGE / "__init__.pyi").read_text())
    imported, other = {}, []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.level == 1
            for alias in node.names:
                assert alias.asname == alias.name, ast.unparse(node)  # explicit re-export
                assert alias.name not in imported
                imported[alias.name] = node.module or ""
        else:
            other.append(node)
    assert imported == {**audiospylt._V1, **dict.fromkeys(SUBPACKAGES, "")}
    assert isinstance(other[0], ast.Expr) and isinstance(other[0].value.value, str)
    assert [ast.unparse(node) for node in other[1:] if not isinstance(node, ast.Assign)] == [
        "__version__: str"]
    assert module_all(tree) == API
    assert len(other) == 3


def test_typing_marker_and_stub_are_package_data():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["setuptools"]["package-data"]["audiospylt"] == ["py.typed", "*.pyi"]
    assert (PACKAGE / "py.typed").read_text() == ""


@pytest.mark.parametrize("name", API[1:])
def test_top_level_name_is_its_defining_object(name):
    module = importlib.import_module(f"audiospylt.{audiospylt._V1[name]}")
    value = getattr(audiospylt, name)
    assert value is getattr(module, name)
    assert value.__module__ == module.__name__
    assert name in module.__all__


def test_version_comes_from_metadata_with_a_checkout_fallback():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert version not in (PACKAGE / "__init__.py").read_text()
    run_script('''
import importlib.metadata

def missing(name):
    raise importlib.metadata.PackageNotFoundError(name)

importlib.metadata.version = missing
import audiospylt
assert audiospylt.__version__ == "0+unknown", audiospylt.__version__
''')


def test_bare_import_loads_no_third_party_module():
    run_script(f'''
import sys
before = set(sys.modules)
import audiospylt
assert audiospylt.__file__ == {str(PACKAGE / "__init__.py")!r}, audiospylt.__file__
new = {{name.partition(".")[0] for name in set(sys.modules) - before}}
assert new <= set(sys.stdlib_module_names) | {{"audiospylt"}}, sorted(new)
assert not {{"numpy", "pandas", "scipy"}} & sys.modules.keys()
''')


def test_compute_access_loads_only_its_own_modules():
    run_script(f'''
import sys
import audiospylt as asp
asp.find_peaks
assert "audiospylt.analysis.peaks" in sys.modules
loaded = set(sys.modules)
assert not {{n for n in loaded if n.startswith(("audiospylt.viz", "audiospylt.nb"))}}
assert not {LEGACY_MODULES | {"audiospylt._compat"}!r} & loaded
tops = {{n.partition(".")[0] for n in loaded}}
assert not {{"librosa", "plotly", "IPython", "verovio", "soxr", "requests"}} & tops, tops
''')


def test_star_import_binds_exactly_all_without_viz_nb_or_legacy():
    run_script(f'''
import sys
namespace = {{}}
exec("from audiospylt import *", namespace)
import audiospylt
assert set(namespace) - {{"__builtins__"}} == set({API!r})
for name in {API!r}:
    assert namespace[name] is getattr(audiospylt, name)
loaded = set(sys.modules)
assert not {{n for n in loaded if n.startswith(("audiospylt.viz", "audiospylt.nb"))}}
assert not {LEGACY_MODULES | {"audiospylt._compat"}!r} & loaded
tops = {{n.partition(".")[0] for n in loaded}}
assert not {{"librosa", "plotly", "IPython", "verovio"}} & tops, tops
''')


def test_dir_hides_legacy_names_even_after_access():
    run_script(f'''
import audiospylt as asp
expected = {sorted(API + SUBPACKAGES)!r}
assert dir(asp) == expected, dir(asp)
asp.render_audio
asp.mei
import audiospylt.audio_utils
assert {{"render_audio", "mei", "audio_utils"}} <= vars(asp).keys()
assert dir(asp) == expected, dir(asp)
''')


# audio_utils guards its IPython import, so a missing IPython surfaces through mei.
@pytest.mark.parametrize("blocked,name", [("librosa", "load_audio_sample_and_preview"), ("IPython", "mei")])
def test_legacy_name_without_its_dependency_names_all_extra(blocked, name):
    run_script(f'''
import sys
sys.modules[{blocked!r}] = None
import audiospylt as asp
try:
    getattr(asp, {name!r})
except ImportError as error:
    assert str(error) == "{name} is 0.6 API and needs {blocked}; install audiospylt[all]", error
    assert isinstance(error.__cause__, ImportError)
else:
    raise AssertionError("missing ImportError")
assert {name!r} not in vars(asp)
''')


def test_legacy_import_error_other_than_a_missing_module_propagates(monkeypatch):
    broken = ImportError("cannot import name 'gone' from 'scipy.signal'", name="scipy.signal")

    def fail(name):
        raise broken

    monkeypatch.setattr(audiospylt, "_import_module", fail)
    with pytest.raises(ImportError) as caught:
        audiospylt._import_legacy("search_f0", "f0_utils")
    assert caught.value is broken


@pytest.mark.parametrize("package,names", [("viz", VIZ), ("nb", NB)])
def test_viz_and_nb_reexport_their_module_functions(package, names):
    namespace = getattr(audiospylt, package)
    assert namespace is importlib.import_module(f"audiospylt.{package}")
    assert namespace.__all__ == list(names)
    for name, module in names.items():
        defining = importlib.import_module(f"audiospylt.{package}.{module}")
        assert getattr(namespace, name) is getattr(defining, name)
        assert getattr(namespace, name).__module__ == defining.__name__


def test_nb_import_loads_no_ipython():
    run_script('''
import sys
import audiospylt as asp
import audiospylt.nb
asp.nb.show
tops = {n.partition(".")[0] for n in sys.modules}
assert not {"IPython", "ipywidgets", "plotly"} & tops, tops
''')


@pytest.mark.parametrize("module", sorted(V1_MODULES))
def test_module_all_lists_public_definitions_and_constants(module):
    tree = ast.parse(V1_MODULES[module].read_text())
    public = {node.name for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")}
    names = module_all(tree)
    assert names is not None, f"{module} has no __all__"
    assert len(set(names)) == len(names)
    assert set(names) == public | CONSTANTS.get(module, set())
    assert importlib.import_module(f"audiospylt.{module}").__all__ == names


@pytest.mark.parametrize("module", sorted(V1_MODULES))
def test_public_functions_are_fully_annotated(module):
    tree = ast.parse(V1_MODULES[module].read_text())
    names = set(module_all(tree))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            arguments = node.args
            params = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                      *filter(None, [arguments.vararg, arguments.kwarg])]
            missing = [param.arg for param in params if param.annotation is None]
            assert not missing, f"{module}.{node.name}: unannotated {missing}"
            assert node.returns is not None, f"{module}.{node.name}: no return annotation"


def _requirement(text):
    return re.match(r"[A-Za-z0-9._-]+", text).group().lower().replace("_", "-")


def _type_checking_only(node, parents):
    ancestor = parents.get(node)
    while ancestor is not None:
        if isinstance(ancestor, ast.If) and ast.unparse(ancestor.test) in {
                "TYPE_CHECKING", "typing.TYPE_CHECKING"}:
            return True
        ancestor = parents.get(ancestor)
    return False


def _in_function(node, parents):
    ancestor = parents.get(node)
    while ancestor is not None:
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        ancestor = parents.get(ancestor)
    return False


def test_declared_dependencies_match_v1_imports():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    core = {_requirement(text) for text in project["dependencies"]}
    raw = project["optional-dependencies"]
    extras = {extra: {_requirement(text) for text in texts}
              for extra, texts in raw.items() if extra not in {"all", "dev"}}
    assert set(extras) == {"plot", "notebook", "notation", "librosa"}
    assert raw["all"] == ["audiospylt[" + ",".join(sorted(extras)) + "]"]
    assert "audiospylt[all]" in raw["dev"]
    declared = core.union(*extras.values())
    everything = [*project["dependencies"], *(text for texts in raw.values() for text in texts)]
    assert not [text for text in everything if "matplotlib" in text.lower()]

    files = [PACKAGE / "__init__.py", PACKAGE / "_optional.py", PACKAGE / "_compat.py", *V1_FILES]
    for path in files:
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "import_optional":
                module = node.args[0].value
                extra = next(k.value.value for k in node.keywords if k.arg == "extra")
                assert _requirement(module) in extras[extra], f"{path}: {module} not in [{extra}]"
                continue
            if isinstance(node, ast.Import):
                tops = [alias.name.partition(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                tops = [node.module.partition(".")[0]]
            else:
                continue
            for top in tops:
                if top in sys.stdlib_module_names or top in {"__future__", "audiospylt"}:
                    continue
                distribution = {"IPython": "ipython"}.get(top, top).lower()
                assert distribution in declared, f"{path}: {top} is not declared"
                runtime = not _in_function(node, parents) and not _type_checking_only(node, parents)
                if runtime and (path.relative_to(PACKAGE).as_posix(), top) != ("viz/_common.py", "plotly"):
                    assert distribution in core, f"{path}: module-level import of {top} ({distribution})"


def test_no_matplotlib_import_anywhere_in_the_package():
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not [name for name in names if name.partition(".")[0] == "matplotlib"], path


# Missing extras: each optional feature names the extra that installs it.

def test_mel_features_without_librosa_name_the_extra(monkeypatch):
    from audiospylt.analysis.spectrogram import mel_spectrogram, spectrogram
    from audiospylt.viz import plot_spectrogram

    tone = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
    spec = spectrogram(tone, 8000, frame_length=256, hop=128)
    mel = mel_spectrogram(spec, n_mels=16)
    monkeypatch.setitem(sys.modules, "librosa", None)
    for call in (lambda: mel_spectrogram(spec, n_mels=16), lambda: plot_spectrogram(mel)):
        with pytest.raises(ImportError, match=f"^{re.escape(MEL_MESSAGE)}$") as raised:
            call()
        assert isinstance(raised.value.__cause__, ImportError)
    plot_spectrogram(spec, freq_scale="log")  # the linear-bin STFT path needs no librosa


def test_import_optional_translates_only_a_missing_module(tmp_path, monkeypatch):
    from audiospylt._optional import import_optional

    (tmp_path / "asp_probe_needs_dep.py").write_text("import asp_probe_missing_dep\n")
    (tmp_path / "asp_probe_broken.py").write_text("raise ImportError('Numba needs NumPy 2.2 or less')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError) as caught:
        import_optional("asp_probe_absent", extra="librosa", feature="mel support")
    assert str(caught.value) == "mel support requires asp_probe_absent; install audiospylt[librosa]"
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    # An installed module whose own imports fail keeps its error.
    for module, text in (("asp_probe_needs_dep", "asp_probe_missing_dep"), ("asp_probe_broken", "Numba")):
        with pytest.raises(ImportError, match=text) as caught:
            import_optional(module, extra="librosa", feature="mel support")
        assert "install audiospylt" not in str(caught.value)


def test_resampling_needs_no_librosa(monkeypatch):
    from audiospylt.io.audio import load_audio

    stream = BytesIO()
    sf.write(stream, np.zeros(100), 8000, format="WAV", subtype="FLOAT")
    stream.seek(0)
    monkeypatch.setitem(sys.modules, "librosa", None)
    audio, rate = load_audio(stream, sr=16000)
    assert audio.shape == (200,) and rate == 16000


def test_viz_without_plotly_names_the_plot_extra():
    run_script('''
import sys
sys.modules["plotly"] = None
import audiospylt as asp
asp.find_peaks
for attempt in (lambda: __import__("audiospylt.viz"), lambda: asp.viz):
    try:
        attempt()
    except ImportError as error:
        assert str(error) == "plotting requires plotly; install audiospylt[plot]", error
    else:
        raise AssertionError("missing ImportError")
''')


def test_help_and_getmembers_work_without_plotly():
    # Both skip only AttributeError, so dir() must not list an unloadable viz.
    run_script(f'''
import inspect
import pydoc
import sys
sys.modules["plotly"] = None
import audiospylt as asp
assert dir(asp) == {sorted(set(API + SUBPACKAGES) - {"viz"})!r}, dir(asp)
pydoc.render_doc(asp)
inspect.getmembers(asp)
''')


def test_core_pipeline_runs_with_every_extra_blocked():
    run_script('''
import io
import sys
for name in ("librosa", "plotly", "IPython", "ipywidgets", "ipykernel", "nbformat", "tqdm", "verovio"):
    sys.modules[name] = None
import numpy as np
import soundfile as sf
import audiospylt as asp

stream = io.BytesIO()
times = np.arange(8000) / 8000
sf.write(stream, 0.5 * np.sin(2 * np.pi * 440 * times), 8000, format="WAV", subtype="FLOAT")
stream.seek(0)
audio, sr = asp.load_audio(stream, sr=16000)
peaks = asp.find_peaks(asp.spectrum(audio, sr), min_db=-30)
assert abs(peaks.loc[peaks["amp"].idxmax(), "freq"] - 440) < 0.01, peaks
events = asp.snapshot_to_events(peaks, 0.0, 0.5)
assert asp.synthesize(events, sr=8000).shape == (4000,)
assert "<mei" in asp.snapshot_to_mei(peaks)
asp.spectrogram(audio, sr)
import audiospylt.nb
for name, extra in (("mel_spectrogram", "librosa"), ("render_svg", "notation")):
    try:
        getattr(asp, name)(asp.spectrogram(audio, sr) if name == "mel_spectrogram" else "<mei/>")
    except ImportError as error:
        assert f"install audiospylt[{extra}]" in str(error), error
    else:
        raise AssertionError(name)
''')
