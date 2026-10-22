"""Check the built sdist and wheel against pyproject.toml before they are installed or published.

Usage:
    python -m build
    python scripts/check_dist.py dist

The sdist must ship the package only (no tests or example data) plus the typing files and
LICENSE; the wheel must ship the typing files and no tests or retired modules. The wheel
METADATA must carry pyproject's version, the PEP 639 license fields, the core dependencies
without a marker and exactly the declared extras. Prints every failure; the exit code is 1
when there is one. Standard library only.
"""

import argparse
import re
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDIST_EXCLUDED = ("tests", "samples", "data", "revision")
SDIST_REQUIRED = ("audiospylt/py.typed", "audiospylt/__init__.pyi", "LICENSE")
WHEEL_REQUIRED = ("audiospylt/py.typed", "audiospylt/__init__.pyi")
RETIRED = ("audiospylt/mei_old.py",)


def requirement_name(text):
    return re.match(r"[A-Za-z0-9._-]+", text.strip()).group(0).lower().replace("_", "-")


def one(dist, pattern, failures):
    found = sorted(dist.glob(pattern))
    if len(found) != 1:
        failures.append(f"expected one {pattern} in {dist}, found {[p.name for p in found]}")
        return None
    return found[0]


def check_sdist(path, failures):
    with tarfile.open(path) as archive:
        # Every member sits under audiospylt-<version>/; compare the paths below it.
        names = {name.partition("/")[2] for name in archive.getnames()} - {""}
    for folder in SDIST_EXCLUDED:
        if any(name == folder or name.startswith(folder + "/") for name in names):
            failures.append(f"sdist contains {folder}/")
    for name in SDIST_REQUIRED:
        if name not in names:
            failures.append(f"sdist lacks {name}")
    for name in RETIRED:
        if name in names:
            failures.append(f"sdist contains {name}")


def check_wheel(path, project, failures):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            failures.append(f"wheel has {len(metadata_files)} METADATA files")
            return
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    for name in WHEEL_REQUIRED:
        if name not in names:
            failures.append(f"wheel lacks {name}")
    for name in RETIRED:
        if name in names:
            failures.append(f"wheel contains {name}")
    if any(name.startswith("tests/") for name in names):
        failures.append("wheel contains tests/")

    expected = {
        "Version": [project["version"]],
        "License-Expression": ["MIT"],
        "License-File": ["LICENSE"],
    }
    for field, value in expected.items():
        if metadata.get_all(field, []) != value:
            failures.append(f"METADATA {field} is {metadata.get_all(field, [])}, expected {value}")
    licence_classifiers = [c for c in metadata.get_all("Classifier", []) if c.startswith("License ::")]
    if licence_classifiers:
        failures.append(f"METADATA has license classifiers {licence_classifiers}")

    core = {requirement_name(r) for r in metadata.get_all("Requires-Dist", []) if ";" not in r}
    declared = {requirement_name(r) for r in project["dependencies"]}
    if core != declared:
        failures.append(f"unconditional Requires-Dist {sorted(core)}, expected {sorted(declared)}")
    extras = sorted(metadata.get_all("Provides-Extra", []))
    if extras != sorted(project["optional-dependencies"]):
        failures.append(f"Provides-Extra {extras}, expected {sorted(project['optional-dependencies'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path, help="folder holding one .tar.gz and one .whl")
    args = parser.parse_args(argv)

    with open(ROOT / "pyproject.toml", "rb") as file:
        project = tomllib.load(file)["project"]
    failures = []
    sdist = one(args.dist, "*.tar.gz", failures)
    wheel = one(args.dist, "*.whl", failures)
    if sdist:
        check_sdist(sdist, failures)
    if wheel:
        check_wheel(wheel, project, failures)

    for failure in failures:
        print(f"FAIL  {failure}")
    if failures:
        return 1
    print(f"ok    {sdist.name}, {wheel.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
