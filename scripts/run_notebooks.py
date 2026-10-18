"""Execute the tutorial and case-study notebooks headless and report pass/fail per notebook.

Each notebook runs with its own folder as the working directory, as it would in Jupyter.
Outputs are discarded; files the notebooks write go to the gitignored tsv/, mei/ and
rendered_audio/ folders.

Usage:
    python scripts/run_notebooks.py                       # all notebooks
    python scripts/run_notebooks.py --offline             # skip notebooks that need the network
    python scripts/run_notebooks.py tutorials_tech/resynth.ipynb

A notebook needs the network when one of its cells carries the `network` tag
(View > Cell Toolbar > Tags in Jupyter). The exit code is 1 when a notebook fails,
or when a notebook listed in EXPECTED_FAILURES passes.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = ("case_studies", "tutorials_tech")
NETWORK_TAG = "network"

# Known failures, relative to the repository root. Remove an entry once the notebook passes.
EXPECTED_FAILURES = {
    "case_studies/showcase_bayle_old.ipynb": "peak-picking parameters leave 0 peaks; fixed in step 3.3",
}


def needs_network(nb):
    return any(NETWORK_TAG in cell.get("metadata", {}).get("tags", []) for cell in nb.cells)


def failing_cell(nb):
    """Index of the first cell with an error output, or None."""
    for index, cell in enumerate(nb.cells):
        if any(out.get("output_type") == "error" for out in cell.get("outputs", [])):
            return index
    return None


def run_notebook(path, timeout, kernel_name):
    """Execute one notebook; return (ok, detail)."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        return False, f"cell {failing_cell(nb)}: {exc.ename}: {exc.evalue}"[:300]
    except CellTimeoutError:
        return False, f"cell timed out after {timeout} s"
    except DeadKernelError as exc:
        return False, f"kernel died: {exc}"[:300]
    return True, ""


def write_step_summary(results):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    lines = ["| Result | Notebook | Time | Detail |", "| --- | --- | --- | --- |"]
    for status, rel, seconds, detail in results:
        lines.append(f"| {status} | `{rel}` | {seconds:.0f} s | {detail.replace('|', '/')} |")
    with open(summary, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path,
                        help="notebooks to run, relative to --root (default: all)")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root (default: the checkout this script is in)")
    parser.add_argument("--offline", action="store_true",
                        help=f"skip notebooks with a cell tagged '{NETWORK_TAG}'")
    parser.add_argument("--timeout", type=int, default=300, help="per-cell timeout in seconds")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.notebooks:
        paths = [root / p for p in args.notebooks]
    else:
        paths = [p for d in NOTEBOOK_DIRS for p in sorted((root / d).glob("*.ipynb"))]

    results = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        expected = EXPECTED_FAILURES.get(rel)
        t0 = time.time()
        if args.offline and needs_network(nbformat.read(path, as_version=4)):
            status, detail = "SKIP", f"tagged '{NETWORK_TAG}'"
        else:
            ok, detail = run_notebook(path, args.timeout, args.kernel)
            if expected:
                status = "XPASS" if ok else "XFAIL"
                detail = (f"passes but is listed in EXPECTED_FAILURES ({expected})" if ok
                          else f"{detail} [{expected}]")
            else:
                status = "PASS" if ok else "FAIL"
        seconds = time.time() - t0
        results.append((status, rel, seconds, detail))
        print(f"{status:5} {seconds:6.1f}s  {rel}  {detail}".rstrip(), flush=True)

    counts = {s: sum(r[0] == s for r in results) for s in ("PASS", "FAIL", "XFAIL", "XPASS", "SKIP")}
    print("--- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items() if n))
    write_step_summary(results)
    return 1 if counts["FAIL"] or counts["XPASS"] else 0


if __name__ == "__main__":
    sys.exit(main())
