"""PM2 rows (Partiels CLI): matched templates, tracks for asp.synthesize (Partiels has no synthesizer).

Usage: python run_partiels.py OUT TEMPLATE_DIR [PARTIELS]   (venv with audiospylt; PARTIELS defaults to
/opt/Partiels/Partiels). TEMPLATE_DIR holds the PM2 .ptldoc templates of the Ircam Vamp Plugins 2.1.0 manuals.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402
from excerpts import points  # noqa: E402

# the 4.1 runner has the same file name, so load it by path: matched_template, run_one, TEMPLATES, MATCHED_NOTE
_spec = importlib.util.spec_from_file_location(
    "accuracy_run_partiels", excerpts.REPO / "benchmarks" / "accuracy" / "run_partiels.py")
acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acc)

RUNS = {"inharmonic": 5, "harmonic": 1}  # PM2 inharmonic output differs from run to run (4.1)


def main(out: pathlib.Path, template_dir: pathlib.Path, partiels: str) -> None:
    tdir = out / "partiels-templates"
    tdir.mkdir(parents=True, exist_ok=True)
    for kind in RUNS:
        src = template_dir / acc.TEMPLATES[f"pm2-{kind}"]
        if not src.is_file():  # a missing template makes the CLI hang
            raise SystemExit(f"missing {src}")
        acc.matched_template(src, tdir / f"matched-{src.name}")
    for name, kind in excerpts.EXCERPTS.items():
        wav = out / "mono" / f"{name}.wav"  # the mono signal every other row analyses
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav, excerpts.load(name).astype(np.float32), excerpts.SR, subtype="FLOAT")
        template = tdir / f"matched-{acc.TEMPLATES[f'pm2-{kind}']}"
        for run in range(1, RUNS[kind] + 1):
            row = f"pm2-{kind}/matched" + (f"/r{run}" if RUNS[kind] > 1 else "")
            tr = acc.run_one(partiels, wav, template, out / "partiels-tmp")
            points.write_points(out / "tracks" / row, name, tr["time"], tr["freq"], tr["amp"], tr["track_id"])
            print(row, name, tr["track_id"].nunique(), "tracks", flush=True)
    (out / "tracks" / "pm2.txt").write_text(
        f"Partiels 2.6.0, Ircam Vamp 2.1.0 (PM2 v5); {acc.MATCHED_NOTE}; read with audiospylt.read_partiels\n")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]),
         sys.argv[3] if len(sys.argv) > 3 else "/opt/Partiels/Partiels")
