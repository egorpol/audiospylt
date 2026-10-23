"""Partiels PM2 rows: inharmonic (all files, 5 runs each: PM2 inharmonic is not reproducible) and harmonic
(harmonic files only), each with the Ircam template and a matched window.

Usage: python run_partiels.py OUT TEMPLATE_DIR [PARTIELS]
TEMPLATE_DIR holds PM2-Partials-Tracking-{Inharmonic,Harmonic}-20.ptldoc from Ircam-Vamp-Plugins-Manuals.zip
(github.com/Ircam-Partiels/ircam-vamp-plugins, release 2.1.0). PARTIELS defaults to /opt/Partiels/Partiels.
Needs audiospylt (read_partiels) in the running venv.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import points  # noqa: E402
import truth  # noqa: E402

import audiospylt as asp  # noqa: E402

RUNS = 5
TEMPLATES = {"pm2-inharmonic": "PM2-Partials-Tracking-Inharmonic-20.ptldoc",
             "pm2-harmonic": "PM2-Partials-Tracking-Harmonic-20.ptldoc"}
# PM2 parameters are list indices: overlap {3,4,8,16,32}, oversampling {1,2,4,8}, window {Blackman,Hann,Hamming}
MATCHED = {"windowsize": 2048 / truth.SR * 1000, "windowoverlapping": 2.0, "fftoversampling": 2.0,
           "windowtype": 1.0}
MATCHED_NOTE = "Hann 46.44 ms (2048 samples), overlap 8x (hop 5.805 ms), FFT oversampling 4x"


def matched_template(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy src with MATCHED values in every PM2 track (text level: .ptldoc writes '&#10' without ';')."""
    text = src.read_text()
    blocks = text.split("<tracks ")
    changed = 0
    for i, block in enumerate(blocks):
        if 'identifier="pm2:' not in block:
            continue
        for key, value in MATCHED.items():
            block, n = re.subn(rf'(<parameters key="{key}" value=")[^"]*(")', rf"\g<1>{value!r}\g<2>", block)
            if n != 2:  # state and defaultState
                raise SystemExit(f"{src.name}: {key} found {n} times in a PM2 track")
        blocks[i] = block
        changed += 1
    if changed not in (19, 20):
        raise SystemExit(f"{src.name}: {changed} PM2 tracks")
    dst.write_text("<tracks ".join(blocks))


def run_one(partiels: str, wav: pathlib.Path, template: pathlib.Path, tmp: pathlib.Path):
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    cmd = [partiels, "--export", "-i", str(wav), "-t", str(template), "-o", str(tmp), "-f", "json", "--description"]
    subprocess.run(cmd, check=True, timeout=300, capture_output=True)
    tracks = asp.read_partiels(tmp)  # skips the F0 and FFT tracks
    shutil.rmtree(tmp)
    return tracks


def main(out: pathlib.Path, template_dir: pathlib.Path, partiels: str) -> None:
    tdir = out / "partiels-templates"
    tdir.mkdir(parents=True, exist_ok=True)
    harmonic = [n for n, s in truth.SIGNALS.items() if s["kind"] == "f0"]
    for tool, fname in TEMPLATES.items():
        src = template_dir / fname
        if not src.is_file():  # a missing template makes the CLI hang
            raise SystemExit(f"missing {src}")
        matched_template(src, tdir / f"matched-{fname}")
        for setting, template in (("template", src), ("matched", tdir / f"matched-{fname}")):
            names = harmonic if tool == "pm2-harmonic" else list(truth.SIGNALS)
            for run in range(1, (RUNS if tool == "pm2-inharmonic" else 1) + 1):
                row = f"{tool}/{setting}" + (f"/r{run}" if tool == "pm2-inharmonic" else "")
                folder = out / "raw" / row
                steps = []
                for name in names:
                    tr = run_one(partiels, out / "suite" / f"{name}.wav", template, out / "partiels-tmp")
                    points.write_points(folder, name, tr["time"], tr["freq"], tr["amp"], tr["track_id"])
                    steps += [np.diff(g["time"].to_numpy()) for _, g in tr.groupby("track_id") if len(g) > 1]
                hop = float(np.median(np.concatenate(steps)))
                points.write_meta(folder, kind="tracks", hop_s=hop, version="Partiels 2.6.0, Ircam Vamp 2.1.0 (PM2 v5)",
                                  params={"template": fname, "matched": MATCHED_NOTE if setting == "matched" else None},
                                  signals=names, hop_note="median frame step within tracks")
                print(row, "done, hop", round(hop, 5), flush=True)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]),
         sys.argv[3] if len(sys.argv) > 3 else "/opt/Partiels/Partiels")
