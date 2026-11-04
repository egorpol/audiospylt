"""Write the synthetic suite: OUT/suite/<name>.wav (float32 mono) + OUT/suite/truth.json.

Usage: python suite.py OUT [--dev]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import truth  # noqa: E402


def main(out: pathlib.Path, dev: bool = False) -> None:
    signals = {**truth.DEV_SIGNALS, **truth.DEV_MOVING} if dev else truth.SIGNALS
    if dev:
        out = out / "dev"
    folder = out / "suite"
    folder.mkdir(parents=True, exist_ok=True)
    for name in signals:
        x = truth.render(name)
        peak = float(np.max(np.abs(x)))
        if peak > 0.9:
            raise SystemExit(f"{name}: peak {peak:.3f} > 0.9")
        sf.write(folder / f"{name}.wav", x.astype(np.float32), truth.SR, subtype="FLOAT")
        print(f"{name}: peak {peak:.3f}")
    meta = {"sr": truth.SR, "duration": truth.DURATION, "fade": truth.FADE, "guard": truth.GUARD,
            "edge": truth.EDGE, "neutral": truth.NEUTRAL, "signals": signals}
    (folder / "truth.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    main(args.out, args.dev)
