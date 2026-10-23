"""Write the synthetic suite: OUT/suite/<name>.wav (float32 mono) + OUT/suite/truth.json.

Usage: python suite.py OUT
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import truth  # noqa: E402


def main(out: pathlib.Path) -> None:
    folder = out / "suite"
    folder.mkdir(parents=True, exist_ok=True)
    for name in truth.SIGNALS:
        x = truth.render(name)
        peak = float(np.max(np.abs(x)))
        if peak > 0.9:
            raise SystemExit(f"{name}: peak {peak:.3f} > 0.9")
        sf.write(folder / f"{name}.wav", x.astype(np.float32), truth.SR, subtype="FLOAT")
        print(f"{name}: peak {peak:.3f}")
    meta = {"sr": truth.SR, "duration": truth.DURATION, "fade": truth.FADE, "guard": truth.GUARD,
            "signals": truth.SIGNALS}
    (folder / "truth.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
