"""sms-tools rows (sineModelAnal): settings of its own tests and matched window.

Usage: python run_sms.py OUT [--new]   (venv with sms-tools 1.2 + soundfile; numpy < 2)
--new analyses only truth.MOVING and adds them to the rows of an earlier full run.
sms-tools magnitudes are 20*log10(A/2); +6.02 dB turns them into D4 peak amplitude.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import sys

import numpy as np
import soundfile as sf
from smstools.models import sineModel as SM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import points  # noqa: E402
import truth  # noqa: E402

ROWS = {
    # tests/test_models_ground_truth.py (sms-tools 1.2), the only documented usage it ships
    "sms-tools/default": {"M": 1025, "N": 2048, "H": 128, "t": -80.0, "maxnSines": 25, "minSineDur": 0.01},
    # -80 dBFS floor in sms dB (A/2 scale); tracking parameters at their signature defaults
    "sms-tools/matched": {"M": 2049, "N": 8192, "H": 256, "t": -86.02, "maxnSines": 100, "minSineDur": 0.01},
}


def main(out: pathlib.Path, new: bool = False) -> None:
    version = "sms-tools " + importlib.metadata.version("sms-tools")
    if new:
        for row in ROWS:
            points.read_meta(out / "raw" / row, version)
    for row, p in ROWS.items():
        folder = out / "raw" / row
        for name in truth.MOVING if new else truth.SIGNALS:
            x, sr = sf.read(out / "suite" / f"{name}.wav")
            w = np.hanning(p["M"])
            tf, tm, _ = SM.sineModelAnal(x, sr, w, p["N"], p["H"], p["t"], maxnSines=p["maxnSines"],
                                         minSineDur=p["minSineDur"])
            ids = points.frames_to_tracks(tf)
            times = np.repeat(np.arange(tf.shape[0]) * p["H"] / sr, tf.shape[1]).reshape(tf.shape)
            on = tf > 0
            amp = 10 ** (tm[on] / 20) * 2
            points.write_points(folder, name, times[on], tf[on], amp, ids[on])
        points.write_meta(folder, kind="tracks", hop_s=p["H"] / truth.SR, version=version,
                          params={**p, "window": "np.hanning(M)"}, scale="amp = 2 * 10**(dB/20)")
        print(row, "done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--new", action="store_true")
    args = parser.parse_args()
    main(args.out, args.new)
