"""sms-tools row (sine model): matched analysis, sineModelSynth, tracks for asp.synthesize.

Usage: python run_sms.py OUT   (venv with sms-tools 1.2, numpy<2, soundfile)
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys

import numpy as np
from smstools.models import sineModel as SM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402
from excerpts import points  # noqa: E402

ROW = "sms-tools/matched"
# 4.1 matched setting; np.hanning(2049) is a periodic Hann of 2048 plus one zero. t -86.02 = -80 dBFS.
MATCHED = {"M": 2049, "N": 8192, "H": 256, "t": -86.02, "maxnSines": 100, "minSineDur": 0.01}
NS = 4 * MATCHED["H"]  # sineModelSynth needs its hop equal to the analysis hop; Ns = 4 H as in sms-tools' examples


def main(out: pathlib.Path) -> None:
    p = MATCHED
    for name in excerpts.EXCERPTS:
        tf, tm, tp = SM.sineModelAnal(excerpts.load(name), excerpts.SR, np.hanning(p["M"]), p["N"], p["H"], p["t"],
                                      maxnSines=p["maxnSines"], minSineDur=p["minSineDur"])
        excerpts.write_render(out, ROW, name, SM.sineModelSynth(tf, tm, tp, NS, p["H"], excerpts.SR))
        ids = points.frames_to_tracks(tf)
        times = np.repeat(np.arange(tf.shape[0]) * p["H"] / excerpts.SR, tf.shape[1]).reshape(tf.shape)
        on = tf > 0
        points.write_points(out / "tracks" / ROW, name, times[on], tf[on], 10 ** (tm[on] / 20) * 2, ids[on])
        print(ROW, name, int(ids.max()) + 1, "tracks", flush=True)
    excerpts.write_meta(out, ROW, version="sms-tools " + importlib.metadata.version("sms-tools"),
                        params={**p, "window": "np.hanning(M)", "Ns": NS}, synth="sineModelSynth",
                        scale="amp = 2 * 10**(dB/20)")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
