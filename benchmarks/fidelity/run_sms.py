"""sms-tools rows: matched sine model on every set, home settings where the sound has some, sineModelSynth,
tracks for asp.synthesize.

Usage: python run_sms.py OUT [--sets case,icmc2000,tools]   (venv with sms-tools 1.2, numpy<2, soundfile)

Home settings: the sms-tools examples' defaults for the three sms-tools test sounds, and for the ICMC 2000 sounds the
SMS analysis files of the panel (`.anl`) mapped onto sms-tools, an approximation: SMS 2000 is not sms-tools, the
residual model is not run, and keys sms-tools has no counterpart for are listed per excerpt in meta.json.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys

import numpy as np
from scipy.signal import get_window
from smstools.models import harmonicModel as HM
from smstools.models import sineModel as SM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402
from excerpts import points  # noqa: E402

# 4.1 matched setting; np.hanning(frame + 1) is a periodic Hann of `frame` plus one zero. t -86.02 = -80 dBFS.
MATCHED = {"model": "sine", "window": "np.hanning", "t": -86.02, "maxnSines": 100, "minSineDur": 0.01}
# sms-tools-materials interface-models/*_function.py @ 227d7c4: sineModel on bendir, harmonicModel on vignesh,
# hprModel on sax-phrase-short (its harmonic part: hprModelSynth adds the subtracted residual, i.e. the input, back)
TOOLS_HOME = {
    "sms-bendir": {"model": "sine", "window": "hamming", "M": 2001, "N": 2048, "H": 128, "t": -80.0,
                   "minSineDur": 0.02, "maxnSines": 150, "freqDevOffset": 10.0, "freqDevSlope": 0.001},
    "sms-vignesh": {"model": "harmonic", "window": "blackman", "M": 1201, "N": 2048, "H": 128, "t": -90.0,
                    "minSineDur": 0.1, "nH": 100, "minf0": 130.0, "maxf0": 300.0, "f0et": 7.0, "harmDevSlope": 0.01},
    "sms-sax-phrase-short": {"model": "harmonic", "window": "blackman", "M": 601, "N": 1024, "H": 128, "t": -100.0,
                             "minSineDur": 0.1, "nH": 100, "minf0": 350.0, "maxf0": 700.0, "f0et": 5.0,
                             "harmDevSlope": 0.01},
}
# .anl keys that set an sms-tools parameter; every other key goes to meta.json as unmapped
ANL_MAPPED = {"InputSoundFile", "OutputSmsFile", "SineModel", "PitchDetection", "nSines", "MagThreshold",
              "LowestPitch", "HighestPitch", "WindowSizeHarm", "WindowSizeInh", "FrameRate", "LowestFreq",
              "HighestFreq"}


def fft_size(frame: int) -> int:
    """The matched zero padding: the next power of 2 at or above 4 frames."""
    return 1 << (4 * frame - 1).bit_length()


def matched(sr: int) -> dict:
    frame, hop = excerpts.samples(excerpts.FRAME_S, sr), excerpts.samples(excerpts.HOP_S, sr)
    return {**MATCHED, "M": frame + 1, "N": fft_size(frame), "H": hop}


def read_anl(path: pathlib.Path) -> dict[str, str]:
    return {k: v for k, _, v in (line.strip().partition(" ") for line in path.read_text().splitlines()) if k}


def anl_home(anl: dict[str, str], sr: int) -> tuple[dict, dict]:
    """sms-tools settings from an SMS 2000 .anl file, and its unmapped keys. Where the file is silent the matched
    value is kept (window type, hop, zero padding, minSineDur); f0et and harmDevSlope are the sms-tools example
    values. WindowSizeHarm counts periods of DefaultPitch, WindowSizeInh samples."""
    harmonic = anl["SineModel"] == "1"
    if harmonic != (anl.get("PitchDetection", "0") == "1"):
        raise SystemExit(f"{anl['InputSoundFile']}: SineModel and PitchDetection disagree")
    p = matched(sr)
    p["t"] = float(anl["MagThreshold"])
    if harmonic:
        p.update(model="harmonic", nH=int(anl["nSines"]), minf0=float(anl["LowestPitch"]),
                 maxf0=float(anl["HighestPitch"]), f0et=5.0, harmDevSlope=0.01)
        del p["maxnSines"]
        if "WindowSizeHarm" in anl:
            p["M"] = round(float(anl["WindowSizeHarm"]) * sr / float(anl["DefaultPitch"]))
    else:
        p["maxnSines"] = int(anl["nSines"])
        if "WindowSizeInh" in anl:
            p["M"] = int(anl["WindowSizeInh"])
    p["N"] = fft_size(p["M"] - 1)
    if "FrameRate" in anl:
        p["H"] = round(sr / float(anl["FrameRate"]))
    p["fmin"], p["fmax"] = float(anl.get("LowestFreq", 0)), float(anl["HighestFreq"])
    used = ANL_MAPPED | ({"DefaultPitch"} if harmonic and "WindowSizeHarm" in anl else set())
    return p, {k: v for k, v in anl.items() if k not in used}


def window(p: dict) -> np.ndarray:
    return np.hanning(p["M"]) if p["window"] == "np.hanning" else get_window(p["window"], p["M"])


def run(out: pathlib.Path, row: str, e: excerpts.Excerpt, p: dict) -> dict:
    x = excerpts.load(e)
    if p["model"] == "harmonic":
        tf, tm, tp = HM.harmonicModelAnal(x, e.sr, window(p), p["N"], p["H"], p["t"], p["nH"], p["minf0"], p["maxf0"],
                                          p["f0et"], p["harmDevSlope"], p["minSineDur"])
    else:
        extra = {k: p[k] for k in ("freqDevOffset", "freqDevSlope") if k in p}
        tf, tm, tp = SM.sineModelAnal(x, e.sr, window(p), p["N"], p["H"], p["t"], maxnSines=p["maxnSines"],
                                      minSineDur=p["minSineDur"], **extra)
    if "fmax" in p:  # the .anl frequency range: points outside it are dropped after the analysis
        off = (tf < p["fmin"]) | (tf > p["fmax"])
        tf[off], tm[off], tp[off] = 0, -100, 0
    # sineModelSynth needs its hop equal to the analysis hop; Ns = 4 H as in sms-tools' examples
    excerpts.write_render(out, row, e, SM.sineModelSynth(tf, tm, tp, 4 * p["H"], p["H"], e.sr))
    ids = points.frames_to_tracks(tf)
    times = np.repeat(np.arange(tf.shape[0]) * p["H"] / e.sr, tf.shape[1]).reshape(tf.shape)
    on = tf > 0
    points.write_points(out / "tracks" / row, e.name, times[on], tf[on], 10 ** (tm[on] / 20) * 2, ids[on])
    n = len(np.unique(ids[on]))
    print(row, e.name, n, "tracks", flush=True)
    return {"set": e.set, "sr": e.sr, "params": {**p, "Ns": 4 * p["H"]}, "tracks": n}


def main(out: pathlib.Path, sets) -> None:
    version = "sms-tools " + importlib.metadata.version("sms-tools")
    common = {"version": version, "synth": "sineModelSynth", "scale": "amp = 2 * 10**(dB/20)"}
    chosen = excerpts.excerpts(out, sets)
    done = {e.name: run(out, "sms-tools/matched", e, matched(e.sr)) for e in chosen}
    excerpts.write_meta(out, "sms-tools/matched", **common, excerpts=done)
    anl_files = excerpts.manifest()["sms_anl"]
    done = {}
    for e in chosen:
        if e.name in TOOLS_HOME:
            p = TOOLS_HOME[e.name]
            done[e.name] = {**run(out, "sms-tools/home", e, p), "source": "sms-tools-materials example @ 227d7c4"}
        elif e.name in anl_files:
            p, unmapped = anl_home(read_anl(out / "reference" / anl_files[e.name]), e.sr)
            done[e.name] = {**run(out, "sms-tools/home", e, p), "unmapped": unmapped,
                            "source": "SMS 2000 .anl mapped onto sms-tools (approximation)"}
    if done:
        excerpts.write_meta(out, "sms-tools/home", **common, excerpts=done)


if __name__ == "__main__":
    args, sets = excerpts.cli(sys.argv[1:])
    main(pathlib.Path(args[0]), sets)
