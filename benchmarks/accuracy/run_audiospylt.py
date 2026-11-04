"""AudioSpylt rows: v1 per-frame QIFFT peaks (defaults, matched) 0.6 peaks (matched window), and v1 tracks.

Usage: python run_audiospylt.py OUT [--dev]   (venv with audiospylt installed)
--dev runs only the tracked rows, on OUT/dev/suite into OUT/dev/raw: score them with evaluate.py OUT/dev RESULTS.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import pathlib
import sys
import warnings

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import points  # noqa: E402
import truth  # noqa: E402

import audiospylt as asp  # noqa: E402

ROWS = {
    # function defaults: spectrogram(2048, 512, hann, zero_pad=1), find_peaks(min_db=-60)
    "audiospylt-v1/default": {"frame_length": 2048, "hop": 512, "zero_pad": 1, "min_db": -60.0},
    "audiospylt-v1/matched": {"frame_length": 2048, "hop": 256, "zero_pad": 4, "min_db": -80.0},
}
# Rows with a gap are diagnostic settings, shown with their noise cost; they are not recommendations.
TRACKED = {
    "audiospylt-v1/tracked": {},
    "audiospylt-v1/tracked-bh": {"window": "blackmanharris"},
    "audiospylt-v1/tracked-lp": {"prediction": "lp"},
    "audiospylt-v1/tracked-gap24": {"max_gap": 24},
    "audiospylt-v1/tracked-lp-gap24": {"prediction": "lp", "max_gap": 24},
    "audiospylt-v1/tracked-gap60": {"max_gap": 60},
    "audiospylt-v1/tracked-lp-gap60": {"prediction": "lp", "max_gap": 60},
}
OLD = {"frame_length": 2048, "hop": 256, "window_type": "hann",
       # 0.6 thresholds act on unscaled rfft/N magnitudes: a Hann sine of amp A reads A/4, so -80 dBFS = 1e-4/4
       "thresh_amp_low": 1e-4 / 4, "thresh_amp_high": 1.0}


def v1_rows(out: pathlib.Path) -> None:
    for row, p in ROWS.items():
        folder = out / "raw" / row
        for name in truth.SIGNALS:
            x, sr = sf.read(out / "suite" / f"{name}.wav")
            S = asp.spectrogram(x, sr, frame_length=p["frame_length"], hop=p["hop"], zero_pad=p["zero_pad"])
            cols = []
            for i, t in enumerate(S.times):
                pk = asp.find_peaks(S.frame(i), min_db=p["min_db"])
                cols.append(np.column_stack([np.full(len(pk), t), pk["freq"], pk["amp"]]))
            data = np.vstack(cols)
            points.write_points(folder, name, data[:, 0], data[:, 1], data[:, 2], -np.ones(len(data)))
        points.write_meta(folder, kind="frames", hop_s=p["hop"] / truth.SR, version=asp.__version__,
                          params={**p, "window": "hann", "center": True})
        print(row, "done")


def tracked_rows(out: pathlib.Path, names: list[str]) -> None:
    defaults = {k: v.default for k, v in inspect.signature(asp.track_partials).parameters.items()
                if v.default is not inspect.Parameter.empty}
    for row, overrides in TRACKED.items():
        params = {**defaults, **overrides}
        folder = out / "raw" / row
        for name in names:
            x, sr = sf.read(out / "suite" / f"{name}.wav")
            tracks = asp.track_partials(x, sr, **overrides)
            points.write_points(folder, name, tracks["time"], tracks["freq"], tracks["amp"], tracks["track_id"])
        points.write_meta(folder, kind="tracks", hop_s=params["hop"] / truth.SR,
                          params=params, version=asp.__version__, signals=names)
        print(row, "done", flush=True)


def old_row(out: pathlib.Path) -> None:
    from audiospylt.dft_analysis import analyze_signal  # 0.6 API (shim)

    p = OLD
    folder = out / "raw" / "audiospylt-0.6/matched"
    half = p["frame_length"] // 2
    for name in truth.SIGNALS:
        x, sr = sf.read(out / "suite" / f"{name}.wav")
        xp = np.concatenate([np.zeros(half), x, np.zeros(half)])
        cols = []
        for k in range(0, len(x) // p["hop"] + 1):
            frame = xp[k * p["hop"]: k * p["hop"] + p["frame_length"]]
            if len(frame) < p["frame_length"]:
                break
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = analyze_signal(frame, sr, "bench", window_type=p["window_type"],
                                    thresh_amp_low=p["thresh_amp_low"], thresh_amp_high=p["thresh_amp_high"],
                                    show_plot=False, show_peaks=False)
            t = k * p["hop"] / sr
            cols.append(np.column_stack([np.full(len(df), t), df["Frequency (Hz)"], df["Amplitude"]]))
        data = np.vstack(cols)
        points.write_points(folder, name, data[:, 0], data[:, 1], data[:, 2], -np.ones(len(data)))
    points.write_meta(folder, kind="frames", hop_s=p["hop"] / truth.SR, version="0.6 API (dft_analysis)",
                      params={**p, "center": "zero-padded, frame k centred at k*hop"})
    print("audiospylt-0.6/matched done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    if args.dev:
        tracked_rows(args.out / "dev", [*truth.DEV_SIGNALS, *truth.DEV_MOVING])
    else:
        v1_rows(args.out)
        old_row(args.out)
        tracked_rows(args.out, list(truth.SIGNALS))
