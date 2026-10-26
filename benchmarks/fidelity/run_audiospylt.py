"""AudioSpylt rows: v1 frame-peak resynthesis, 0.6 frame resynthesis, asp.synthesize on other tools' tracks.

Usage: python run_audiospylt.py OUT   (venv with audiospylt[plot]; run after the other runners so that their
track tables in OUT/tracks/ are replayed too)
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import excerpts  # noqa: E402
from excerpts import points  # noqa: E402

import audiospylt as asp  # noqa: E402

SR = excerpts.SR
V1 = {
    # function defaults: spectrogram(2048, 512, hann, zero_pad=1), find_peaks(min_db=-60), synthesize()
    "audiospylt-v1/default": {"frame_length": 2048, "hop": 512, "zero_pad": 1, "min_db": -60.0, "fade": 0.0},
    "audiospylt-v1/matched": {"frame_length": 2048, "hop": 256, "zero_pad": 4, "min_db": -80.0, "fade": 0.0},
    "audiospylt-v1/matched-fade": {"frame_length": 2048, "hop": 256, "zero_pad": 4, "min_db": -80.0, "fade": 0.0025},
}
# 0.6: the 4.1 matched analysis, then the tutorial render path (tutorials_tech/resynth.ipynb, as in G1)
OLD = {"frame_length": 2048, "hop": 256, "window_type": "hann", "thresh_amp_low": 1e-4 / 4, "thresh_amp_high": 1.0,
       "k": 0.00015, "phase_mode": "chirp", "edge_fade_s": 0.0025}
TRACK_FADE = 0.005  # as in G2


def v1_rows(out: pathlib.Path) -> None:
    for row, p in V1.items():
        h = p["hop"] / SR
        counts = {}
        for name in excerpts.EXCERPTS:
            S = asp.spectrogram(excerpts.load(name), SR, frame_length=p["frame_length"], hop=p["hop"],
                                zero_pad=p["zero_pad"])
            # each frame's peaks are held over its centre +- hop/2
            ev = pd.concat([asp.snapshot_to_events(pk[["freq", "amp"]], max(0.0, t - h / 2), t + h / 2)
                            for i, t in enumerate(S.times)
                            if len(pk := asp.find_peaks(S.frame(i), min_db=p["min_db"]))], ignore_index=True)
            excerpts.write_render(out, row, name, asp.synthesize(ev, SR, fade=p["fade"]))
            counts[name] = len(ev)
            print(row, name, len(ev), "events", flush=True)
        excerpts.write_meta(out, row, version=asp.__version__, params={**p, "window": "hann", "center": True},
                            synth="audiospylt.synthesize (continuous, link_cents 100, link_ramp 0.005)", events=counts)


def old_row(out: pathlib.Path) -> None:
    from audiospylt.dft_analysis import analyze_signal  # 0.6 API (shims)
    from audiospylt.generate_wave_file import render_audio
    from audiospylt.plot_wave import plot_waves

    p, row = OLD, "audiospylt-0.6/matched"
    half, h = p["frame_length"] // 2, p["hop"] / SR
    counts = {}
    for name in excerpts.EXCERPTS:
        x = excerpts.load(name)
        xp = np.concatenate([np.zeros(half), x, np.zeros(half)])
        rows = []
        for k in range(0, len(x) // p["hop"] + 1):
            frame = xp[k * p["hop"]: k * p["hop"] + p["frame_length"]]
            if len(frame) < p["frame_length"]:
                break
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = analyze_signal(frame, SR, "bench", window_type=p["window_type"],
                                    thresh_amp_low=p["thresh_amp_low"], thresh_amp_high=p["thresh_amp_high"],
                                    show_plot=False, show_peaks=False)
            t = k * p["hop"] / SR
            f, a = df["Frequency (Hz)"].to_numpy(), df["Amplitude"].to_numpy()
            rows.append(pd.DataFrame({"time_start": max(0.0, t - h / 2), "time_stop": t + h / 2,
                                      "freq_start": f, "freq_stop": f, "amp_min": a, "amp_max": a}))
        table = pd.concat(rows, ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = plot_waves(table, p["k"], edge_fade_s=p["edge_fade_s"], phase_mode=p["phase_mode"],
                           show_waveform=False, show_fft=False)
            pcm, fs = render_audio(y, 1 / p["k"], fs_target_name="44.1kHz", alias_above_nyquist=True,
                                   save_audio=False, player=False, verbose=False, output_dir=str(out / "tmp-0.6"))
        if fs != SR:
            raise SystemExit(f"0.6 render_audio returned {fs} Hz")
        excerpts.write_render(out, row, name, pcm.astype(np.float64) / 2**31)  # 24-bit PCM in int32
        counts[name] = len(table)
        print(row, name, len(table), "rows", flush=True)
    excerpts.write_meta(out, row, version="0.6 API (dft_analysis, plot_wave, generate_wave_file)",
                        params={**p, "center": "zero-padded, frame k centred at k*hop"}, events=counts)


def track_rows(out: pathlib.Path) -> None:
    for folder in sorted(p for p in (out / "tracks").glob("**/*") if any(p.glob("*.csv"))):
        tracks_row = folder.relative_to(out / "tracks").as_posix()
        row = re.sub(r"(/r\d+)?$", r"+asp\1", tracks_row, count=1)
        for csv in sorted(folder.glob("*.csv")):
            data = points.read_points(folder, csv.stem)
            tracks = pd.DataFrame({"track_id": data[:, 3].astype("int64"), "time": data[:, 0],
                                   "freq": data[:, 1], "amp": data[:, 2]})
            y = asp.synthesize(asp.tracks_to_events(tracks), SR, fade=TRACK_FADE)
            excerpts.write_render(out, row, csv.stem, y)
            print(row, csv.stem, tracks["track_id"].nunique(), "tracks", flush=True)
        excerpts.write_meta(out, row, version=asp.__version__, tracks=tracks_row,
                            synth=f"audiospylt.synthesize(tracks_to_events(tracks), fade={TRACK_FADE})")


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    v1_rows(out)
    old_row(out)
    track_rows(out)
