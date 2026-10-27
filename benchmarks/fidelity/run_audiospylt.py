"""AudioSpylt rows: v1 frame-peak resynthesis, 0.6 frame resynthesis, asp.synthesize on other tools' tracks.

Usage: python run_audiospylt.py OUT [--sets case,icmc2000,tools]   (venv with audiospylt[plot]; run after the other
runners so that their track tables in OUT/tracks/ are replayed too)

Every row runs at the excerpt's own rate. The matched rows take their window and hop in seconds (2048 and 256
samples at 44.1 kHz); `audiospylt-v1/default` keeps the function defaults in samples at every rate, as a user gets.
"""

from __future__ import annotations

import contextlib
import inspect
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

MATCHED = {"frame_s": excerpts.FRAME_S, "hop_s": excerpts.HOP_S}
DEFAULT_FADE = inspect.signature(asp.synthesize).parameters["fade"].default
V1 = {
    # function defaults: spectrogram(2048, 512, hann, zero_pad=1), find_peaks(min_db=-60), synthesize()
    "audiospylt-v1/default": {"frame_length": 2048, "hop": 512, "zero_pad": 1, "min_db": -60.0},
    "audiospylt-v1/matched": {**MATCHED, "zero_pad": 4, "min_db": -80.0},
    "audiospylt-v1/matched-nofade": {**MATCHED, "zero_pad": 4, "min_db": -80.0, "fade": 0.0},
}
# 0.6: the 4.1 matched analysis, then the tutorial render path (tutorials_tech/resynth.ipynb, as in G1); the
# synthesis grid k stays 0.15 ms (6.67 kHz) at every rate
OLD = {**MATCHED, "window_type": "hann", "thresh_amp_low": 1e-4 / 4, "thresh_amp_high": 1.0, "k": 0.00015,
       "phase_mode": "chirp", "edge_fade_s": 0.0025}
TRACK_FADE = 0.005  # as in G2


def frame_hop(p: dict, sr: int) -> tuple[int, int]:
    """Frame and hop in samples at `sr`: as given, or from seconds."""
    if "hop" in p:
        return p["frame_length"], p["hop"]
    return excerpts.samples(p["frame_s"], sr), excerpts.samples(p["hop_s"], sr)


def v1_rows(out: pathlib.Path, chosen: list[excerpts.Excerpt]) -> None:
    for row, p in V1.items():
        done = {}
        fade = p.get("fade", DEFAULT_FADE)
        if fade == 0:
            placement = "no fades"
        else:
            placement = "centred on voice ends"
        for e in chosen:
            frame, hop = frame_hop(p, e.sr)
            h = hop / e.sr
            S = asp.spectrogram(excerpts.load(e), e.sr, frame_length=frame, hop=hop, zero_pad=p["zero_pad"])
            # each frame's peaks are held over its centre +- hop/2
            ev = pd.concat([asp.snapshot_to_events(pk[["freq", "amp"]], max(0.0, t - h / 2), t + h / 2)
                            for i, t in enumerate(S.times)
                            if len(pk := asp.find_peaks(S.frame(i), min_db=p["min_db"]))], ignore_index=True)
            excerpts.write_render(out, row, e, asp.synthesize(ev, e.sr, fade=fade))
            done[e.name] = {"set": e.set, "sr": e.sr, "frame": frame, "hop": hop, "events": len(ev)}
            print(row, e.name, len(ev), "events", flush=True)
        params = {**p, "fade": fade, "window": "hann", "center": True}
        excerpts.write_meta(
            out, row, version=asp.__version__, params=params, excerpts=done,
            synth=(f"audiospylt.synthesize (continuous, link_cents 100, link_ramp 0.005, "
                   f"fade={fade}, {placement})"),
        )


def old_row(out: pathlib.Path, chosen: list[excerpts.Excerpt]) -> None:
    from audiospylt.dft_analysis import analyze_signal  # 0.6 API (shims)
    from audiospylt.generate_wave_file import render_audio
    from audiospylt.plot_wave import plot_waves

    p, row = OLD, "audiospylt-0.6/matched"
    done = {}
    for e in chosen:
        x = excerpts.load(e)
        size, hop = frame_hop(p, e.sr)
        half, h = size // 2, hop / e.sr
        xp = np.concatenate([np.zeros(half), x, np.zeros(half)])
        rows = []
        for k in range(0, len(x) // hop + 1):
            frame = xp[k * hop: k * hop + size]
            if len(frame) < size:
                break
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = analyze_signal(frame, e.sr, "bench", window_type=p["window_type"],
                                    thresh_amp_low=p["thresh_amp_low"], thresh_amp_high=p["thresh_amp_high"],
                                    show_plot=False, show_peaks=False)
            t = k * hop / e.sr
            f, a = df["Frequency (Hz)"].to_numpy(), df["Amplitude"].to_numpy()
            rows.append(pd.DataFrame({"time_start": max(0.0, t - h / 2), "time_stop": t + h / 2,
                                      "freq_start": f, "freq_stop": f, "amp_min": a, "amp_max": a}))
        table = pd.concat(rows, ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = plot_waves(table, p["k"], edge_fade_s=p["edge_fade_s"], phase_mode=p["phase_mode"],
                           show_waveform=False, show_fft=False)
            # the 0.6 render path at the excerpt's rate (render_audio takes an int rate in Hz)
            pcm, fs = render_audio(y, 1 / p["k"], fs_target_name=e.sr, alias_above_nyquist=True,
                                   save_audio=False, player=False, verbose=False, output_dir=str(out / "tmp-0.6"))
        if fs != e.sr:
            raise SystemExit(f"0.6 render_audio returned {fs} Hz for {e.name} at {e.sr} Hz")
        excerpts.write_render(out, row, e, pcm.astype(np.float64) / 2**31)  # 24-bit PCM in int32
        done[e.name] = {"set": e.set, "sr": e.sr, "frame": size, "hop": hop, "rows": len(table)}
        print(row, e.name, len(table), "rows", flush=True)
    excerpts.write_meta(out, row, version="0.6 API (dft_analysis, plot_wave, generate_wave_file)",
                        params={**p, "center": "zero-padded, frame k centred at k*hop"}, excerpts=done)


def track_rows(out: pathlib.Path, chosen: list[excerpts.Excerpt]) -> None:
    by_name = {e.name: e for e in chosen}
    for folder in sorted(p for p in (out / "tracks").glob("**/*") if any(p.glob("*.csv"))):
        tracks_row = folder.relative_to(out / "tracks").as_posix()
        row = re.sub(r"(/r\d+)?$", r"+asp\1", tracks_row, count=1)
        done = {}
        for csv in sorted(folder.glob("*.csv")):
            if (e := by_name.get(csv.stem)) is None:
                continue
            data = points.read_points(folder, csv.stem)
            tracks = pd.DataFrame({"track_id": data[:, 3].astype("int64"), "time": data[:, 0],
                                   "freq": data[:, 1], "amp": data[:, 2]})
            y = asp.synthesize(asp.tracks_to_events(tracks), e.sr, fade=TRACK_FADE)
            excerpts.write_render(out, row, e, y)
            done[e.name] = {"set": e.set, "sr": e.sr, "tracks": int(tracks["track_id"].nunique())}
            print(row, e.name, done[e.name]["tracks"], "tracks", flush=True)
        if done:
            excerpts.write_meta(out, row, version=asp.__version__, tracks=tracks_row, excerpts=done,
                                synth=f"audiospylt.synthesize(tracks_to_events(tracks), fade={TRACK_FADE})")


if __name__ == "__main__":
    args, sets = excerpts.cli(sys.argv[1:])
    out, chosen = pathlib.Path(args[0]), excerpts.excerpts(pathlib.Path(args[0]), sets)
    v1_rows(out, chosen)
    old_row(out, chosen)
    track_rows(out, chosen)
