#!/usr/bin/env python3
"""Deliberately replace 0.6.3 goldens from the checked-out package (offline)."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["MPLBACKEND"] = "Agg"
os.environ["PLOTLY_RENDERER"] = "json"

from contextlib import redirect_stdout
import io
import json
import tempfile
from tests import _cases as cases
from tests._support import GOLDEN_DIR, offline_display, snapshot


def main():
    results = {}
    # All transient WAV/MEI/SVG files stay in the owned tests tree.
    with tempfile.TemporaryDirectory(prefix="regenerate-", dir=GOLDEN_DIR) as scratch:
        old_cwd = Path.cwd()
        try:
            os.chdir(scratch)
            with offline_display(), redirect_stdout(io.StringIO()):
                for mono in (True, False):
                    results[f"audio/load/{mono}"] = cases.audio_load_case(scratch, mono)
                for fades in (True, False):
                    results[f"audio/trim/{fades}"] = cases.audio_trim_case(fades)
                for axis in ("linear", "log", "mixed", "mel"):
                    _, info = cases.spectrogram_case(axis)
                    results[f"audio/spec/{axis}"] = info
                _, info = cases.spectrogram_case(boundary="zeros_end", padded=True, time_reference="start")
                results["audio/spec/padded"] = info
                results["dft"] = cases.dft_case()
                for name in cases.F0_CASES:
                    results[f"f0/{name}"] = cases.f0_numeric(cases.f0_case(name))
                results["f0/summary"] = cases.f0_summary_case()
                for name in cases.TRANSFORM_CASES:
                    results[f"df/{name}"] = cases.transform_case(name)
                for name in cases.WAVE_CASES:
                    results[f"wave/{name}"] = cases.wave_case(name)
                for phase in ("chirp", "reset", "global"):
                    results[f"wave/overlap/{phase}"] = cases.overlap_case(phase)
                results["wave/row_fades"] = cases.row_fades_case()
                for bits in (16, 24):
                    results[f"export/{bits}"] = cases.export_case(scratch, bits)
                for alias in (False, True):
                    results[f"export/resample/{alias}"] = cases.export_case(scratch, 16, 6000, alias)
                for mode in ("sum", "mean"):
                    results[f"export/mix/{mode}"] = cases.mix_case(mode)
                results["io/events"] = cases.io_events_case()
                results["io/tsv"] = cases.io_tsv_case(scratch)
                results["mei/pitch"] = cases.mei_pitch_case()
                for mode in ("sequence", "chord"):
                    results[f"mei/{mode}"] = cases.mei_string_case(mode)
                for label in ("fing", "dir"):
                    results[f"mei/temporal/{label}"] = cases.mei_temporal_case(label)
                for temporal in (False, True):
                    results[f"mei/process/{temporal}"] = cases.mei_process_case(scratch, temporal)
        finally:
            os.chdir(old_cwd)
    target = GOLDEN_DIR / "current.json"
    target.write_text(json.dumps(snapshot(results), sort_keys=True, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(f"Wrote {len(results)} snapshots to {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
