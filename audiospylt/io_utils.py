import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd


def save_df_tsv(df, filename, *, index=False, verbose=True, make_dirs=True):
    """
    Save a pandas DataFrame to a TSV file with a consistent log message.

    - Creates parent directories by default (make_dirs=True).
    - Raises on error (so notebooks fail loudly when something is wrong).
    """
    if make_dirs:
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)

    df.to_csv(filename, sep="\t", index=index)

    if verbose:
        cwd = os.getcwd()
        ts = datetime.now()
        print(f"Data saved successfully to {cwd}\\{filename} at {ts}.")

    return filename


def events_df_from_peaks_by_interval(
    peaks_by_interval: Sequence[Mapping[str, Any]],
    *,
    freq_col: str = "Frequency (Hz)",
    amp_col: str = "Amplitude",
) -> pd.DataFrame:
    """
    Convert a `peaks_by_interval` list (as produced in notebooks) into a unified event table.

    Expected input shape:
      peaks_by_interval = [
        {"time_start": float, "time_stop": float, "peaks_df": pd.DataFrame, ...},
        ...
      ]

    Expected peaks_df columns (defaults match `audiospylt.dft_analysis.analyze_signal`):
      - freq_col: "Frequency (Hz)"
      - amp_col: "Amplitude"

    Returns DataFrame with columns:
      freq_start, freq_stop, time_start, time_stop, amp_min, amp_max
    """
    rows: List[pd.DataFrame] = []
    for item in peaks_by_interval:
        if "time_start" not in item or "time_stop" not in item or "peaks_df" not in item:
            raise KeyError("Each item must have keys: 'time_start', 'time_stop', 'peaks_df'")

        t0 = float(item["time_start"])
        t1 = float(item["time_stop"])
        peaks_df = item["peaks_df"]

        if peaks_df is None:
            continue
        if not isinstance(peaks_df, pd.DataFrame):
            peaks_df = pd.DataFrame(peaks_df)
        if peaks_df.empty:
            continue
        if freq_col not in peaks_df.columns or amp_col not in peaks_df.columns:
            raise KeyError(f"peaks_df must contain columns {freq_col!r} and {amp_col!r}")

        freqs = peaks_df[freq_col].astype(float)
        amps = peaks_df[amp_col].astype(float)
        rows.append(
            pd.DataFrame(
                {
                    "freq_start": freqs,
                    "freq_stop": freqs,
                    "time_start": t0,
                    "time_stop": t1,
                    "amp_min": amps,
                    "amp_max": amps,
                }
            )
        )

    events_df = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["freq_start", "freq_stop", "time_start", "time_stop", "amp_min", "amp_max"])
    )

    return events_df

