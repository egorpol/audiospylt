import pandas as pd
import numpy as np
from scipy.special import expit

def detect_dataframe_type(df: pd.DataFrame) -> dict:
    """
    Detects if the dataframe is single-frame (spectral snapshot) or multi-frame (temporal).
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        Dictionary with keys:
        - 'type': 'single_frame', 'multi_frame', or 'unknown'
        - 'time_cues': List of unique time_start values (for multi_frame)
        - 'count': Number of time cues
        - 'columns': List of columns in the dataframe
        - 'pairs': Inferred number of freq/amp pairs (for single_frame)
    """
    columns = df.columns.tolist()
    
    # Check for multi-frame columns
    # Must have freq start/stop and time start/stop
    multi_cols = ['freq_start', 'freq_stop', 'time_start', 'time_stop']
    if all(col in columns for col in multi_cols):
        unique_starts = sorted(df['time_start'].unique())
        return {
            'type': 'multi_frame',
            'count': len(unique_starts),
            'time_cues': [float(t) for t in unique_starts]
        }
        
    # Check for single-frame columns
    # Heuristic: look for 'freq' and 'amp' in column names (case insensitive)
    # e.g. "Frequency (Hz)", "Amplitude"
    lower_cols = [c.lower() for c in columns]
    if any('freq' in c for c in lower_cols) and any('amp' in c for c in lower_cols):
        freq_cols = [c for c in columns if "freq" in c.lower()]
        amp_cols = [c for c in columns if "amp" in c.lower()]

        pairs = None
        try:
            # Common "long" format: one freq col + one amp col, one row per partial
            if len(freq_cols) == 1 and len(amp_cols) == 1:
                f = df[freq_cols[0]]
                a = df[amp_cols[0]]
                pairs = int((f.notna() & a.notna()).sum())
            else:
                # Heuristic for "wide" format: freq1..freqN and amp1..ampN
                # If we can match numbered suffixes, count matched indices; otherwise fall back.
                import re

                def suffix_int(col: str):
                    m = re.search(r"(\d+)\s*$", str(col))
                    return int(m.group(1)) if m else None

                freq_idx = {suffix_int(c): c for c in freq_cols if suffix_int(c) is not None}
                amp_idx = {suffix_int(c): c for c in amp_cols if suffix_int(c) is not None}
                common = sorted(set(freq_idx) & set(amp_idx))
                if common:
                    # Count per-row matches where both values exist, then sum.
                    pairs = 0
                    for i in common:
                        f = df[freq_idx[i]]
                        a = df[amp_idx[i]]
                        pairs += int((f.notna() & a.notna()).sum())
                else:
                    # Fallback: assume columns are already aligned; count min(freq, amp) per row.
                    pairs = int(min(len(freq_cols), len(amp_cols)) * len(df))
        except Exception:
            pairs = None

        return {
            'type': 'single_frame',
            'pairs': pairs,
            'freq_cols': freq_cols,
            'amp_cols': amp_cols,
        }
        
    return {
        'type': 'unknown',
        'columns': columns
    }

def process_spectral_dataframe(
    df: pd.DataFrame,
    stretch_factor=1.0,
    pitch_factor=1.0,
    pitch_cents=None,
    pitch_min_hz=None,
    pitch_max_hz=None,
    pitch_overshoot="clip",
    print_overshoot_freqs=False,
) -> pd.DataFrame:
    """
    Applies time stretch and pitch shift to a spectral dataframe.
    Handles both single-frame and multi-frame structures automatically.
    
    Args:
        df: Input DataFrame
        stretch_factor: float or list/array of floats. 
                       If list, must match length of detected time cues (multi-frame).
        pitch_factor: float or list/array of floats.
                     If list, must match length of detected time cues (multi-frame).
        pitch_cents: float or list/array of floats in cents.
                     If list, must match length of detected time cues (multi-frame).
                     If provided, it multiplies pitch_factor by 2**(cents/1200).
        pitch_min_hz: float minimum frequency threshold (Hz). If None, no min bound.
        pitch_max_hz: float maximum frequency threshold (Hz). If None, no max bound.
        pitch_overshoot: How to handle out-of-range values when thresholds set.
            - "clip": clamp to min/max
            - "allow": keep overshot values
            - "threshold": drop rows outside thresholds
        print_overshoot_freqs: If True, display out-of-bounds freq pairs.
                      
    Returns:
        Transformed DataFrame copy.
    """
    df_out = df.copy()
    info = detect_dataframe_type(df_out)

    def cents_to_factor(cents):
        return 2 ** (np.array(cents, dtype=float) / 1200.0)

    def apply_pitch_bounds(df_local, freq_cols, df_original=None):
        if pitch_min_hz is None and pitch_max_hz is None:
            return df_local
        if pitch_overshoot not in {"clip", "allow", "threshold"}:
            raise ValueError("pitch_overshoot must be 'clip', 'allow', or 'threshold'.")
        if pitch_min_hz is not None and pitch_max_hz is not None:
            if float(pitch_min_hz) > float(pitch_max_hz):
                raise ValueError("pitch_min_hz must be <= pitch_max_hz.")

        below_mask = pd.Series(False, index=df_local.index)
        above_mask = pd.Series(False, index=df_local.index)
        if pitch_min_hz is not None:
            below_mask = df_local[freq_cols].lt(float(pitch_min_hz)).any(axis=1)
        if pitch_max_hz is not None:
            above_mask = df_local[freq_cols].gt(float(pitch_max_hz)).any(axis=1)
        out_of_bounds = below_mask | above_mask
        if out_of_bounds.any():
            min_str = str(pitch_min_hz) if pitch_min_hz is not None else "-inf"
            max_str = str(pitch_max_hz) if pitch_max_hz is not None else "inf"
            min_seen = df_local[freq_cols].min().min()
            max_seen = df_local[freq_cols].max().max()
            below_count = int(below_mask.sum())
            above_count = int(above_mask.sum())
            extra = (
                f" (below: {below_count}, above: {above_count}; "
                f"observed min/max: {min_seen:g}/{max_seen:g})"
            )
            print(
                f"Warning: {int(out_of_bounds.sum())} rows outside pitch bounds "
                f"[{min_str}, {max_str}].{extra}"
            )
            if print_overshoot_freqs:
                overshot_pitched = df_local.loc[out_of_bounds, freq_cols]
                if df_original is not None:
                    overshot_initial = df_original.loc[out_of_bounds, freq_cols]
                else:
                    overshot_initial = overshot_pitched

                rows = []
                for col in freq_cols:
                    for idx in overshot_pitched.index:
                        rows.append(
                            {
                                "freq_initial": float(overshot_initial.at[idx, col]),
                                "freq_pitched": float(overshot_pitched.at[idx, col]),
                            }
                        )
                overshot_df = pd.DataFrame(rows)
                print("Overshot frequencies (initial -> pitched):")
                try:
                    from IPython.display import display

                    display(overshot_df)
                except Exception:
                    print(overshot_df.to_string(index=False))

        if pitch_overshoot == "allow":
            return df_local

        if pitch_overshoot == "clip":
            for col in freq_cols:
                if pitch_min_hz is not None:
                    df_local[col] = df_local[col].clip(lower=float(pitch_min_hz))
                if pitch_max_hz is not None:
                    df_local[col] = df_local[col].clip(upper=float(pitch_max_hz))
            return df_local

        # "threshold" -> drop rows outside bounds
        mask = pd.Series(True, index=df_local.index)
        if pitch_min_hz is not None:
            for col in freq_cols:
                mask &= df_local[col] >= float(pitch_min_hz)
        if pitch_max_hz is not None:
            for col in freq_cols:
                mask &= df_local[col] <= float(pitch_max_hz)
        return df_local.loc[mask].reset_index(drop=True)
    
    if info['type'] == 'unknown':
        print(f"Warning: Could not detect DataFrame type. Columns: {info['columns']}")
        return df_out
        
    if info['type'] == 'single_frame':
        # --- Single Frame Processing ---
        
        # Resolve pitch factor (scalar only for single frame)
        p_factor = pitch_factor
        if hasattr(pitch_factor, '__len__') and not isinstance(pitch_factor, str):
             if len(pitch_factor) > 0:
                 p_factor = pitch_factor[0]
             else:
                 p_factor = 1.0

        if pitch_cents is not None:
            c_factor = pitch_cents
            if hasattr(pitch_cents, '__len__') and not isinstance(pitch_cents, str):
                if len(pitch_cents) > 0:
                    c_factor = pitch_cents[0]
                else:
                    c_factor = 0.0
            p_factor = p_factor * float(cents_to_factor(c_factor))
        
        # Apply pitch
        freq_col = next((c for c in df_out.columns if 'freq' in c.lower()), None)
        df_before_pitch = df_out.copy()
        if freq_col:
            df_out[freq_col] *= p_factor

        if freq_col:
            df_out = apply_pitch_bounds(df_out, [freq_col], df_before_pitch)
            
        # Warn about stretch if it's not 1.0
        has_stretch = False
        if hasattr(stretch_factor, '__len__') and not isinstance(stretch_factor, str):
            if any(s != 1.0 for s in stretch_factor):
                has_stretch = True
        elif stretch_factor != 1.0:
            has_stretch = True
            
        if has_stretch:
            print("Info: Time stretch ignored for single-frame DataFrame (no time dimension).")
            
        return df_out

    elif info['type'] == 'multi_frame':
        # --- Multi Frame Processing ---
        time_cues = info['time_cues']
        n_cues = info['count']
        
        # Helper to normalize factors to list
        def normalize_factor(factor, count, name):
            if np.isscalar(factor):
                return [factor] * count
            elif hasattr(factor, '__len__'):
                if len(factor) != count:
                    raise ValueError(f"{name} factor length ({len(factor)}) must match number of time cues ({count}).")
                return factor
            else:
                 return [factor] * count

        s_factors = normalize_factor(stretch_factor, n_cues, "Stretch")
        p_factors = normalize_factor(pitch_factor, n_cues, "Pitch")

        if pitch_cents is None:
            c_factors = [1.0] * n_cues
        else:
            c_cents = normalize_factor(pitch_cents, n_cues, "Pitch cents")
            c_factors = [float(c) for c in cents_to_factor(c_cents)]

        p_factors = [pf * cf for pf, cf in zip(p_factors, c_factors)]
        
        # Maps for quick lookup: time_start -> factor
        stretch_map = dict(zip(time_cues, s_factors))
        pitch_map = dict(zip(time_cues, p_factors))
        
        # 1. Apply Pitch Shift
        df_before_pitch = df_out.copy()
        row_pitch_factors = df_out['time_start'].map(pitch_map)
        df_out['freq_start'] *= row_pitch_factors
        df_out['freq_stop'] *= row_pitch_factors

        df_out = apply_pitch_bounds(df_out, ['freq_start', 'freq_stop'], df_before_pitch)
        
        # 2. Apply Time Stretch (Cumulative)
        
        sorted_cues = sorted(time_cues)
        new_start_map = {}
        
        # Anchor: The first time cue stays fixed (standard stretch behavior relative to start)
        prev_old = sorted_cues[0]
        prev_new = sorted_cues[0]
        new_start_map[prev_old] = prev_new
        
        for i in range(1, len(sorted_cues)):
            curr_old = sorted_cues[i]
            
            # Duration of the gap between this cue and the previous one
            delta = curr_old - prev_old
            
            # Use the stretch factor of the previous segment
            factor = stretch_map[prev_old]
            
            new_delta = delta * factor
            curr_new = prev_new + new_delta
            
            new_start_map[curr_old] = curr_new
            
            prev_old = curr_old
            prev_new = curr_new
            
        # Map new start times
        new_starts = df_out['time_start'].map(new_start_map)
        
        # Calculate new stop times based on new starts + stretched durations
        row_stretch_factors = df_out['time_start'].map(stretch_map)
        old_durations = df_out['time_stop'] - df_out['time_start']
        new_durations = old_durations * row_stretch_factors
        
        df_out['time_start'] = new_starts
        df_out['time_stop'] = new_starts + new_durations
        
        return df_out

def expand_to_multi(df: pd.DataFrame, duration: float) -> pd.DataFrame:
    """
    Expands a single-frame spectral dataframe to a multi-frame dataframe with a single time segment.
    The spectral content is assumed constant from time 0 to duration.
    
    Args:
        df: Single-frame DataFrame (Frequency/Amplitude).
        duration: Total duration of the segment in seconds.
        
    Returns:
        Multi-frame DataFrame with columns:
        freq_start, freq_stop, time_start, time_stop, amp_min, amp_max
    """
    info = detect_dataframe_type(df)
    if info['type'] != 'single_frame':
        raise ValueError("Input dataframe must be single-frame.")
        
    freq_col = next((c for c in df.columns if 'freq' in c.lower()), None)
    amp_col = next((c for c in df.columns if 'amp' in c.lower()), None)
    
    if not freq_col or not amp_col:
        raise ValueError("Could not identify Frequency or Amplitude columns.")
        
    new_df = pd.DataFrame()
    new_df['freq_start'] = df[freq_col]
    new_df['freq_stop'] = df[freq_col]
    new_df['time_start'] = 0.0
    new_df['time_stop'] = float(duration)
    new_df['amp_min'] = df[amp_col]
    new_df['amp_max'] = df[amp_col]
    
    return new_df

def add_time_cues(df: pd.DataFrame, cues: list) -> pd.DataFrame:
    """
    Adds time slices to a multi-frame dataframe by splitting existing segments.
    
    Args:
        df: Multi-frame DataFrame.
        cues: List of time points (floats) where cuts should happen.
        
    Returns:
        New Multi-frame DataFrame with additional segments.
    """
    info = detect_dataframe_type(df)
    if info['type'] != 'multi_frame':
        raise ValueError("Input dataframe must be multi-frame.")
    
    # Clean cues: sorted, unique, ignoring existing starts/stops to avoid zero-length segments or duplication issues?
    # Actually, we need to split segments.
    # A cue splits a segment if start < cue < stop.
    
    # Current unique boundaries
    # We really only care about 'time_start' defining the segments for processing,
    # but strictly speaking each row is a partial definition.
    # Usually in this format, rows with same time_start share the same time_stop.
    
    df_out = df.copy()
    
    # Filter valid cues: must be > min_start and < max_stop of the whole structure?
    # Or should we just iterate through rows?
    # Faster approach:
    # 1. Identify all unique segments (start, stop).
    # 2. For each segment, check which cues fall inside (start < cue < stop).
    # 3. If cues fall inside, split the segment into parts.
    
    # It's safer to process per-row or per-segment-group.
    # Let's iterate over unique (time_start, time_stop) pairs to handle groups of partials together.
    
    valid_cues = sorted(list(set(cues)))
    
    # We will build a list of new DataFrames to concatenate
    new_rows = []
    
    # Group by time structure to avoid mismatched row splitting if data is consistent
    # But rows might have different durations? usually not in this project's context (vertical slices).
    # Let's assume vertical slices are consistent for now, or just handle row by row.
    # Handling row-by-row is safest for general cases.
    
    for _, row in df_out.iterrows():
        t_start = row['time_start']
        t_stop = row['time_stop']
        
        # Find cues that are strictly inside this row's interval
        internal_cues = [c for c in valid_cues if t_start < c < t_stop]
        
        if not internal_cues:
            # No split needed
            new_rows.append(row.to_frame().T)
        else:
            # Need to split
            # e.g. start=0, stop=1, cue=0.5
            # Seg 1: 0 -> 0.5. Freq/Amp interpolation?
            # User said "single frame values at beginning and at the end" for expand,
            # but for split, we should interpolate freq/amp linear to time.
            
            # Interpolation logic:
            # frac = (t_current - t_start) / (t_stop - t_start)
            # val_current = val_start + frac * (val_stop - val_start)
            
            boundaries = [t_start] + internal_cues + [t_stop]
            
            for i in range(len(boundaries) - 1):
                b1 = boundaries[i]
                b2 = boundaries[i+1]
                
                # Interpolate values at b1 and b2
                # Note: b1 is start of new sub-segment, b2 is end.
                
                # However, b1 matches previous end (except for first), so we can reuse?
                # Calculating explicitly is safer.
                
                # Original values
                f_s = row['freq_start']
                f_e = row['freq_stop']
                a_s = row['amp_min']
                a_e = row['amp_max']
                orig_dur = t_stop - t_start
                
                # Interpolate Start of sub-segment
                frac1 = (b1 - t_start) / orig_dur
                new_f_s = f_s + frac1 * (f_e - f_s)
                new_a_s = a_s + frac1 * (a_e - a_s)
                
                # Interpolate End of sub-segment
                frac2 = (b2 - t_start) / orig_dur
                new_f_e = f_s + frac2 * (f_e - f_s)
                new_a_e = a_s + frac2 * (a_e - a_s)
                
                new_row = row.copy()
                new_row['time_start'] = b1
                new_row['time_stop'] = b2
                new_row['freq_start'] = new_f_s
                new_row['freq_stop'] = new_f_e
                new_row['amp_min'] = new_a_s
                new_row['amp_max'] = new_a_e
                
                new_rows.append(new_row.to_frame().T)
                
    if not new_rows:
        return df_out # Should only happen if input empty
        
    result_df = pd.concat(new_rows, ignore_index=True)
    
    # Sort by time_start for cleanliness
    result_df = result_df.sort_values(by=['time_start', 'freq_start']).reset_index(drop=True)
    
    return result_df

def merge_equal_split(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Upsamples the smaller of two DataFrames to match the length of the larger one using nearest-neighbor interpolation,
    then merges them side-by-side.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.

    Returns:
        A new DataFrame with columns ['freq_data1', 'amp_data1', 'freq_data2', 'amp_data2'].
    """
    def upsample_df(df, target_len):
        if len(df) >= target_len:
            return df.copy().reset_index(drop=True)
        # Calculate indices to repeat. np.linspace distributes the repeats evenly
        indices = np.linspace(0, len(df) - 1, target_len).round().astype(int)
        return df.iloc[indices].reset_index(drop=True)

    # Determine target length
    target_len = max(len(df1), len(df2))

    # Upsample both to target length ensuring index alignment
    df1_equal = upsample_df(df1, target_len)
    df2_equal = upsample_df(df2, target_len)

    # Rename columns
    df1_equal.columns = ['freq_data1', 'amp_data1']
    df2_equal.columns = ['freq_data2', 'amp_data2']

    # Combine side-by-side
    result_equal = pd.concat([df1_equal, df2_equal], axis=1)

    return result_equal

def create_time_transition_df(input_df: pd.DataFrame, duration: float, scale_amp: bool = True, override_amp: float = None) -> pd.DataFrame:
    """
    Transforms merged spectral data into a time-based transition DataFrame.
    
    Args:
        input_df: DataFrame with columns ['freq_data1', 'amp_data1', 'freq_data2', 'amp_data2']
        duration: Total duration of the transition in seconds
        scale_amp: If True, normalizes amplitudes to 0-1 range
        override_amp: If provided, sets all amplitudes to this value
    """
    df = input_df.copy()
    
    # Handle amplitude processing
    if override_amp is not None:
        amp_start = np.full(len(df), override_amp)
        amp_stop = np.full(len(df), override_amp)
    elif scale_amp:
        # Avoid division by zero
        max1 = df['amp_data1'].max() or 1.0
        max2 = df['amp_data2'].max() or 1.0
        amp_start = df['amp_data1'] / max1
        amp_stop = df['amp_data2'] / max2
    else:
        amp_start = df['amp_data1']
        amp_stop = df['amp_data2']

    return pd.DataFrame({
        "freq_start": df['freq_data1'],
        "freq_stop": df['freq_data2'],
        "time_start": 0.0,
        "time_stop": float(duration),
        "amp_min": amp_start,  # Using 'min' to represent start value
        "amp_max": amp_stop    # Using 'max' to represent stop value
    })

def _cdf_mapping_indices(n_src: int, n_dst: int, power: float) -> np.ndarray:
    if n_src == 0 or n_dst == 0:
        return np.array([], dtype=int)
    # Negative power flips the mapping direction (mirrors the CDF)
    flip = power < 0
    p = abs(power)
    bins = np.linspace(0, 1, n_src) ** p
    bins[0], bins[-1] = 0, 1
    thresholds = (bins[:-1] + bins[1:]) / 2
    targets = np.linspace(0, 1, n_dst, endpoint=False) + 0.5 / n_dst
    indices = np.digitize(targets, thresholds)
    indices = np.clip(indices, 0, n_src - 1)
    indices[0] = 0
    indices[-1] = n_src - 1
    if flip:
        indices = (n_src - 1) - indices
        indices[0] = n_src - 1
        indices[-1] = 0
    return indices

def merge_cdf_analysis(df1: pd.DataFrame, df2: pd.DataFrame, plot: bool = False):
    """
    Analyzes two DataFrames to find the valid power range for CDF-based merging.
    Ideally, we map the smaller dataframe into the larger one such that all original
    values are represented.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        plot: If True, plots the CDF for min/max valid powers using Plotly.

    Returns:
        list: [min_power, max_power] range.
    """
    # Determine which DataFrame has fewer rows and prepare copies
    if len(df1) < len(df2):
        df_less_src, df_more_src = df1.copy(), df2.copy()
    else:
        df_less_src, df_more_src = df2.copy(), df1.copy()

    n_less = len(df_less_src)
    n_more = len(df_more_src)

    if n_less == 0 or n_more == 0:
        return [1.0, 1.0]

    def check_coverage(indices, n_src):
        """Checks if all source indices are used at least once."""
        return len(np.unique(indices)) == n_src

    # Optimize finding the valid power range
    # Define search space with high resolution near 1, coarser for larger powers
    power_candidates = np.concatenate([
        np.linspace(0.01, 2, 1000),
        np.linspace(2.01, 100, 1000)
    ])
    valid_powers = []

    # Check validity using vectorized operations
    if n_less == n_more:
        power_range = [1.0, 1.0]
    else:
        for p in power_candidates:
            idxs = _cdf_mapping_indices(n_less, n_more, p)
            if check_coverage(idxs, n_less):
                valid_powers.append(p)

        if not valid_powers:
            # Fallback if no full coverage found
            print("Warning: No fully covering power found. Using default [1.0, 1.0].")
            power_range = [1.0, 1.0]
        else:
            power_range = [valid_powers[0], valid_powers[-1]]

    if plot:
        import plotly.graph_objects as go
        
        # We need values to plot CDF. Let's use the freq column of the smaller DF.
        cols = df_less_src.columns
        f_col = next((c for c in cols if 'freq' in c.lower()), cols[0])
        
        # Get mapped values for min and max power
        # Note: mapping produces a sequence of length n_more
        vals_min = df_less_src[f_col].iloc[_cdf_mapping_indices(n_less, n_more, power_range[0])].values
        vals_max = df_less_src[f_col].iloc[_cdf_mapping_indices(n_less, n_more, power_range[1])].values
        
        sorted_freq_min = np.sort(vals_min)
        sorted_freq_max = np.sort(vals_max)
        
        # Create Y axis (CDF probability)
        y = np.arange(1, len(sorted_freq_min) + 1) / len(sorted_freq_min)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_freq_min, y=y, mode='lines', name=f'Min Power: {power_range[0]:.3f}'))
        fig.add_trace(go.Scatter(x=sorted_freq_max, y=y, mode='lines', name=f'Max Power: {power_range[1]:.3f}'))
        
        fig.update_layout(
            title='Cumulative Distribution Function for Min and Max Power',
            xaxis_title='Frequency',
            yaxis_title='CDF',
            template='plotly_white',
            legend_title='Power'
        )
        fig.show()

    return power_range

def merge_cdf(df1: pd.DataFrame, df2: pd.DataFrame, power_factor: float = None, plot: bool = False):
    """
    Merges two spectral DataFrames using CDF-based distribution mapping with a specific power factor.
    
    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        power_factor: The power exponent for the mapping distribution.
                      Negative values flip the mapping direction (mirror the CDF).
                      If None, calculates the mean valid power automatically.
        plot: If True, plots the distribution curve for the used power_factor.

    Returns:
        DataFrame: Merged DataFrame with columns
        ['freq_data1', 'amp_data1', 'freq_data2', 'amp_data2'].
    """
    # Determine which DataFrame has fewer rows and prepare copies
    if len(df1) < len(df2):
        df_less_src, df_more_src = df1.copy(), df2.copy()
        df1_role, df2_role = "less", "more"
    else:
        df_less_src, df_more_src = df2.copy(), df1.copy()
        df1_role, df2_role = "more", "less"

    # Find freq/amp cols - assume first freq-like and amp-like columns found
    def get_cols(df):
        cols = df.columns
        f = next((c for c in cols if 'freq' in c.lower()), cols[0])
        a = next((c for c in cols if 'amp' in c.lower()), cols[1])
        return f, a

    f1, a1 = get_cols(df_less_src)
    f2, a2 = get_cols(df_more_src)

    # Create standardized working copies
    df_less_temp = pd.DataFrame({
        'freq_data_less': df_less_src[f1],
        'amp_data_less': df_less_src[a1]
    })
    
    # Prepare result container based on larger dataframe
    df_result = pd.DataFrame({
        'freq_data_more': df_more_src[f2],
        'amp_data_more': df_more_src[a2]
    })
    
    n_less = len(df_less_temp)
    n_more = len(df_result)
    
    # Calculate mapping
    if power_factor is None:
        pr = merge_cdf_analysis(df1, df2, plot=False)
        power_factor = float(np.mean(pr))

    if plot:
        import plotly.graph_objects as go
        
        # Calculate min/max range for context
        pr = merge_cdf_analysis(df1, df2, plot=False)
        min_p, max_p = pr[0], pr[1]
        
        # We need values to plot CDF. Let's use the freq column of the smaller DF.
        f_col = next((c for c in df_less_src.columns if 'freq' in c.lower()), df_less_src.columns[0])
        
        # Function to get sorted values for a given power
        def get_sorted_vals(p):
            idxs = _cdf_mapping_indices(n_less, n_more, p)
            vals = df_less_src[f_col].iloc[idxs].values
            return np.sort(vals)
            
        vals_min = get_sorted_vals(min_p)
        vals_max = get_sorted_vals(max_p)
        vals_sel = get_sorted_vals(power_factor)
        
        # Create Y axis (CDF probability)
        y = np.arange(1, len(vals_sel) + 1) / len(vals_sel)

        fig = go.Figure()
        
        # Add Min/Max context
        fig.add_trace(go.Scatter(
            x=vals_min, y=y, 
            mode='lines', name=f'Min Power: {min_p:.3f}',
            line=dict(dash='dot', width=1, color='gray')
        ))
        fig.add_trace(go.Scatter(
            x=vals_max, y=y, 
            mode='lines', name=f'Max Power: {max_p:.3f}',
            line=dict(dash='dot', width=1, color='gray')
        ))

        # Add Selected
        fig.add_trace(go.Scatter(
            x=vals_sel, 
            y=y, 
            mode='lines', 
            name=f'Selected Power: {float(power_factor):.3f}',
            line=dict(width=3, color='#1f77b4')
        ))
        
        fig.update_layout(
            title=f'CDF Power Mapping (Selected={float(power_factor):.3f})',
            xaxis_title='Frequency',
            yaxis_title='CDF',
            template='plotly_white',
            legend_title='Power'
        )
        fig.show()

    idxs = _cdf_mapping_indices(n_less, n_more, power_factor)
    
    # Assign mapped values to result dataframe
    df_result['freq_data_less'] = df_less_temp['freq_data_less'].iloc[idxs].values
    df_result['amp_data_less'] = df_less_temp['amp_data_less'].iloc[idxs].values

    # Reorder columns to be consistent and store used power
    df_out = df_result[['freq_data_less', 'amp_data_less', 'freq_data_more', 'amp_data_more']].copy()
    
    if df1_role == "less":
        rename_map = {
            "freq_data_less": "freq_data1",
            "amp_data_less": "amp_data1",
            "freq_data_more": "freq_data2",
            "amp_data_more": "amp_data2",
        }
    else:
        # df1 was "more", so "more" columns correspond to data1
        rename_map = {
            "freq_data_more": "freq_data1",
            "amp_data_more": "amp_data1",
            "freq_data_less": "freq_data2",
            "amp_data_less": "amp_data2",
        }

    df_out = df_out.rename(columns=rename_map)
    
    # Reorder to standard [freq1, amp1, freq2, amp2]
    df_out = df_out[['freq_data1', 'amp_data1', 'freq_data2', 'amp_data2']]

    df_out.attrs["power_factor_used"] = float(power_factor)
    df_out.attrs["df1_role"] = df1_role
    df_out.attrs["df2_role"] = df2_role
    return df_out

def _sigmoid_mapping_indices(n_src: int, n_dst: int, sigmoid_range: float, sigmoid_slope: float = 1.0) -> np.ndarray:
    if n_src == 0 or n_dst == 0:
        return np.array([], dtype=int)
    
    # We want to map each target to nearest source in sigmoid space
    # Target domain: [-6, 6] (standard approx for full sigmoid transition)
    # The slope parameter steepens or flattens the curve
    targets = expit(np.linspace(-6, 6, n_dst))
    
    # Source domain: [-sigmoid_range, sigmoid_range] with slope adjustment
    # Applying slope to the input space effectively changes the steepness
    sources = expit(np.linspace(-sigmoid_range * sigmoid_slope, sigmoid_range * sigmoid_slope, n_src))
    
    # Normalize sources to match target range [0, 1] if slope pushed them out or kept them in
    # Actually, expit always returns [0, 1].
    # But if range is small, expit output range is small (e.g. 0.45-0.55).
    # If we want to "retain shape" but just stretch it, we might need to normalize the output of expit?
    # Original logic relied on 'sigmoid_range' to define how much of the sigmoid curve we traverse.
    # A large range (e.g. 10) traverses 0 to 1. A small range (0.1) traverses 0.48 to 0.52 (linear-ish).
    
    # If the user wants to "adjust the S curve", they usually mean steepness (slope).
    # In the logistic function 1 / (1 + exp(-k * x)), k is the slope.
    # Our current implementation uses 1 / (1 + exp(-x)) where x goes from -range to +range.
    # This is equivalent to 1 / (1 + exp(-k * t)) where t goes from -1 to 1 and k = range.
    
    # So 'sigmoid_range' IS effectively the slope/steepness control over the normalized domain.
    # If 'sigmoid_range' is small, the curve is flat (linear).
    # If 'sigmoid_range' is large, the curve is steep (step-like).
    
    # However, to ensure we cover the full output space [0, 1] regardless of range/slope,
    # we should normalize the SOURCE values to [0, 1] before matching with TARGET values (which are 0-1).
    
    # Let's adjust the logic:
    # 1. Generate source distribution based on range (shape).
    # 2. Normalize it to [0, 1] so min matches 0 and max matches 1.
    # 3. Match with targets (which are uniform/sigmoid in their own space? No, targets are just quantiles).
    
    # WAIT: The original logic matched expit(target_space) with expit(source_space).
    # Target space was fixed [-6, 6] -> approx [0, 1].
    # Source space was [-range, range].
    # If range is small, source values are [0.48, 0.52].
    # Targets [0, 0.47] and [0.53, 1] would map to the endpoints 0 and n_src-1.
    # This caused "clipping" where the middle of the target maps to the linear middle of source,
    # but the edges of target map to edges of source.
    
    # If the goal is "retain S shape", we want the source distribution to Look like an S-curve
    # distributed across the indices.
    
    # Let's add a 'slope' parameter that works alongside range?
    # Or just interpret the user request: "retain general sigmoid shape" means we shouldn't use the linear-ish
    # middle part (small range). We should probably use a wider range (large S shape) but maybe compress it?
    
    # Actually, `sigmoid_range` controls the shape.
    # Large range = Step function (Wall).
    # Medium range (~6) = Standard S.
    # Small range = Linear / Diagonal.
    
    # If the user wants to "adjust the S curve", they likely want to change the steepness
    # while KEEPING the start/end points at 0 and 1.
    # The current implementation fails this for small ranges because it outputs [0.48, 0.52].
    
    # REVISED LOGIC:
    # Always normalize source distribution to [0, 1].
    # Then `sigmoid_range` solely controls the curvature (Linear vs S-curve vs Step).
    
    raw_sources = expit(np.linspace(-sigmoid_range, sigmoid_range, n_src))
    
    # Normalize to [0, 1]
    s_min, s_max = raw_sources.min(), raw_sources.max()
    if s_max - s_min < 1e-9:
        # Avoid div by zero if range is 0 (shouldn't happen with linspace unless n_src=1)
        sources = np.linspace(0, 1, n_src)
    else:
        sources = (raw_sources - s_min) / (s_max - s_min)
        
    # Targets are just evenly spaced quantiles [0, 1]
    # We want to map these linear quantiles to the source distribution
    targets = np.linspace(0, 1, n_dst)
    
    # Midpoints for classification
    thresholds = (sources[:-1] + sources[1:]) / 2
    
    indices = np.digitize(targets, thresholds)
    indices = np.clip(indices, 0, n_src - 1)
    
    return indices

def merge_sigmoid_analysis(df1: pd.DataFrame, df2: pd.DataFrame, plot: bool = False):
    """
    Analyzes two DataFrames to find the valid sigmoid range for merging.
    Ideally, we map the smaller dataframe into the larger one such that all original
    values are represented.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        plot: If True, plots the Sigmoid distributions for min/max valid ranges using Plotly.

    Returns:
        list: [min_range, max_range] valid sigmoid range.
    """
    if len(df1) < len(df2):
        df_less_src, df_more_src = df1.copy(), df2.copy()
        n_less, n_more = len(df1), len(df2)
    else:
        df_less_src, df_more_src = df2.copy(), df1.copy()
        n_less, n_more = len(df2), len(df1)

    if n_less == 0 or n_more == 0:
        return [1.0, 1.0]

    def check_coverage(indices, n_src):
        return len(np.unique(indices)) == n_src

    # Search space - now purely shape control
    # 0.1 (Linear) -> 10 (Steep S)
    range_candidates = np.concatenate([
        np.linspace(0.01, 10, 1000),
        np.linspace(10.01, 100, 1000)
    ])

    valid_ranges = []

    if n_less == n_more:
        valid_ranges = [1.0]
    else:
        for r in range_candidates:
            idxs = _sigmoid_mapping_indices(n_less, n_more, r)
            if check_coverage(idxs, n_less):
                valid_ranges.append(r)

    if not valid_ranges:
        print("Warning: No fully covering sigmoid range found. Using default [6.0, 6.0].")
        res_range = [6.0, 6.0]
    else:
        res_range = [valid_ranges[0], valid_ranges[-1]]

    if plot:
        import plotly.graph_objects as go
        
        cols = df_less_src.columns
        f_col = next((c for c in cols if 'freq' in c.lower()), cols[0])
        
        min_freq = float(df_less_src[f_col].min())
        max_freq = float(df_less_src[f_col].max())
        
        # Plot distribution curves
        sigmoid_bins = np.linspace(min_freq, max_freq, 500)
        
        # Map bins to [0, 1]
        normalized = (sigmoid_bins - min_freq) / (max_freq - min_freq)
        
        # Helper to get normalized curve
        def get_curve(r):
            # Same logic as indices: generate standard expit, then normalize to 0-1
            # But here we are plotting y vs x. 
            # x is linear (bins). y should be the mapping curve.
            
            # The "shape" r implies mapping from domain [-r, r]
            val_in_range = normalized * 2 * r - r
            raw_y = expit(val_in_range)
            
            # Normalize raw_y to start at 0 and end at 1
            s_min = expit(-r)
            s_max = expit(r)
            return (raw_y - s_min) / (s_max - s_min)

        y_min = get_curve(res_range[0])
        y_max = get_curve(res_range[1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sigmoid_bins, y=y_min, mode='lines', name=f'Min Range (Shape): {res_range[0]:.3f}'))
        fig.add_trace(go.Scatter(x=sigmoid_bins, y=y_max, mode='lines', name=f'Max Range (Shape): {res_range[1]:.3f}'))
        
        fig.update_layout(
            title='Normalized Sigmoid Mapping',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Normalized CDF (0-1)',
            template='plotly_white'
        )
        fig.show()

    return res_range

def merge_sigmoid(df1: pd.DataFrame, df2: pd.DataFrame, sigmoid_range: float = None, plot: bool = False):
    """
    Merges two spectral DataFrames using Sigmoid-based distribution mapping.
    
    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        sigmoid_range: The range factor for the sigmoid distribution.
                      If None, calculates the mean valid range automatically.
        plot: If True, plots the distribution curve for the used sigmoid_range.
    
    Returns:
        DataFrame: Merged DataFrame with columns ['freq_data1', 'amp_data1', 'freq_data2', 'amp_data2'].
    """
    if len(df1) < len(df2):
        df_less_src, df_more_src = df1.copy(), df2.copy()
        df1_role = "less"
    else:
        df_less_src, df_more_src = df2.copy(), df1.copy()
        df1_role = "more"
        
    def get_cols(df):
        cols = df.columns
        f = next((c for c in cols if 'freq' in c.lower()), cols[0])
        a = next((c for c in cols if 'amp' in c.lower()), cols[1])
        return f, a
        
    f1, a1 = get_cols(df_less_src)
    f2, a2 = get_cols(df_more_src)
    
    df_less_temp = pd.DataFrame({
        'freq_data_less': df_less_src[f1],
        'amp_data_less': df_less_src[a1]
    })
    
    df_result = pd.DataFrame({
        'freq_data_more': df_more_src[f2],
        'amp_data_more': df_more_src[a2]
    })
    
    n_less = len(df_less_temp)
    n_more = len(df_result)
    
    if sigmoid_range is None:
        pr = merge_sigmoid_analysis(df1, df2, plot=False)
        sigmoid_range = float(np.mean(pr))

    if plot:
        import plotly.graph_objects as go
        from scipy.special import expit
        
        # Calculate min/max range for context
        pr = merge_sigmoid_analysis(df1, df2, plot=False)
        min_r, max_r = pr[0], pr[1]
        
        # Use freq column of the source (less) DF for x-axis range
        min_freq = float(df_less_src[f1].min())
        max_freq = float(df_less_src[f1].max())
        
        sigmoid_bins = np.linspace(min_freq, max_freq, 500)
        normalized = (sigmoid_bins - min_freq) / (max_freq - min_freq)
        
        def get_curve_y(r):
            val_in_range = normalized * 2 * r - r
            raw_y = expit(val_in_range)
            s_min = expit(-r)
            s_max = expit(r)
            return (raw_y - s_min) / (s_max - s_min)

        fig = go.Figure()
        
        # Add Min/Max context
        fig.add_trace(go.Scatter(
            x=sigmoid_bins, y=get_curve_y(min_r), 
            mode='lines', name=f'Min Range: {min_r:.3f}',
            line=dict(dash='dot', width=1, color='gray')
        ))
        fig.add_trace(go.Scatter(
            x=sigmoid_bins, y=get_curve_y(max_r), 
            mode='lines', name=f'Max Range: {max_r:.3f}',
            line=dict(dash='dot', width=1, color='gray')
        ))

        # Add Selected
        fig.add_trace(go.Scatter(
            x=sigmoid_bins, 
            y=get_curve_y(float(sigmoid_range)), 
            mode='lines', 
            name=f'Selected Range: {float(sigmoid_range):.3f}',
            line=dict(width=3, color='#1f77b4')
        ))
        
        fig.update_layout(
            title=f'Sigmoid Mapping Curve (Selected={float(sigmoid_range):.3f})',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Normalized Distribution (0-1)',
            template='plotly_white'
        )
        fig.show()
        
    idxs = _sigmoid_mapping_indices(n_less, n_more, sigmoid_range)
    
    df_result['freq_data_less'] = df_less_temp['freq_data_less'].iloc[idxs].values
    df_result['amp_data_less'] = df_less_temp['amp_data_less'].iloc[idxs].values
    
    # Reorder columns and handle flipping to preserve df1 -> df2 order
    df_out = df_result[['freq_data_less', 'amp_data_less', 'freq_data_more', 'amp_data_more']].copy()
    
    if df1_role == "less":
        rename_map = {
            "freq_data_less": "freq_data1",
            "amp_data_less": "amp_data1",
            "freq_data_more": "freq_data2",
            "amp_data_more": "amp_data2",
        }
    else:
        rename_map = {
            "freq_data_more": "freq_data1",
            "amp_data_more": "amp_data1",
            "freq_data_less": "freq_data2",
            "amp_data_less": "amp_data2",
        }
        
    df_out = df_out.rename(columns=rename_map)
    df_out = df_out[['freq_data1', 'amp_data1', 'freq_data2', 'amp_data2']]
    df_out.attrs["sigmoid_range_used"] = float(sigmoid_range)
    
    return df_out
