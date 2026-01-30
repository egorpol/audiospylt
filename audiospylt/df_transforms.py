import pandas as pd
import numpy as np

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
