import numpy as np
import os
from scipy.signal import resample
from datetime import datetime
import soundfile as sf
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

def ensure_finite_audio(audio_data, name="audio_data", sanitize=True, verbose=True):
    """
    Ensure an audio array contains only finite values.

    - If already finite: returns the original array (no copy).
    - If non-finite values exist:
      - sanitize=True  -> returns a sanitized copy via np.nan_to_num
      - sanitize=False -> raises ValueError
    """
    arr = np.asarray(audio_data)
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return audio_data

    n_bad = int(arr.size - finite_mask.sum())
    if verbose:
        msg = f"Warning: {name} contains {n_bad} NaN/Inf samples"
        msg += "; sanitizing." if sanitize else "."
        print(msg)

    if sanitize:
        return np.nan_to_num(arr)

    raise ValueError(f"{name} contains {n_bad} NaN/Inf samples")

def scale_samples(samples, bit_rate):
    if bit_rate == 16:
        return np.int16(samples * (2**15 - 1))
    elif bit_rate == 24:
        return (samples * (2**23 - 1)).astype(np.int32) << 8
    else:
        raise ValueError(f"Unsupported bit rate: {bit_rate}")

_WIN_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')

def _sanitize_filename(filename: str) -> str:
    """
    Make a filename safe across platforms (especially Windows).

    Notes:
      - This function operates on a *basename* (no directory parts).
      - Invalid Windows filename characters are replaced with '_'.
    """
    name = _WIN_INVALID_FILENAME_CHARS.sub("_", filename)
    # Windows also forbids trailing spaces/dots in filenames.
    name = name.rstrip(" .")
    return name

def _preset_name_from_sr(fs_hz: int) -> str | None:
    """
    Convert a sample rate (Hz) into a known preset label, if available.

    Example: 44100 -> "44.1kHz"
    """
    fs_hz = int(fs_hz)
    mapping = {
        44100: "44.1kHz",
        48000: "48kHz",
        88200: "88.2kHz",
        96000: "96kHz",
        192000: "192kHz",
    }
    return mapping.get(fs_hz)

def _resolve_fs_target(fs_initial, fs_target_name):
    """
    Resolve the target sample rate and a friendly label.

    Supported fs_target_name values:
      - preset string: "44.1kHz", "48kHz", "88.2kHz", "96kHz", "192kHz"
      - "source" / "native" / "input" / "original" / None: keep fs_initial (no resample)
      - int: explicit Hz (e.g., 44100)
    """
    try:
        fs_initial = int(fs_initial)
    except Exception as e:
        raise ValueError(f"fs_initial must be an integer sample rate in Hz; got {fs_initial!r}") from e
    if fs_initial <= 0:
        raise ValueError(f"fs_initial must be > 0; got {fs_initial}")

    fs_target_dict = {
        "44.1kHz": 44100,
        "48kHz": 48000,
        "88.2kHz": 88200,
        "96kHz": 96000,
        "192kHz": 192000,
    }

    if fs_target_name is None:
        fs_target = fs_initial
        label = _preset_name_from_sr(fs_target) or f"{fs_target}Hz"
        return fs_target, label

    if isinstance(fs_target_name, (int, np.integer)):
        fs_target = int(fs_target_name)
        if fs_target <= 0:
            raise ValueError(f"fs_target_name as int must be > 0; got {fs_target}")
        label = _preset_name_from_sr(fs_target) or f"{fs_target}Hz"
        return fs_target, label

    if not isinstance(fs_target_name, str):
        raise ValueError(
            "fs_target_name must be a preset string, 'source'/None, or an int Hz; "
            f"got {type(fs_target_name).__name__}"
        )

    fs_target_name_norm = fs_target_name.strip().lower()
    if fs_target_name_norm in {"source", "native", "input", "original"}:
        fs_target = fs_initial
        label = _preset_name_from_sr(fs_target) or f"{fs_target}Hz"
        return fs_target, label

    if fs_target_name in fs_target_dict:
        fs_target = fs_target_dict[fs_target_name]
        return fs_target, fs_target_name

    raise ValueError(
        f"Unsupported sampling rate preset: {fs_target_name!r}. "
        "Use one of: '44.1kHz','48kHz','88.2kHz','96kHz','192kHz' or 'source'."
    )

def _build_output_filename(
    *,
    fs_target_name: str,
    bit_rate: int,
    custom_filename: str | None,
    filename_template: str | None,
    timestamp_format: str,
) -> str:
    """
    Decide the final filename for the rendered wave file.

    Priority:
      1) custom_filename (exact override)
      2) filename_template (format string with placeholders)
      3) default filename

    Available template placeholders:
      - {fs_target_name}
      - {bit_rate}
      - {timestamp}          (formatted with timestamp_format)
      - {timestamp_format}   (the format string itself)
    """
    timestamp = datetime.now().strftime(timestamp_format)

    if custom_filename is not None and filename_template is not None:
        raise ValueError("Provide only one of custom_filename or filename_template (not both).")

    if custom_filename is not None:
        base = Path(custom_filename).name
    elif filename_template is not None:
        try:
            base = filename_template.format(
                fs_target_name=fs_target_name,
                bit_rate=bit_rate,
                timestamp=timestamp,
                timestamp_format=timestamp_format,
            )
        except KeyError as e:
            raise ValueError(
                f"filename_template contains an unknown placeholder: {e}. "
                "Allowed: {fs_target_name}, {bit_rate}, {timestamp}, {timestamp_format}."
            ) from e
        base = Path(base).name
    else:
        base = f"generated_wave_file_{fs_target_name}_{bit_rate}bit_{timestamp}.wav"

    if not base.lower().endswith(".wav"):
        base = f"{base}.wav"

    return _sanitize_filename(base)

def generate_wave_file(
    y_combined,
    fs_initial,
    fs_target_name='44.1kHz',
    bit_rate=16,
    custom_filename=None,
    filename_template=None,
    timestamp_format="%H_%M_%S",
    output_dir: str | Path | None = None,
    save_to_file=True,
):
    """
    Export audio to a WAV file (or return the exported samples when save_to_file=False).

    fs_target_name controls resampling/export rate:
      - preset strings: '44.1kHz','48kHz','88.2kHz','96kHz','192kHz'
      - 'source' (or None): keep fs_initial (no resampling)
      - int: explicit Hz (e.g., 44100)
    """
    # Normalize/validate input early so downstream operations are predictable.
    y_combined = np.asarray(y_combined, dtype=float).reshape(-1)
    if y_combined.size == 0:
        raise ValueError("y_combined is empty")
    if not np.isfinite(y_combined).all():
        raise ValueError("y_combined contains NaN/Inf; sanitize upstream before exporting.")

    fs_target, fs_target_label = _resolve_fs_target(fs_initial, fs_target_name)
    fs_initial = int(fs_initial)

    if fs_target == fs_initial:
        y_resampled = y_combined
    else:
        # calculate the number of samples in the resampled signal
        num_samples_resampled = int(len(y_combined) * fs_target / fs_initial)
        # resample the signal to the target sample rate
        y_resampled = resample(y_combined, num_samples_resampled)

    # Normalize the resampled signal to the range of -1 to 1.
    # Guard against silent/all-zero signals to avoid division by 0 -> NaNs.
    max_amp = float(np.max(np.abs(y_resampled)))
    if not np.isfinite(max_amp) or max_amp <= 0.0:
        y_normalized = np.zeros_like(y_resampled)
    else:
        y_normalized = y_resampled / max_amp

    # scale the resampled signal to the desired bit rate range
    y_scaled = scale_samples(y_normalized, bit_rate)

    output_filename = _build_output_filename(
        fs_target_name=fs_target_label,
        bit_rate=bit_rate,
        custom_filename=custom_filename,
        filename_template=filename_template,
        timestamp_format=timestamp_format,
    )

    # Decide output directory.
    # Default behavior (backwards compatible): write into ./rendered_audio relative to CWD.
    if output_dir is None:
        output_directory = os.path.join(os.getcwd(), "rendered_audio")
    else:
        output_directory = str(Path(output_dir))

    # Create output directory if it doesn't exist.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # construct the full path to save the audio
    file_path = os.path.join(output_directory, output_filename)

    if save_to_file:
        # write the resampled waveform to a wave audio file
        sf.write(file_path, y_scaled, fs_target, subtype=f'PCM_{bit_rate}')

        # get the current timestamp for log
        timestamp_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # print a message about the wave file being saved successfully with a timestamp
        print(
            f"[{timestamp_log}] {bit_rate}-bit wave file with {fs_target_label} sampling rate saved successfully to: {file_path}"
        )

        return file_path
    else:
        return y_scaled, fs_target


def render_audio(
    y_combined,
    fs_initial,
    fs_target_name="44.1kHz",
    bit_rate=24,
    custom_filename=None,
    filename_template=None,
    timestamp_format="%H_%M_%S",
    output_dir: str | Path | None = None,
    save_audio=True,
    player=True,
    sanitize=True,
    verbose=True,
):
    """
    Convenience wrapper around `generate_wave_file`:

    - sanitizes NaN/Inf (optional)
    - exports audio (to file or array)
    - optionally plays it via IPython's Audio widget

    fs_target_name supports the same values as `generate_wave_file`, including 'source' to keep
    the original sample rate (no resampling).

    Returns:
      - file path (str) if save_audio=True
      - (audio_data, fs_target) if save_audio=False
    """
    y_export = ensure_finite_audio(
        y_combined, name="y_combined", sanitize=sanitize, verbose=verbose
    )
    result = generate_wave_file(
        y_export,
        fs_initial,
        fs_target_name=fs_target_name,
        bit_rate=bit_rate,
        custom_filename=custom_filename,
        filename_template=filename_template,
        timestamp_format=timestamp_format,
        output_dir=output_dir,
        save_to_file=save_audio,
    )

    if player:
        from IPython.display import Audio, display

        if save_audio:
            display(Audio(filename=result))
        else:
            audio_data, fs_target = result
            display(Audio(audio_data, rate=fs_target))

    return result


def _fs_initial_from_config(config: Mapping[str, Any], *, verbose: bool = True) -> int:
    """
    Infer an integer sample rate (Hz) from a config dict containing either:
      - dt: sample spacing in seconds (preferred)
      - n: legacy alias for dt
    """
    if "dt" in config:
        dt = float(config["dt"])
    elif "n" in config:
        dt = float(config["n"])
    else:
        raise KeyError("Config must include 'dt' (preferred) or legacy 'n'.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0; got {dt}")

    fs_float = 1.0 / dt
    fs_int = int(round(fs_float))
    if verbose and abs(fs_float - fs_int) > 1e-9:
        print(
            f"Warning: 1/dt={fs_float} is not an integer Hz; using rounded fs_initial={fs_int}."
        )
    return fs_int


def mix_selected_waveforms(
    waveform_data: Mapping[str, np.ndarray],
    selected_waveforms: Iterable[str],
    *,
    mode: str = "sum",
    strict: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Combine multiple named waveforms into a single 1D float array.

    - mode='sum': additive mix (recommended; export path normalizes to [-1, 1] anyway)
    - mode='mean': average mix (keeps amplitude comparable when adding many signals)
    """
    names = list(selected_waveforms)
    ys: list[np.ndarray] = []
    used: list[str] = []

    for name in names:
        if name in waveform_data:
            ys.append(np.asarray(waveform_data[name], dtype=float).reshape(-1))
            used.append(name)
        elif strict:
            raise KeyError(f"Requested waveform '{name}' not found in waveform_data.")
        elif verbose:
            print(f"Warning: requested waveform '{name}' not found in waveform_data; skipping.")

    if not ys:
        raise ValueError(
            "No valid waveforms to mix. "
            "Check config['selected_waveforms'] and waveform_data keys."
        )

    min_len = min(int(y.size) for y in ys)
    if any(y.size != min_len for y in ys):
        if verbose:
            print(f"Warning: waveform lengths differ; trimming all to {min_len} samples.")
        ys = [y[:min_len] for y in ys]

    y_mix = np.zeros(min_len, dtype=float)
    for y in ys:
        y_mix += y

    mode_norm = mode.strip().lower()
    if mode_norm == "sum":
        return y_mix
    if mode_norm == "mean":
        return y_mix / float(len(ys))
    raise ValueError("mode must be 'sum' or 'mean'")


def render_selected_waveforms(
    waveform_data: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    mix_mode: str = "sum",
    fs_target_name: str | int | None = "44.1kHz",
    bit_rate: int = 24,
    custom_filename: str | None = None,
    filename_template: str | None = None,
    timestamp_format: str = "%H_%M_%S",
    output_dir: str | Path | None = None,
    save_audio: bool = True,
    player: bool = True,
    sanitize: bool = True,
    verbose: bool = True,
):
    """
    Notebook-friendly convenience function:

    - mixes ALL waveforms listed in config['selected_waveforms'] into a single stream
    - derives fs_initial from config['dt'] (or legacy 'n')
    - calls render_audio(...) with the usual export/playback options

    Parameters (high-level):
      - waveform_data: dict-like mapping {name -> 1D array} (e.g. from waveform_utils.generate_waveforms)
      - config: must contain 'selected_waveforms' (list[str]) and 'dt' (or legacy 'n')

    Parameters (audio/rendering):
      - mix_mode: how to combine selected waveforms into one stream:
          - "sum": additive mix (recommended; export normalizes anyway)
          - "mean": average mix (keeps level similar when adding many signals)
      - fs_target_name: export sample rate preset, "source" to keep original, or an int Hz.
          Examples: "44.1kHz", "48kHz", "96kHz", "source", 44100
      - bit_rate: PCM bit depth (supported: 16, 24)
      - custom_filename: exact output filename (basename); mutually exclusive with filename_template
      - filename_template: format string with placeholders:
          {fs_target_name}, {bit_rate}, {timestamp}, {timestamp_format}
      - timestamp_format: strftime format used to fill {timestamp}
      - output_dir: directory to write the .wav into when save_audio=True. If None, defaults to
          ./rendered_audio relative to the current working directory.
      - save_audio: True -> write .wav and return file path; False -> return (audio_data, fs_target)
      - player: if True, display an IPython Audio widget
      - sanitize: if True, replace NaN/Inf with finite values before exporting
      - verbose: print warnings/info (missing waveforms, length mismatches, dt rounding, etc.)

    Note:
      - In notebooks, if this is the last expression in a cell, Jupyter will print the return value.
        Assign it (e.g. `_ = render_selected_waveforms(...)`) or add a trailing ';' to suppress output.
    """
    fs_initial = _fs_initial_from_config(config, verbose=verbose)
    y = mix_selected_waveforms(
        waveform_data,
        config.get("selected_waveforms", []),
        mode=mix_mode,
        strict=False,
        verbose=verbose,
    )
    return render_audio(
        y,
        fs_initial,
        fs_target_name=fs_target_name,
        bit_rate=bit_rate,
        custom_filename=custom_filename,
        filename_template=filename_template,
        timestamp_format=timestamp_format,
        output_dir=output_dir,
        save_audio=save_audio,
        player=player,
        sanitize=sanitize,
        verbose=verbose,
    )
