import requests
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import os
import io
import tempfile
import librosa
from scipy.signal import spectrogram, get_window
from urllib.parse import urlparse
try:  # pragma: no cover
    from IPython.display import Audio as IPyAudio, display as ipy_display
except Exception:  # pragma: no cover
    # Allow using this module outside Jupyter/IPython.
    IPyAudio = None
    ipy_display = None

try:  # pragma: no cover
    from .plotting import show_plotly
except Exception:  # pragma: no cover
    # Allow running this file directly (outside package context).
    from audiospylt.plotting import show_plotly


def _infer_num_channels(audio_data):
    """Return an estimated channel count for mono or stacked channel arrays."""
    if audio_data.ndim == 1:
        return 1
    # For librosa, channel dimension is first (channels, samples); fall back to the smaller axis.
    return audio_data.shape[0] if audio_data.shape[0] <= audio_data.shape[-1] else audio_data.shape[-1]

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

def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(str(s))
    except Exception:
        return False
    return p.scheme in {"http", "https"} and bool(p.netloc)


def load_wav_from_source(
    wav_source,
    *,
    verbose: bool = False,
    download_mode: str = "cwd",
    download_dir: str | None = None,
    show_load_bar: bool = False,
    download_timeout_s: float | None = 60.0,
    download_chunk_size: int = 1024 * 256,
):
    """
    Load an audio file either from a local path or a URL.

    This function is intentionally side-effect-light:
    - No printing unless verbose=True
    - For URLs, it downloads the file (by default into the current working directory).

    Parameters
    - download_mode:
      - "cwd": write the downloaded file into the current working directory (back-compat default)
      - "dir": write into download_dir (directory is created if missing)
      - "temp": download into a temporary file (caller may delete it after load)
    - show_load_bar: if True, show a tqdm progress bar for URL downloads (notebook-friendly).
    """
    if os.path.exists(wav_source):
        full_path = os.path.abspath(wav_source)
        if verbose:
            print(f"WAV file loaded from local path: {full_path}")
        return full_path

    if not _is_http_url(wav_source):
        raise ValueError(
            f"wav_source must be an existing local path or an http(s) URL; got {wav_source!r}"
        )

    download_mode_norm = str(download_mode).lower().strip().replace("-", "_")

    # Prefer the URL path basename, but fall back to something stable.
    p = urlparse(str(wav_source))
    url_base = os.path.basename((p.path or "").rstrip("/")) or "downloaded_audio"
    url_root, url_ext = os.path.splitext(url_base)
    if not url_ext:
        # Keep librosa/ffmpeg helpers happy by using a generic extension.
        url_base = f"{url_base}.bin"
        url_root, url_ext = os.path.splitext(url_base)

    if download_mode_norm == "cwd":
        target_path = os.path.abspath(url_base)
    elif download_mode_norm == "dir":
        if not download_dir:
            raise ValueError("download_dir must be provided when download_mode='dir'")
        os.makedirs(download_dir, exist_ok=True)
        target_path = os.path.abspath(os.path.join(download_dir, url_base))
    elif download_mode_norm == "temp":
        # On Windows, NamedTemporaryFile must be closed before reopening.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=url_ext)
        target_path = tmp.name
        tmp.close()
    else:
        raise ValueError("download_mode must be one of: 'cwd' | 'dir' | 'temp'")

    try:
        timeout = download_timeout_s
        with requests.get(wav_source, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            total = response.headers.get("Content-Length") or response.headers.get("content-length")
            try:
                total_bytes = int(total) if total is not None else None
            except Exception:
                total_bytes = None

            pbar = None
            if show_load_bar:
                desc = f"Downloading {url_base}"
                # tqdm notebook widgets require ipywidgets; when missing, tqdm.notebook can
                # raise ImportError at *runtime* ("IProgress not found"). Fall back to auto.
                try:  # pragma: no cover
                    from tqdm.notebook import tqdm as _tqdm_notebook
                    try:  # pragma: no cover
                        pbar = _tqdm_notebook(
                            total=total_bytes,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=desc,
                            leave=False,
                        )
                    except Exception:  # pragma: no cover
                        from tqdm.auto import tqdm as _tqdm_auto
                        if verbose:
                            print("Note: ipywidgets missing; falling back to a text progress bar.")
                        pbar = _tqdm_auto(
                            total=total_bytes,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=desc,
                            leave=False,
                        )
                except Exception:  # pragma: no cover
                    try:  # pragma: no cover
                        from tqdm.auto import tqdm as _tqdm_auto
                        pbar = _tqdm_auto(
                            total=total_bytes,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=desc,
                            leave=False,
                        )
                    except Exception:  # pragma: no cover
                        pbar = None

            try:
                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=int(download_chunk_size)):
                        if not chunk:
                            continue
                        f.write(chunk)
                        if pbar is not None:
                            pbar.update(len(chunk))
            finally:
                if pbar is not None:
                    pbar.close()

        if verbose:
            if download_mode_norm == "temp":
                print(f"Audio downloaded from URL into a temporary file: {target_path}")
            else:
                print(f"Audio file downloaded from URL and saved to: {target_path}")

        return target_path
    except requests.RequestException as e:
        # Clean up partial files.
        try:
            if os.path.exists(target_path):
                os.remove(target_path)
        except Exception:
            pass
        raise ValueError(f"Failed to load audio from URL. Error: {e}") from e

def load_audio_data(wav_filename, desired_sample_rate, convert_to_mono):
    """Load audio data, convert it to mono (if desired), and resample it (if a desired sample rate is provided)."""
    return librosa.load(wav_filename, sr=desired_sample_rate, mono=convert_to_mono)

def display_audio_properties(audio_data, sample_rate, wav_source):
    """Display the properties of the loaded audio."""
    num_channels = _infer_num_channels(audio_data)
    print(f"Number of audio channels: {num_channels}")
    print(f"Sampling rate: {sample_rate} Hz")
    print(f"WAV file loaded from {wav_source}")

    return num_channels


def _infer_num_samples(audio_data):
    """Infer the number of samples for mono or 2D channel arrays."""
    if audio_data.ndim == 1:
        return int(audio_data.shape[0])
    # (channels, samples) vs (samples, channels) -> samples are on the larger axis.
    return int(audio_data.shape[-1] if audio_data.shape[0] <= audio_data.shape[-1] else audio_data.shape[0])

def _basename_for_title(name: str) -> str:
    """
    Return a "basename-like" string for plot titles.

    - For local paths: `os.path.basename(...)`
    - For URLs: basename of the URL path (ignores query/fragment)
    """
    s = str(name).strip().rstrip("/\\")
    if not s:
        return s
    try:
        p = urlparse(s)
        if p.scheme and p.netloc:
            base = os.path.basename((p.path or "").rstrip("/"))
            return base or s
    except Exception:
        # If parsing fails, fall back to os.path.basename below.
        pass
    base = os.path.basename(s)
    return base or s

def plot_waveform(
    audio_data,
    sample_rate,
    wav_filename,
    *,
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict | None = None,
    title_name: str | None = None,
    title_mode: str = "full",
):
    """Plot the waveform of the audio.

    Parameters
    - plot_width / plot_height: optional explicit Plotly figure size (pixels)
    - plotly_layout: optional dict forwarded to `fig.update_layout(**plotly_layout)`
    - title_name: optional name used in the figure title (defaults to wav_filename)
    - title_mode: "full" | "basename" (applied to title_name / wav_filename)
    """
    duration = len(audio_data) / float(sample_rate)
    print(f"Total duration: {duration:.3f} seconds")

    time_axis = np.linspace(0, duration, len(audio_data))
    fig = go.Figure(go.Scatter(x=time_axis, y=audio_data))

    title_mode_norm = str(title_mode).lower().strip().replace("-", "_")
    name = wav_filename if title_name is None else title_name
    if title_mode_norm == "basename":
        name = _basename_for_title(name)
    elif title_mode_norm != "full":
        raise ValueError(f"title_mode must be 'full' or 'basename'; got {title_mode!r}")

    fig.update_layout(
        title=f"Waveform of {name}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
    )

    if plot_width is not None or plot_height is not None:
        fig.update_layout(width=plot_width, height=plot_height, autosize=False)

    if plotly_layout:
        fig.update_layout(**plotly_layout)

    show_plotly(fig)

    return duration
       
def trim_and_fade_audio(
    audio_data,
    sample_rate,
    num_channels,
    duration,
    wav_source,
    start_time=0.4,
    end_time=1.6,
    add_fades=True,
    fade_in_duration=0.2,
    fade_out_duration=0.3,
    fade_in_exponent=0.8,
    fade_out_exponent=1.5,
    *,
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict | None = None,
    title_name: str | None = None,
    title_mode: str = "basename",
):
    
    # Ensure start_time and end_time are within the duration of the audio
    start_time = max(0, start_time)
    end_time = min(duration, end_time)

    # Calculate start and end frame indices
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)

    # Cut the audio data
    cut_audio_data = audio_data[start_frame:end_frame].copy()

    # Adjust the time axis for the cut waveform
    cut_duration = end_time - start_time
    cut_time_axis = np.linspace(0, cut_duration, len(cut_audio_data))

    title_mode_norm = str(title_mode).lower().strip().replace("-", "_")
    name = wav_source if title_name is None else title_name
    if title_mode_norm == "basename":
        name = _basename_for_title(name)
    elif title_mode_norm != "full":
        raise ValueError(f"title_mode must be 'full' or 'basename'; got {title_mode!r}")

    if add_fades:
        # Calculate the number of frames for the fade-in and fade-out
        fade_in_frames = int(fade_in_duration * sample_rate)
        fade_out_frames = int(fade_out_duration * sample_rate)

        # Create exponential fade-in and fade-out curves
        fade_in_curve = np.linspace(0, 1, fade_in_frames) ** fade_in_exponent
        fade_out_curve = np.linspace(1, 0, fade_out_frames) ** fade_out_exponent

        # Apply the fade-in and fade-out curves to the cut audio data
        if num_channels > 1:
            for channel in range(num_channels):
                channel_data = cut_audio_data[channel::num_channels]
                channel_data[:fade_in_frames] *= fade_in_curve
                channel_data[-fade_out_frames:] *= fade_out_curve
        else:
            cut_audio_data[:fade_in_frames] *= fade_in_curve
            cut_audio_data[-fade_out_frames:] *= fade_out_curve

        # Plot the waveform with fade-in and fade-out applied
        if num_channels > 1:
            overlay_fig = go.Figure()
            for channel in range(num_channels):
                channel_data = cut_audio_data[channel::num_channels]
                overlay_fig.add_trace(go.Scatter(x=cut_time_axis, y=channel_data, name=f'Channel {channel+1}', yaxis='y1'))
        else:
            overlay_fig = go.Figure(go.Scatter(x=cut_time_axis, y=cut_audio_data, name='Waveform', yaxis='y1'))
        overlay_fig.update_layout(
            title=f"Waveform with Fade-in and Fade-out {name}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            showlegend=False,
        )

        # Add fade-in and fade-out curve traces to the plot
        fade_in_trace = go.Scatter(x=cut_time_axis[:fade_in_frames],
                                   y=fade_in_curve,
                                   name='Fade-in Curve',
                                   yaxis='y2',
                                   fill='tozeroy',
                                   fillcolor='rgba(255,0,0,0.2)',
                                   line=dict(color='rgba(255,0,0,0.8)'))

        fade_out_trace = go.Scatter(x=cut_time_axis[-fade_out_frames:],
                                     y=fade_out_curve,
                                     name='Fade-out Curve',
                                     yaxis='y2',
                                     fill='tozeroy',
                                     fillcolor='rgba(255,0,0,0.2)',
                                     line=dict(color='rgba(255,0,0,0.8)'))

        overlay_fig.add_trace(fade_in_trace)
        overlay_fig.add_trace(fade_out_trace)

        # Configure the layout for dual y-axes
        overlay_fig.update_layout(
            yaxis1=dict(
                title='Amplitude',
                side='left',
                showgrid=False,
                zeroline=False
            ),
            yaxis2=dict(
                title='Fade-in and Fade-out',
                range=[0, 1],
                side='right',
                overlaying='y',
                showgrid=False,
                zeroline=False
            )
        )

        if plot_width is not None or plot_height is not None:
            overlay_fig.update_layout(width=plot_width, height=plot_height, autosize=False)
        if plotly_layout:
            overlay_fig.update_layout(**plotly_layout)

        # Display the waveform plot with fade-in and fade-out applied
        show_plotly(overlay_fig)

    else:
        # Plot the cut waveform only
        if num_channels > 1:
            cut_fig = go.Figure()
            for channel in range(num_channels):
                channel_data = cut_audio_data[channel::num_channels]
                cut_fig.add_trace(go.Scatter(x=cut_time_axis, y=channel_data, name=f'Channel {channel+1}'))
        else:
            cut_fig = go.Figure(go.Scatter(x=cut_time_axis, y=cut_audio_data))
        cut_fig.update_layout(
            title=f"Cut Waveform of {name}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
        )
        if plot_width is not None or plot_height is not None:
            cut_fig.update_layout(width=plot_width, height=plot_height, autosize=False)
        if plotly_layout:
            cut_fig.update_layout(**plotly_layout)
        # Display the cut waveform plot
        show_plotly(cut_fig)

    return cut_audio_data, cut_duration


def trim_and_fade_and_render(
    *,
    audio_data,
    sample_rate,
    audio_info: dict,
    start_time=0.4,
    end_time=1.6,
    add_fades=True,
    fade_in_duration=0.2,
    fade_out_duration=0.3,
    fade_in_exponent=0.8,
    fade_out_exponent=1.5,
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict | None = None,
    plot_title_source: str = "filename",
    plot_title_mode: str = "basename",
    # Keep downstream analysis stable (independent of render_audio's internal sanitization).
    cut_sanitize: bool = True,
    # Back-compat alias (deprecated): kept so existing notebooks/calls don't break.
    sanitize_cut: bool | None = None,
    cut_verbose: bool = True,
    # Optional render/playback (same parameter names as render_audio for familiarity).
    render: bool = True,
    fs_target_name: str | int | None = "source",
    bit_rate: int = 24,
    save_audio: bool = True,
    player: bool = True,
    sanitize: bool = True,
    verbose: bool = True,
):
    """
    One-call convenience wrapper for notebooks: trim/fade + sanitize + optional render/playback.

    Parameters are designed to match the existing notebook cell style:
      - trim params match `trim_and_fade_audio`
      - render params match `render_audio`

    Returns (cut_audio_data, cut_duration, render_result).
      - render_result is whatever `render_audio(...)` returns (file path or (audio, fs)),
        or None when render=False.
    """
    if sanitize_cut is not None:
        cut_sanitize = bool(sanitize_cut)

    if not isinstance(audio_info, dict):
        raise TypeError("audio_info must be a dict (as returned by load_audio_sample)")

    for key in ("num_channels", "duration"):
        if key not in audio_info:
            raise KeyError(f"audio_info missing required key: {key!r}")

    plot_title_source_norm = str(plot_title_source).lower().strip().replace("-", "_")
    if plot_title_source_norm in {"filename", "file", "wav_filename"}:
        title_name = audio_info.get("wav_filename") or audio_info.get("wav_source") or "audio"
    elif plot_title_source_norm in {"source", "wav_source", "url"}:
        title_name = audio_info.get("wav_source") or audio_info.get("wav_filename") or "audio"
    else:
        raise ValueError(
            "plot_title_source must be one of: 'filename' | 'source' "
            f"(got {plot_title_source!r})"
        )

    # Back-compat: keep passing a label positionally, but titles are now controlled via title_* args.
    wav_label = audio_info.get("wav_filename") or audio_info.get("wav_source") or "audio"

    cut_audio_data, cut_duration = trim_and_fade_audio(
        audio_data,
        sample_rate,
        audio_info["num_channels"],
        audio_info["duration"],
        wav_label,
        start_time=start_time,
        end_time=end_time,
        add_fades=add_fades,
        fade_in_duration=fade_in_duration,
        fade_out_duration=fade_out_duration,
        fade_in_exponent=fade_in_exponent,
        fade_out_exponent=fade_out_exponent,
        plot_width=plot_width,
        plot_height=plot_height,
        plotly_layout=plotly_layout,
        title_name=title_name,
        title_mode=plot_title_mode,
    )

    cut_audio_data = ensure_finite_audio(
        cut_audio_data, name="cut_audio_data", sanitize=cut_sanitize, verbose=cut_verbose
    )

    render_result = None
    if render:
        from .generate_wave_file import render_audio

        render_result = render_audio(
            cut_audio_data,
            sample_rate,
            fs_target_name=fs_target_name,
            bit_rate=bit_rate,
            save_audio=save_audio,
            player=player,
            sanitize=sanitize,
            verbose=verbose,
        )

    return cut_audio_data, cut_duration, render_result


def load_audio_sample(
    wav_source,
    desired_sample_rate=None,
    convert_to_mono=True,
    *,
    verbose: bool = False,
    download_mode: str = "cwd",
    download_dir: str | None = None,
    show_load_bar: bool = False,
):
    """
    Load-only helper for audio samples (local path or URL).

    This function intentionally performs NO playback and NO plotting.
    Use `render_audio(...)` for playback/export and `plot_waveform(...)` for visualization.

    Returns (audio_data, sample_rate, info_dict) where info contains:
      wav_source, wav_filename, num_channels, duration, playback_audio.
    """
    is_url = _is_http_url(wav_source)
    is_temp_download = bool(is_url and str(download_mode).lower().strip().replace("-", "_") == "temp")

    source_path = load_wav_from_source(
        wav_source,
        verbose=verbose,
        download_mode=download_mode,
        download_dir=download_dir,
        show_load_bar=show_load_bar,
    )

    audio_data, sample_rate = load_audio_data(source_path, desired_sample_rate, convert_to_mono)

    # If we downloaded to a temporary file, delete it after decoding so we don't litter
    # the notebook working directory.
    temp_deleted = False
    if is_temp_download:
        try:
            os.remove(source_path)
            temp_deleted = True
        except Exception:
            temp_deleted = False

    audio_arr = np.asarray(audio_data)
    num_channels = _infer_num_channels(audio_arr)
    num_samples = _infer_num_samples(audio_arr)
    duration = (num_samples / float(sample_rate)) if sample_rate else 0.0

    # Choose a single channel for playback/quick inspection when multichannel.
    if num_channels > 1 and audio_data.ndim > 1:
        playback_audio = audio_data[0] if audio_data.shape[0] <= audio_data.shape[-1] else audio_data[:, 0]
    else:
        playback_audio = audio_data

    return audio_data, sample_rate, {
        "wav_source": wav_source,
        # Back-compat: keep the key name, but in temp-download mode this is a display label
        # (not a stable file path).
        "wav_filename": (_basename_for_title(wav_source) if is_temp_download else source_path),
        "source_local_path": source_path,
        "source_is_url": bool(is_url),
        "source_is_tempfile": bool(is_temp_download),
        "source_tempfile_deleted": bool(temp_deleted),
        "num_channels": num_channels,
        "duration": duration,
        "playback_audio": playback_audio,
    }


def load_audio_sample_and_preview(
    *,
    wav_source,
    desired_sample_rate=None,
    convert_to_mono=True,
    show_properties: bool = True,
    show_waveform: bool = True,
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict | None = None,
    plot_title_source: str = "filename",
    plot_title_mode: str = "full",
    play_audio: bool = True,
    playback_fs_target_name: str | None = "source",
    playback_bit_rate: int = 24,
    playback_save_audio: bool = False,
    playback_sanitize: bool = True,
    source_verbose: bool = False,
    download_mode: str = "cwd",
    download_dir: str | None = None,
    show_load_bar: bool = False,
    # Back-compat alias (deprecated): kept so existing notebooks/calls don't break.
    verbose: bool | None = None,
    playback_verbose: bool = False,
):
    """
    One-call convenience wrapper for notebooks: load + optional info/plot/playback.

    This intentionally *does* have side effects (printing/plotting/audio widget) based on flags.
    For a load-only helper, use `load_audio_sample(...)`.

    Returns (audio_data, sample_rate, info_dict). The returned info_dict is the same as
    `load_audio_sample`, plus optional keys:
      - waveform_duration_s
      - playback_result

    Notes:
      - playback_fs_target_name='source' keeps the loaded sample rate (no resampling in render_audio).
      - source_verbose controls printing during source loading/downloading.
    """
    if verbose is not None:
        source_verbose = bool(verbose)

    audio_data, sample_rate, audio_info = load_audio_sample(
        wav_source=wav_source,
        desired_sample_rate=desired_sample_rate,
        convert_to_mono=convert_to_mono,
        verbose=source_verbose,
        download_mode=download_mode,
        download_dir=download_dir,
        show_load_bar=show_load_bar,
    )

    if show_properties:
        _ = display_audio_properties(audio_data, sample_rate, wav_source)

    if show_waveform:
        # Plot a single channel even when `convert_to_mono=False`.
        plot_title_source_norm = str(plot_title_source).lower().strip().replace("-", "_")
        if plot_title_source_norm in {"filename", "file", "wav_filename"}:
            title_name = audio_info["wav_filename"]
        elif plot_title_source_norm in {"source", "wav_source", "url"}:
            title_name = audio_info["wav_source"]
        else:
            raise ValueError(
                "plot_title_source must be one of: 'filename' | 'source' "
                f"(got {plot_title_source!r})"
            )

        audio_info["waveform_duration_s"] = plot_waveform(
            audio_info["playback_audio"],
            sample_rate,
            audio_info["wav_filename"],
            plot_width=plot_width,
            plot_height=plot_height,
            plotly_layout=plotly_layout,
            title_name=title_name,
            title_mode=plot_title_mode,
        )

    if play_audio:
        # Local import to keep `load_audio_sample` lightweight and avoid unnecessary notebook deps.
        from .generate_wave_file import render_audio

        audio_info["playback_result"] = render_audio(
            audio_info["playback_audio"],
            sample_rate,
            fs_target_name=playback_fs_target_name,
            bit_rate=playback_bit_rate,
            save_audio=playback_save_audio,
            player=True,
            sanitize=playback_sanitize,
            verbose=playback_verbose,
        )

    return audio_data, sample_rate, audio_info


def plot_spectrogram(
    audio_data,
    sample_rate,
    y_axis_mode="linear",
    y_axis_mix=0.5,
    mixed_log_floor_hz=1.0,
    amp_scale="db",
    power_law_gamma=None,
    n_fft=2048,
    window_type="hann",
    overlap=0.75,
    oversample_factor=1.0,
    mel_bins=128,
    mel_fmax=None,
    scaling="density",
    mode="magnitude",
    cmap="Viridis",
    boundary=None,
    padded=False,
    time_range="signal",
    time_reference="center",
    show=True,
):
    """
    Build a Plotly spectrogram figure with optional log/mel scaling and FFT controls.

    Returns (fig, info) where info has frequencies, times, Sxx_dB.

    Edge framing / time base parameters (important for fades and “does the first frame start at 0?”):

    - boundary:
      - None: no padding; scipy uses “valid” framing (first/last ~n_fft/2 may be missing in view)
      - "zeros" (centered): pad by n_fft//2 on both sides, so a frame is centered at t=0.
        This makes the first frame contain pre-zero padding (lower energy near start).
      - "zeros_end" (start-aligned): pad only the end, so the first *full* window starts at t=0.
        Useful when you want onset/fade-in to align to the start without pre-padding artifacts.
      - any other string: forwarded as a numpy.pad mode (pads both sides by n_fft//2)

    - padded:
      - False: keep scipy’s “valid” behavior (no extra right padding beyond boundary padding)
      - True: pad the end just enough so the last hop boundary inside the original signal has a
        corresponding frame (avoids overlap-dependent gaps or extra all-padding frames at the end)

    - time_reference:
      - "center": x positions are window centers (scipy default)
      - "start": x positions are start-aligned for display; we keep heatmap cell edges aligned
        to 0 by placing the first *cell center* at hop/2 (not 0). Start times are returned in info.

    - time_range:
      - "stft": show only the time span covered by returned STFT frames
      - "signal": force x-axis to [0, duration] for easier comparison with waveform
    """
    def _mixed_warp(freqs_hz, mix, fmax_hz, log_floor_hz):
        """
        Warp frequency coordinates to continuously blend linear->log spacing.

        mix=0   => y = f (linear)
        mix=1   => y = log-scaled coordinate normalized to [0, fmax_hz]
        """
        mix = float(mix)
        if not (0.0 <= mix <= 1.0):
            raise ValueError(f"y_axis_mix must be in [0,1]; got {mix}")
        log_floor_hz = float(log_floor_hz)
        if log_floor_hz <= 0:
            raise ValueError(f"mixed_log_floor_hz must be > 0; got {log_floor_hz}")

        freqs_hz = np.asarray(freqs_hz, dtype=float)
        if freqs_hz.ndim != 1:
            raise ValueError("freqs_hz must be a 1D array")
        fmax_hz = float(fmax_hz)
        if fmax_hz <= 0:
            # Degenerate case; return something sensible.
            return freqs_hz.copy()

        # Log map with a floor so 0 Hz is well-defined; normalize to match endpoints.
        log0 = np.log10(log_floor_hz)
        logf = np.log10(freqs_hz + log_floor_hz)
        log_max = np.log10(fmax_hz + log_floor_hz)
        denom = max(1e-12, (log_max - log0))
        log_scaled = (logf - log0) / denom * fmax_hz

        return (1.0 - mix) * freqs_hz + mix * log_scaled

    def _default_tick_freqs_hz(fmax_hz):
        fmax_hz = float(fmax_hz)
        if fmax_hz <= 0:
            return np.array([0.0])
        if fmax_hz <= 200:
            return np.linspace(0.0, fmax_hz, num=6)
        start = 20.0
        if fmax_hz < start:
            start = max(1.0, fmax_hz / 10.0)
        # 0 plus a few log-ish ticks up to fmax
        return np.concatenate(([0.0], np.geomspace(start, fmax_hz, num=6)))

    def _format_hz(v):
        v = float(v)
        if v >= 1000:
            return f"{v/1000:.1f} kHz"
        return f"{v:.0f} Hz"

    # Validate parameters to avoid silent breakage.
    if not (0 <= overlap < 1):
        raise ValueError(f"overlap must be in [0,1); got {overlap}")
    if overlap >= 0.95:
        raise ValueError("overlap too high; keep below 0.95 to avoid zero hop length")
    if oversample_factor < 1.0:
        raise ValueError("oversample_factor must be >= 1.0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if mel_bins <= 0:
        raise ValueError("mel_bins must be > 0")
    if amp_scale not in {"db", "linear", "linear_norm"}:
        raise ValueError(f"amp_scale must be db|linear|linear_norm; got {amp_scale}")
    if y_axis_mode not in {"linear", "log", "mel", "mixed"}:
        raise ValueError(f"y_axis_mode must be linear|log|mel|mixed; got {y_axis_mode}")
    if scaling not in {"density", "spectrum"}:
        raise ValueError(f"scaling must be density|spectrum; got {scaling}")
    if mode not in {"magnitude", "psd"}:
        raise ValueError(f"mode must be magnitude|psd; got {mode}")
    if time_range not in {"stft", "signal"}:
        raise ValueError(f"time_range must be stft|signal; got {time_range}")
    if time_reference not in {"center", "start"}:
        raise ValueError(f"time_reference must be center|start; got {time_reference}")

    mel_fmax = mel_fmax or (sample_rate / 2)
    if mel_fmax <= 0 or mel_fmax > sample_rate / 2:
        raise ValueError("mel_fmax must be in (0, Nyquist]")

    # Normalize input shape for downstream tooling (scipy/librosa).
    audio_data = np.asarray(audio_data, dtype=float).reshape(-1)
    # Match the waveform time base used by `plot_waves` (which includes the end sample):
    # last sample time is (N-1)/sr, not N/sr.
    duration_s = (audio_data.size - 1) / float(sample_rate) if audio_data.size else 0.0

    hop_length = max(1, int(n_fft * (1 - overlap)))
    noverlap = n_fft - hop_length
    nfft = int(max(n_fft, n_fft * oversample_factor))
    window = get_window(window_type, n_fft, fftbins=True)

    # Optional padding to enable full-duration analysis (i.e., frames at the beginning/end).
    #
    # scipy.signal.spectrogram uses "valid" framing: it only returns frames where the full
    # n_fft window fits inside the input array, and its time bins are *segment centers*.
    #
    # To emulate STFT-like centered framing, we can pad by n_fft//2 on both sides and then
    # shift the returned time vector by that left-pad amount.
    pad_left = 0
    audio_for_spec = audio_data

    if boundary is not None:
        pad_half = int(n_fft // 2)
        pad_left = pad_half
        pad_right = pad_half

        # Common boundary modes for STFT-like framing.
        # - "zeros": centered framing (pads both sides; first frame is half padding -> lower energy)
        # - "zeros_end": start-aligned framing (pads only at the end; first full window begins at t=0)
        boundary_norm = boundary.strip().lower() if isinstance(boundary, str) else boundary

        if boundary_norm in {"zeros_end", "zeros_right", "end", "pad_end"}:
            pad_left = 0
            pad_right = pad_half
            audio_for_spec = np.pad(
                audio_for_spec, (pad_left, pad_right), mode="constant", constant_values=0.0
            )
        elif boundary_norm in {"zeros_start", "zeros_left", "start", "pad_start"}:
            pad_left = pad_half
            pad_right = 0
            audio_for_spec = np.pad(
                audio_for_spec, (pad_left, pad_right), mode="constant", constant_values=0.0
            )
        elif boundary_norm in {"zeros", "zero", "center", "centered"}:
            audio_for_spec = np.pad(
                audio_for_spec, (pad_left, pad_right), mode="constant", constant_values=0.0
            )
        else:
            # Defer to numpy pad modes, e.g. "reflect", "symmetric", "edge".
            audio_for_spec = np.pad(audio_for_spec, (pad_left, pad_right), mode=boundary)

        if padded:
            # Pad at the end so that the *last frame start* lands on the last hop boundary
            # within the original signal duration. This avoids:
            # - leaving a "gap" at the end for small hop sizes
            # - adding an extra (mostly padded) frame beyond the end for large hop sizes
            #
            # Let N be the original (un-padded) sample count. Define:
            #   last_start = floor((N-1)/hop) * hop
            # We then ensure the padded array is long enough for a window starting at:
            #   pad_left + last_start
            # i.e. length >= pad_left + last_start + n_fft.
            orig_n = int(audio_data.size)
            if orig_n > 0:
                # We calculate the start index of the last frame such that it starts
                # at or before the last sample of the signal.
                #
                # last_start = floor((N-1)/hop) * hop
                #
                # This ensures consistent coverage for all overlaps:
                # - The last frame always contains at least one signal sample.
                # - We pad enough to allow this frame to exist (valid framing).
                last_start = int(((orig_n - 1) // hop_length) * hop_length)
                
                required_len = int(pad_left + last_start + n_fft)
                pad_end = max(0, required_len - int(audio_for_spec.size))
                if pad_end:
                    audio_for_spec = np.pad(
                        audio_for_spec, (0, int(pad_end)), mode="constant", constant_values=0.0
                    )

    freqs, times, Sxx = spectrogram(
        audio_for_spec,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
        mode="magnitude" if mode == "magnitude" else "psd",
        axis=-1,
    )

    # If we centered the framing via padding, shift times back so 0 aligns to the original
    # (un-padded) signal start.
    if pad_left:
        times = times - (pad_left / float(sample_rate))

    # scipy.signal.spectrogram reports segment *centers*.
    times_center = times
    times_start = times_center - (int(n_fft // 2) / float(sample_rate))

    if time_reference == "center":
        times_for_plot = times_center
    else:
        # "start" reference for display:
        #
        # Plotly renders each heatmap column centered at x[i] and extends half a column
        # to the left/right. If we set the first center to 0.0, you’ll see a half-column
        # clipped at the left edge (looks like a “half frame”).
        #
        # To make the *cell edges* align to 0.0, we keep the first center at hop/2.
        hop_s = hop_length / float(sample_rate)

        # Drop negative start-times (can happen with centered padding like boundary="zeros").
        if times_start.size:
            keep = times_start >= 0.0
            if not keep.all():
                times_center = times_center[keep]
                times_start = times_start[keep]
                Sxx = Sxx[:, keep]

        times_for_plot = times_start + (hop_s / 2.0)

    # Prepare plot arrays. Keep `freqs_hz` for reporting/tick labels even if we warp.
    if y_axis_mode == "mel":
        # Use the same time grid/framing as the scipy spectrogram above, then project
        # the (power) spectrum into mel bands via a mel filter bank.
        S_power = (
            np.square(np.maximum(Sxx, 0.0)) if mode == "magnitude" else np.maximum(Sxx, 0.0)
        )
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=nfft, n_mels=mel_bins, fmax=mel_fmax
        )
        mel_spec = mel_basis @ S_power
        
        if amp_scale == "db":
            S_plot = 10 * np.log10(np.maximum(mel_spec, 1e-12))
            cbar_label = "Amplitude (dB)"
        else:
            S_plot = mel_spec
            if amp_scale == "linear_norm":
                m_val = np.max(S_plot)
                if m_val > 0:
                    S_plot /= m_val
                cbar_label = "Normalized Amplitude"
            else:
                cbar_label = "Amplitude"
            
            # Optional power-law compression (Gamma correction) for linear modes
            if power_law_gamma is not None:
                gamma = float(power_law_gamma)
                if gamma > 0 and gamma != 1.0:
                    S_plot = np.power(S_plot, gamma)
                    cbar_label += f" (gamma={gamma})"

        freqs_hz = librosa.mel_frequencies(n_mels=mel_bins, fmax=mel_fmax)
        y_label = "Frequency"

        # Important: px.imshow hover labels use the *actual y values*, not ticktext,
        # so if we use numeric mel-bin indices you'll see mixed hover like:
        #   "Frequency 96" (bin index) vs "Frequency 9.5 kHz" (only when y hits a tickval).
        #
        # To make every bin map to a frequency label (and keep equal mel-bin spacing),
        # we use a categorical y axis with one label per bin.
        y_for_plot = [_format_hz(v) for v in freqs_hz]
        if len(set(y_for_plot)) != len(y_for_plot):
            # Rare case (very large mel_bins): avoid duplicate category labels due to rounding.
            y_for_plot = [f"{float(v):.3f} Hz" if float(v) < 1000 else f"{float(v)/1000:.3f} kHz" for v in freqs_hz]

        y_axis_type = "category"

        tick_idx = np.linspace(0, len(freqs_hz) - 1, num=min(6, len(freqs_hz)), dtype=int)
        y_tickvals = [y_for_plot[i] for i in tick_idx]
        y_ticktext = [_format_hz(freqs_hz[i]) for i in tick_idx]
        title = "Spectrogram (mel y-axis)"
    else:
        if amp_scale == "db":
            S_plot = 10 * np.log10(np.maximum(Sxx, 1e-12))
            cbar_label = "Amplitude (dB)"
        else:
            S_plot = Sxx
            if amp_scale == "linear_norm":
                m_val = np.max(S_plot)
                if m_val > 0:
                    S_plot /= m_val
                cbar_label = "Normalized Amplitude"
            else:
                cbar_label = "Amplitude"
            
            # Optional power-law compression (Gamma correction) for linear modes
            if power_law_gamma is not None:
                gamma = float(power_law_gamma)
                if gamma > 0 and gamma != 1.0:
                    S_plot = np.power(S_plot, gamma)
                    cbar_label += f" (gamma={gamma})"

        freqs_hz = freqs
        y_axis_type = "linear"
        y_label = "Frequency (Hz)"
        y_tickvals = None
        y_ticktext = None
        title = f"Spectrogram ({y_axis_mode} y-axis)"

        if y_axis_mode == "linear":
            y_for_plot = freqs_hz
        elif y_axis_mode == "log":
            # Drop 0 Hz to avoid log axis issues.
            if freqs_hz[0] == 0:
                freqs_plot = freqs_hz[1:]
                S_plot = S_plot[1:, :]
            else:
                freqs_plot = freqs_hz
            y_for_plot = freqs_plot
            y_axis_type = "log"
        elif y_axis_mode == "mixed":
            fmax_hz = float(freqs_hz[-1]) if len(freqs_hz) else 0.0
            y_for_plot = _mixed_warp(freqs_hz, y_axis_mix, fmax_hz=fmax_hz, log_floor_hz=mixed_log_floor_hz)
            # Keep axis linear but show Hz ticks at their warped positions.
            tick_freqs = _default_tick_freqs_hz(fmax_hz)
            y_tickvals = _mixed_warp(tick_freqs, y_axis_mix, fmax_hz=fmax_hz, log_floor_hz=mixed_log_floor_hz)
            y_ticktext = [_format_hz(v) for v in tick_freqs]
            title = f"Spectrogram (mixed y-axis, mix={y_axis_mix:.2f})"

    fig = px.imshow(
        S_plot,
        x=times_for_plot,
        y=y_for_plot,
        origin="lower",
        aspect="auto",
        color_continuous_scale=cmap,
        labels={"x": "Time (s)", "y": y_label, "color": cbar_label},
        title=title,
    )
    fig.update_yaxes(type=y_axis_type)
    if y_tickvals is not None and y_ticktext is not None:
        fig.update_yaxes(tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext)

    # Hide the colorbar ("amplitude legend") on the right.
    fig.update_layout(coloraxis_showscale=False)

    # With "valid" STFT framing (no padding), the last time bin is often < signal duration.
    # Showing the full signal duration on the x-axis avoids confusion when comparing
    # waveform duration vs. spectrogram extent.
    #
    # With padded STFT framing (padded=True, overlap=0, etc.), the last time bin can extend
    # *beyond* the signal duration (showing padding as signal).
    # We default to clipping the view to the actual signal duration.
    #
    # You can override this by passing time_range="stft" or explicitly setting x-axis range
    # on the returned figure.
    final_x_range = None
    if time_range == "signal" and duration_s > 0:
        final_x_range = [0.0, duration_s]
        fig.update_xaxes(range=final_x_range, autorange=False)
    elif time_range == "stft":
        # Let plotly auto-scale to show all generated frames (including padding tails)
        pass

    if show:
        show_plotly(fig)

    return fig, {
        # The x values actually used for plotting (cell centers).
        "times": times_for_plot,
        # Additional time bases for interpretation.
        "times_center": times_center,
        "times_start": times_start,
        "frequencies": freqs_hz,
        "Sxx_plot": S_plot,
        "Sxx_raw": Sxx,
        "duration_s": duration_s,
        "boundary": boundary,
        "padded": padded,
        "time_range": time_range,
        "time_reference": time_reference,
        "plot_x_range": final_x_range,
        "amp_scale": amp_scale,
        "power_law_gamma": power_law_gamma,
    }


def plot_spectrogram_with_waveform(
    audio_data,
    sample_rate,
    *,
    audio_info: dict | None = None,
    plot_width: int | None = None,
    plot_height: int | None = None,
    plotly_layout: dict | None = None,
    plot_title_source: str = "filename",
    plot_title_mode: str = "basename",
    plot_title_name: str | None = None,
    waveform_mode: str = "integrated",
    # Back-compat alias (deprecated): kept so existing notebooks/calls don't break.
    waveform_under: bool | None = None,
    waveform_height_ratio: float = 0.25,
    waveform_line_width: float = 1.0,
    waveform_max_points: int | str | None = 10000,
    show: bool = True,
    print_info: bool = False,
    **spectrogram_kwargs,
):
    """
    Convenience wrapper: spectrogram with an optional synchronous waveform plot.

    The x-axes are synchronized (shared), so you can visually align fade-in/out with spectral changes.

    waveform_mode:
      - "integrated": waveform is a subplot under the spectrogram (single figure)
      - "separate": waveform is shown as a separate figure (same x-range)
      - "none": spectrogram only

    waveform_max_points:
      - int: maximum number of points to plot (downsamples for speed)
      - "sample_rate": use sample_rate (≈ 1 second at original sample resolution)
        (alias: "sr")
      - None: no downsampling

    `spectrogram_kwargs` are forwarded to `plot_spectrogram(...)`.
    """
    if waveform_under is not None:
        waveform_mode = "integrated" if waveform_under else "none"

    waveform_mode = str(waveform_mode).strip().lower()
    if waveform_mode not in {"integrated", "separate", "none"}:
        raise ValueError("waveform_mode must be 'integrated', 'separate', or 'none'")

    if isinstance(waveform_max_points, str):
        norm = waveform_max_points.strip().lower()
        if norm in {"sample_rate", "sr"}:
            waveform_max_points = int(sample_rate)
        else:
            raise ValueError("waveform_max_points as a string must be 'sample_rate' (alias: 'sr')")

    # Build the spectrogram figure without displaying it yet.
    spec_fig, spec_info = plot_spectrogram(
        audio_data,
        sample_rate,
        show=False,
        **spectrogram_kwargs,
    )

    # Optional: apply consistent naming and sizing to plots (for notebook consistency).
    title_name = plot_title_name
    if title_name is None and audio_info is not None:
        plot_title_source_norm = str(plot_title_source).lower().strip().replace("-", "_")
        if plot_title_source_norm in {"filename", "file", "wav_filename"}:
            title_name = audio_info.get("wav_filename") or audio_info.get("wav_source")
        elif plot_title_source_norm in {"source", "wav_source", "url"}:
            title_name = audio_info.get("wav_source") or audio_info.get("wav_filename")
        else:
            raise ValueError(
                "plot_title_source must be one of: 'filename' | 'source' "
                f"(got {plot_title_source!r})"
            )

    title_suffix = None
    if title_name:
        plot_title_mode_norm = str(plot_title_mode).lower().strip().replace("-", "_")
        if plot_title_mode_norm == "basename":
            title_suffix = _basename_for_title(title_name)
        elif plot_title_mode_norm == "full":
            title_suffix = str(title_name)
        else:
            raise ValueError(
                "plot_title_mode must be one of: 'full' | 'basename' "
                f"(got {plot_title_mode!r})"
            )

    if title_suffix:
        base_title = None
        try:
            base_title = getattr(spec_fig.layout.title, "text", None)
        except Exception:
            base_title = None
        if not base_title:
            base_title = "Spectrogram"
        spec_fig.update_layout(title=f"{base_title} {title_suffix}")

    if plotly_layout:
        spec_fig.update_layout(**plotly_layout)

    if plot_width is not None or plot_height is not None:
        spec_fig.update_layout(width=plot_width, height=plot_height, autosize=False)

    if waveform_mode == "none":
        if show:
            show_plotly(spec_fig)
        if print_info:
            _print_spectrogram_info(sample_rate, spec_info, spectrogram_kwargs)
        return spec_fig, spec_info

    from plotly.subplots import make_subplots

    waveform_height_ratio = float(waveform_height_ratio)
    if not (0.05 <= waveform_height_ratio <= 0.95):
        raise ValueError("waveform_height_ratio must be in [0.05, 0.95]")

    # Downsample waveform for speed if needed.
    y = np.asarray(audio_data, dtype=float).reshape(-1)
    n = int(y.size)
    if n <= 1:
        t = np.array([0.0])
        y_plot = y if n else np.array([0.0])
    else:
        if waveform_max_points is None:
            step = 1
        else:
            step = max(1, int(np.ceil(n / max(1, int(waveform_max_points)))))
        y_plot = y[::step]
        t = (np.arange(y_plot.size, dtype=float) * step) / float(sample_rate)

    # Use the explicitly calculated range from the spectrogram info if available.
    # Fall back to checking the figure layout if not found (legacy compat).
    x_range = spec_info.get("plot_x_range", getattr(spec_fig.layout.xaxis, "range", None))

    # Separate waveform figure option.
    wave_fig = go.Figure(go.Scatter(x=t, y=y_plot, name="Waveform", line=dict(width=waveform_line_width)))
    wave_title = "Waveform" if not title_suffix else f"Waveform — {title_suffix}"
    wave_fig.update_layout(title=wave_title, xaxis_title="Time (s)", yaxis_title="Amplitude")
    if x_range is not None:
        wave_fig.update_xaxes(range=x_range)
    if plotly_layout:
        wave_fig.update_layout(**plotly_layout)
    if plot_width is not None or plot_height is not None:
        wave_fig.update_layout(width=plot_width, height=plot_height, autosize=False)

    if waveform_mode == "separate":
        if show:
            show_plotly(spec_fig)
            show_plotly(wave_fig)
        if print_info:
            _print_spectrogram_info(sample_rate, spec_info, spectrogram_kwargs)
        spec_info["waveform_fig"] = wave_fig
        return spec_fig, spec_info

    # Integrated: create a single subplot figure.
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[1.0 - waveform_height_ratio, waveform_height_ratio],
    )

    # Row 1: spectrogram (copy traces from the existing fig).
    for tr in spec_fig.data:
        fig.add_trace(tr, row=1, col=1)

    # Preserve relevant layout settings from the spectrogram figure (coloraxis, titles, axis config).
    fig.update_layout(
        title=spec_fig.layout.title,
        coloraxis=spec_fig.layout.coloraxis,
        coloraxis_showscale=getattr(spec_fig.layout, "coloraxis_showscale", False),
    )

    # Apply the spectrogram's y-axis configuration to the top subplot.
    fig.update_yaxes(
        type=spec_fig.layout.yaxis.type,
        tickmode=spec_fig.layout.yaxis.tickmode,
        tickvals=spec_fig.layout.yaxis.tickvals,
        ticktext=spec_fig.layout.yaxis.ticktext,
        title_text=spec_fig.layout.yaxis.title.text if spec_fig.layout.yaxis.title else None,
        row=1,
        col=1,
    )

    # Row 2: waveform.
    fig.add_trace(
        go.Scatter(x=t, y=y_plot, name="Waveform", line=dict(width=waveform_line_width)),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    # Keep the same x-axis range the spectrogram chose (e.g., [0, duration] when time_range='signal').
    # NOTE: When shared_xaxes=True in make_subplots, updating x-axis on any subplot (or without row/col)
    # affects the shared axis. However, explicitly targeting the axis index is safer.
    # The spectrogram is on row 1, waveform on row 2.
    if x_range is not None:
        fig.update_xaxes(range=x_range, autorange=False)

    if plotly_layout:
        fig.update_layout(**plotly_layout)
    if plot_width is not None or plot_height is not None:
        fig.update_layout(width=plot_width, height=plot_height, autosize=False)

    if show:
        show_plotly(fig)
    if print_info:
        _print_spectrogram_info(sample_rate, spec_info, spectrogram_kwargs)

    return fig, spec_info


def _print_spectrogram_info(sample_rate, spec_info: dict, spectrogram_kwargs: dict):
    """
    Print a compact, notebook-friendly summary of STFT/spectrogram settings.

    This is intended for interactive exploration (helps avoid “what did I run?” confusion).
    """
    try:
        sr = float(sample_rate)
    except Exception:
        sr = 0.0

    # Prefer values from kwargs; fall back to info where possible.
    n_fft = spectrogram_kwargs.get("n_fft", None)
    overlap = spectrogram_kwargs.get("overlap", None)
    oversample_factor = spectrogram_kwargs.get("oversample_factor", None)
    window_type = spectrogram_kwargs.get("window_type", None)

    boundary = spec_info.get("boundary", None)
    padded = spec_info.get("padded", None)
    time_reference = spec_info.get("time_reference", None)
    time_range = spec_info.get("time_range", None)

    times = spec_info.get("times", None)
    n_frames = int(len(times)) if hasattr(times, "__len__") else None

    # Derived quantities (when possible).
    hop = None
    redundancy = None
    if n_fft is not None and overlap is not None:
        try:
            hop = max(1, int(int(n_fft) * (1 - float(overlap))))
            redundancy = float(int(n_fft)) / float(hop) if hop else None  # = 1/(1-overlap)
        except Exception:
            hop = None
            redundancy = None
    
    amp_scale = spec_info.get("amp_scale", "db")
    
    power_law_gamma = spec_info.get("power_law_gamma", None)

    # Calculate padded samples at the end (from info if available)
    # The spectrogram function pads to match the frames; we can infer total padding
    # if we know signal duration vs last frame end, but simpler is to use the raw
    # array info if passed, or just omit if too complex to reconstruct here.
    # However, if 'boundary' and 'padded' are used, we can estimate.
    # Actually, we can just infer from the duration difference if we had the original signal len.
    # But we don't have signal len here easily.
    #
    # Better approach: check if 'plot_x_range' (signal duration) is available in info.
    duration_s = spec_info.get("duration_s", None)
    times = spec_info.get("times", [])
    padding_info_str = ""
    
    if duration_s is not None and len(times) > 0 and n_fft is not None:
         # Last frame end time (approx, assuming centered/start logic matches)
         # In audio_utils, we return cell centers.
         # For start-aligned: t_start = t_center - n_fft/2/sr
         # For signal coverage: end of last frame = t_start_last + n_fft/sr
         
         # We need to know if times are centers or starts.
         # spec_info['times'] are the plotted x-values.
         # The code says:
         # if time_reference == "center": plotted = centers
         # else: plotted = start + hop/2
         
         # Let's use the explicit "times_start" if available in spec_info, otherwise infer.
         t_starts = spec_info.get("times_start", None)
         if t_starts is not None:
             last_start = t_starts[-1]
             last_end = last_start + (n_fft / sr)
             excess_s = last_end - duration_s
             if excess_s > 0:
                 excess_samples = int(round(excess_s * sr))
                 padding_info_str = f"  (padded at end: ~{excess_samples} samples, {excess_s:.4f}s)"

    # Frequency resolution
    freq_res_str = ""
    if n_fft is not None and sr > 0:
         # Base resolution = sr / n_fft
         # With oversampling (zero padding): res = sr / nfft_total
         # nfft_total = n_fft * oversample_factor
         os_factor = float(oversample_factor) if oversample_factor else 1.0
         nfft_total = int(n_fft * os_factor)
         df = sr / nfft_total
         freq_res_str = f"  (bin width: {df:.2f} Hz)"

    overlap_pct = None
    if overlap is not None:
        try:
            overlap_pct = 100.0 * float(overlap)
        except Exception:
            overlap_pct = None

    def _fmt(x):
        return "?" if x is None else str(x)

    def _fmt_s(x):
        if x is None or sr <= 0:
            return "?"
        return f"{(float(x)/sr):.4f}s"

    print("Spectrogram settings:")
    print(f"- sample_rate: {_fmt(sr) if sr else 'unknown'} Hz")
    print(f"- n_fft: {_fmt(n_fft)}  (window: {_fmt_s(n_fft)})")
    print(f"- window_type: {_fmt(window_type)}")
    if hop is not None:
        print(f"- hop_length: {hop}  (hop: {_fmt_s(hop)})")
    if overlap is not None:
        if overlap_pct is not None:
            print(f"- overlap: {float(overlap):.3f} ({overlap_pct:.1f}%)")
        else:
            print(f"- overlap: {_fmt(overlap)}")
    if redundancy is not None:
        print(f"- redundancy (n_fft / hop): {redundancy:.3f}x")
    print(f"- oversample_factor: {_fmt(oversample_factor)}{freq_res_str}")
    print(f"- boundary: {_fmt(boundary)}; padded: {_fmt(padded)}")
    print(f"- time_reference: {_fmt(time_reference)}; time_range: {_fmt(time_range)}")
    print(f"- amplitude: {amp_scale}" + (f" (gamma={power_law_gamma})" if power_law_gamma else ""))
    if n_frames is not None:
        print(f"- frames: {n_frames}{padding_info_str}")

