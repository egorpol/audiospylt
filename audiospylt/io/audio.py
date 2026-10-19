"""Audio loading and pure edits with an explicit channel-first layout."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO, TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    import requests


def _sample_rate(sr: int) -> int:
    if isinstance(sr, (bool, np.bool_)) or not isinstance(sr, Integral) or sr <= 0:
        raise ValueError(f"sr must be a positive integer; got {sr!r}")
    return int(sr)


def _audio_array(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"audio shape {array.shape} must be 1-D (samples,) or 2-D (channels, samples)"
        )
    return array


def _layout(array: np.ndarray) -> str:
    layout = "(samples,)" if array.ndim == 1 else "(channels, samples)"
    return f"audio shape {array.shape} is read as {layout}"


def _time_samples(value: float, sr: int, name: str, array: np.ndarray) -> int:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite seconds; got {value!r} ({_layout(array)})")
    return round(value * sr)


def _decode(source: str | os.PathLike[str] | BinaryIO, label: str) -> tuple[np.ndarray, int]:
    try:
        # Reading into memory also supports binary streams with no seek/tell.
        if hasattr(source, "read"):
            source = BytesIO(source.read())
        data, rate = sf.read(source, dtype="float64", always_2d=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError(f"Cannot decode audio from {label!r}: {error}") from error
    return np.ascontiguousarray(data.T), int(rate)


def _download_name(url: str) -> str:
    segment = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", segment)
    stem, extension = os.path.splitext(safe_name)
    digest = sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{stem or 'audio'}-{digest}{extension}"


def _keep_download(body: bytes, destination: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_url(
    url: str,
    download_dir: str | os.PathLike[str] | None,
    refresh: bool,
    timeout: float,
    session: requests.Session | None,
) -> tuple[np.ndarray, int]:
    destination = None
    if download_dir is not None:
        directory = Path(download_dir)
        if not directory.exists():
            raise FileNotFoundError(f"download_dir does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"download_dir is not a directory: {directory}")
        destination = directory / _download_name(url)
        if destination.exists() and not refresh:
            return _decode(destination, url)
    elif refresh:
        raise ValueError("refresh=True requires download_dir")

    import requests

    get = requests.get if session is None else session.get
    with get(url, timeout=timeout) as response:
        response.raise_for_status()
        body = response.content
    decoded = _decode(BytesIO(body), url)
    if destination is not None:
        _keep_download(body, destination)
    return decoded


def load_audio(
    source: str | os.PathLike[str] | BinaryIO,
    *,
    sr: int | None = None,
    mono: bool = True,
    download_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> tuple[np.ndarray, int]:
    """Load a local path, http(s) URL, or binary stream without displaying it.

    Parameters
    ----------
    source : str, os.PathLike, or binary file object
        Existing local path, http(s) URL, or stream read from its current
        position. Caller-owned streams are not closed.
    sr : int, optional
        Positive target sample rate. None retains the file's rate. Resampling
        uses librosa's ``soxr_hq`` on the last axis, only when rates differ.
    mono : bool, default True
        Average channels. With False, a one-channel file is still 1-D.
    download_dir : str or os.PathLike, optional
        Existing directory in which to keep URL response bytes. By default
        URLs are decoded entirely in memory and nothing is written. Kept
        filenames use a sanitized, percent-decoded basename and an eight-digit
        SHA-256 hash of the exact URL, including its query string.
    refresh : bool, default False
        Download again instead of reusing a kept copy. A kept copy is reused
        until refresh=True; changes to the remote file are not noticed.
        Requires download_dir for URLs. Both options are ignored for non-URLs.
    timeout : float, default 60.0
        Passed unchanged to the HTTP request.
    session : requests.Session, optional
        HTTP session; otherwise use requests.get. Imported only for downloads.

    Returns
    -------
    audio : numpy.ndarray
        C-contiguous float64 samples: mono ``(samples,)`` or multichannel
        ``(channels, samples)`` in file channel order.
    sr : int
        Sample rate as a Python int.

    Raises
    ------
    FileNotFoundError
        Local source or download_dir does not exist.
    NotADirectoryError
        download_dir is a file.
    ValueError
        Invalid target rate, refresh without download_dir, or decoding failure
        (naming the source and retaining the original exception as its cause).
    requests.RequestException
        HTTP failures propagate unchanged. A download is decoded before any
        kept copy is written atomically, so failures leave existing files intact.
    """
    target_rate = None if sr is None else _sample_rate(sr)
    if isinstance(source, (str, os.PathLike)):
        if os.path.exists(source):
            audio, rate = _decode(source, os.fspath(source))
        elif (isinstance(source, str) and urlsplit(source).scheme.lower() in {"http", "https"}
              and urlsplit(source).netloc):
            audio, rate = _load_url(source, download_dir, refresh, timeout, session)
        else:
            raise FileNotFoundError(f"Audio source does not exist: {os.fspath(source)!r}")
    else:
        audio, rate = _decode(source, str(getattr(source, "name", "<buffer>")))

    if mono:
        audio = to_mono(audio)
    elif audio.shape[0] == 1:
        audio = audio[0]
    if target_rate is not None and target_rate != rate:
        import librosa

        audio = librosa.resample(
            audio, orig_sr=rate, target_sr=target_rate, res_type="soxr_hq", axis=-1,
        )
        rate = target_rate
    return np.ascontiguousarray(audio, dtype=np.float64), int(rate)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Return a float64 mono copy, averaging axis 0 for multichannel audio.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``(samples,)`` or multichannel ``(channels, samples)``. A 2-D
        array always uses this layout, regardless of its shape.

    Returns
    -------
    numpy.ndarray
        New 1-D float64 array; the input is never modified.

    Raises
    ------
    ValueError
        The input is neither 1-D nor 2-D.
    """
    array = _audio_array(audio)
    return array.copy() if array.ndim == 1 else array.mean(axis=0)


def trim(
    audio: np.ndarray, sr: int, start: float = 0.0, stop: float | None = None,
) -> np.ndarray:
    """Copy a half-open interval from the last (sample) axis of audio.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``(samples,)`` or multichannel ``(channels, samples)``.
    sr : int
        Positive sample rate.
    start, stop : float, optional
        Seconds, converted to indices with ``round(time * sr)``. Start defaults
        to zero; None for stop uses the number of samples. No bounds are clamped.

    Returns
    -------
    numpy.ndarray
        New float64 array with the same ndim; the input is never modified.

    Raises
    ------
    ValueError
        Invalid ndim/rate, nonfinite times, negative start, stop beyond the
        clip, or an empty interval after rounding. Bounds errors give seconds,
        sample indices, and the array's interpreted layout.
    """
    array = _audio_array(audio)
    sr = _sample_rate(sr)
    n_samples = array.shape[-1]
    i0 = _time_samples(start, sr, "start", array)
    stop_seconds = n_samples / sr if stop is None else stop
    i1 = n_samples if stop is None else _time_samples(stop, sr, "stop", array)
    hint = f" ({_layout(array)})"
    if start < 0:
        raise ValueError(f"start {start:g} s ({i0} samples) must be >= 0 s (0 samples){hint}")
    if i1 > n_samples:
        raise ValueError(
            f"stop {stop_seconds:g} s ({i1} samples) exceeds the clip "
            f"{n_samples / sr:g} s ({n_samples} samples){hint}"
        )
    if i1 <= i0:
        raise ValueError(
            f"stop {stop_seconds:g} s ({i1} samples) must be after "
            f"start {start:g} s ({i0} samples); empty clip{hint}"
        )
    return array[..., i0:i1].copy()


def fade(
    audio: np.ndarray,
    sr: int,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    *,
    in_exponent: float = 0.8,
    out_exponent: float = 1.5,
) -> np.ndarray:
    """Copy audio and apply power-law fades identically on every channel.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``(samples,)`` or multichannel ``(channels, samples)``.
    sr : int
        Positive sample rate.
    fade_in, fade_out : float, default 0.0
        Nonnegative finite seconds, rounded to samples. Zero samples means no
        fade. Each fade must fit the clip, and the two fades may not overlap.
    in_exponent, out_exponent : float, optional
        Positive powers applied to ``linspace(0, 1, n_in)`` and
        ``linspace(1, 0, n_out)``, respectively. Defaults retain the 0.6 curves.

    Returns
    -------
    numpy.ndarray
        New float64 array with the same ndim; the input is never modified.

    Raises
    ------
    ValueError
        Invalid ndim/rate, durations, exponents, or fades that exceed the clip.
        Bounds errors include the array's interpreted layout.
    """
    array = _audio_array(audio)
    sr = _sample_rate(sr)
    n_samples = array.shape[-1]
    hint = f" ({_layout(array)})"
    counts = []
    for name, duration in (("fade_in", fade_in), ("fade_out", fade_out)):
        count = _time_samples(duration, sr, name, array)
        if duration < 0:
            raise ValueError(f"{name} {duration:g} s ({count} samples) must be >= 0{hint}")
        if count > n_samples:
            raise ValueError(
                f"{name} {duration:g} s ({count} samples) is longer than the clip "
                f"{n_samples / sr:g} s ({n_samples} samples){hint}"
            )
        counts.append(count)
    for name, exponent in (("in_exponent", in_exponent), ("out_exponent", out_exponent)):
        if not exponent > 0:
            raise ValueError(f"{name} must be > 0; got {exponent!r}{hint}")
    n_in, n_out = counts
    if n_in + n_out > n_samples:
        raise ValueError(
            f"fade_in {fade_in:g} s + fade_out {fade_out:g} s = "
            f"{fade_in + fade_out:g} s exceeds the clip ({n_samples / sr:g} s){hint}"
        )
    result = array.copy()
    if n_in:
        result[..., :n_in] *= np.linspace(0, 1, n_in) ** in_exponent
    if n_out:
        result[..., -n_out:] *= np.linspace(1, 0, n_out) ** out_exponent
    return result
