"""Offline audio loading, explicit channel layout, and pure sample-axis edits."""

import builtins
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from types import MethodType
import wave

import numpy as np
import pytest
import requests
import soundfile as sf
import soxr

from audiospylt.io.audio import fade, load_audio, to_mono, trim, write_audio


SAMPLES = Path(__file__).resolve().parents[1] / "samples"
AUDIO_MODULE = Path(__file__).resolve().parents[1] / "audiospylt" / "io" / "audio.py"
# Capture before the autouse offline fixture patches the class method. Only our
# fake-adapter session restores it; all socket guards remain in force.
_SESSION_REQUEST = requests.Session.request
REAL_AUDIO = {
    "bayle.wav": (44100, 2, 339099, (-6.131206, -6.189307)),
    "cello_arco_d_sul_c_01.wav": (44100, 2, 374850, (-0.067321, -0.067321)),
    "de_natura_son.wav": (44100, 2, 441000, (-528.186907, -808.241563)),
    "o-lang.wav": (44100, 2, 658532, (-0.114746, -0.117279)),
    "saariaho_flute.wav": (44100, 2, 325417, (-350.487129, -350.487163)),
}


def _wav_bytes(data=None, rate=8000):
    if data is None:
        data = np.array([[0.25, -0.5], [0.5, 0.125], [-0.25, 0.75], [0, -0.125]])
    stream = BytesIO()
    sf.write(stream, data, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def _kept_name(url, stem, extension=".wav"):
    return f"{stem}-{sha256(url.encode('utf-8')).hexdigest()[:8]}{extension}"


class FakeAdapter(requests.adapters.BaseAdapter):
    """Use real requests preparation/status handling without touching sockets."""

    def __init__(self):
        self.replies = {}
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append((request.url, kwargs))
        reply = self.replies[request.url]
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        response = requests.Response()
        response.status_code = status
        response._content = body
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


@pytest.fixture
def transport(monkeypatch):
    with requests.Session() as session:
        monkeypatch.setattr(session, "request", MethodType(_SESSION_REQUEST, session))
        adapter = FakeAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        yield session, adapter


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_synthetic_formats_layout_exact_data_and_channel_mean(channels, subtype, tmp_path):
    # Dyadic values are represented exactly by FLOAT as well as float64.
    left = np.linspace(-0.75, 0.75, 17)
    right = 0.125 - (np.arange(17) % 5) / 8
    written = left if channels == 1 else np.column_stack([left, right])
    path = tmp_path / f"{channels}-{subtype}.wav"
    sf.write(path, written, 44100, subtype=subtype)
    reference, _ = sf.read(path, dtype="float64", always_2d=True)
    expected = reference[:, 0] if channels == 1 else reference.T
    result, rate = load_audio(path, mono=False)
    assert result.shape == ((17,) if channels == 1 else (2, 17))
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert type(rate) is int and rate == 44100
    np.testing.assert_array_equal(result, expected)
    if subtype == "FLOAT":
        np.testing.assert_array_equal(result.view(np.uint64), written.T.copy().view(np.uint64))
    mono, mono_rate = load_audio(path)
    assert mono.shape == (17,)
    assert mono.dtype == np.float64 and mono.flags.c_contiguous
    assert mono_rate == rate
    np.testing.assert_array_equal(mono, reference.mean(axis=1))


@pytest.mark.parametrize("source_kind", ["str", "path", "file", "buffer", "read_only"])
def test_local_paths_and_binary_streams(source_kind, tmp_path):
    body = _wav_bytes()
    path = tmp_path / "source.wav"
    path.write_bytes(body)
    if source_kind in {"str", "path"}:
        source = str(path) if source_kind == "str" else path
        actual, _ = load_audio(source, mono=False)
    else:
        class ReadOnly:
            def read(self):
                return body

        source = (path.open("rb") if source_kind == "file" else
                  BytesIO(body) if source_kind == "buffer" else ReadOnly())
        try:
            actual, _ = load_audio(source, mono=False)
            if hasattr(source, "closed"):
                assert not source.closed
        finally:
            if hasattr(source, "close"):
                source.close()
    reference, _ = sf.read(BytesIO(body), dtype="float64", always_2d=True)
    np.testing.assert_array_equal(actual, reference.T)


def test_binary_stream_current_position():
    stream = BytesIO(b"skip" + _wav_bytes())
    stream.read(4)
    result, rate = load_audio(stream, mono=False)
    assert result.shape == (2, 4) and rate == 8000
    assert not stream.closed


def test_multichannel_file_never_guesses_orientation(tmp_path):
    written = np.arange(10, dtype=np.float64).reshape(2, 5) / 16
    path = tmp_path / "five-channels-two-frames.wav"
    sf.write(path, written, 8000, subtype="FLOAT")
    result, _ = load_audio(path, mono=False)
    assert result.shape == (5, 2)
    np.testing.assert_array_equal(result, written.T)
    np.testing.assert_array_equal(load_audio(path)[0], written.mean(axis=1))


@pytest.mark.parametrize("source", ["absent.wav", "http:absent.wav", "ftp://example.test/absent.wav"])
def test_missing_non_url_names_source(source):
    with pytest.raises(FileNotFoundError) as caught:
        load_audio(source)
    assert source in str(caught.value)


@pytest.mark.parametrize("source_kind", ["path", "buffer"])
def test_decode_failure_names_source_and_chains_error(source_kind, tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not audio")
    source = path if source_kind == "path" else BytesIO(b"not audio")
    with pytest.raises(ValueError) as caught:
        load_audio(source)
    label = str(path) if source_kind == "path" else "<buffer>"
    assert label in str(caught.value)
    assert isinstance(caught.value.__cause__, sf.LibsndfileError)


def test_non_url_ignores_download_options(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(_wav_bytes())
    result, rate = load_audio(source, download_dir=tmp_path / "missing", refresh=True)
    assert result.shape == (4,) and rate == 8000
    assert set(tmp_path.iterdir()) == {source}


def test_module_and_local_loading_do_not_import_requests_librosa_or_soxr(monkeypatch, tmp_path):
    # Load the file directly to isolate v1 imports from the package initializer.
    # sr equal to the file rate: soxr is imported only when resampling happens.
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"requests", "librosa", "soxr"}:
            raise AssertionError(f"unexpected eager import: {name}")
        return original_import(name, *args, **kwargs)

    path = tmp_path / "local.wav"
    path.write_bytes(_wav_bytes())
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = spec_from_file_location("audio_import_check", AUDIO_MODULE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    result, rate = module.load_audio(path, sr=8000)
    assert result.shape == (4,) and rate == 8000


@pytest.fixture
def soxr_calls(monkeypatch):
    calls = []
    resample = soxr.resample

    def recording_resample(audio, *args, **kwargs):
        calls.append((audio.shape, args, kwargs))
        return resample(audio, *args, **kwargs)

    monkeypatch.setattr(soxr, "resample", recording_resample)
    return calls


def test_stereo_resampling_preserves_channels_and_frequency(soxr_calls, tmp_path):
    times = np.arange(44100) / 44100
    written = np.column_stack([np.sin(2 * np.pi * 440 * times),
                               0.5 * np.sin(2 * np.pi * 880 * times)])
    path = tmp_path / "sines.wav"
    sf.write(path, written, 44100, subtype="FLOAT")
    result, rate = load_audio(path, mono=False, sr=22050)
    assert result.shape == (2, 22050)
    assert result.dtype == np.float64 and result.flags.c_contiguous
    assert type(rate) is int and rate == 22050
    assert soxr_calls == [((44100, 2), (44100, 22050), {"quality": "HQ"})]
    for channel, frequency in zip(result, [440, 880]):
        peak = np.argmax(np.abs(np.fft.rfft(channel)))
        assert np.fft.rfftfreq(channel.size, 1 / rate)[peak] == frequency


def test_mono_resampling_passes_samples_as_one_axis(soxr_calls):
    result, rate = load_audio(BytesIO(_wav_bytes()), sr=16000)
    assert result.shape == (8,) and rate == 16000
    assert soxr_calls == [((4,), (8000, 16000), {"quality": "HQ"})]


def test_equal_rate_does_not_resample(soxr_calls, tmp_path):
    path = tmp_path / "same-rate.wav"
    path.write_bytes(_wav_bytes())
    result, rate = load_audio(path, sr=np.int64(8000))
    assert result.shape == (4,)
    assert type(rate) is int and rate == 8000
    assert soxr_calls == []


def test_short_soxr_output_is_zero_padded_to_the_exact_ceil(monkeypatch):
    # 4 samples at 8000 Hz -> 20000 Hz: ceil(4·20000/8000) = 10 samples.
    monkeypatch.setattr(soxr, "resample", lambda audio, *args, **kwargs: np.ones(9))
    result, _ = load_audio(BytesIO(_wav_bytes()), sr=20000)
    np.testing.assert_array_equal(result, [1] * 9 + [0])


def test_long_soxr_output_raises_instead_of_trimming(monkeypatch):
    monkeypatch.setattr(soxr, "resample", lambda audio, *args, **kwargs: np.ones(11))
    with pytest.raises(RuntimeError, match=r"soxr returned 11 samples, more than ceil\(n·sr/rate\) = 10"):
        load_audio(BytesIO(_wav_bytes()), sr=20000)


@pytest.mark.parametrize("rates", [(44100, 48000), (48000, 44100), (44100, 22050), (22050, 44100),
                                   (8000, 44100), (44100, 8000), (22050, 48000), (44100, 44099)])
def test_resampling_matches_librosa_soxr_hq_except_its_float_ceil(rates):
    librosa = pytest.importorskip("librosa")
    rate, target = rates
    noise = np.random.default_rng(rate + target).uniform(-0.5, 0.5, (44101, 2))
    for n in (1, 2, 3, 7, 101, 1001, 44101):
        for data in (noise[:n, 0], noise[:n]):
            stream = BytesIO()
            sf.write(stream, data, rate, format="WAV", subtype="DOUBLE")
            original, _ = load_audio(BytesIO(stream.getvalue()), mono=False)
            ours, _ = load_audio(BytesIO(stream.getvalue()), mono=False, sr=target)
            theirs = librosa.resample(original, orig_sr=rate, target_sr=target,
                                      res_type="soxr_hq", axis=-1)
            assert ours.shape[:-1] == theirs.shape[:-1] == original.shape[:-1]
            assert ours.shape[-1] == -(-n * target // rate), (n, data.ndim)
            common = min(ours.shape[-1], theirs.shape[-1])
            np.testing.assert_array_equal(ours[..., :common], theirs[..., :common])
            assert not ours[..., common:].any() and not theirs[..., common:].any()


def test_exact_ceil_is_one_sample_shorter_than_librosa_in_rare_cases():
    librosa = pytest.importorskip("librosa")
    stream = BytesIO()
    sf.write(stream, np.random.default_rng(0).uniform(-0.5, 0.5, 44100), 22050,
             format="WAV", subtype="DOUBLE")
    original, _ = load_audio(BytesIO(stream.getvalue()))
    ours, _ = load_audio(BytesIO(stream.getvalue()), sr=48000)
    theirs = librosa.resample(original, orig_sr=22050, target_sr=48000, res_type="soxr_hq", axis=-1)
    assert ours.shape == (96000,) and theirs.shape == (96001,)
    np.testing.assert_array_equal(ours, theirs[:96000])
    assert theirs[-1] == 0


@pytest.mark.parametrize("rate", [0, -1, 44100.0, True, False, np.bool_(True),
                                 "44100", np.nan, np.inf, 1 + 0j])
def test_bad_target_rates_raise_value_error(rate):
    with pytest.raises(ValueError, match="sr must be a positive integer"):
        load_audio(BytesIO(_wav_bytes()), sr=rate)


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_url_memory_default_writes_nothing(scheme, transport, tmp_path):
    session, adapter = transport
    url = f"{scheme}://example.test/music.wav"
    adapter.replies[url] = (200, _wav_bytes())
    marker = tmp_path / "keep.txt"
    marker.write_bytes(b"unchanged")
    before = list(tmp_path.rglob("*"))
    result, rate = load_audio(url, mono=False, session=session, timeout=12.5)
    assert result.shape == (2, 4) and rate == 8000
    assert Path.cwd() == tmp_path
    assert list(tmp_path.rglob("*")) == before
    assert marker.read_bytes() == b"unchanged"
    assert len(adapter.calls) == 1
    assert adapter.calls[0][1]["timeout"] == 12.5


def test_plain_requests_get_is_used_without_session(monkeypatch, transport):
    session, adapter = transport
    url = "https://example.test/default.wav"
    adapter.replies[url] = (200, _wav_bytes())
    calls = []

    def get(source, **kwargs):
        calls.append((source, kwargs))
        return session.get(source, **kwargs)

    monkeypatch.setattr(requests, "get", get)
    load_audio(url)
    assert calls == [(url, {"timeout": 60.0})]
    assert len(adapter.calls) == 1


def test_url_kept_name_reuse_and_refresh(transport, tmp_path):
    session, adapter = transport
    url = "https://example.test/music/caf%C3%A9%20take%3F.wav?version=2"
    body = _wav_bytes()
    adapter.replies[url] = (200, body)
    directory = tmp_path / "downloads"
    directory.mkdir()
    kept = directory / _kept_name(url, "caf__take_")
    first, rate = load_audio(url, mono=False, download_dir=directory, session=session)
    assert list(directory.iterdir()) == [kept]
    assert kept.read_bytes() == body
    assert len(adapter.calls) == 1
    adapter.calls.clear()
    changed_body = _wav_bytes(np.full((4, 2), -0.25))
    adapter.replies[url] = (200, changed_body)
    reused, reused_rate = load_audio(url, mono=False, download_dir=directory, session=session)
    assert adapter.calls == []
    assert reused_rate == rate
    np.testing.assert_array_equal(reused, first)
    refreshed, refreshed_rate = load_audio(
        url, mono=False, download_dir=directory, refresh=True, session=session,
    )
    assert len(adapter.calls) == 1 and refreshed_rate == rate
    np.testing.assert_array_equal(refreshed, np.full((2, 4), -0.25))
    assert kept.read_bytes() == changed_body
    assert list(directory.iterdir()) == [kept]


@pytest.mark.parametrize("suffix,stem,extension", [
    ("/", "audio", ""),
    ("/no-extension?format=wav", "no-extension", ""),
    ("/a%2Fb%3Ac.wav", "a_b_c", ".wav"),
    ("/sound%7E.wav", "sound_", ".wav"),
])
def test_url_filename_sanitization_and_exact_url_hash(suffix, stem, extension, transport, tmp_path):
    session, adapter = transport
    url = "https://example.test" + suffix
    prepared_url = requests.Request("GET", url).prepare().url
    adapter.replies[prepared_url] = (200, _wav_bytes())
    load_audio(url, download_dir=tmp_path, session=session)
    assert [p.name for p in tmp_path.iterdir()] == [_kept_name(url, stem, extension)]


def test_same_basename_urls_keep_distinct_files(transport, tmp_path):
    session, adapter = transport
    urls = ["https://example.test/a/music.wav?take=1", "https://example.test/b/music.wav?take=2"]
    for index, url in enumerate(urls):
        body = _wav_bytes(np.full((4, 2), index / 4))
        adapter.replies[url] = (200, body)
        load_audio(url, download_dir=tmp_path, session=session)
        assert (tmp_path / _kept_name(url, "music")).read_bytes() == body
    assert len(list(tmp_path.iterdir())) == 2
    assert len(adapter.calls) == 2


@pytest.mark.parametrize("failure", ["timeout", "connection", "404", "html"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_url_leaves_no_new_or_temporary_files(failure, existing, transport, tmp_path):
    session, adapter = transport
    url = "https://example.test/error.wav"
    kept = tmp_path / _kept_name(url, "error")
    if existing:
        kept.write_bytes(_wav_bytes())
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    if failure in {"timeout", "connection"}:
        error = (requests.Timeout if failure == "timeout" else requests.ConnectionError)("failed")
        adapter.replies[url] = error
        error_type = type(error)
    else:
        adapter.replies[url] = (404 if failure == "404" else 200, b"<html>error</html>")
        error_type = requests.HTTPError if failure == "404" else ValueError
    with pytest.raises(error_type) as caught:
        load_audio(url, download_dir=tmp_path, refresh=existing, session=session)
    if failure in {"timeout", "connection"}:
        assert caught.value is error
    else:
        assert url in str(caught.value)
        if failure == "html":
            assert isinstance(caught.value.__cause__, sf.LibsndfileError)
    assert len(adapter.calls) == 1
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("directory_kind", ["missing", "file"])
def test_invalid_download_directory_before_request(directory_kind, transport, tmp_path):
    session, adapter = transport
    directory = tmp_path / directory_kind
    if directory_kind == "file":
        directory.write_bytes(b"keep")
    exception = FileNotFoundError if directory_kind == "missing" else NotADirectoryError
    with pytest.raises(exception, match=directory_kind):
        load_audio("https://example.test/music.wav", download_dir=directory, session=session)
    assert adapter.calls == []
    if directory_kind == "missing":
        assert not directory.exists()
    else:
        assert directory.read_bytes() == b"keep"


def test_refresh_without_download_directory_before_request(transport):
    session, adapter = transport
    with pytest.raises(ValueError, match="refresh=True requires download_dir"):
        load_audio("https://example.test/music.wav", refresh=True, session=session)
    assert adapter.calls == []


@pytest.mark.parametrize("fail_replace", [False, True])
def test_atomic_kept_copy_and_cleanup_on_replace_failure(monkeypatch, fail_replace, transport, tmp_path):
    import os

    session, adapter = transport
    url = "https://example.test/atomic.wav"
    body = _wav_bytes()
    adapter.replies[url] = (200, body)
    kept = tmp_path / _kept_name(url, "atomic")
    kept.write_bytes(b"previous version")
    replace = os.replace
    replacements = []

    def checked_replace(source, destination):
        assert Path(source).parent == tmp_path
        assert Path(source).read_bytes() == body
        assert destination == kept
        assert kept.read_bytes() == b"previous version"
        replacements.append(destination)
        if fail_replace:
            raise OSError("replace failed")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", checked_replace)
    if fail_replace:
        with pytest.raises(OSError, match="replace failed"):
            load_audio(url, download_dir=tmp_path, refresh=True, session=session)
    else:
        load_audio(url, download_dir=tmp_path, refresh=True, session=session)
    assert replacements == [kept]
    assert list(tmp_path.iterdir()) == [kept]
    assert kept.read_bytes() == (b"previous version" if fail_replace else body)


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_to_mono_copies_or_means_without_modifying_input(ndim, dtype):
    data = np.arange(20, dtype=dtype)
    if ndim == 2:
        data = data.reshape(2, 10)
    original = data.copy()
    result = to_mono(data)
    expected = data.astype(np.float64) if ndim == 1 else data.astype(np.float64).mean(axis=0)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(data, original)
    assert result.dtype == np.float64 and result.ndim == 1
    assert not np.shares_memory(result, data)


@pytest.mark.parametrize("function", [to_mono, trim, fade])
@pytest.mark.parametrize("shape", [(), (2, 3, 4)])
def test_wrong_ndim_raises(function, shape):
    with pytest.raises(ValueError, match="1-D.*2-D.*channels, samples"):
        function(np.zeros(shape), *([] if function is to_mono else [1000]))


@pytest.mark.parametrize("stereo", [False, True])
def test_trim_dsp6_sample_axis_and_immutable_input(stereo):
    data = np.arange(1000)
    if stereo:
        data = np.vstack([data, -data])
    original = data.copy()
    result = trim(data, 1000, 0.1, 0.3)
    assert result.shape == ((2, 200) if stereo else (200,))
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, data[..., 100:300])
    np.testing.assert_array_equal(data, original)
    assert not np.shares_memory(result, data)


def test_trim_defaults_rounding_and_noncontiguous_input():
    data = np.arange(40, dtype=np.float32).reshape(2, 20)[:, ::2]
    whole = trim(data, 10)
    np.testing.assert_array_equal(whole, data)
    assert not np.shares_memory(whole, data)
    np.testing.assert_array_equal(trim(data, 10, 0.15, 0.65), data[:, 2:6])
    np.testing.assert_array_equal(trim(data, 10, 0.2), data[:, 2:])


@pytest.mark.parametrize("start,stop,fragments", [
    (-0.1, None, ["start -0.1 s (-100 samples)", ">= 0"]),
    (-0.0001, None, ["start -0.0001 s (0 samples)", ">= 0"]),
    (0, 1.1, ["stop 1.1 s (1100 samples)", "1 s (1000 samples)"]),
    (0.3, 0.3, ["stop 0.3 s (300 samples)", "start 0.3 s (300 samples)", "empty clip"]),
    (0.3, 0.2, ["stop 0.2 s (200 samples)", "start 0.3 s (300 samples)", "empty clip"]),
    (1.0, None, ["stop 1 s (1000 samples)", "start 1 s (1000 samples)", "empty clip"]),
    (0, 0.0001, ["stop 0.0001 s (0 samples)", "start 0 s (0 samples)", "empty clip"]),
])
def test_trim_bounds_report_seconds_samples_and_layout(start, stop, fragments):
    with pytest.raises(ValueError) as caught:
        trim(np.zeros((2, 1000)), 1000, start, stop)
    for fragment in fragments + ["audio shape (2, 1000) is read as (channels, samples)"]:
        assert fragment in str(caught.value)


@pytest.mark.parametrize("function", [trim, fade])
def test_samples_channels_shape_gets_layout_hint(function):
    data = np.zeros((339099, 2))
    with pytest.raises(ValueError) as caught:
        if function is trim:
            trim(data, 44100, 0.4, 1.6)
        else:
            fade(data, 44100, 0.2, 0.3)
    assert "audio shape (339099, 2) is read as (channels, samples)" in str(caught.value)


def test_2d_orientation_is_never_rejected_or_inferred():
    data = np.arange(14).reshape(7, 2)
    np.testing.assert_array_equal(to_mono(data), data.mean(axis=0))
    np.testing.assert_array_equal(trim(data, 1, 0, 1), data[:, :1])
    expected = data.astype(np.float64)
    expected[:, 0] = 0
    np.testing.assert_array_equal(fade(data, 1, 1, 1), expected)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("exponents", [None, (1.0, 2.0)])
def test_fades_match_legacy_curves_on_every_channel(stereo, exponents):
    data = np.arange(1000, dtype=np.int16)
    if stereo:
        data = np.vstack([data, -data])
    original = data.copy()
    powers = (0.8, 1.5) if exponents is None else exponents
    options = {} if exponents is None else dict(zip(["in_exponent", "out_exponent"], powers))
    expected = data.astype(np.float64)
    expected[..., :100] *= np.linspace(0, 1, 100) ** powers[0]
    expected[..., -300:] *= np.linspace(1, 0, 300) ** powers[1]
    result = fade(data, 1000, 0.1, 0.3, **options)
    assert result.dtype == np.float64 and result.shape == data.shape
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(data, original)
    assert not np.shares_memory(result, data)


@pytest.mark.parametrize("fade_in,fade_out", [(0, 0), (0.0004, 0.0004), (0.1, 0), (0, 0.1)])
def test_zero_sample_fades_and_one_sided_fades(fade_in, fade_out):
    data = np.ones((2, 1000), dtype=np.float32)
    expected = np.ones((2, 1000), dtype=np.float64)
    if round(fade_in * 1000):
        expected[:, :100] *= np.linspace(0, 1, 100) ** 0.8
    if round(fade_out * 1000):
        expected[:, -100:] *= np.linspace(1, 0, 100) ** 1.5
    result = fade(data, 1000, fade_in, fade_out)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(data, np.ones_like(data))
    assert result.dtype == np.float64 and not np.shares_memory(result, data)


def test_fade_rounding_and_exactly_adjacent_fades():
    data = np.ones(8)
    expected = data.copy()
    expected[:2] *= np.linspace(0, 1, 2) ** 0.8
    expected[-6:] *= np.linspace(1, 0, 6) ** 1.5
    np.testing.assert_array_equal(fade(data, 10, 0.25, 0.55), expected)


@pytest.mark.parametrize("options,fragment", [
    ({"fade_in": -0.1}, "fade_in -0.1 s (-4410 samples) must be >= 0"),
    ({"fade_out": -0.000001}, "fade_out -1e-06 s (0 samples) must be >= 0"),
    ({"in_exponent": 0}, "in_exponent must be > 0"),
    ({"out_exponent": -1}, "out_exponent must be > 0"),
    ({"in_exponent": np.nan}, "in_exponent must be > 0"),
    ({"fade_in": 0.5}, "fade_in 0.5 s (22050 samples) is longer than the clip 0.4 s (17640 samples)"),
    ({"fade_out": 0.5}, "fade_out 0.5 s (22050 samples) is longer than the clip 0.4 s (17640 samples)"),
    ({"fade_in": 0.2, "fade_out": 0.3}, "fade_in 0.2 s + fade_out 0.3 s = 0.5 s exceeds the clip (0.4 s)"),
])
def test_fade_errors_and_length_messages(options, fragment):
    with pytest.raises(ValueError) as caught:
        fade(np.ones((2, 17640)), 44100, **options)
    assert fragment in str(caught.value)
    assert "audio shape (2, 17640) is read as (channels, samples)" in str(caught.value)


@pytest.mark.parametrize("function,option", [(trim, "start"), (trim, "stop"),
                                            (fade, "fade_in"), (fade, "fade_out")])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_times_raise_value_error(function, option, value):
    with pytest.raises(ValueError, match=f"{option} must be finite seconds"):
        function(np.ones(10), 10, **{option: value})


def _read_pcm(path):
    """Decode the samples' 16/24-bit integer PCM independently of soundfile."""
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        frames = stream.getnframes()
        rate = stream.getframerate()
        width = stream.getsampwidth()
        payload = stream.readframes(frames)
    if width == 3:
        octets = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        integers = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        integers = (integers ^ 0x800000) - 0x800000
    else:
        assert width == 2
        integers = np.frombuffer(payload, dtype="<i2")
    return integers.reshape(frames, channels).T / 2 ** (8 * width - 1), rate


def test_every_sample_wav_is_covered():
    assert {path.name for path in SAMPLES.glob("*.wav")} == set(REAL_AUDIO)


@pytest.mark.parametrize("name", REAL_AUDIO)
def test_real_audio_against_independent_pcm_and_fixed_channel_sums(name):
    rate, channels, frames, sums = REAL_AUDIO[name]
    reference, reference_rate = _read_pcm(SAMPLES / name)
    assert reference_rate == rate and reference.shape == (channels, frames)
    assert reference.sum(axis=1) == pytest.approx(sums, abs=1e-6, rel=0)
    result, result_rate = load_audio(SAMPLES / name, mono=False, sr=None)
    assert result.shape == (2, frames)
    assert type(result_rate) is int and result_rate == rate
    assert result.dtype == np.float64 and result.flags.c_contiguous
    np.testing.assert_array_equal(result, reference)
    assert result.sum(axis=1) == pytest.approx(sums, abs=1e-6, rel=0)
    mono, mono_rate = load_audio(SAMPLES / name)
    assert mono_rate == rate
    np.testing.assert_array_equal(mono, reference.mean(axis=0))
    cropped = trim(result, result_rate, 0.4, 1.6)
    assert cropped.shape == (2, 52920)
    np.testing.assert_array_equal(cropped, reference[:, 17640:70560])


# ---------------------------------------------------------------- write_audio


def _sine(amplitude=0.5, n=4410, rate=44100):
    return amplitude * np.sin(2 * np.pi * 441.3 * np.arange(n) / rate)


@pytest.mark.parametrize("subtype, width", [("PCM_16", 16), ("PCM_24", 24)])
def test_write_audio_keeps_level_in_pcm(tmp_path, subtype, width):
    audio = _sine()
    path = tmp_path / "x.wav"
    assert write_audio(path, audio, 44100, subtype=subtype) == path
    decoded, rate = _read_pcm(path)
    assert rate == 44100 and decoded.shape == (1, audio.size)
    assert np.abs(decoded[0] - audio).max() <= 2.0 ** -(width - 1)
    assert write_audio(str(path), audio, 44100) == Path(str(path))


def test_write_audio_stereo_layout_and_float_overrange(tmp_path):
    stereo = np.stack([_sine(0.5), _sine(1.5)])
    with pytest.raises(ValueError, match=r"\+3\.52 dBFS"):
        write_audio(tmp_path / "x.wav", stereo, 44100)
    write_audio(tmp_path / "x.wav", stereo, 44100, subtype="FLOAT")
    decoded, rate = sf.read(tmp_path / "x.wav", dtype="float64", always_2d=True)
    assert rate == 44100
    np.testing.assert_allclose(decoded.T, stereo, rtol=0, atol=1e-7)
    assert np.abs(decoded[:, 1]).max() > 1.4


def test_write_audio_full_scale_is_allowed_and_clipping_raises(tmp_path):
    write_audio(tmp_path / "x.wav", np.array([1.0, -1.0, 0.0]), 8000, subtype="PCM_16")
    path = tmp_path / "y.wav"
    write_audio(path, _sine(0.25), 44100)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="clip.*normalize=-1.0"):
        write_audio(path, _sine(1.0001), 44100)
    with pytest.raises(ValueError):
        write_audio(path, _sine(0.25), 44100, subtype="NOT_A_SUBTYPE")
    assert path.read_bytes() == before


def test_write_audio_normalize_sets_the_peak(tmp_path):
    audio = _sine(6.3)
    original = audio.copy()
    write_audio(tmp_path / "x.wav", audio, 44100, subtype="FLOAT", normalize=-1.0)
    decoded, _ = sf.read(tmp_path / "x.wav", dtype="float64")
    assert np.abs(decoded).max() == pytest.approx(10 ** (-1 / 20), abs=1e-7)
    np.testing.assert_allclose(decoded / np.abs(decoded).max(), audio / np.abs(audio).max(), atol=1e-6)
    np.testing.assert_array_equal(audio, original)
    write_audio(tmp_path / "q.wav", _sine(0.1), 44100, normalize=0.0)
    decoded, _ = _read_pcm(tmp_path / "q.wav")
    assert np.abs(decoded).max() == pytest.approx(1.0, abs=2.0 ** -22)


def test_write_audio_normalizes_a_subnormal_peak(tmp_path):
    write_audio(tmp_path / "x.wav", np.array([1e-310, 0.0, -5e-311]), 8000, normalize=-1.0)
    decoded, _ = _read_pcm(tmp_path / "x.wav")
    assert np.all(np.isfinite(decoded))
    assert np.abs(decoded).max() == pytest.approx(10 ** (-1 / 20), abs=2.0 ** -22)


@pytest.mark.parametrize("audio, options, error, fragment", [
    (np.zeros(10), {"normalize": -1.0}, ValueError, "silence"),
    (np.full(10, 0.1), {"normalize": 1.0}, ValueError, "normalize"),
    (np.full(10, 0.1), {"normalize": np.nan}, ValueError, "normalize"),
    (np.full(10, 0.1), {"normalize": "loud"}, ValueError, "normalize"),
    (np.full(10, 0.1), {"normalize": False}, ValueError, "normalize"),
    (np.array([0.1, np.nan]), {}, ValueError, "finite"),
    (np.zeros(0), {}, ValueError, "no samples"),
    (np.zeros((2, 2, 2)), {}, ValueError, "1-D"),
    (np.full(10, 0.1), {"subtype": 24}, ValueError, "subtype"),
    (np.array([0, 2, -1]), {}, ValueError, "clip"),
    (np.array([0.1, np.inf]), {}, ValueError, "finite"),
])
def test_write_audio_errors(tmp_path, audio, options, error, fragment):
    with pytest.raises(error, match=fragment):
        write_audio(tmp_path / "x.wav", audio, 44100, **options)
    assert not (tmp_path / "x.wav").exists()


def test_write_audio_rate_extension_and_parent(tmp_path):
    for rate in (0, 44100.0, True):
        with pytest.raises(ValueError, match="sr"):
            write_audio(tmp_path / "x.wav", _sine(), rate)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_audio(tmp_path / "missing" / "x.wav", _sine(), 44100)
    with pytest.raises(ValueError, match="Cannot write"):
        write_audio(tmp_path / "x", _sine(), 44100)
    write_audio(tmp_path / "x.flac", _sine(), 44100, subtype="PCM_16")
    assert sf.info(tmp_path / "x.flac").format == "FLAC"
