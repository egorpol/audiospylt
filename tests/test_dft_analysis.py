import numpy as np
import plotly.graph_objects as go
import pytest
from audiospylt import dft_analysis as dft
from tests import _cases as cases


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def test_fft_window_filter_and_analysis(golden):
    golden("dft", cases.dft_case())


@pytest.mark.parametrize("frequency_axis,amplitude_axis", [
    ("linear", "linear"), ("log", "log"), ("mel", "linear"), ("mixed", "mixed")])
def test_plot_spectrum_returns_none(inert_environment, frequency_axis, amplitude_axis):
    freqs, spec = dft.compute_fft(cases.tone(), 256)
    peaks = dft.filter_peaks(spec, freqs, 0.05, 0.6, 20, 60)
    result = dft.plot_spectrum(freqs, spec, peaks, 0.05, 0.6, 20, 60,
        freq_axis_mode=frequency_axis, amp_axis_mode=amplitude_axis, auto_plot_range=True,
        plot_partials=True, partials_hz=[16., 32., 48.], plot_width=600, plot_height=300)
    assert result is None
    fig = inert_environment[-1]
    assert isinstance(fig, go.Figure) and len(fig.data) >= 6


def test_windowed_waveform(goldens):
    fig = dft.plot_windowed_waveform(cases.tone(), 256, window_type="hann", show=True)
    assert isinstance(fig, go.Figure) and len(fig.data) == 2
    np.testing.assert_allclose(fig.data[0].y, goldens["dft"]["hann"], rtol=1e-5, atol=1e-8)
