import numpy as np
import pytest
from audiospylt import plot_wave
from tests import _cases as cases


@pytest.mark.parametrize("name", cases.WAVE_CASES)
def test_synthesized_phases_intervals_and_fades(golden, name):
    y = cases.wave_case(name)
    golden(f"wave/{name}", y)
    assert y.ndim == 1 and np.isfinite(y).all()


@pytest.mark.parametrize("phase", ["chirp", "reset", "global"])
def test_overlap_crossfade(golden, phase):
    actual = cases.overlap_case(phase)
    golden(f"wave/overlap/{phase}", actual)
    frame = actual["events"]
    assert frame.loc[0, "time_stop"] > frame.loc[1, "time_start"]


def test_per_row_fades_override_defaults(golden):
    golden("wave/row_fades", cases.row_fades_case())


def test_synthesis_with_spectrogram():
    y = plot_wave.plot_waves(cases.events(), k=0.002, show_fft=True,
                             spectrogram_kwargs={"n_fft": 16, "overlap": 0.5})
    assert len(y) == 61
