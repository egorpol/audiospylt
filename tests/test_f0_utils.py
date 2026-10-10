import numpy as np
import pytest
import plotly.graph_objects as go
from audiospylt import f0_utils
from tests import _cases as cases


@pytest.mark.parametrize("name", cases.F0_CASES)
def test_fundamental_search(golden, name):
    result = cases.f0_case(name, plot=True)
    golden(f"f0/{name}", cases.f0_numeric(result))
    assert isinstance(result["fig"], go.Figure)
    assert len(result["top_candidates"]) == 3
    if name == "harmonic":
        assert result["best_f0_hz"] == 100
        np.testing.assert_array_equal(result["best_partial_numbers"], [1, 2, 3, 4])
    if name in {"trim_cents", "trim_fraction"}:
        assert result["best_rms_cents"] < 1e-8
        assert abs(result["best_cents_errors"][-1]) > 50


def test_summary(golden):
    golden("f0/summary", cases.f0_summary_case())


def test_peak_weights_match_explicit_weights():
    a = cases.f0_case("weighted")
    b = cases.f0_case("peaks_df")
    np.testing.assert_allclose(a["rms_cents_all"], b["rms_cents_all"], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("kwargs", [{"weights": [-1, 1]}, {"top_k": 0}, {"step": 0},
                                    {"robust_trim_frac": 1}, {"trim_outlier_cents": 0}])
def test_invalid_search_parameters(kwargs):
    with pytest.raises(ValueError):
        f0_utils.search_f0([100, 200], **kwargs)
