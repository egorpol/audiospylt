import numpy as np
import pytest
import plotly.graph_objects as go
from audiospylt import multiplotter
from tests import _cases as cases


@pytest.mark.parametrize("three_d", [False, True], ids=["plot_combined", "plot_combined_3d"])
def test_event_plots_from_tsv(tmp_path, inert_environment, three_d):
    frame = cases.events()
    paths = [tmp_path / "events_a.tsv", tmp_path / "events_b.tsv"]
    for path in paths:
        frame.to_csv(path, sep="\t", index=False)
    function = multiplotter.plot_combined_3d if three_d else multiplotter.plot_combined
    result = function(files=[str(p) for p in paths], plot_width=600, plot_height=400)
    assert result is None
    fig = inert_environment[-1]
    assert isinstance(fig, go.Figure) and len(fig.data) == 6
    np.testing.assert_allclose(fig.data[0].x, [0, 0.06])
    np.testing.assert_allclose(fig.data[0].y, [0.8, 0.5] if three_d else [13, 21])
    if three_d:
        np.testing.assert_allclose(fig.data[0].z, [13, 21])


@pytest.mark.parametrize("agg", ["max", "sum", "mean"])
def test_equalizer_bars_from_tsv(tmp_path, inert_environment, agg):
    path = tmp_path / "peaks.tsv"
    cases.peaks().to_csv(path, sep="\t", index=False)
    result = multiplotter.plot_equalizer_bars(files=[str(path)], x_bin_size=200, agg=agg,
                                             plot_width=600, plot_height=400)
    assert result is None
    fig = inert_environment[-1]
    assert isinstance(fig, go.Figure) and len(fig.data) == 1
    np.testing.assert_allclose(fig.data[0].x, [100, 300])
    np.testing.assert_allclose(fig.data[0].y, [0.8, {"max": 0.4, "sum": 0.6, "mean": 0.3}[agg]])
