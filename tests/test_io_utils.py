from pathlib import Path
import pandas as pd
import pytest
from audiospylt import io_utils
from tests import _cases as cases


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


def test_interval_events(golden):
    golden("io/events", cases.io_events_case())
    assert list(io_utils.events_df_from_peaks_by_interval([]).columns) == [
        "freq_start", "freq_stop", "time_start", "time_stop", "amp_min", "amp_max"]


def test_tsv_save(tmp_path, golden):
    golden("io/tsv", cases.io_tsv_case(tmp_path))
    frame = cases.peaks()
    result = io_utils.save_df_tsv(frame, tmp_path / "with_index.tsv", index=True, verbose=True)
    assert result == str((tmp_path / "with_index.tsv").resolve())
    loaded = pd.read_csv(result, sep="\t", index_col=0)
    pd.testing.assert_frame_equal(loaded, frame)
    assert Path(result).is_file()
