import pandas as pd
import pytest
from audiospylt import df_transforms as transforms
from tests import _cases as cases


pytestmark = pytest.mark.filterwarnings(
    "ignore::audiospylt._compat.AudiospyltDeprecationWarning"
)


@pytest.mark.parametrize("name", cases.TRANSFORM_CASES)
def test_spectral_transforms(golden, name):
    golden(f"df/{name}", cases.transform_case(name))


def test_transforms_do_not_mutate_inputs():
    frame = cases.events()
    original = frame.copy(deep=True)
    transforms.process_spectral_dataframe(frame, stretch_factor=2, pitch_cents=1200)
    transforms.add_time_cues(frame, [0.03, 0.09])
    pd.testing.assert_frame_equal(frame, original)


def test_unknown_structure_is_preserved():
    frame = pd.DataFrame({"label": ["a", "b"]})
    pd.testing.assert_frame_equal(transforms.process_spectral_dataframe(frame), frame)
