import xml.etree.ElementTree as ET
import pytest
from tests import _cases as cases
from tests._support import normalize_xml


def test_pitch_helpers(golden):
    golden("mei/pitch", cases.mei_pitch_case())


@pytest.mark.parametrize("mode", ["sequence", "chord"])
def test_mei_string(golden, mode):
    golden(f"mei/{mode}", cases.mei_string_case(mode))


@pytest.mark.parametrize("label_mode", ["fing", "dir"])
def test_temporal_mei_string(golden, label_mode):
    golden(f"mei/temporal/{label_mode}", cases.mei_temporal_case(label_mode))


@pytest.mark.parametrize("temporal", [False, True], ids=["process_and_visualize", "process_temporal_chords"])
def test_process_and_real_verovio_render(tmp_path, golden, temporal):
    golden(f"mei/process/{temporal}", cases.mei_process_case(tmp_path, temporal))


def test_xml_normalization_preserves_references_and_musical_time():
    source = '<mei><note xml:id="random1"/><fing startid="#random1" tstamp="1"/><date>2026-10-10</date></mei>'
    renamed = source.replace("random1", "random2").replace("2026-10-10", "2026-10-11")
    assert normalize_xml(source) == normalize_xml(renamed)
    assert normalize_xml(source) != normalize_xml(source.replace('tstamp="1"', 'tstamp="2"'))
    assert ET.fromstring(normalize_xml(source)).find("fing").get("startid").startswith("#id")
