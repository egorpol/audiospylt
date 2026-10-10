import os
from pathlib import Path

# Set before importing plotting/audio dependencies.
os.environ["MPLBACKEND"] = "Agg"
os.environ["PLOTLY_RENDERER"] = "json"

import pytest
from tests._support import assert_snapshot, offline_display, read_goldens


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(__file__).parent / ".pytest-tmp")
    config.addinivalue_line("markers", "slow: computationally expensive local tests")
    config.addinivalue_line("markers", "network: requires network; always skipped here")


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(pytest.mark.skip(reason="This suite runs without network access"))


@pytest.fixture(autouse=True)
def inert_environment(monkeypatch, tmp_path):
    # Convenience wrappers without output_dir are confined to this test's temp directory.
    monkeypatch.chdir(tmp_path)
    with offline_display() as shown:
        yield shown


@pytest.fixture(scope="session")
def goldens():
    return read_goldens()


@pytest.fixture
def golden(goldens):
    def compare(key, actual, **tolerances):
        assert_snapshot(actual, goldens[key], **tolerances)
    return compare
