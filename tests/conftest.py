"""
Shared pytest fixtures. Import factories and expose as fixtures.
"""
import pytest
from tests.factories import (
    make_normal_market, make_hostile_market, make_fragile_market,
    make_edge_case_thresholds, make_sector_rotation, make_breadth_divergence,
    make_flat_choppy_market, make_single_sector_pump, make_missing_data,
    make_momentum_crash, make_all_sectors_identical,
)


@pytest.fixture
def normal_market():
    return make_normal_market()


@pytest.fixture
def hostile_market():
    return make_hostile_market()


@pytest.fixture
def fragile_market():
    return make_fragile_market()


@pytest.fixture
def edge_thresholds():
    return make_edge_case_thresholds()


@pytest.fixture
def sector_rotation():
    return make_sector_rotation()


@pytest.fixture
def breadth_divergence():
    return make_breadth_divergence()


@pytest.fixture
def flat_market():
    return make_flat_choppy_market()


@pytest.fixture
def single_sector_pump():
    return make_single_sector_pump()


@pytest.fixture
def momentum_crash():
    return make_momentum_crash()


@pytest.fixture
def missing_data():
    return make_missing_data()


@pytest.fixture
def identical_sectors():
    return make_all_sectors_identical()


@pytest.fixture
def settings():
    """Load settings.yaml and return as dict."""
    import yaml
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def universe():
    """Load universe.yaml and return as dict."""
    import yaml
    with open("config/universe.yaml") as f:
        return yaml.safe_load(f)
