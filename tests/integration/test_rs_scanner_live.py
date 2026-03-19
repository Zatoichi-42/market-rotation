"""
Integration tests for RS scanner — live yfinance data.
"""
import pytest
import yaml
from engine.schemas import RSReading
from engine.rs_scanner import compute_rs_readings

pytestmark = pytest.mark.integration

SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


@pytest.fixture(scope="module")
def live_data():
    from data.fetcher import fetch_all
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)
    config = {**settings, **universe}
    data = fetch_all(config)
    return data, settings


class TestRSScannerLive:

    def test_all_11_sectors_returned(self, live_data):
        data, settings = live_data
        rs_cfg = settings["rs"]
        readings = compute_rs_readings(
            data["prices"], SECTOR_NAMES,
            windows=rs_cfg["windows"],
            slope_window=rs_cfg["slope_window"],
            composite_weights=rs_cfg["composite_weights"],
        )
        assert len(readings) == 11
        tickers = {r.ticker for r in readings}
        assert tickers == set(SECTOR_NAMES.keys())

    def test_ranks_are_1_through_11(self, live_data):
        data, settings = live_data
        rs_cfg = settings["rs"]
        readings = compute_rs_readings(
            data["prices"], SECTOR_NAMES,
            windows=rs_cfg["windows"],
            slope_window=rs_cfg["slope_window"],
            composite_weights=rs_cfg["composite_weights"],
        )
        ranks = sorted(r.rs_rank for r in readings)
        assert ranks == list(range(1, 12))

    def test_composite_in_range(self, live_data):
        data, settings = live_data
        rs_cfg = settings["rs"]
        readings = compute_rs_readings(
            data["prices"], SECTOR_NAMES,
            windows=rs_cfg["windows"],
            slope_window=rs_cfg["slope_window"],
            composite_weights=rs_cfg["composite_weights"],
        )
        for r in readings:
            assert 0 <= r.rs_composite <= 100, f"{r.ticker} composite={r.rs_composite}"

    def test_rs_values_are_reasonable(self, live_data):
        """RS values should be in a reasonable range (not > 100% or < -100%)."""
        data, settings = live_data
        rs_cfg = settings["rs"]
        readings = compute_rs_readings(
            data["prices"], SECTOR_NAMES,
            windows=rs_cfg["windows"],
            slope_window=rs_cfg["slope_window"],
            composite_weights=rs_cfg["composite_weights"],
        )
        for r in readings:
            assert -1.0 < r.rs_20d < 1.0, f"{r.ticker} rs_20d={r.rs_20d} seems extreme"

    def test_readings_are_rs_reading_type(self, live_data):
        data, settings = live_data
        rs_cfg = settings["rs"]
        readings = compute_rs_readings(
            data["prices"], SECTOR_NAMES,
            windows=rs_cfg["windows"],
            slope_window=rs_cfg["slope_window"],
            composite_weights=rs_cfg["composite_weights"],
        )
        for r in readings:
            assert isinstance(r, RSReading)
