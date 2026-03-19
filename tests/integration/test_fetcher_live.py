"""
Integration tests for data fetcher — live yfinance data.
"""
import pytest
import yaml

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config():
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    return {**settings, **universe}


@pytest.fixture(scope="module")
def live_data(config):
    from data.fetcher import fetch_all
    return fetch_all(config)


class TestFetcherLive:

    def test_prices_not_empty(self, live_data):
        """Prices DataFrame should have rows."""
        assert len(live_data["prices"]) > 100

    def test_all_sector_tickers_present(self, live_data):
        """All 11 GICS sector ETFs should be in prices."""
        sectors = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]
        for t in sectors:
            assert t in live_data["prices"].columns, f"Missing {t}"

    def test_spy_rsp_present(self, live_data):
        """SPY and RSP must be in prices."""
        assert "SPY" in live_data["prices"].columns
        assert "RSP" in live_data["prices"].columns

    def test_vix_has_data(self, live_data):
        """VIX series should have data."""
        assert len(live_data["vix"]) > 100

    def test_vix3m_has_data(self, live_data):
        """VIX3M series should have data."""
        assert len(live_data["vix3m"]) > 50

    def test_volumes_present(self, live_data):
        """Volume data should exist."""
        assert not live_data["volumes"].empty

    def test_credit_has_hyg_lqd(self, live_data):
        """Credit DataFrame has HYG and LQD."""
        credit = live_data["credit"]
        assert "hyg_close" in credit.columns
        assert "lqd_close" in credit.columns

    def test_metadata_has_fetch_timestamp(self, live_data):
        """Metadata includes a fetch timestamp."""
        assert "fetch_timestamp" in live_data["metadata"]
        assert len(live_data["metadata"]["fetch_timestamp"]) > 0

    def test_prices_no_all_nan_columns(self, live_data):
        """No ticker column should be entirely NaN."""
        prices = live_data["prices"]
        for col in prices.columns:
            assert prices[col].notna().any(), f"Column {col} is all NaN"

    def test_fred_does_not_crash(self, live_data):
        """FRED fetch should either return data or None, not crash."""
        fred = live_data["fred_hy_oas"]
        assert fred is None or len(fred) > 0
