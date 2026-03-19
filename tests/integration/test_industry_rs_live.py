"""Integration tests for industry RS — live data."""
import pytest
import yaml
from engine.industry_rs import compute_industry_rs

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_data():
    from data.fetcher import fetch_all
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)
    config = {**settings, **universe}
    return fetch_all(config), settings, universe


class TestIndustryRSLive:

    def test_industry_etfs_have_data(self, live_data):
        data, settings, universe = live_data
        industries = universe.get("industries", [])
        missing = [i["ticker"] for i in industries if i["ticker"] not in data["prices"].columns]
        assert len(missing) <= 2, f"Missing industry ETFs: {missing}"

    def test_rs_vs_parent_reasonable(self, live_data):
        data, settings, universe = live_data
        industries = universe.get("industries", [])
        available = [i for i in industries if i["ticker"] in data["prices"].columns]
        results = compute_industry_rs(data["prices"], available)
        for r in results:
            assert -0.5 < r.rs_20d_vs_parent < 0.5, f"{r.ticker} vs parent {r.rs_20d_vs_parent:.3f} extreme"

    def test_rankings_complete(self, live_data):
        data, settings, universe = live_data
        industries = universe.get("industries", [])
        available = [i for i in industries if i["ticker"] in data["prices"].columns]
        results = compute_industry_rs(data["prices"], available)
        ranks = sorted(r.rs_rank for r in results)
        assert ranks == list(range(1, len(results) + 1))
