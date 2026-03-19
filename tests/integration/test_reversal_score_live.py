"""Integration tests for reversal score — live data."""
import pytest
import yaml
from engine.reversal_score import compute_reversal_scores_batch
from tests.factories import SECTOR_TICKERS

pytestmark = pytest.mark.integration

WEIGHTS = {"breadth_det_weight": 0.40, "price_break_weight": 0.30, "crowding_weight": 0.30}


@pytest.fixture(scope="module")
def live_data():
    from data.fetcher import fetch_all
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)
    config = {**settings, **universe}
    return fetch_all(config), settings


class TestReversalScoreLive:

    def test_reversal_scores_for_all_sectors(self, live_data):
        data, settings = live_data
        results = compute_reversal_scores_batch(
            data["prices"], data["highs"], data["lows"], data["volumes"],
            SECTOR_TICKERS, settings=settings.get("reversal", {}), weights=WEIGHTS,
        )
        assert len(results) == 11
        for r in results:
            assert 0.0 <= r.reversal_score <= 1.0, f"{r.ticker} score {r.reversal_score} out of range"

    def test_sub_signals_all_present(self, live_data):
        data, settings = live_data
        results = compute_reversal_scores_batch(
            data["prices"], data["highs"], data["lows"], data["volumes"],
            SECTOR_TICKERS, settings=settings.get("reversal", {}), weights=WEIGHTS,
        )
        for r in results:
            assert len(r.sub_signals) > 0, f"{r.ticker} has no sub-signals"

    def test_scores_differentiate(self, live_data):
        data, settings = live_data
        results = compute_reversal_scores_batch(
            data["prices"], data["highs"], data["lows"], data["volumes"],
            SECTOR_TICKERS, settings=settings.get("reversal", {}), weights=WEIGHTS,
        )
        scores = [r.reversal_score for r in results]
        assert max(scores) != min(scores), "All reversal scores identical — likely a bug"
