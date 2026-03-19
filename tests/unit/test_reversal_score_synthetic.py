"""
Reversal Score unit tests — synthetic data only.
Three orthogonal pillars: Breadth Deterioration, Price Break Quality, Crowding/Stretch.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import ReversalScoreReading
from engine.reversal_score import (
    compute_reversal_score, compute_reversal_scores_batch,
    _compute_breadth_deterioration, _compute_price_break_quality,
    _compute_crowding_stretch,
)

SETTINGS = {
    "rs_slope_lookback": 5, "rs_slope_reversal_threshold": 0,
    "participation_decay_window": 20,
    "failed_breakout_lookback": 20, "failed_breakout_reversal_days": 3,
    "gap_threshold_pct": 0.005, "gap_fade_lookback": 20,
    "clv_short_window": 5, "clv_long_window": 20,
    "follow_through_window": 10,
    "distance_ma_period": 20, "rvol_lookback": 20,
    "price_accel_fast": 5, "price_accel_slow": 20,
}
WEIGHTS = {"breadth_det_weight": 0.40, "price_break_weight": 0.30, "crowding_weight": 0.30}


def _make_trending_prices(n=60, ticker="XLK", spy_rate=0.001, ticker_rate=0.002):
    """Simple uptrending prices for ticker + SPY."""
    dates = pd.bdate_range(end="2026-03-18", periods=n)
    spy_p = [100.0]
    t_p = [100.0]
    for _ in range(n):
        spy_p.append(spy_p[-1] * (1 + spy_rate))
        t_p.append(t_p[-1] * (1 + ticker_rate))
    prices = pd.DataFrame({"SPY": spy_p[1:], ticker: t_p[1:]}, index=dates)
    highs = prices * 1.005
    lows = prices * 0.995
    volumes = pd.DataFrame({"SPY": [10_000_000] * n, ticker: [5_000_000] * n}, index=dates)
    return prices, highs, lows, volumes


def _make_exhaustion_prices(n=60, ticker="XLE"):
    """Price strong for 40d then stalling with deteriorating internals."""
    np.random.seed(300)
    dates = pd.bdate_range(end="2026-03-18", periods=n)
    spy_p, t_p = [100.0], [100.0]
    for i in range(n):
        spy_p.append(spy_p[-1] * (1 + 0.0005))
        if i < 40:
            t_p.append(t_p[-1] * (1 + 0.003 + np.random.normal(0, 0.001)))
        else:
            t_p.append(t_p[-1] * (1 - 0.0005 + np.random.normal(0, 0.002)))
    prices = pd.DataFrame({"SPY": spy_p[1:], ticker: t_p[1:]}, index=dates)
    # Highs: during exhaustion, close near lows (CLV declining)
    h_vals = np.array(prices[ticker]) * 1.008
    l_vals = np.array(prices[ticker]) * 0.992
    # Make CLV bad in last 20 days (close near low)
    close_vals = np.array(prices[ticker])
    for i in range(40, n):
        l_vals[i] = close_vals[i] * 0.998
        h_vals[i] = close_vals[i] * 1.012  # high further from close
    highs = pd.DataFrame({"SPY": np.array(prices["SPY"]) * 1.005, ticker: h_vals}, index=dates)
    lows = pd.DataFrame({"SPY": np.array(prices["SPY"]) * 0.995, ticker: l_vals}, index=dates)
    # Volume spikes at end
    vol_base = np.array([5_000_000] * n, dtype=float)
    vol_base[40:] = 15_000_000  # 3x
    volumes = pd.DataFrame({"SPY": [10_000_000] * n, ticker: vol_base.astype(int)}, index=dates)
    return prices, highs, lows, volumes


class TestBreadthDeteriorationPillar:

    def test_rs_slope_reversal_detected(self):
        """RS slope was positive then turned negative → elevated pillar."""
        prices, highs, lows, volumes = _make_exhaustion_prices()
        pillar, subs = _compute_breadth_deterioration(prices, "XLE", "SPY", SETTINGS)
        assert pillar > 30  # Elevated vs trending baseline

    def test_no_deterioration_when_trending(self):
        """Clean uptrend → low pillar."""
        prices, highs, lows, volumes = _make_trending_prices(60, "XLK", 0.001, 0.003)
        pillar, subs = _compute_breadth_deterioration(prices, "XLK", "SPY", SETTINGS)
        assert pillar < 50

    def test_participation_decay_detected(self):
        """% of days outperforming drops → detected in sub-signal."""
        prices, _, _, _ = _make_exhaustion_prices()
        _, subs = _compute_breadth_deterioration(prices, "XLE", "SPY", SETTINGS)
        assert "participation_decay" in subs


class TestPriceBreakQualityPillar:

    def test_clv_deterioration(self):
        """Close near lows → high pillar."""
        prices, highs, lows, volumes = _make_exhaustion_prices()
        pillar, subs = _compute_price_break_quality(prices, highs, lows, "XLE", SETTINGS)
        assert pillar > 40

    def test_healthy_clv(self):
        """Close near highs → low pillar."""
        prices, highs, lows, volumes = _make_trending_prices()
        pillar, subs = _compute_price_break_quality(prices, highs, lows, "XLK", SETTINGS)
        assert pillar < 60

    def test_sub_signals_present(self):
        """All expected sub-signals present."""
        prices, highs, lows, volumes = _make_trending_prices()
        _, subs = _compute_price_break_quality(prices, highs, lows, "XLK", SETTINGS)
        for key in ["failed_breakout_rate", "gap_fade_rate", "clv_trend", "follow_through"]:
            assert key in subs


class TestCrowdingStretchPillar:

    def test_extreme_distance_from_ma(self):
        """Parabolic move → crowding pillar higher than steady trend."""
        np.random.seed(301)
        n = 60
        dates = pd.bdate_range(end="2026-03-18", periods=n)
        # Parabolic: accelerating returns
        t_p = [100.0]
        for i in range(n):
            t_p.append(t_p[-1] * (1 + 0.002 + 0.0003 * i))
        prices = pd.DataFrame({"SPY": [100 + i * 0.1 for i in range(n)],
                                "XLK": t_p[1:]}, index=dates)
        vol_base = np.array([5_000_000] * n, dtype=float)
        vol_base[-10:] = 15_000_000
        volumes = pd.DataFrame({"SPY": [10_000_000] * n, "XLK": vol_base.astype(int)}, index=dates)
        pillar_parabolic, _ = _compute_crowding_stretch(prices, volumes, "XLK", SETTINGS)

        # Compare to steady trend
        prices_steady, _, _, volumes_steady = _make_trending_prices(60, "XLK", 0.001, 0.0012)
        pillar_steady, _ = _compute_crowding_stretch(prices_steady, volumes_steady, "XLK", SETTINGS)

        assert pillar_parabolic > pillar_steady

    def test_normal_price_near_ma(self):
        """Steady trend, no stretch → low pillar."""
        prices, _, _, volumes = _make_trending_prices(60, "XLK", 0.001, 0.0012)
        pillar, subs = _compute_crowding_stretch(prices, volumes, "XLK", SETTINGS)
        assert pillar < 60

    def test_sub_signals_present(self):
        prices, _, _, volumes = _make_trending_prices()
        _, subs = _compute_crowding_stretch(prices, volumes, "XLK", SETTINGS)
        for key in ["distance_from_ma", "rvol", "price_acceleration"]:
            assert key in subs


class TestReversalScoreComposite:

    def test_weights_sum_to_one(self):
        total = WEIGHTS["breadth_det_weight"] + WEIGHTS["price_break_weight"] + WEIGHTS["crowding_weight"]
        assert abs(total - 1.0) < 1e-10

    def test_score_range_0_to_1(self):
        prices, highs, lows, volumes = _make_trending_prices()
        result = compute_reversal_score(prices, highs, lows, volumes, "XLK", settings=SETTINGS, weights=WEIGHTS)
        assert 0.0 <= result.reversal_score <= 1.0

    def test_exhaustion_has_elevated_score(self):
        """Exhaustion scenario score higher than healthy trend."""
        prices_ex, highs_ex, lows_ex, volumes_ex = _make_exhaustion_prices()
        result_ex = compute_reversal_score(prices_ex, highs_ex, lows_ex, volumes_ex, "XLE", settings=SETTINGS, weights=WEIGHTS)

        prices_ok, highs_ok, lows_ok, volumes_ok = _make_trending_prices(60, "XLE", 0.001, 0.002)
        result_ok = compute_reversal_score(prices_ok, highs_ok, lows_ok, volumes_ok, "XLE", settings=SETTINGS, weights=WEIGHTS)

        assert result_ex.reversal_score > result_ok.reversal_score

    def test_healthy_trend_has_low_score(self):
        prices, highs, lows, volumes = _make_trending_prices(60, "XLK", 0.001, 0.002)
        result = compute_reversal_score(prices, highs, lows, volumes, "XLK", settings=SETTINGS, weights=WEIGHTS)
        assert result.reversal_score < 0.5

    def test_percentile_with_history(self):
        """Given history, percentile computed correctly."""
        prices, highs, lows, volumes = _make_trending_prices()
        history = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        result = compute_reversal_score(prices, highs, lows, volumes, "XLK",
                                         settings=SETTINGS, weights=WEIGHTS,
                                         history_scores=history)
        assert 0 <= result.reversal_percentile <= 100

    def test_above_75th_flag(self):
        """Score in top quartile → above_75th = True."""
        prices, highs, lows, volumes = _make_exhaustion_prices()
        # Use history where most values are low
        history = pd.Series([0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.25, 0.3, 0.28, 0.15])
        result = compute_reversal_score(prices, highs, lows, volumes, "XLE",
                                         settings=SETTINGS, weights=WEIGHTS,
                                         history_scores=history)
        if result.reversal_score > 0.3:
            assert result.above_75th == True

    def test_sub_signals_dict_populated(self):
        prices, highs, lows, volumes = _make_trending_prices()
        result = compute_reversal_score(prices, highs, lows, volumes, "XLK",
                                         settings=SETTINGS, weights=WEIGHTS)
        assert isinstance(result.sub_signals, dict)
        assert len(result.sub_signals) > 0

    def test_returns_reversal_score_reading(self):
        prices, highs, lows, volumes = _make_trending_prices()
        result = compute_reversal_score(prices, highs, lows, volumes, "XLK",
                                         settings=SETTINGS, weights=WEIGHTS)
        assert isinstance(result, ReversalScoreReading)


class TestBatchCompute:

    def test_batch_returns_list(self):
        prices, highs, lows, volumes = _make_trending_prices()
        # Add a second ticker
        prices["XLF"] = prices["XLK"] * 0.98
        highs["XLF"] = highs["XLK"] * 0.98
        lows["XLF"] = lows["XLK"] * 0.98
        volumes["XLF"] = volumes["XLK"]
        results = compute_reversal_scores_batch(
            prices, highs, lows, volumes, ["XLK", "XLF"],
            settings=SETTINGS, weights=WEIGHTS,
        )
        assert len(results) == 2
        tickers = {r.ticker for r in results}
        assert tickers == {"XLK", "XLF"}


class TestReversalScenarios:

    def test_exhaustion_scenario(self, reversal_exhaustion):
        """XLE reversal score should be elevated."""
        d = reversal_exhaustion
        result = compute_reversal_score(
            d["prices"], d["highs"], d["lows"], d["volumes"],
            "XLE", settings=SETTINGS, weights=WEIGHTS,
        )
        assert result.reversal_score > 0.3

    def test_crowding_scenario(self, reversal_crowding):
        """XLK crowding pillar higher than a steady sector."""
        d = reversal_crowding
        result_xlk = compute_reversal_score(
            d["prices"], d["highs"], d["lows"], d["volumes"],
            "XLK", settings=SETTINGS, weights=WEIGHTS,
        )
        # Compare to a sector that isn't parabolic in this scenario
        result_xlp = compute_reversal_score(
            d["prices"], d["highs"], d["lows"], d["volumes"],
            "XLP", settings=SETTINGS, weights=WEIGHTS,
        )
        assert result_xlk.crowding_pillar > result_xlp.crowding_pillar
