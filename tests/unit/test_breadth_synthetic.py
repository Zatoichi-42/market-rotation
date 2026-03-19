"""
Breadth unit tests — synthetic data only.
Tests RSP/SPY ratio computation, z-score classification, boundary cases, missing data.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import BreadthSignal, BreadthReading
from engine.breadth import compute_breadth


def _make_spy_rsp_prices(spy_daily: float, rsp_daily: float, n_days: int = 120,
                         start: float = 100.0) -> pd.DataFrame:
    """Build a simple SPY + RSP price DataFrame from constant daily returns."""
    dates = pd.bdate_range(end="2026-03-18", periods=n_days)
    spy_prices = [start]
    rsp_prices = [start]
    for _ in range(n_days):
        spy_prices.append(spy_prices[-1] * (1 + spy_daily))
        rsp_prices.append(rsp_prices[-1] * (1 + rsp_daily))
    return pd.DataFrame({
        "SPY": spy_prices[1:],
        "RSP": rsp_prices[1:],
    }, index=dates)


def _make_diverging_prices(n_days: int = 120) -> pd.DataFrame:
    """RSP underperforming SPY consistently → divergence."""
    dates = pd.bdate_range(end="2026-03-18", periods=n_days)
    spy_prices = [100.0]
    rsp_prices = [100.0]
    for _ in range(n_days):
        spy_prices.append(spy_prices[-1] * 1.001)
        rsp_prices.append(rsp_prices[-1] * 0.9995)
    return pd.DataFrame({
        "SPY": spy_prices[1:],
        "RSP": rsp_prices[1:],
    }, index=dates)


class TestBreadthClassification:

    def test_healthy_breadth(self):
        """RSP outperforming SPY, z-score > 0, 20d change > 0 → HEALTHY."""
        # RSP gains more than SPY → ratio rising → healthy
        prices = _make_spy_rsp_prices(spy_daily=0.0005, rsp_daily=0.001, n_days=504)
        result = compute_breadth(prices)
        assert result.signal == BreadthSignal.HEALTHY

    def test_narrowing_breadth(self):
        """RSP underperforming with z-score between -1 and 0 → NARROWING."""
        # Build a ratio series that drifts down just enough for z-score ~ -0.5
        # Use noise so z-score distribution is well-behaved, but with
        # a recent dip that puts the tail in NARROWING territory.
        np.random.seed(77)
        n = 504
        dates = pd.bdate_range(end="2026-03-18", periods=n)
        # Both have noise, but RSP dips in the final 60 days
        spy_rets = np.random.normal(0.001, 0.005, n)
        rsp_rets = np.random.normal(0.001, 0.005, n)
        # Push RSP slightly negative in the recent window
        rsp_rets[-60:] -= 0.0008
        spy_p = [100.0]
        rsp_p = [100.0]
        for i in range(n):
            spy_p.append(spy_p[-1] * (1 + spy_rets[i]))
            rsp_p.append(rsp_p[-1] * (1 + rsp_rets[i]))
        prices = pd.DataFrame({"SPY": spy_p[1:], "RSP": rsp_p[1:]}, index=dates)
        result = compute_breadth(prices)
        # The z-score should be mildly negative
        assert result.rsp_spy_ratio_zscore < 0
        assert result.signal == BreadthSignal.NARROWING

    def test_diverging_breadth(self):
        """RSP sharply underperforming, z-score < -1 → DIVERGING."""
        prices = _make_diverging_prices(n_days=504)
        result = compute_breadth(prices)
        assert result.signal == BreadthSignal.DIVERGING

    def test_boundary_healthy_narrowing(self):
        """z-score exactly 0.0 → NARROWING (boundary goes to worse case)."""
        # Test the classification function directly with known z-score
        from engine.breadth import _classify_breadth_signal
        signal = _classify_breadth_signal(zscore=0.0, ratio_20d_change=0.0)
        assert signal == BreadthSignal.NARROWING

    def test_boundary_narrowing_diverging(self):
        """z-score exactly -1.0 → DIVERGING."""
        from engine.breadth import _classify_breadth_signal
        signal = _classify_breadth_signal(zscore=-1.0, ratio_20d_change=-0.01)
        assert signal == BreadthSignal.DIVERGING

    def test_positive_zscore_is_healthy(self):
        """z-score = 1.5 → HEALTHY."""
        from engine.breadth import _classify_breadth_signal
        signal = _classify_breadth_signal(zscore=1.5, ratio_20d_change=0.01)
        assert signal == BreadthSignal.HEALTHY

    def test_negative_zscore_mid_is_narrowing(self):
        """z-score = -0.5 → NARROWING."""
        from engine.breadth import _classify_breadth_signal
        signal = _classify_breadth_signal(zscore=-0.5, ratio_20d_change=-0.005)
        assert signal == BreadthSignal.NARROWING

    def test_deeply_negative_zscore_is_diverging(self):
        """z-score = -2.0 → DIVERGING."""
        from engine.breadth import _classify_breadth_signal
        signal = _classify_breadth_signal(zscore=-2.0, ratio_20d_change=-0.02)
        assert signal == BreadthSignal.DIVERGING


class TestRatioComputation:

    def test_rsp_spy_ratio_calculation(self):
        """Known prices → verify ratio computed correctly."""
        prices = _make_spy_rsp_prices(spy_daily=0.001, rsp_daily=0.001, n_days=504)
        result = compute_breadth(prices)
        # Both same return → ratio ≈ 1.0
        assert abs(result.rsp_spy_ratio - 1.0) < 0.01

    def test_rsp_outperforming_ratio_above_one(self):
        """RSP outperforming → ratio > starting ratio."""
        prices = _make_spy_rsp_prices(spy_daily=0.0005, rsp_daily=0.002, n_days=504)
        result = compute_breadth(prices)
        assert result.rsp_spy_ratio > 1.0

    def test_rsp_underperforming_ratio_below_one(self):
        """RSP underperforming → ratio < starting ratio."""
        prices = _make_spy_rsp_prices(spy_daily=0.002, rsp_daily=0.0005, n_days=504)
        result = compute_breadth(prices)
        assert result.rsp_spy_ratio < 1.0

    def test_20d_change_positive_when_improving(self):
        """RSP gaining on SPY → 20d change positive."""
        prices = _make_spy_rsp_prices(spy_daily=0.0005, rsp_daily=0.001, n_days=504)
        result = compute_breadth(prices)
        assert result.rsp_spy_ratio_20d_change > 0

    def test_20d_change_negative_when_declining(self):
        """RSP losing to SPY → 20d change negative."""
        prices = _make_spy_rsp_prices(spy_daily=0.001, rsp_daily=0.0005, n_days=504)
        result = compute_breadth(prices)
        assert result.rsp_spy_ratio_20d_change < 0


class TestInsufficientData:

    def test_insufficient_history_for_zscore(self):
        """< 252 days of data → z-score = NaN, signal = NARROWING (conservative default)."""
        prices = _make_spy_rsp_prices(spy_daily=0.001, rsp_daily=0.001, n_days=30)
        result = compute_breadth(prices)
        assert np.isnan(result.rsp_spy_ratio_zscore)
        assert result.signal == BreadthSignal.NARROWING

    def test_very_short_data(self):
        """Only 5 days → still returns a BreadthReading without crashing."""
        prices = _make_spy_rsp_prices(spy_daily=0.001, rsp_daily=0.001, n_days=5)
        result = compute_breadth(prices)
        assert isinstance(result, BreadthReading)
        assert result.signal == BreadthSignal.NARROWING

    def test_missing_rsp_column(self):
        """RSP column missing → conservative default NARROWING."""
        dates = pd.bdate_range(end="2026-03-18", periods=60)
        prices = pd.DataFrame({"SPY": [100 + i * 0.1 for i in range(60)]}, index=dates)
        result = compute_breadth(prices)
        assert result.signal == BreadthSignal.NARROWING

    def test_nan_in_prices(self):
        """NaN in price data → still produces a reading."""
        prices = _make_spy_rsp_prices(spy_daily=0.001, rsp_daily=0.001, n_days=504)
        prices.iloc[100:105, prices.columns.get_loc("RSP")] = np.nan
        result = compute_breadth(prices)
        assert isinstance(result, BreadthReading)


class TestReturnType:

    def test_returns_breadth_reading(self):
        """Return type is BreadthReading."""
        prices = _make_spy_rsp_prices(spy_daily=0.001, rsp_daily=0.001, n_days=504)
        result = compute_breadth(prices)
        assert isinstance(result, BreadthReading)

    def test_explanation_nonempty(self):
        """Explanation string is non-empty."""
        prices = _make_spy_rsp_prices(spy_daily=0.001, rsp_daily=0.001, n_days=504)
        result = compute_breadth(prices)
        assert len(result.explanation) > 0


class TestFactoryScenarios:

    def test_normal_market_healthy_or_narrowing(self, normal_market):
        """Normal market: RSP slightly outperforms → HEALTHY or NARROWING."""
        result = compute_breadth(normal_market["prices"])
        assert result.signal in (BreadthSignal.HEALTHY, BreadthSignal.NARROWING)

    def test_breadth_divergence_scenario(self, breadth_divergence):
        """Breadth divergence factory: SPY up, RSP flat → DIVERGING or NARROWING."""
        result = compute_breadth(breadth_divergence["prices"])
        # With only 60 days, z-score may be NaN → NARROWING default
        # But the 20d change should be negative
        assert result.rsp_spy_ratio_20d_change < 0

    def test_hostile_market_breadth_declining(self, hostile_market):
        """Hostile market: RSP declining faster → ratio declining."""
        result = compute_breadth(hostile_market["prices"])
        assert result.rsp_spy_ratio_20d_change < 0

    def test_identical_sectors_neutral_breadth(self, identical_sectors):
        """All sectors identical → RSP/SPY near 1.0, change near 0."""
        result = compute_breadth(identical_sectors["prices"])
        assert abs(result.rsp_spy_ratio - 1.0) < 0.05
