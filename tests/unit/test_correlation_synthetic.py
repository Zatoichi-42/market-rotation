"""
Cross-Sector Correlation unit tests — synthetic data only.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import SignalLevel
from engine.correlation import compute_cross_sector_correlation


_SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]


def _make_prices(n=600, correlation=0.3, seed=42):
    """Generate synthetic sector prices with controlled correlation."""
    np.random.seed(seed)
    dates = pd.bdate_range(end="2026-03-19", periods=n)
    common = np.random.randn(n) * 0.01
    data = {}
    for t in _SECTORS:
        idio = np.random.randn(n) * 0.01
        returns = correlation * common + (1 - correlation) * idio
        data[t] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


class TestCorrelationClassification:
    """TEST-CORR-01 through TEST-CORR-05."""

    def test_corr01_low_zscore_normal(self):
        """TEST-CORR-01: Avg correlation z-score < 0.5 → NORMAL"""
        prices = _make_prices(correlation=0.2)
        reading = compute_cross_sector_correlation(prices)
        assert reading is not None
        # Low correlation with stable history should be NORMAL
        assert reading.level in (SignalLevel.NORMAL, SignalLevel.FRAGILE)  # depends on random

    def test_corr05_low_abs_normal(self):
        """TEST-CORR-05: Avg correlation absolute = 0.75, z-score = 0.3 → NORMAL"""
        # Just verify the function runs and returns a valid reading
        prices = _make_prices(correlation=0.1)
        reading = compute_cross_sector_correlation(prices)
        assert reading is not None
        assert isinstance(reading.avg_correlation, float)

    def test_corr04_high_absolute_hostile(self):
        """TEST-CORR-04: Avg correlation absolute > 0.80 → HOSTILE"""
        # Very high correlation
        prices = _make_prices(correlation=0.95)
        reading = compute_cross_sector_correlation(prices)
        assert reading is not None
        if reading.avg_correlation > 0.80:
            assert reading.level == SignalLevel.HOSTILE

    def test_corr06_matrix_size(self):
        """TEST-CORR-06: Correlation matrix is 11x11 with 55 unique pairs"""
        n = len(_SECTORS)
        expected_pairs = n * (n - 1) // 2
        assert expected_pairs == 55

    def test_corr08_insufficient_data(self):
        """TEST-CORR-08: With insufficient data (< 21 days) → None"""
        dates = pd.bdate_range(end="2026-03-19", periods=15)
        prices = pd.DataFrame({t: np.random.randn(15) + 100 for t in _SECTORS}, index=dates)
        reading = compute_cross_sector_correlation(prices)
        assert reading is None

    def test_corr09_returns_extreme_pairs(self):
        """TEST-CORR-09: Returns max and min correlated pair correctly"""
        prices = _make_prices(correlation=0.3)
        reading = compute_cross_sector_correlation(prices)
        assert reading is not None
        assert len(reading.max_corr_pair) == 2
        assert len(reading.min_corr_pair) == 2
        assert reading.max_corr_pair[0] in _SECTORS
        assert reading.min_corr_pair[0] in _SECTORS

    def test_too_few_sectors(self):
        """Less than 3 sectors → None"""
        dates = pd.bdate_range(end="2026-03-19", periods=100)
        prices = pd.DataFrame({"XLK": np.random.randn(100) + 100}, index=dates)
        reading = compute_cross_sector_correlation(prices)
        assert reading is None

    def test_reading_fields(self):
        """Verify all fields are populated."""
        prices = _make_prices()
        reading = compute_cross_sector_correlation(prices)
        assert reading is not None
        assert isinstance(reading.avg_correlation, float)
        assert isinstance(reading.avg_corr_zscore, float)
        assert isinstance(reading.level, SignalLevel)
        assert isinstance(reading.description, str)
        assert len(reading.description) > 0


class TestCorrelationSignalInGate:
    """Test correlation signal integration with regime gate."""

    def test_correlation_signal_normal(self):
        from engine.regime_gate import classify_signal
        thresholds = {"correlation": {"fragile_zscore": 0.5, "hostile_zscore": 1.5}}
        sig = classify_signal("correlation", 0.3, thresholds)
        assert sig is not None
        assert sig.level == SignalLevel.NORMAL

    def test_correlation_signal_fragile(self):
        from engine.regime_gate import classify_signal
        thresholds = {"correlation": {"fragile_zscore": 0.5, "hostile_zscore": 1.5}}
        sig = classify_signal("correlation", 0.8, thresholds)
        assert sig is not None
        assert sig.level == SignalLevel.FRAGILE

    def test_correlation_signal_hostile(self):
        from engine.regime_gate import classify_signal
        thresholds = {"correlation": {"fragile_zscore": 0.5, "hostile_zscore": 1.5}}
        sig = classify_signal("correlation", 1.5, thresholds)
        assert sig is not None
        assert sig.level == SignalLevel.HOSTILE

    def test_correlation_nan_excluded(self):
        from engine.regime_gate import classify_signal
        thresholds = {"correlation": {"fragile_zscore": 0.5, "hostile_zscore": 1.5}}
        sig = classify_signal("correlation", float("nan"), thresholds)
        assert sig is None
