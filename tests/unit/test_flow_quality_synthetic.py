"""Test flow quality pillar computation."""
import pytest
import pandas as pd
import numpy as np
from engine.flow_quality import compute_flow_pillar


class TestFlowPillar:
    def _make_ohlcv(self, closes, highs, lows, volumes, n_days=60):
        dates = pd.bdate_range(end="2026-03-19", periods=n_days)
        n = min(n_days, len(closes))
        return (
            pd.DataFrame({"T": closes[-n:]}, index=dates[-n:]),
            pd.DataFrame({"T": highs[-n:]}, index=dates[-n:]),
            pd.DataFrame({"T": lows[-n:]}, index=dates[-n:]),
            pd.DataFrame({"T": volumes[-n:]}, index=dates[-n:]),
        )

    def test_high_rvol_high_clv_returns_high(self):
        n = 60
        closes = [100 + i * 0.5 for i in range(n)]
        highs = [c + 0.3 for c in closes]
        lows = [c - 1.0 for c in closes]
        volumes = [1_000_000] * (n - 5) + [3_000_000] * 5
        p, h, l, v = self._make_ohlcv(closes, highs, lows, volumes, n)
        result = compute_flow_pillar(p, h, l, v, "T")
        assert result > 60.0

    def test_low_rvol_low_clv_returns_low(self):
        n = 60
        closes = [100 - i * 0.5 for i in range(n)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 0.1 for c in closes]
        volumes = [2_000_000] * (n - 5) + [500_000] * 5
        p, h, l, v = self._make_ohlcv(closes, highs, lows, volumes, n)
        result = compute_flow_pillar(p, h, l, v, "T")
        assert result < 45.0

    def test_missing_ticker_returns_neutral(self):
        dates = pd.bdate_range(end="2026-03-19", periods=25)
        empty = pd.DataFrame({"OTHER": [100]*25}, index=dates)
        result = compute_flow_pillar(empty, empty, empty, empty, "T")
        assert result == 50.0

    def test_insufficient_data_returns_neutral(self):
        dates = pd.bdate_range(end="2026-03-19", periods=10)
        df = pd.DataFrame({"T": [100]*10}, index=dates)
        result = compute_flow_pillar(df, df, df, df, "T")
        assert result == 50.0

    def test_result_clamped_0_100(self):
        n = 60
        closes = [100] * n
        highs = [101] * n
        lows = [99] * n
        volumes = [1_000_000] * n
        p, h, l, v = self._make_ohlcv(closes, highs, lows, volumes, n)
        result = compute_flow_pillar(p, h, l, v, "T")
        assert 0.0 <= result <= 100.0

    def test_clv_close_at_high_is_bullish(self):
        """Uptrending with close near high → CLV high → bullish flow."""
        n = 60
        closes = [100 + i * 0.2 for i in range(n)]  # Uptrend
        highs = [c + 0.1 for c in closes]  # Close near high
        lows = [c - 1.5 for c in closes]   # Low well below close
        volumes = [1_000_000] * n
        p, h, l, v = self._make_ohlcv(closes, highs, lows, volumes, n)
        result = compute_flow_pillar(p, h, l, v, "T")
        assert result >= 50.0
