"""Concentration monitor unit tests."""
import pytest
import pandas as pd
import numpy as np
from engine.schemas import ConcentrationRegime
from engine.concentration_monitor import compute_concentration, compute_concentration_all

SETTINGS = {
    "top_n_leaders": 5,
    "elevated_zscore": -0.5,
    "extreme_zscore": -1.5,
    "leader_strong_min_avg_rs": 0.0,
    "leader_deteriorating_max_rs": -0.02,
    "leader_dispersion_mixed": 0.03,
    "healthy_modifier": 15,
    "unhealthy_modifier": -15,
}


def _make_prices(n=60, leaders_strong=True):
    """Build prices for sectors + their leaders."""
    np.random.seed(600)
    dates = pd.bdate_range(end="2026-03-18", periods=n)
    data = {"SPY": [], "RSP": [], "XLK": []}
    spy_p, rsp_p, xlk_p = 100.0, 100.0, 100.0
    for _ in range(n):
        spy_p *= 1 + np.random.normal(0.001, 0.005)
        rsp_p *= 1 + np.random.normal(0.0005, 0.005)
        xlk_p *= 1 + np.random.normal(0.0015, 0.005)
        data["SPY"].append(spy_p)
        data["RSP"].append(rsp_p)
        data["XLK"].append(xlk_p)
    # Add leaders — strong signal relative to sector
    for leader in ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"]:
        lp = 100.0
        # Leaders significantly above sector when strong, below when weak
        rate = 0.004 if leaders_strong else -0.002
        for _ in range(n):
            lp *= 1 + np.random.normal(rate, 0.005)
            data.setdefault(leader, []).append(lp)
    return pd.DataFrame(data, index=dates)


class TestConcentration:

    def test_broad_healthy(self):
        """Normal breadth, normal concentration → BROAD_HEALTHY."""
        prices = _make_prices(leaders_strong=True)
        result = compute_concentration(
            prices, "XLK", ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"],
            ew_cw_zscore=0.5, settings=SETTINGS,
        )
        assert result.regime == ConcentrationRegime.BROAD_HEALTHY

    def test_concentrated_healthy(self):
        """Narrow breadth but strong leaders → CONCENTRATED_HEALTHY."""
        prices = _make_prices(leaders_strong=True)
        result = compute_concentration(
            prices, "XLK", ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"],
            ew_cw_zscore=-0.8, settings=SETTINGS,
        )
        assert result.regime == ConcentrationRegime.CONCENTRATED_HEALTHY
        assert result.participation_modifier > 0

    def test_concentrated_unhealthy(self):
        """Narrow breadth and weak leaders → CONCENTRATED_UNHEALTHY."""
        prices = _make_prices(leaders_strong=False)
        result = compute_concentration(
            prices, "XLK", ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"],
            ew_cw_zscore=-0.8, settings=SETTINGS,
        )
        assert result.regime in (ConcentrationRegime.CONCENTRATED_UNHEALTHY,
                                  ConcentrationRegime.CONCENTRATED_FRAGILE)
        assert result.participation_modifier <= 0

    def test_modifier_in_range(self):
        prices = _make_prices()
        result = compute_concentration(
            prices, "XLK", ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"],
            ew_cw_zscore=-0.8, settings=SETTINGS,
        )
        assert -15 <= result.participation_modifier <= 15


class TestComputeAll:

    def test_returns_list(self):
        prices = _make_prices()
        sector_leaders = {"XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"]}
        results = compute_concentration_all(prices, sector_leaders, ew_cw_zscore=0.0, settings=SETTINGS)
        assert len(results) >= 1
        assert results[0].sector_ticker == "XLK"
