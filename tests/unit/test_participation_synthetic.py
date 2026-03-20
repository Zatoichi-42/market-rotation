"""Test participation pillar computation."""
import pytest
import pandas as pd
import numpy as np
from engine.participation import compute_participation_pillar


class TestParticipationPillar:
    def _make_prices(self, sector_rets, child_rets_dict, n_days=25):
        dates = pd.bdate_range(end="2026-03-19", periods=n_days)
        data = {}
        for ticker, rets in [("SECTOR", sector_rets)] + list(child_rets_dict.items()):
            prices = [100.0]
            for r in rets:
                prices.append(prices[-1] * (1 + r))
            data[ticker] = prices[-n_days:]
        return pd.DataFrame(data, index=dates)

    def test_all_children_outperforming_returns_high(self):
        sector = [0.001] * 25
        children = {"C1": [0.002]*25, "C2": [0.002]*25, "C3": [0.002]*25}
        prices = self._make_prices(sector, children)
        result = compute_participation_pillar(prices, "SECTOR", list(children.keys()))
        assert result > 70.0

    def test_no_children_outperforming_returns_low(self):
        sector = [0.003] * 25
        children = {"C1": [0.001]*25, "C2": [0.001]*25, "C3": [0.001]*25}
        prices = self._make_prices(sector, children)
        result = compute_participation_pillar(prices, "SECTOR", list(children.keys()))
        assert result < 30.0

    def test_mixed_children_returns_mid(self):
        sector = [0.002] * 25
        children = {"C1": [0.003]*25, "C2": [0.001]*25}
        prices = self._make_prices(sector, children)
        result = compute_participation_pillar(prices, "SECTOR", list(children.keys()))
        assert 30.0 < result < 70.0

    def test_no_children_returns_neutral(self):
        prices = pd.DataFrame({"SECTOR": [100]*25},
                              index=pd.bdate_range(end="2026-03-19", periods=25))
        result = compute_participation_pillar(prices, "SECTOR", [])
        assert result == 50.0

    def test_children_not_in_prices_returns_neutral(self):
        prices = pd.DataFrame({"SECTOR": [100]*25},
                              index=pd.bdate_range(end="2026-03-19", periods=25))
        result = compute_participation_pillar(prices, "SECTOR", ["MISSING1", "MISSING2"])
        assert result == 50.0

    def test_result_clamped_0_100(self):
        sector = [0.001] * 25
        children = {"C1": [0.01]*25}
        prices = self._make_prices(sector, children)
        result = compute_participation_pillar(prices, "SECTOR", ["C1"])
        assert 0.0 <= result <= 100.0

    def test_insufficient_data_returns_neutral(self):
        prices = pd.DataFrame({"SECTOR": [100]*10, "C1": [100]*10},
                              index=pd.bdate_range(end="2026-03-19", periods=10))
        result = compute_participation_pillar(prices, "SECTOR", ["C1"])
        assert result == 50.0
