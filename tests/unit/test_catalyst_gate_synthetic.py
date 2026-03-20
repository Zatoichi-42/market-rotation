"""Catalyst gate unit tests."""
import pytest
import pandas as pd
import numpy as np
from engine.schemas import CatalystAction, ShockType
from engine.catalyst_gate import (
    load_catalyst_calendar, assess_catalyst, detect_shock,
)

SETTINGS = {
    "caution_confidence_penalty": 15,
    "embargo_confidence_penalty": 25,
    "earnings_confidence_penalty": 10,
    "shock_multi_sector_threshold": 8,
    "shock_pause_threshold": 2.0,
    "shock_regime_recheck_threshold": 3.0,
}


class TestScheduledCatalysts:

    def _dummy_prices(self):
        np.random.seed(700)
        dates = pd.bdate_range(end="2026-03-18", periods=60)
        tickers = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]
        data = {}
        for t in tickers:
            p = [100.0]
            for _ in range(59):
                p.append(p[-1] * (1 + np.random.normal(0.0005, 0.01)))
            data[t] = p
        return pd.DataFrame(data, index=dates)

    def test_fomc_day_is_embargo(self):
        catalysts = load_catalyst_calendar()
        prices = self._dummy_prices()
        result = assess_catalyst("2026-03-18", prices, catalysts,
                                  catalyst_settings=SETTINGS, shock_settings=SETTINGS)
        assert result.action == CatalystAction.EMBARGO

    def test_no_catalyst_is_clear(self):
        catalysts = load_catalyst_calendar()
        prices = self._dummy_prices()
        result = assess_catalyst("2026-02-15", prices, catalysts,
                                  catalyst_settings=SETTINGS, shock_settings=SETTINGS)
        assert result.action == CatalystAction.CLEAR

    def test_embargo_returns_negative_confidence(self):
        catalysts = load_catalyst_calendar()
        prices = self._dummy_prices()
        result = assess_catalyst("2026-03-18", prices, catalysts,
                                  catalyst_settings=SETTINGS, shock_settings=SETTINGS)
        assert result.confidence_modifier < 0


class TestShockDetection:

    def _make_prices(self, n=60, shock_day_returns=None):
        np.random.seed(500)
        dates = pd.bdate_range(end="2026-03-18", periods=n)
        tickers = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]
        data = {}
        for t in tickers:
            p = [100.0]
            for i in range(n):
                p.append(p[-1] * (1 + np.random.normal(0.0005, 0.01)))
            data[t] = p[1:]
        prices = pd.DataFrame(data, index=dates)
        # Apply shock on last day
        if shock_day_returns:
            for t, ret in shock_day_returns.items():
                if t in prices.columns:
                    prices.iloc[-1, prices.columns.get_loc(t)] = prices.iloc[-2, prices.columns.get_loc(t)] * (1 + ret)
        return prices

    def test_broad_selloff_detected(self):
        # 9 of 11 sectors down >3%
        shock_rets = {t: -0.04 for t in ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY"]}
        prices = self._make_prices(shock_day_returns=shock_rets)
        result = detect_shock(prices, shock_settings=SETTINGS)
        assert result.shock_type == ShockType.BROAD_SELLOFF

    def test_normal_day_no_shock(self):
        prices = self._make_prices()
        result = detect_shock(prices, shock_settings=SETTINGS)
        assert result.shock_type == ShockType.NONE

    def test_sector_dislocation_detected(self):
        # XLE +8%, rest flat
        shock_rets = {"XLE": 0.08}
        prices = self._make_prices(shock_day_returns=shock_rets)
        result = detect_shock(prices, shock_settings=SETTINGS)
        assert result.shock_type in (ShockType.SECTOR_DISLOCATION, ShockType.NONE)


class TestEmbargoTiming:

    def test_day_before_fomc_is_embargo(self):
        from engine.catalyst_gate import load_catalyst_calendar, check_scheduled_catalyst
        catalysts = load_catalyst_calendar()
        action, name, _ = check_scheduled_catalyst("2026-03-17", catalysts)
        assert action == CatalystAction.EMBARGO

    def test_fomc_day_is_embargo(self):
        from engine.catalyst_gate import load_catalyst_calendar, check_scheduled_catalyst
        catalysts = load_catalyst_calendar()
        action, name, _ = check_scheduled_catalyst("2026-03-18", catalysts)
        assert action == CatalystAction.EMBARGO

    def test_day_after_fomc_is_caution(self):
        from engine.catalyst_gate import load_catalyst_calendar, check_scheduled_catalyst
        catalysts = load_catalyst_calendar()
        action, name, _ = check_scheduled_catalyst("2026-03-19", catalysts)
        assert action == CatalystAction.CAUTION

    def test_two_days_after_fomc_is_clear(self):
        from engine.catalyst_gate import load_catalyst_calendar, check_scheduled_catalyst
        catalysts = load_catalyst_calendar()
        action, name, _ = check_scheduled_catalyst("2026-03-20", catalysts)
        assert action == CatalystAction.CLEAR

    def test_day_before_cpi_is_embargo(self):
        from engine.catalyst_gate import load_catalyst_calendar, check_scheduled_catalyst
        catalysts = load_catalyst_calendar()
        action, name, _ = check_scheduled_catalyst("2026-04-09", catalysts)
        assert action == CatalystAction.EMBARGO

    def test_cpi_day_is_embargo(self):
        from engine.catalyst_gate import load_catalyst_calendar, check_scheduled_catalyst
        catalysts = load_catalyst_calendar()
        action, name, _ = check_scheduled_catalyst("2026-04-10", catalysts)
        assert action == CatalystAction.EMBARGO

    def test_normal_day_is_clear(self):
        from engine.catalyst_gate import load_catalyst_calendar, check_scheduled_catalyst
        catalysts = load_catalyst_calendar()
        action, _, _ = check_scheduled_catalyst("2026-02-15", catalysts)
        assert action == CatalystAction.CLEAR
