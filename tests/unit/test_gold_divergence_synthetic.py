"""
Gold/VIX Divergence unit tests — synthetic data only.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import SignalLevel, RegimeState
from engine.gold_divergence import compute_gold_vix_divergence, apply_gold_divergence_modifier


def _make_series(n=100, base=180.0, last_5d_return=0.0):
    """Make a price series with controlled 5d return."""
    dates = pd.bdate_range(end="2026-03-19", periods=n)
    prices = pd.Series(np.full(n, base), index=dates)
    # Set the -6th value to control 5d return
    if last_5d_return != 0:
        prices.iloc[-6] = base / (1 + last_5d_return)
    return prices


class TestGoldDivergenceClassification:
    """TEST-GOLD-01 through TEST-GOLD-04."""

    def test_gold01_margin_call_triggered(self):
        """TEST-GOLD-01: Gold -3%, SPY -3%, VIX 28 → HOSTILE"""
        gold = _make_series(last_5d_return=-0.03)
        spy = _make_series(base=450.0, last_5d_return=-0.03)
        reading = compute_gold_vix_divergence(gold, spy, 28.0)
        assert reading is not None
        assert reading.is_margin_call_regime == True
        assert reading.level == SignalLevel.HOSTILE

    def test_gold02_vix_below_threshold(self):
        """TEST-GOLD-02: Gold -3%, SPY -3%, VIX 22 → NOT triggered"""
        gold = _make_series(last_5d_return=-0.03)
        spy = _make_series(base=450.0, last_5d_return=-0.03)
        reading = compute_gold_vix_divergence(gold, spy, 22.0)
        assert reading is not None
        assert reading.is_margin_call_regime == False

    def test_gold03_gold_above_threshold(self):
        """TEST-GOLD-03: Gold -1%, SPY -3%, VIX 28 → NOT triggered"""
        gold = _make_series(last_5d_return=-0.01)
        spy = _make_series(base=450.0, last_5d_return=-0.03)
        reading = compute_gold_vix_divergence(gold, spy, 28.0)
        assert reading is not None
        assert reading.is_margin_call_regime == False

    def test_gold04_gold_up(self):
        """TEST-GOLD-04: Gold +2%, SPY -3%, VIX 28 → NOT triggered"""
        gold = _make_series(last_5d_return=0.02)
        spy = _make_series(base=450.0, last_5d_return=-0.03)
        reading = compute_gold_vix_divergence(gold, spy, 28.0)
        assert reading is not None
        assert reading.is_margin_call_regime == False
        assert reading.level == SignalLevel.NORMAL


class TestGoldDivergenceModifier:
    """TEST-GOLD-05 through TEST-GOLD-07."""

    def _make_hostile_reading(self):
        from engine.schemas import GoldDivergenceReading
        return GoldDivergenceReading(
            gold_5d_return=-0.03, spy_5d_return=-0.03, vix_level=28.0,
            is_margin_call_regime=True, level=SignalLevel.HOSTILE,
            description="test margin call",
        )

    def _make_normal_reading(self):
        from engine.schemas import GoldDivergenceReading
        return GoldDivergenceReading(
            gold_5d_return=0.02, spy_5d_return=-0.01, vix_level=18.0,
            is_margin_call_regime=False, level=SignalLevel.NORMAL,
            description="normal",
        )

    def test_gold05_gate_at_least_fragile(self):
        """TEST-GOLD-05: Margin-call triggered → gate is AT LEAST FRAGILE"""
        reading = self._make_hostile_reading()
        new_state, explanation = apply_gold_divergence_modifier(RegimeState.NORMAL, reading)
        assert new_state == RegimeState.FRAGILE

    def test_gold06_hostile_stays_hostile(self):
        """TEST-GOLD-06: Gate was already HOSTILE + margin-call → still HOSTILE"""
        reading = self._make_hostile_reading()
        new_state, explanation = apply_gold_divergence_modifier(RegimeState.HOSTILE, reading)
        assert new_state == RegimeState.HOSTILE

    def test_gold07_normal_to_fragile(self):
        """TEST-GOLD-07: Gate NORMAL + margin-call → FRAGILE"""
        reading = self._make_hostile_reading()
        new_state, explanation = apply_gold_divergence_modifier(RegimeState.NORMAL, reading)
        assert new_state == RegimeState.FRAGILE
        assert "MARGIN CALL" in explanation

    def test_gold08_none_reading(self):
        """TEST-GOLD-08: GLD data unavailable → modifier inactive"""
        new_state, explanation = apply_gold_divergence_modifier(RegimeState.NORMAL, None)
        assert new_state == RegimeState.NORMAL
        assert explanation == ""

    def test_normal_reading_no_change(self):
        """Normal reading does not modify gate."""
        reading = self._make_normal_reading()
        new_state, explanation = apply_gold_divergence_modifier(RegimeState.NORMAL, reading)
        assert new_state == RegimeState.NORMAL

    def test_fragile_stays_fragile(self):
        """TEST-GOLD-06 variant: Gate FRAGILE + margin-call → stays FRAGILE"""
        reading = self._make_hostile_reading()
        new_state, explanation = apply_gold_divergence_modifier(RegimeState.FRAGILE, reading)
        assert new_state == RegimeState.FRAGILE


class TestGracefulDegradation:
    def test_none_gold(self):
        reading = compute_gold_vix_divergence(None, pd.Series([450] * 100), 25.0)
        assert reading is None

    def test_none_spy(self):
        reading = compute_gold_vix_divergence(pd.Series([180] * 100), None, 25.0)
        assert reading is None

    def test_short_series(self):
        gold = pd.Series([180, 179, 178])
        spy = pd.Series([450, 449, 448])
        reading = compute_gold_vix_divergence(gold, spy, 25.0)
        assert reading is None

    def test_reading_fields(self):
        gold = _make_series()
        spy = _make_series(base=450.0)
        reading = compute_gold_vix_divergence(gold, spy, 20.0)
        assert reading is not None
        assert isinstance(reading.gold_5d_return, float)
        assert isinstance(reading.spy_5d_return, float)
        assert isinstance(reading.vix_level, float)
        assert isinstance(reading.is_margin_call_regime, bool)
        assert isinstance(reading.level, SignalLevel)
        assert isinstance(reading.description, str)
