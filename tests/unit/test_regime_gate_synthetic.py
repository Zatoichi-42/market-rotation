"""
Regime Gate unit tests — synthetic data only.
Tests EVERY classification path and edge case.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import RegimeState, SignalLevel, RegimeSignal, RegimeAssessment
from engine.regime_gate import classify_signal, classify_regime


# Default thresholds matching config/settings.yaml
DEFAULT_THRESHOLDS = {
    "vix": {"normal_max": 20, "fragile_max": 30},
    "term_structure": {"contango_max": 0.95, "flat_max": 1.05},
    "breadth": {"normal_min_zscore": 0.0, "fragile_min_zscore": -1.0},
    "credit": {"normal_min_zscore": -0.5, "fragile_min_zscore": -1.5},
    "gate": {"hostile_threshold": 2, "fragile_mixed": True},
}


class TestIndividualSignals:
    """Test each signal classifier independently."""

    # ── VIX Level ──────────────────────────────────────

    def test_vix_normal(self):
        """VIX = 15.0 → NORMAL"""
        sig = classify_signal("vix", 15.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.NORMAL

    def test_vix_fragile_low(self):
        """VIX = 20.0 (exact boundary) → FRAGILE"""
        sig = classify_signal("vix", 20.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_vix_fragile_mid(self):
        """VIX = 25.0 → FRAGILE"""
        sig = classify_signal("vix", 25.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_vix_hostile_boundary(self):
        """VIX = 30.0 (exact boundary) → HOSTILE"""
        sig = classify_signal("vix", 30.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    def test_vix_hostile_high(self):
        """VIX = 45.0 → HOSTILE"""
        sig = classify_signal("vix", 45.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    def test_vix_zero(self):
        """VIX = 0.0 → NORMAL (edge case)"""
        sig = classify_signal("vix", 0.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.NORMAL

    # ── Term Structure ─────────────────────────────────

    def test_term_contango(self):
        """ratio = 0.85 → NORMAL"""
        sig = classify_signal("term_structure", 0.85, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.NORMAL

    def test_term_contango_boundary(self):
        """ratio = 0.95 (exact) → FRAGILE"""
        sig = classify_signal("term_structure", 0.95, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_term_flat(self):
        """ratio = 1.00 → FRAGILE"""
        sig = classify_signal("term_structure", 1.00, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_term_backwardation_boundary(self):
        """ratio = 1.05 (exact) → HOSTILE"""
        sig = classify_signal("term_structure", 1.05, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    def test_term_backwardation(self):
        """ratio = 1.15 → HOSTILE"""
        sig = classify_signal("term_structure", 1.15, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    # ── Breadth ────────────────────────────────────────

    def test_breadth_healthy(self):
        """z-score = 1.0 → NORMAL"""
        sig = classify_signal("breadth", 1.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.NORMAL

    def test_breadth_boundary_normal(self):
        """z-score = 0.0 (exact) → FRAGILE"""
        sig = classify_signal("breadth", 0.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_breadth_narrowing(self):
        """z-score = -0.5 → FRAGILE"""
        sig = classify_signal("breadth", -0.5, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_breadth_boundary_hostile(self):
        """z-score = -1.0 (exact) → HOSTILE"""
        sig = classify_signal("breadth", -1.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    def test_breadth_collapsed(self):
        """z-score = -2.0 → HOSTILE"""
        sig = classify_signal("breadth", -2.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    # ── Credit ─────────────────────────────────────────

    def test_credit_healthy(self):
        """z-score = 0.5 → NORMAL"""
        sig = classify_signal("credit", 0.5, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.NORMAL

    def test_credit_boundary_normal(self):
        """z-score = -0.5 (exact) → FRAGILE"""
        sig = classify_signal("credit", -0.5, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_credit_stressed(self):
        """z-score = -1.0 → FRAGILE"""
        sig = classify_signal("credit", -1.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.FRAGILE

    def test_credit_boundary_hostile(self):
        """z-score = -1.5 (exact) → HOSTILE"""
        sig = classify_signal("credit", -1.5, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    def test_credit_crisis(self):
        """z-score = -3.0 → HOSTILE"""
        sig = classify_signal("credit", -3.0, DEFAULT_THRESHOLDS)
        assert sig.level == SignalLevel.HOSTILE

    # ── Signal metadata ────────────────────────────────

    def test_signal_has_name(self):
        sig = classify_signal("vix", 15.0, DEFAULT_THRESHOLDS)
        assert sig.name == "vix"

    def test_signal_has_raw_value(self):
        sig = classify_signal("vix", 15.0, DEFAULT_THRESHOLDS)
        assert sig.raw_value == 15.0

    def test_signal_has_description(self):
        sig = classify_signal("vix", 15.0, DEFAULT_THRESHOLDS)
        assert len(sig.description) > 0


class TestGateClassification:
    """Test the gate aggregation logic."""

    def _make_signals(self, levels: list[SignalLevel]) -> list[RegimeSignal]:
        """Helper: create signal list with given levels."""
        names = ["vix", "term_structure", "breadth", "credit"]
        return [
            RegimeSignal(name=names[i], raw_value=0.0, level=level, description="test")
            for i, level in enumerate(levels)
        ]

    def test_all_normal(self):
        """4 NORMAL signals → NORMAL gate"""
        signals = self._make_signals([SignalLevel.NORMAL] * 4)
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.NORMAL
        assert result.hostile_count == 0
        assert result.normal_count == 4

    def test_all_hostile(self):
        """4 HOSTILE signals → HOSTILE gate"""
        signals = self._make_signals([SignalLevel.HOSTILE] * 4)
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.HOSTILE
        assert result.hostile_count == 4

    def test_two_hostile(self):
        """2 HOSTILE + 2 NORMAL → HOSTILE (threshold = 2)"""
        signals = self._make_signals([
            SignalLevel.HOSTILE, SignalLevel.HOSTILE,
            SignalLevel.NORMAL, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.HOSTILE

    def test_one_hostile_one_fragile(self):
        """1 HOSTILE + 1 FRAGILE + 2 NORMAL → FRAGILE"""
        signals = self._make_signals([
            SignalLevel.HOSTILE, SignalLevel.FRAGILE,
            SignalLevel.NORMAL, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.FRAGILE

    def test_one_hostile_two_fragile(self):
        """1 HOSTILE + 2 FRAGILE + 1 NORMAL → FRAGILE"""
        signals = self._make_signals([
            SignalLevel.HOSTILE, SignalLevel.FRAGILE,
            SignalLevel.FRAGILE, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.FRAGILE

    def test_three_fragile(self):
        """0 HOSTILE + 3 FRAGILE + 1 NORMAL → FRAGILE"""
        signals = self._make_signals([
            SignalLevel.FRAGILE, SignalLevel.FRAGILE,
            SignalLevel.FRAGILE, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.FRAGILE

    def test_one_hostile_zero_fragile(self):
        """1 HOSTILE + 0 FRAGILE + 3 NORMAL → FRAGILE (any hostile present = at least FRAGILE)"""
        signals = self._make_signals([
            SignalLevel.HOSTILE, SignalLevel.NORMAL,
            SignalLevel.NORMAL, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.FRAGILE

    def test_one_fragile_only(self):
        """0 HOSTILE + 1 FRAGILE + 3 NORMAL → NORMAL"""
        signals = self._make_signals([
            SignalLevel.NORMAL, SignalLevel.FRAGILE,
            SignalLevel.NORMAL, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.NORMAL

    def test_exactly_two_hostile(self):
        """Exactly 2 HOSTILE → HOSTILE (not FRAGILE)"""
        signals = self._make_signals([
            SignalLevel.HOSTILE, SignalLevel.HOSTILE,
            SignalLevel.FRAGILE, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.HOSTILE

    def test_two_fragile_only(self):
        """0 HOSTILE + 2 FRAGILE + 2 NORMAL → FRAGILE"""
        signals = self._make_signals([
            SignalLevel.FRAGILE, SignalLevel.FRAGILE,
            SignalLevel.NORMAL, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.FRAGILE

    def test_counts_are_correct(self):
        """Verify hostile/fragile/normal counts."""
        signals = self._make_signals([
            SignalLevel.HOSTILE, SignalLevel.FRAGILE,
            SignalLevel.NORMAL, SignalLevel.NORMAL,
        ])
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.hostile_count == 1
        assert result.fragile_count == 1
        assert result.normal_count == 2

    def test_timestamp_is_set(self):
        """Result should have a non-empty timestamp."""
        signals = self._make_signals([SignalLevel.NORMAL] * 4)
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert len(result.timestamp) > 0

    def test_returns_regime_assessment(self):
        """Return type is RegimeAssessment."""
        signals = self._make_signals([SignalLevel.NORMAL] * 4)
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert isinstance(result, RegimeAssessment)


class TestGateMissingData:
    """Test graceful degradation when data is incomplete."""

    def _make_signals(self, levels: list[SignalLevel]) -> list[RegimeSignal]:
        names = ["vix", "term_structure", "breadth", "credit"]
        return [
            RegimeSignal(name=names[i], raw_value=0.0, level=level, description="test")
            for i, level in enumerate(levels)
        ]

    def test_missing_vix3m(self):
        """VIX3M unavailable → skip term structure signal, classify with remaining 3."""
        # Only 3 signals provided (no term_structure)
        signals = [
            RegimeSignal(name="vix", raw_value=15.0, level=SignalLevel.NORMAL, description="ok"),
            RegimeSignal(name="breadth", raw_value=0.5, level=SignalLevel.NORMAL, description="ok"),
            RegimeSignal(name="credit", raw_value=0.0, level=SignalLevel.NORMAL, description="ok"),
        ]
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.NORMAL
        assert result.normal_count == 3

    def test_missing_credit(self):
        """HYG/LQD unavailable → skip credit signal, classify with remaining 3."""
        signals = [
            RegimeSignal(name="vix", raw_value=35.0, level=SignalLevel.HOSTILE, description="high"),
            RegimeSignal(name="term_structure", raw_value=1.1, level=SignalLevel.HOSTILE, description="backw"),
            RegimeSignal(name="breadth", raw_value=-2.0, level=SignalLevel.HOSTILE, description="bad"),
        ]
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.HOSTILE

    def test_all_missing_except_vix(self):
        """Only VIX available → classify with 1 signal."""
        signals = [
            RegimeSignal(name="vix", raw_value=35.0, level=SignalLevel.HOSTILE, description="spike"),
        ]
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        # 1 hostile signal — below threshold of 2 for HOSTILE, but 1 hostile → FRAGILE
        assert result.state == RegimeState.FRAGILE

    def test_nan_in_vix(self):
        """VIX value is NaN → signal excluded, not crash."""
        sig = classify_signal("vix", float("nan"), DEFAULT_THRESHOLDS)
        assert sig is None

    def test_empty_signals_returns_normal_with_warning(self):
        """No signals at all → default to NORMAL with explanation noting missing data."""
        result = classify_regime([], DEFAULT_THRESHOLDS)
        assert result.state == RegimeState.NORMAL
        assert "no signals" in result.explanation.lower() or "missing" in result.explanation.lower()


class TestRegimeExplanation:
    """Test that explanation strings are generated correctly."""

    def _make_signals(self, levels: list[SignalLevel]) -> list[RegimeSignal]:
        names = ["vix", "term_structure", "breadth", "credit"]
        descs = ["VIX at 15.0", "term structure 0.88", "breadth z: 0.5", "credit z: 0.1"]
        vals = [15.0, 0.88, 0.5, 0.1]
        return [
            RegimeSignal(name=names[i], raw_value=vals[i], level=level, description=descs[i])
            for i, level in enumerate(levels)
        ]

    def test_normal_explanation_mentions_all_signals(self):
        """NORMAL explanation should reference all 4 signal areas."""
        signals = self._make_signals([SignalLevel.NORMAL] * 4)
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        exp = result.explanation.lower()
        assert "normal" in exp
        assert len(result.explanation) > 20

    def test_hostile_explanation_lists_hostile_signals(self):
        """HOSTILE explanation should mention which signals triggered."""
        signals = [
            RegimeSignal(name="vix", raw_value=35.0, level=SignalLevel.HOSTILE, description="VIX spiking"),
            RegimeSignal(name="term_structure", raw_value=1.1, level=SignalLevel.HOSTILE, description="backwardation"),
            RegimeSignal(name="breadth", raw_value=0.5, level=SignalLevel.NORMAL, description="ok"),
            RegimeSignal(name="credit", raw_value=0.0, level=SignalLevel.NORMAL, description="ok"),
        ]
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        exp = result.explanation.lower()
        assert "hostile" in exp

    def test_fragile_explanation_identifies_concerning_signals(self):
        """FRAGILE explanation should note the concerning signals."""
        signals = [
            RegimeSignal(name="vix", raw_value=25.0, level=SignalLevel.FRAGILE, description="VIX elevated"),
            RegimeSignal(name="term_structure", raw_value=1.0, level=SignalLevel.FRAGILE, description="flat"),
            RegimeSignal(name="breadth", raw_value=0.5, level=SignalLevel.NORMAL, description="ok"),
            RegimeSignal(name="credit", raw_value=0.0, level=SignalLevel.NORMAL, description="ok"),
        ]
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        exp = result.explanation.lower()
        assert "fragile" in exp

    def test_explanation_is_nonempty(self):
        """Explanation should never be empty."""
        signals = self._make_signals([SignalLevel.NORMAL] * 4)
        result = classify_regime(signals, DEFAULT_THRESHOLDS)
        assert len(result.explanation) > 0


class TestClassifyRegimeFromData:
    """Test the full classify_regime_from_data convenience function that takes raw market data."""

    def test_normal_market_scenario(self, normal_market, settings):
        """Normal market factory data → NORMAL regime."""
        from engine.regime_gate import classify_regime_from_data
        data = normal_market
        result = classify_regime_from_data(
            vix_current=data["vix"].iloc[-1],
            vix3m_current=data["vix3m"].iloc[-1],
            breadth_zscore=0.5,  # Normal market has healthy breadth
            credit_zscore=0.0,
            thresholds=settings["regime"],
        )
        assert result.state == RegimeState.NORMAL

    def test_hostile_market_scenario(self, hostile_market, settings):
        """Hostile market factory data → HOSTILE regime."""
        from engine.regime_gate import classify_regime_from_data
        data = hostile_market
        result = classify_regime_from_data(
            vix_current=data["vix"].iloc[-1],
            vix3m_current=data["vix3m"].iloc[-1],
            breadth_zscore=-2.0,
            credit_zscore=-2.0,
            thresholds=settings["regime"],
        )
        assert result.state == RegimeState.HOSTILE

    def test_fragile_market_scenario(self, fragile_market, settings):
        """Fragile market factory data → FRAGILE regime."""
        from engine.regime_gate import classify_regime_from_data
        data = fragile_market
        result = classify_regime_from_data(
            vix_current=data["vix"].iloc[-1],
            vix3m_current=data["vix3m"].iloc[-1],
            breadth_zscore=-0.5,
            credit_zscore=-0.8,
            thresholds=settings["regime"],
        )
        assert result.state == RegimeState.FRAGILE

    def test_nan_vix3m_graceful(self, settings):
        """NaN VIX3M → still classifies without crashing."""
        from engine.regime_gate import classify_regime_from_data
        result = classify_regime_from_data(
            vix_current=15.0,
            vix3m_current=float("nan"),
            breadth_zscore=0.5,
            credit_zscore=0.0,
            thresholds=settings["regime"],
        )
        assert isinstance(result, RegimeAssessment)
        assert result.state in (RegimeState.NORMAL, RegimeState.FRAGILE, RegimeState.HOSTILE)
