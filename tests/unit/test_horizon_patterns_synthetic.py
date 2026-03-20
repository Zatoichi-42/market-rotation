"""
Horizon pattern classification tests — cross-horizon divergence patterns.
"""
import pytest
from engine.schemas import HorizonPattern
from engine.horizon_patterns import (
    classify_horizon_pattern,
    compute_horizon_conviction,
    classify_horizon_reading,
    classify_all_horizon_patterns,
)


class TestHorizonPatternClassification:
    """TEST-HRZ-01 through TEST-HRZ-08: Pattern from RS signs."""

    def test_full_confirm(self):
        """TEST-HRZ-01: ↑↑↑ → FULL_CONFIRM."""
        assert classify_horizon_pattern(0.02, 0.05, 0.10) == HorizonPattern.FULL_CONFIRM

    def test_rotation_in(self):
        """TEST-HRZ-02: ↑↑↓ → ROTATION_IN."""
        assert classify_horizon_pattern(0.02, 0.03, -0.05) == HorizonPattern.ROTATION_IN

    def test_rotation_out(self):
        """TEST-HRZ-03: ↓↓↑ → ROTATION_OUT."""
        assert classify_horizon_pattern(-0.02, -0.03, 0.08) == HorizonPattern.ROTATION_OUT

    def test_full_reject(self):
        """TEST-HRZ-04: ↓↓↓ → FULL_REJECT."""
        assert classify_horizon_pattern(-0.02, -0.05, -0.10) == HorizonPattern.FULL_REJECT

    def test_dead_cat(self):
        """TEST-HRZ-05: ↑↓↓ → DEAD_CAT."""
        assert classify_horizon_pattern(0.02, -0.03, -0.08) == HorizonPattern.DEAD_CAT

    def test_healthy_dip(self):
        """TEST-HRZ-06: ↓↑↑ → HEALTHY_DIP."""
        assert classify_horizon_pattern(-0.01, 0.03, 0.08) == HorizonPattern.HEALTHY_DIP

    def test_flat_5d_positive_20d_60d(self):
        """TEST-HRZ-07: ~↑↑ → HEALTHY_DIP (5d flat → conservative)."""
        assert classify_horizon_pattern(0.001, 0.03, 0.08) == HorizonPattern.HEALTHY_DIP

    def test_all_flat(self):
        """TEST-HRZ-08: ~~~ → NO_PATTERN (all flat)."""
        assert classify_horizon_pattern(0.001, -0.001, 0.001) == HorizonPattern.NO_PATTERN


class TestHorizonPatternEnumCompleteness:
    """TEST-HRZ-09: All 7 patterns are distinct enum values."""

    def test_seven_distinct_patterns(self):
        assert len(HorizonPattern) == 7
        values = [p.value for p in HorizonPattern]
        assert len(set(values)) == 7


class TestHorizonReadingFlags:
    """TEST-HRZ-10 through TEST-HRZ-13: Boolean flags on HorizonReading."""

    def test_dead_cat_is_trap(self):
        """TEST-HRZ-10: DEAD_CAT has is_trap=True, is_entry_zone=False."""
        hr = classify_horizon_reading("XLY", "Cons Disc", 0.02, -0.03, -0.08)
        assert hr.is_trap is True
        assert hr.is_entry_zone is False

    def test_healthy_dip_is_entry_zone(self):
        """TEST-HRZ-11: HEALTHY_DIP has is_entry_zone=True, is_trap=False."""
        hr = classify_horizon_reading("XLK", "Tech", -0.01, 0.03, 0.08)
        assert hr.is_entry_zone is True
        assert hr.is_trap is False

    def test_rotation_in_is_rotation_signal(self):
        """TEST-HRZ-12: ROTATION_IN has is_rotation_signal=True."""
        hr = classify_horizon_reading("XLK", "Tech", 0.02, 0.03, -0.05)
        assert hr.is_rotation_signal is True

    def test_rotation_out_is_rotation_signal(self):
        """TEST-HRZ-13: ROTATION_OUT has is_rotation_signal=True."""
        hr = classify_horizon_reading("XLB", "Materials", -0.02, -0.03, 0.08)
        assert hr.is_rotation_signal is True


class TestHorizonConviction:
    """TEST-HRZ-14 through TEST-HRZ-16: Conviction scoring."""

    def test_large_rs_higher_conviction(self):
        """TEST-HRZ-14: FULL_CONFIRM with large RS > conviction with small RS."""
        large = compute_horizon_conviction(0.10, 0.15, 0.20, HorizonPattern.FULL_CONFIRM)
        small = compute_horizon_conviction(0.005, 0.006, 0.007, HorizonPattern.FULL_CONFIRM)
        assert large > small

    def test_full_confirm_high_conviction(self):
        """TEST-HRZ-15: FULL_CONFIRM with strong RS → ≥ 80."""
        c = compute_horizon_conviction(0.10, 0.15, 0.20, HorizonPattern.FULL_CONFIRM)
        assert c >= 80

    def test_no_pattern_low_conviction(self):
        """TEST-HRZ-16: NO_PATTERN conviction → ≤ 30."""
        c = compute_horizon_conviction(0.001, -0.001, 0.001, HorizonPattern.NO_PATTERN)
        assert c <= 30


class TestFlatHandling:
    """Edge cases for flat (~) horizons."""

    def test_two_flats_no_pattern(self):
        """2+ horizons flat → NO_PATTERN."""
        assert classify_horizon_pattern(0.001, 0.001, 0.05) == HorizonPattern.NO_PATTERN

    def test_flat_60d_with_positive_5d_20d(self):
        """↑↑~ → lean FULL_CONFIRM."""
        assert classify_horizon_pattern(0.02, 0.03, 0.001) == HorizonPattern.FULL_CONFIRM

    def test_flat_60d_with_negative_5d_20d(self):
        """↓↓~ → lean FULL_REJECT."""
        assert classify_horizon_pattern(-0.02, -0.03, 0.001) == HorizonPattern.FULL_REJECT

    def test_flat_20d_positive_5d_negative_60d(self):
        """↑~↓ → lean ROTATION_IN."""
        assert classify_horizon_pattern(0.02, 0.001, -0.05) == HorizonPattern.ROTATION_IN


class TestClassifyAllHorizonPatterns:
    """Test batch classification."""

    def test_returns_dict(self):
        from engine.schemas import RSReading
        rs = [RSReading("XLK", "Tech", 0.02, 0.03, 0.05, 0.01, 1, 0, 80.0)]
        result = classify_all_horizon_patterns(rs)
        assert "XLK" in result
        assert result["XLK"].pattern == HorizonPattern.FULL_CONFIRM

    def test_includes_industries(self):
        from engine.schemas import RSReading, IndustryRSReading, GroupType
        rs = [RSReading("XLK", "Tech", 0.02, 0.03, 0.05, 0.01, 1, 0, 80.0)]
        ind = [IndustryRSReading(
            "SMH", "Semis", "XLK", GroupType.INDUSTRY,
            -0.02, -0.03, -0.08, -0.01, 20.0,
            0.01, 0.01, 0.02, 0.005, 60.0, 20.0,
            5, 0, 1,
        )]
        result = classify_all_horizon_patterns(rs, ind)
        assert "XLK" in result
        assert "SMH" in result
        assert result["SMH"].pattern == HorizonPattern.FULL_REJECT


class TestXLYAntiRegression:
    """TEST-HRZ-ANTI-01: XLY March 20 values → NOT Accumulation via DEAD_CAT."""

    def test_xly_negative_rs_pattern(self):
        """XLY rs_5d=-0.007, rs_20d=-0.023, rs_60d=-0.067 → FULL_REJECT."""
        pattern = classify_horizon_pattern(-0.007, -0.023, -0.067)
        assert pattern == HorizonPattern.FULL_REJECT


class TestHorizonVetoInClassifier:
    """TEST-HRZ-VETO-01/02/03: Horizon pattern vetoes in state classifier."""

    def _make_pump(self, ticker="XLY", delta=0.02, delta_5d=0.01, score=0.45):
        from engine.schemas import PumpScoreReading
        return PumpScoreReading(
            ticker=ticker, name=ticker, rs_pillar=50.0,
            participation_pillar=50.0, flow_pillar=50.0,
            pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta_5d,
        )

    def _default_settings(self):
        return {
            "broadening": {"rs_delta_positive_sessions": 5},
            "overt_pump": {"min_pump_percentile": 75},
            "distribution": {"pump_delta_negative_sessions": 3},
            "exhaustion": {"pump_delta_nonpositive_sessions": 3},
            "ambiguous": {"max_duration": 15},
        }

    def test_dead_cat_vetoes_accumulation(self):
        """TEST-HRZ-VETO-01: DEAD_CAT + positive delta → NOT Accumulation."""
        from engine.state_classifier import classify_state
        from engine.schemas import RegimeState, AnalysisState, HorizonPattern
        pump = self._make_pump(delta=0.02, delta_5d=0.01)
        result = classify_state(
            pump=pump, prior=None, regime=RegimeState.NORMAL,
            rs_rank=5, pump_percentile=50.0,
            delta_history=[0.01, 0.02, 0.01, 0.02, 0.02],
            settings=self._default_settings(),
            horizon_pattern=HorizonPattern.DEAD_CAT,
        )
        assert result.state != AnalysisState.ACCUMULATION

    def test_dead_cat_vetoes_broadening(self):
        """TEST-HRZ-VETO-02: DEAD_CAT + positive delta → NOT Broadening."""
        from engine.state_classifier import classify_state
        from engine.schemas import RegimeState, AnalysisState, HorizonPattern
        pump = self._make_pump(delta=0.02, delta_5d=0.01)
        result = classify_state(
            pump=pump, prior=None, regime=RegimeState.NORMAL,
            rs_rank=5, pump_percentile=60.0,
            delta_history=[0.01, 0.02, 0.01, 0.02, 0.02, 0.01, 0.02],
            settings=self._default_settings(),
            horizon_pattern=HorizonPattern.DEAD_CAT,
        )
        assert result.state != AnalysisState.BROADENING

    def test_full_reject_vetoes_accumulation(self):
        """TEST-HRZ-VETO-03: FULL_REJECT + positive delta → NOT Accumulation."""
        from engine.state_classifier import classify_state
        from engine.schemas import RegimeState, AnalysisState, HorizonPattern
        pump = self._make_pump(delta=0.01, delta_5d=0.005)
        result = classify_state(
            pump=pump, prior=None, regime=RegimeState.NORMAL,
            rs_rank=5, pump_percentile=50.0,
            delta_history=[0.01, 0.01, 0.01],
            settings=self._default_settings(),
            horizon_pattern=HorizonPattern.FULL_REJECT,
        )
        assert result.state != AnalysisState.ACCUMULATION
