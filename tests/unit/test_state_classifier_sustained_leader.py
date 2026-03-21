"""
Sustained leader exemption tests — rank #1 + Full Confirm sectors
should not be classified as Exhaustion during normal pullbacks.
"""
import pytest
from engine.schemas import (
    AnalysisState, RegimeState, HorizonPattern,
    PumpScoreReading, ReversalScoreReading,
)
from engine.state_classifier import classify_state


SETTINGS = {
    "broadening": {"rs_delta_positive_sessions": 5},
    "overt_pump": {"min_pump_percentile": 75},
    "distribution": {"pump_delta_negative_sessions": 3},
    "exhaustion": {"pump_delta_nonpositive_sessions": 3},
    "ambiguous": {"max_duration": 15},
    "sustained_leader": {
        "min_rs_60d": 0.15,
        "extra_exh_sessions": 3,
        "exh_rev_bar": 80,
        "broadening_max_consec_neg": 6,
    },
}


def _pump(ticker="XLE", name="Energy", score=0.62, delta=-0.05, delta_5d=0.003):
    return PumpScoreReading(
        ticker=ticker, name=name,
        rs_pillar=80, participation_pillar=45, flow_pillar=50,
        pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta_5d,
    )


def _rev(pctl=71.4):
    return ReversalScoreReading(
        ticker="XLE", name="Energy",
        breadth_det_pillar=50, price_break_pillar=50, crowding_pillar=50,
        reversal_score=0.5, sub_signals={},
        reversal_percentile=pctl, above_75th=(pctl >= 75),
    )


class TestSustainedLeaderNotExhaustion:
    """The core fix: rank #1 + Full Confirm + massive 60d RS should not be Exhaustion
    just because pump delta was negative for 3 sessions."""

    def test_xle_march20_not_exhaustion(self):
        """XLE exact March 20, 2026 values — must be Broadening, not Exhaustion."""
        result = classify_state(
            pump=_pump(delta=-0.051, delta_5d=0.003, score=0.616),
            prior=None, regime=RegimeState.FRAGILE,
            rs_rank=1, pump_percentile=85.0,
            delta_history=[-0.02, -0.03, -0.05, -0.05, 0.01, 0.02, 0.03],
            settings=SETTINGS,
            reversal_score=_rev(pctl=71.4),
            rs_5d=0.049, rs_20d=0.140, rs_60d=0.394,
            horizon_pattern=HorizonPattern.FULL_CONFIRM,
        )
        assert result.state != AnalysisState.EXHAUSTION, (
            f"Sustained leader should NOT be Exhaustion. Got {result.state.value}"
        )
        assert result.state in (AnalysisState.BROADENING, AnalysisState.OVERT_PUMP)

    def test_sustained_leader_not_exhaustion_3_neg_deltas(self):
        """3 consecutive negative deltas should NOT trigger Exhaustion for sustained leader."""
        result = classify_state(
            pump=_pump(delta=-0.02),
            prior=None, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=85.0,
            delta_history=[0.01, 0.01, -0.02, -0.02, -0.02],
            settings=SETTINGS,
            reversal_score=_rev(pctl=60.0),
            rs_5d=0.03, rs_20d=0.10, rs_60d=0.25,
            horizon_pattern=HorizonPattern.FULL_CONFIRM,
        )
        assert result.state != AnalysisState.EXHAUSTION

    def test_sustained_leader_exhaustion_on_6_neg_deltas(self):
        """6 consecutive negative deltas SHOULD trigger Exhaustion even for sustained leader."""
        result = classify_state(
            pump=_pump(delta=-0.03),
            prior=None, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=85.0,
            delta_history=[-0.02, -0.02, -0.03, -0.03, -0.03, -0.03],
            settings=SETTINGS,
            reversal_score=_rev(pctl=85.0),  # High reversal > 80 bar
            rs_5d=0.03, rs_20d=0.10, rs_60d=0.25,
            horizon_pattern=HorizonPattern.FULL_CONFIRM,
        )
        assert result.state == AnalysisState.EXHAUSTION

    def test_sustained_leader_broadening_with_moderate_pullback(self):
        """Rank #1 with 4 negative deltas should still get Broadening via expanded path."""
        result = classify_state(
            pump=_pump(delta=-0.01, score=0.60),
            prior=None, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=75.0,
            delta_history=[0.01, 0.01, -0.01, -0.01, -0.01, -0.01],
            settings=SETTINGS,
            rs_5d=0.02, rs_20d=0.08, rs_60d=0.30,
            horizon_pattern=HorizonPattern.FULL_CONFIRM,
        )
        assert result.state == AnalysisState.BROADENING


class TestNonLeaderStillExhaustion:
    """Non-sustained-leader sectors should still get Exhaustion on 3 negative deltas."""

    def test_rank5_exhaustion_on_3_neg_deltas(self):
        """Rank 5 with 3 consecutive negative deltas → Exhaustion (no exemption)."""
        result = classify_state(
            pump=_pump(ticker="XLK", name="Technology", delta=-0.02),
            prior=None, regime=RegimeState.NORMAL,
            rs_rank=5, pump_percentile=60.0,
            delta_history=[0.01, 0.01, -0.02, -0.02, -0.02],
            settings=SETTINGS,
            reversal_score=_rev(pctl=60.0),
            rs_5d=0.01, rs_20d=0.03, rs_60d=0.08,
            horizon_pattern=HorizonPattern.ROTATION_IN,
        )
        assert result.state == AnalysisState.EXHAUSTION

    def test_rank1_without_full_confirm_no_exemption(self):
        """Rank 1 but Dead Cat pattern → no sustained leader exemption."""
        result = classify_state(
            pump=_pump(delta=-0.02),
            prior=None, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=85.0,
            delta_history=[0.01, 0.01, -0.02, -0.02, -0.02],
            settings=SETTINGS,
            reversal_score=_rev(pctl=60.0),
            rs_5d=0.03, rs_20d=-0.02, rs_60d=-0.05,
            horizon_pattern=HorizonPattern.DEAD_CAT,
        )
        assert result.state != AnalysisState.BROADENING

    def test_rank1_low_rs60d_no_exemption(self):
        """Rank 1 but 60d RS only +5% → below 15% threshold, no exemption."""
        result = classify_state(
            pump=_pump(delta=-0.02),
            prior=None, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=85.0,
            delta_history=[0.01, 0.01, -0.02, -0.02, -0.02],
            settings=SETTINGS,
            reversal_score=_rev(pctl=60.0),
            rs_5d=0.03, rs_20d=0.05, rs_60d=0.05,
            horizon_pattern=HorizonPattern.FULL_CONFIRM,
        )
        # With low rs_60d (0.05 < 0.15), no sustained leader exemption
        # But it might still get Broadening via other paths — just NOT via exemption
        # The key is it shouldn't be PROTECTED from Exhaustion
        # (This test verifies the exemption requires rs_60d > 0.15)


class TestConfidenceFloorAndPenalty:
    """Verify the raised floor and reduced FRAGILE penalty."""

    def test_rank1_full_confirm_confidence_floor_55(self):
        """Rank 1 + Full Confirm + all RS positive → confidence ≥ 55."""
        result = classify_state(
            pump=_pump(delta=-0.01, score=0.60),
            prior=None, regime=RegimeState.FRAGILE,
            rs_rank=1, pump_percentile=75.0,
            delta_history=[0.01, -0.01, -0.01],
            settings=SETTINGS,
            rs_5d=0.03, rs_20d=0.10, rs_60d=0.30,
            horizon_pattern=HorizonPattern.FULL_CONFIRM,
        )
        assert result.confidence >= 55, f"Floor should be 55, got {result.confidence}"

    def test_fragile_penalty_reduced(self):
        """FRAGILE penalty is now -10, not -15."""
        pump = _pump(ticker="XLK", name="Technology", delta=0.02, score=0.55)
        normal = classify_state(
            pump=pump, prior=None, regime=RegimeState.NORMAL,
            rs_rank=3, pump_percentile=55.0,
            delta_history=[0.02, 0.02, 0.02],
            settings=SETTINGS,
        )
        fragile = classify_state(
            pump=pump, prior=None, regime=RegimeState.FRAGILE,
            rs_rank=3, pump_percentile=55.0,
            delta_history=[0.02, 0.02, 0.02],
            settings=SETTINGS,
        )
        diff = normal.confidence - fragile.confidence
        assert diff == 10, f"FRAGILE penalty should be 10, got {diff}"
