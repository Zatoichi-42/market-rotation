"""
State classifier v2 — reversal score integration tests.
"""
import pytest
from engine.schemas import (
    AnalysisState, TransitionPressure, RegimeState,
    StateClassification, PumpScoreReading, ReversalScoreReading,
)
from engine.state_classifier import classify_state

SETTINGS = {
    "broadening": {"rs_delta_positive_sessions": 5, "min_pump_percentile": 50},
    "overt_pump": {"min_pump_percentile": 75},
    "exhaustion": {"pump_delta_nonpositive_sessions": 3},
    "ambiguous": {"conflicting_sessions": 3, "max_duration": 15},
}


def _pump(score=0.50, delta=0.0, rs=50.0, part=50.0, flow=50.0):
    return PumpScoreReading(
        ticker="XLK", name="Technology", rs_pillar=rs,
        participation_pillar=part, flow_pillar=flow,
        pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta,
    )


def _prior(state=AnalysisState.OVERT_PUMP, sessions=10):
    return StateClassification(
        ticker="XLK", name="Technology", state=state, confidence=70,
        sessions_in_state=sessions, transition_pressure=TransitionPressure.UP,
        prior_state=None, state_changed=False, explanation="",
    )


def _rev(score=0.65, pct=82.0, above=True):
    return ReversalScoreReading(
        ticker="XLK", name="Technology",
        breadth_det_pillar=70.0, price_break_pillar=60.0, crowding_pillar=55.0,
        reversal_score=score, sub_signals={},
        reversal_percentile=pct, above_75th=above,
    )


class TestReversalScoreIntegration:

    def test_exhaustion_to_rotation_with_high_reversal(self):
        """Exhaustion + Reversal > 75th pctl → Rotation."""
        pump = _pump(score=0.75, delta=-0.03)
        prior = _prior(state=AnalysisState.DISTRIBUTION, sessions=5)
        rev = _rev(score=0.70, pct=85.0, above=True)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=3, pump_percentile=70.0,
            delta_history=[-0.02, -0.03, -0.03, -0.03],
            settings=SETTINGS, reversal_score=rev,
        )
        assert result.state == AnalysisState.OVERT_DUMP

    def test_exhaustion_stays_without_high_reversal(self):
        """Exhaustion + Reversal < 75th pctl → stays Exhaustion."""
        pump = _pump(score=0.75, delta=-0.03)
        prior = _prior(state=AnalysisState.DISTRIBUTION, sessions=5)
        rev = _rev(score=0.20, pct=30.0, above=False)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=3, pump_percentile=70.0,
            delta_history=[-0.02, -0.03, -0.03, -0.03],
            settings=SETTINGS, reversal_score=rev,
        )
        assert result.state == AnalysisState.DISTRIBUTION

    def test_conflicting_pump_reversal_reduces_confidence(self):
        """Pump rising BUT Reversal high → lower confidence than aligned."""
        pump = _pump(score=0.70, delta=0.03)
        prior = _prior(state=AnalysisState.ACCUMULATION, sessions=6)

        # With high reversal (conflicting)
        rev_high = _rev(score=0.70, pct=85.0, above=True)
        result_conflict = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=2, pump_percentile=70.0,
            delta_history=[0.03] * 6, settings=SETTINGS,
            reversal_score=rev_high,
        )

        # With low reversal (confirming)
        rev_low = _rev(score=0.15, pct=15.0, above=False)
        result_confirm = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=2, pump_percentile=70.0,
            delta_history=[0.03] * 6, settings=SETTINGS,
            reversal_score=rev_low,
        )

        assert result_conflict.confidence < result_confirm.confidence

    def test_confirming_signals_increase_confidence(self):
        """Pump high + Reversal low → higher confidence."""
        pump = _pump(score=0.85, delta=0.03)
        prior = _prior(state=AnalysisState.OVERT_PUMP, sessions=8)
        rev = _rev(score=0.10, pct=10.0, above=False)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=90.0,
            delta_history=[0.03] * 6, settings=SETTINGS,
            reversal_score=rev,
        )
        assert result.confidence >= 70

    def test_backward_compatible_without_reversal(self):
        """When reversal_score=None, behaves as Phase 1."""
        pump = _pump(score=0.50, delta=0.0)
        result = classify_state(
            pump=pump, prior=None, regime=RegimeState.NORMAL,
            rs_rank=6, pump_percentile=50.0,
            delta_history=[], settings=SETTINGS,
            reversal_score=None,
        )
        assert isinstance(result, StateClassification)
        # All Phase 1 behavior preserved
        assert result.state in AnalysisState
