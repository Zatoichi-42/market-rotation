"""
State classifier unit tests — synthetic data only.
Tests every state, every transition, confidence bands, edge cases.
"""
import pytest
from engine.schemas import (
    AnalysisState, TransitionPressure, RegimeState,
    StateClassification, PumpScoreReading,
)
from engine.state_classifier import classify_state, classify_all_sectors


SETTINGS = {
    "broadening": {"rs_delta_positive_sessions": 5, "min_pump_percentile": 50},
    "overt_pump": {"min_pump_percentile": 75},
    "exhaustion": {"pump_delta_nonpositive_sessions": 3},
    "ambiguous": {"conflicting_sessions": 3, "max_duration": 15},
}


def _pump(ticker="XLK", name="Technology", score=0.50, delta=0.0, delta_5d=0.0,
          rs=50.0, part=50.0, flow=50.0):
    return PumpScoreReading(
        ticker=ticker, name=name,
        rs_pillar=rs, participation_pillar=part, flow_pillar=flow,
        pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta_5d,
    )


def _prior(ticker="XLK", name="Technology", state=AnalysisState.ACCUMULATION,
           sessions=3, confidence=60, pressure=TransitionPressure.UP,
           prior_state=None):
    return StateClassification(
        ticker=ticker, name=name, state=state, confidence=confidence,
        sessions_in_state=sessions, transition_pressure=pressure,
        prior_state=prior_state, state_changed=False, explanation="",
    )


# ═══════════════════════════════════════════════════════
# STATE ASSIGNMENT
# ═══════════════════════════════════════════════════════

class TestStateAssignment:

    def test_accumulation(self):
        """Low-to-mid pump score + positive delta few sessions → Accumulation."""
        pump = _pump(score=0.35, delta=0.02, delta_5d=0.015)
        prior = _prior(state=AnalysisState.ACCUMULATION, sessions=3)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=7, pump_percentile=35.0,
            delta_history=[0.01, 0.02, 0.01, 0.02],
            settings=SETTINGS,
        )
        assert result.state == AnalysisState.ACCUMULATION

    def test_broadening(self):
        """Pump delta positive 5+ sessions + score above 50th pctl → Broadening."""
        pump = _pump(score=0.58, delta=0.03, delta_5d=0.025)
        prior = _prior(state=AnalysisState.ACCUMULATION, sessions=6)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=4, pump_percentile=58.0,
            delta_history=[0.02, 0.03, 0.02, 0.03, 0.02, 0.03],
            settings=SETTINGS,
        )
        assert result.state == AnalysisState.BROADENING

    def test_overt_pump(self):
        """Top quartile pump score + top 3 rank + delta positive → Overt Pump."""
        pump = _pump(score=0.85, delta=0.02, delta_5d=0.018)
        prior = _prior(state=AnalysisState.BROADENING, sessions=8)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=88.0,
            delta_history=[0.02, 0.01, 0.03, 0.02, 0.02],
            settings=SETTINGS,
        )
        assert result.state == AnalysisState.OVERT_PUMP

    def test_exhaustion(self):
        """Was top quartile + delta negative 3+ sessions → Exhaustion."""
        pump = _pump(score=0.78, delta=-0.03, delta_5d=-0.02)
        prior = _prior(state=AnalysisState.OVERT_PUMP, sessions=10)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=2, pump_percentile=76.0,
            delta_history=[0.01, -0.01, -0.02, -0.03],
            settings=SETTINGS,
        )
        assert result.state == AnalysisState.EXHAUSTION

    def test_rotation(self):
        """Score declining + rank dropping → Rotation."""
        pump = _pump(score=0.45, delta=-0.06, delta_5d=-0.04)
        prior = _prior(state=AnalysisState.EXHAUSTION, sessions=4)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=8, pump_percentile=40.0,
            delta_history=[-0.04, -0.05, -0.06, -0.06],
            settings=SETTINGS,
        )
        assert result.state == AnalysisState.ROTATION

    def test_ambiguous(self):
        """Flip-flopping delta + mid-pack rank → Ambiguous."""
        pump = _pump(score=0.50, delta=0.005, delta_5d=0.0)
        prior = _prior(state=AnalysisState.AMBIGUOUS, sessions=4)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=6, pump_percentile=50.0,
            delta_history=[0.01, -0.01, 0.01, -0.01, 0.005],
            settings=SETTINGS,
        )
        assert result.state == AnalysisState.AMBIGUOUS


# ═══════════════════════════════════════════════════════
# TRANSITION PRESSURE
# ═══════════════════════════════════════════════════════

class TestTransitionPressure:

    def test_up_pressure(self):
        """Delta > 0 for 3+ sessions → UP."""
        pump = _pump(score=0.60, delta=0.02, delta_5d=0.02)
        prior = _prior(state=AnalysisState.ACCUMULATION, sessions=3)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=5, pump_percentile=55.0,
            delta_history=[0.02, 0.02, 0.02],
            settings=SETTINGS,
        )
        assert result.transition_pressure == TransitionPressure.UP

    def test_stable_pressure(self):
        """Delta near zero → STABLE."""
        pump = _pump(score=0.50, delta=0.001, delta_5d=0.0)
        prior = _prior(state=AnalysisState.ACCUMULATION, sessions=5)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=6, pump_percentile=50.0,
            delta_history=[0.001, -0.001, 0.001, -0.001, 0.001],
            settings=SETTINGS,
        )
        assert result.transition_pressure == TransitionPressure.STABLE

    def test_down_pressure(self):
        """Delta < 0 for 3+ sessions, state unchanged → DOWN."""
        pump = _pump(score=0.50, delta=-0.03, delta_5d=-0.02)
        # Prior already EXHAUSTION so state stays the same → no BREAK
        prior = _prior(state=AnalysisState.EXHAUSTION, sessions=4)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=3, pump_percentile=50.0,
            delta_history=[-0.02, -0.03, -0.03],
            settings=SETTINGS,
        )
        assert result.transition_pressure == TransitionPressure.DOWN

    def test_break_on_state_change(self):
        """State changed this session → BREAK."""
        pump = _pump(score=0.85, delta=0.04, delta_5d=0.03)
        prior = _prior(state=AnalysisState.BROADENING, sessions=5)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=88.0,
            delta_history=[0.03, 0.04, 0.04, 0.04, 0.04],
            settings=SETTINGS,
        )
        # If state transitions from BROADENING to OVERT_PUMP → BREAK
        if result.state != prior.state:
            assert result.transition_pressure == TransitionPressure.BREAK


# ═══════════════════════════════════════════════════════
# CONFIDENCE
# ═══════════════════════════════════════════════════════

class TestConfidence:

    def test_high_confidence_all_confirming(self):
        """All signals aligned → confidence > 70."""
        pump = _pump(score=0.85, delta=0.03, delta_5d=0.025)
        prior = _prior(state=AnalysisState.BROADENING, sessions=6)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=88.0,
            delta_history=[0.03, 0.03, 0.03, 0.03, 0.03],
            settings=SETTINGS,
        )
        assert result.confidence > 70

    def test_low_confidence_fragile_regime(self):
        """Same signals + FRAGILE regime → confidence reduced."""
        pump = _pump(score=0.85, delta=0.03, delta_5d=0.025)
        prior = _prior(state=AnalysisState.BROADENING, sessions=6)

        normal_result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=88.0,
            delta_history=[0.03, 0.03, 0.03, 0.03, 0.03],
            settings=SETTINGS,
        )
        fragile_result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.FRAGILE,
            rs_rank=1, pump_percentile=88.0,
            delta_history=[0.03, 0.03, 0.03, 0.03, 0.03],
            settings=SETTINGS,
        )
        assert fragile_result.confidence < normal_result.confidence

    def test_very_low_confidence_hostile(self):
        """HOSTILE regime → confidence reduced further."""
        pump = _pump(score=0.85, delta=0.03, delta_5d=0.025)
        prior = _prior(state=AnalysisState.BROADENING, sessions=6)

        fragile_result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.FRAGILE,
            rs_rank=1, pump_percentile=88.0,
            delta_history=[0.03, 0.03, 0.03, 0.03, 0.03],
            settings=SETTINGS,
        )
        hostile_result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.HOSTILE,
            rs_rank=1, pump_percentile=88.0,
            delta_history=[0.03, 0.03, 0.03, 0.03, 0.03],
            settings=SETTINGS,
        )
        assert hostile_result.confidence < fragile_result.confidence

    def test_confidence_clamped_to_range(self):
        """Never below 10, never above 95."""
        # Max confidence scenario
        pump = _pump(score=0.95, delta=0.05, delta_5d=0.04)
        prior = _prior(state=AnalysisState.OVERT_PUMP, sessions=15)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=1, pump_percentile=99.0,
            delta_history=[0.05] * 10,
            settings=SETTINGS,
        )
        assert 10 <= result.confidence <= 95

        # Min confidence scenario
        pump2 = _pump(score=0.50, delta=0.001, delta_5d=0.0)
        prior2 = _prior(state=AnalysisState.AMBIGUOUS, sessions=14)
        result2 = classify_state(
            pump=pump2, prior=prior2, regime=RegimeState.HOSTILE,
            rs_rank=6, pump_percentile=50.0,
            delta_history=[0.001, -0.001, 0.001, -0.001, 0.001],
            settings=SETTINGS,
        )
        assert 10 <= result2.confidence <= 95

    def test_mixed_signals_reduce_confidence(self):
        """Conflicting pillar signals → lower confidence."""
        # High RS but low participation and flow — conflicting
        pump = _pump(score=0.55, delta=0.01, delta_5d=0.005, rs=90.0, part=20.0, flow=25.0)
        prior = _prior(state=AnalysisState.ACCUMULATION, sessions=3)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=3, pump_percentile=55.0,
            delta_history=[0.01, 0.01, 0.01],
            settings=SETTINGS,
        )
        # Compare to aligned pillars
        pump_aligned = _pump(score=0.55, delta=0.01, delta_5d=0.005, rs=60.0, part=55.0, flow=50.0)
        result_aligned = classify_state(
            pump=pump_aligned, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=3, pump_percentile=55.0,
            delta_history=[0.01, 0.01, 0.01],
            settings=SETTINGS,
        )
        assert result.confidence <= result_aligned.confidence


# ═══════════════════════════════════════════════════════
# AMBIGUOUS STATE
# ═══════════════════════════════════════════════════════

class TestAmbiguousState:

    def test_enters_ambiguous_on_mixed_signals(self):
        """Flip-flopping delta → enters Ambiguous."""
        pump = _pump(score=0.50, delta=0.005, delta_5d=0.0)
        # Prior was not ambiguous, but signals are mixed
        prior = _prior(state=AnalysisState.ACCUMULATION, sessions=5)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=6, pump_percentile=50.0,
            delta_history=[0.02, -0.02, 0.01, -0.01, 0.005],
            settings=SETTINGS,
        )
        assert result.state == AnalysisState.AMBIGUOUS

    def test_ambiguous_forced_exit_at_max_duration(self):
        """After max_duration sessions in Ambiguous → forced reclassification."""
        pump = _pump(score=0.55, delta=0.02, delta_5d=0.01)
        prior = _prior(state=AnalysisState.AMBIGUOUS, sessions=15)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=5, pump_percentile=55.0,
            delta_history=[0.02, 0.02, 0.02],
            settings=SETTINGS,
        )
        # Should be forced out of ambiguous
        assert result.state != AnalysisState.AMBIGUOUS

    def test_ambiguous_sessions_tracked(self):
        """Sessions in Ambiguous tracked correctly — increments when staying."""
        pump = _pump(score=0.50, delta=0.005, delta_5d=0.0)
        prior = _prior(state=AnalysisState.AMBIGUOUS, sessions=7)
        result = classify_state(
            pump=pump, prior=prior, regime=RegimeState.NORMAL,
            rs_rank=6, pump_percentile=50.0,
            delta_history=[0.01, -0.01, 0.01, -0.01, 0.005],
            settings=SETTINGS,
        )
        if result.state == AnalysisState.AMBIGUOUS:
            assert result.sessions_in_state == 8


# ═══════════════════════════════════════════════════════
# FIRST CLASSIFICATION (NO PRIOR)
# ═══════════════════════════════════════════════════════

class TestFirstClassification:

    def test_no_prior_state_does_not_crash(self):
        """First-ever classification (no prior) should work."""
        pump = _pump(score=0.50, delta=0.0, delta_5d=0.0)
        result = classify_state(
            pump=pump, prior=None, regime=RegimeState.NORMAL,
            rs_rank=6, pump_percentile=50.0,
            delta_history=[],
            settings=SETTINGS,
        )
        assert isinstance(result, StateClassification)
        assert result.prior_state is None

    def test_first_classification_is_accumulation_or_ambiguous(self):
        """With no history, should default to Accumulation or Ambiguous."""
        pump = _pump(score=0.40, delta=0.0, delta_5d=0.0)
        result = classify_state(
            pump=pump, prior=None, regime=RegimeState.NORMAL,
            rs_rank=6, pump_percentile=40.0,
            delta_history=[],
            settings=SETTINGS,
        )
        assert result.state in (AnalysisState.ACCUMULATION, AnalysisState.AMBIGUOUS)


# ═══════════════════════════════════════════════════════
# MULTI-SECTOR
# ═══════════════════════════════════════════════════════

class TestClassifyAllSectors:

    def test_returns_dict_of_classifications(self):
        """classify_all_sectors returns dict[str, StateClassification]."""
        pumps = {
            "XLK": _pump("XLK", "Technology", score=0.80, delta=0.02, delta_5d=0.02),
            "XLE": _pump("XLE", "Energy", score=0.30, delta=-0.01, delta_5d=-0.005),
        }
        priors = {}
        ranks = {"XLK": 1, "XLE": 10}
        percentiles = {"XLK": 85.0, "XLE": 25.0}
        delta_histories = {"XLK": [0.02] * 5, "XLE": [-0.01] * 5}

        results = classify_all_sectors(
            pumps=pumps, priors=priors, regime=RegimeState.NORMAL,
            rs_ranks=ranks, pump_percentiles=percentiles,
            delta_histories=delta_histories, settings=SETTINGS,
        )
        assert isinstance(results, dict)
        assert "XLK" in results
        assert "XLE" in results
        assert isinstance(results["XLK"], StateClassification)

    def test_all_sectors_classified(self):
        """Every sector in input gets a classification."""
        tickers = ["XLK", "XLV", "XLF"]
        pumps = {t: _pump(t, t, score=0.50) for t in tickers}
        results = classify_all_sectors(
            pumps=pumps, priors={}, regime=RegimeState.NORMAL,
            rs_ranks={t: i+1 for i, t in enumerate(tickers)},
            pump_percentiles={t: 50.0 for t in tickers},
            delta_histories={t: [0.0] * 3 for t in tickers},
            settings=SETTINGS,
        )
        assert set(results.keys()) == set(tickers)

    def test_explanation_is_populated(self):
        """Each classification has a non-empty explanation."""
        pumps = {"XLK": _pump("XLK", "Technology", score=0.70, delta=0.02)}
        results = classify_all_sectors(
            pumps=pumps, priors={}, regime=RegimeState.NORMAL,
            rs_ranks={"XLK": 1}, pump_percentiles={"XLK": 70.0},
            delta_histories={"XLK": [0.02] * 5}, settings=SETTINGS,
        )
        assert len(results["XLK"].explanation) > 0
