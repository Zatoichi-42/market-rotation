"""
Explain module unit tests — synthetic data only.
Tests human-readable explanation generation for regime, state, and breadth.
"""
import pytest
from engine.schemas import (
    RegimeState, SignalLevel, RegimeSignal, RegimeAssessment,
    AnalysisState, TransitionPressure, StateClassification,
    BreadthSignal, BreadthReading, PumpScoreReading,
)
from engine.explain import explain_regime, explain_state, explain_breadth


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def _make_regime(state: RegimeState, signals: list[tuple[str, float, SignalLevel]]) -> RegimeAssessment:
    sigs = [RegimeSignal(name=n, raw_value=v, level=l, description=f"{n}={v}") for n, v, l in signals]
    h = sum(1 for s in sigs if s.level == SignalLevel.HOSTILE)
    f = sum(1 for s in sigs if s.level == SignalLevel.FRAGILE)
    n = sum(1 for s in sigs if s.level == SignalLevel.NORMAL)
    return RegimeAssessment(
        state=state, signals=sigs, hostile_count=h, fragile_count=f,
        normal_count=n, timestamp="2026-03-18T10:00:00", explanation="",
    )


def _make_state(
    ticker: str = "XLK", name: str = "Technology",
    state: AnalysisState = AnalysisState.ACCUMULATION,
    confidence: int = 72, sessions: int = 6,
    pressure: TransitionPressure = TransitionPressure.UP,
    prior: AnalysisState = AnalysisState.ACCUMULATION,
) -> StateClassification:
    return StateClassification(
        ticker=ticker, name=name, state=state, confidence=confidence,
        sessions_in_state=sessions, transition_pressure=pressure,
        prior_state=prior, state_changed=False, explanation="",
    )


def _make_pump(
    ticker: str = "XLK", name: str = "Technology",
    rs: float = 75.0, part: float = 65.0, flow: float = 70.0,
    score: float = 0.70, delta: float = 0.03, delta_5d: float = 0.02,
) -> PumpScoreReading:
    return PumpScoreReading(
        ticker=ticker, name=name,
        rs_pillar=rs, participation_pillar=part, flow_pillar=flow,
        pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta_5d,
    )


def _make_breadth(
    signal: BreadthSignal = BreadthSignal.HEALTHY,
    ratio: float = 1.02, change: float = 0.005, zscore: float = 0.5,
) -> BreadthReading:
    return BreadthReading(
        rsp_spy_ratio=ratio, rsp_spy_ratio_20d_change=change,
        rsp_spy_ratio_zscore=zscore, signal=signal, explanation="",
    )


# ═══════════════════════════════════════════════════════
# REGIME EXPLANATION
# ═══════════════════════════════════════════════════════

class TestExplainRegime:

    def test_normal_mentions_normal(self):
        regime = _make_regime(RegimeState.NORMAL, [
            ("vix", 15.0, SignalLevel.NORMAL),
            ("term_structure", 0.88, SignalLevel.NORMAL),
            ("breadth", 0.5, SignalLevel.NORMAL),
            ("credit", 0.1, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime)
        assert "NORMAL" in result

    def test_normal_references_all_signals(self):
        regime = _make_regime(RegimeState.NORMAL, [
            ("vix", 15.0, SignalLevel.NORMAL),
            ("term_structure", 0.88, SignalLevel.NORMAL),
            ("breadth", 0.5, SignalLevel.NORMAL),
            ("credit", 0.1, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime)
        assert "vix" in result.lower() or "15" in result

    def test_hostile_mentions_hostile(self):
        regime = _make_regime(RegimeState.HOSTILE, [
            ("vix", 35.0, SignalLevel.HOSTILE),
            ("term_structure", 1.12, SignalLevel.HOSTILE),
            ("breadth", -2.0, SignalLevel.HOSTILE),
            ("credit", -1.8, SignalLevel.HOSTILE),
        ])
        result = explain_regime(regime)
        assert "HOSTILE" in result

    def test_hostile_includes_action_guidance(self):
        regime = _make_regime(RegimeState.HOSTILE, [
            ("vix", 35.0, SignalLevel.HOSTILE),
            ("term_structure", 1.12, SignalLevel.HOSTILE),
            ("breadth", 0.5, SignalLevel.NORMAL),
            ("credit", 0.1, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime)
        lower = result.lower()
        assert "hedge" in lower or "cash" in lower or "no new" in lower

    def test_fragile_mentions_fragile(self):
        regime = _make_regime(RegimeState.FRAGILE, [
            ("vix", 24.0, SignalLevel.FRAGILE),
            ("term_structure", 1.0, SignalLevel.FRAGILE),
            ("breadth", 0.5, SignalLevel.NORMAL),
            ("credit", 0.0, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime)
        assert "FRAGILE" in result

    def test_fragile_includes_caution_guidance(self):
        regime = _make_regime(RegimeState.FRAGILE, [
            ("vix", 24.0, SignalLevel.FRAGILE),
            ("term_structure", 1.0, SignalLevel.FRAGILE),
            ("breadth", 0.5, SignalLevel.NORMAL),
            ("credit", 0.0, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime)
        lower = result.lower()
        assert "reduce" in lower or "caution" in lower or "tighten" in lower

    def test_returns_string(self):
        regime = _make_regime(RegimeState.NORMAL, [("vix", 15.0, SignalLevel.NORMAL)])
        result = explain_regime(regime)
        assert isinstance(result, str)
        assert len(result) > 10

    def test_empty_signals_still_works(self):
        regime = _make_regime(RegimeState.NORMAL, [])
        result = explain_regime(regime)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_signal_values(self):
        """Explanation should include actual numeric values for transparency."""
        regime = _make_regime(RegimeState.HOSTILE, [
            ("vix", 34.2, SignalLevel.HOSTILE),
            ("credit", -1.8, SignalLevel.HOSTILE),
            ("breadth", 0.5, SignalLevel.NORMAL),
            ("term_structure", 0.90, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime)
        assert "34.2" in result or "34.20" in result
        assert "-1.8" in result or "-1.80" in result

    def test_includes_source_citations(self):
        """Explanation should cite data sources."""
        regime = _make_regime(RegimeState.NORMAL, [
            ("vix", 15.0, SignalLevel.NORMAL),
            ("term_structure", 0.88, SignalLevel.NORMAL),
            ("breadth", 0.5, SignalLevel.NORMAL),
            ("credit", 0.1, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime)
        assert "yfinance" in result.lower() or "live" in result.lower()

    def test_fred_oas_included_when_provided(self):
        """When FRED OAS value is passed, it appears in the explanation with lag disclaimer."""
        regime = _make_regime(RegimeState.NORMAL, [
            ("vix", 15.0, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime, fred_hy_oas_value=350.0)
        assert "350" in result
        assert "FRED" in result or "lag" in result.lower()

    def test_fred_oas_omitted_when_none(self):
        """When FRED OAS is None, explanation still works."""
        regime = _make_regime(RegimeState.NORMAL, [
            ("vix", 15.0, SignalLevel.NORMAL),
        ])
        result = explain_regime(regime, fred_hy_oas_value=None)
        assert isinstance(result, str)
        assert "NORMAL" in result

    def test_breadth_explanation_cites_source(self):
        """Breadth explanation should cite RSP/SPY source."""
        br = _make_breadth(BreadthSignal.HEALTHY, ratio=1.02, zscore=0.5)
        result = explain_breadth(br)
        assert "yfinance" in result.lower() or "live" in result.lower()


# ═══════════════════════════════════════════════════════
# STATE EXPLANATION
# ═══════════════════════════════════════════════════════

class TestExplainState:

    def test_broadening_explanation(self):
        sc = _make_state(state=AnalysisState.ACCUMULATION, confidence=72, sessions=6)
        pump = _make_pump(delta=0.03)
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "Accumulation" in result
        assert "72" in result or "confidence" in result.lower()

    def test_exhaustion_explanation(self):
        sc = _make_state(state=AnalysisState.EXHAUSTION, confidence=58, sessions=4,
                         pressure=TransitionPressure.DOWN)
        pump = _make_pump(score=0.82, delta=-0.04)
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "Exhaustion" in result

    def test_overt_pump_explanation(self):
        sc = _make_state(state=AnalysisState.OVERT_PUMP, confidence=80, sessions=10)
        pump = _make_pump(score=0.88, delta=0.02)
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "Overt Pump" in result

    def test_accumulation_explanation(self):
        sc = _make_state(state=AnalysisState.ACCUMULATION, confidence=55, sessions=3)
        pump = _make_pump(score=0.35, delta=0.02)
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "Accumulation" in result

    def test_rotation_explanation(self):
        sc = _make_state(state=AnalysisState.OVERT_DUMP, confidence=60, sessions=2,
                         pressure=TransitionPressure.DOWN)
        pump = _make_pump(score=0.40, delta=-0.05)
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "Overt Dump" in result

    def test_ambiguous_explanation(self):
        sc = _make_state(state=AnalysisState.AMBIGUOUS, confidence=30, sessions=5)
        pump = _make_pump(score=0.50, delta=0.0)
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "Ambiguous" in result

    def test_hostile_regime_noted_in_explanation(self):
        """When regime is HOSTILE, explanation should mention it."""
        sc = _make_state(state=AnalysisState.ACCUMULATION, confidence=50)
        pump = _make_pump()
        result = explain_state(sc, pump, RegimeState.HOSTILE)
        assert "HOSTILE" in result

    def test_fragile_regime_noted(self):
        sc = _make_state()
        pump = _make_pump()
        result = explain_state(sc, pump, RegimeState.FRAGILE)
        assert "FRAGILE" in result

    def test_includes_ticker(self):
        sc = _make_state(ticker="XLE", name="Energy")
        pump = _make_pump(ticker="XLE", name="Energy")
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "XLE" in result or "Energy" in result

    def test_includes_pump_score(self):
        sc = _make_state()
        pump = _make_pump(score=0.72)
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "0.72" in result or "72" in result

    def test_includes_sessions_in_state(self):
        sc = _make_state(sessions=8)
        pump = _make_pump()
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert "8" in result

    def test_returns_string(self):
        sc = _make_state()
        pump = _make_pump()
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert isinstance(result, str)
        assert len(result) > 20

    def test_no_prior_state(self):
        """First classification (no prior) should not crash."""
        sc = _make_state(prior=None)
        pump = _make_pump()
        result = explain_state(sc, pump, RegimeState.NORMAL)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════
# BREADTH EXPLANATION
# ═══════════════════════════════════════════════════════

class TestExplainBreadth:

    def test_healthy_explanation(self):
        br = _make_breadth(BreadthSignal.HEALTHY, ratio=1.03, change=0.008, zscore=0.7)
        result = explain_breadth(br)
        assert "HEALTHY" in result

    def test_narrowing_explanation(self):
        br = _make_breadth(BreadthSignal.NARROWING, ratio=0.99, change=-0.003, zscore=-0.4)
        result = explain_breadth(br)
        assert "NARROWING" in result

    def test_diverging_explanation(self):
        br = _make_breadth(BreadthSignal.DIVERGING, ratio=0.95, change=-0.015, zscore=-1.8)
        result = explain_breadth(br)
        assert "DIVERGING" in result

    def test_includes_ratio_value(self):
        br = _make_breadth(ratio=1.0342)
        result = explain_breadth(br)
        assert "1.03" in result

    def test_includes_zscore(self):
        br = _make_breadth(zscore=0.75)
        result = explain_breadth(br)
        assert "0.75" in result or "0.7" in result

    def test_nan_zscore_handled(self):
        br = _make_breadth(zscore=float("nan"))
        result = explain_breadth(br)
        assert isinstance(result, str)
        assert "N/A" in result or "unavailable" in result.lower() or "insufficient" in result.lower()

    def test_returns_string(self):
        br = _make_breadth()
        result = explain_breadth(br)
        assert isinstance(result, str)
        assert len(result) > 10


class TestOilSourceTag:
    def test_oil_signal_has_source_tag(self):
        from engine.schemas import RegimeAssessment, RegimeSignal, SignalLevel, RegimeState
        assessment = RegimeAssessment(
            state=RegimeState.FRAGILE,
            signals=[RegimeSignal("oil", 2.93, SignalLevel.HOSTILE, "Oil z-score 2.93")],
            hostile_count=1, fragile_count=0, normal_count=0,
            timestamp="2026-03-19T00:00:00Z", explanation="",
        )
        explanation = explain_regime(assessment)
        assert "unknown source" not in explanation.lower()
        assert "CL=F" in explanation
