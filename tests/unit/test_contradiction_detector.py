"""
Contradiction Detector — unit tests.
Covers regime override, crisis beneficiary, horizon conflict, delta mismatch,
and system-level checks (coverage gap, all-same-trade).
"""
import pytest
from engine.contradiction_detector import detect_contradictions
from engine.schemas import (
    AnalysisState, RegimeState, HorizonPattern, CrisisType,
    StateClassification, TransitionPressure, PumpScoreReading,
    ReversalScoreReading, TradeStateAssignment, TradeState,
    RegimeAssessment, RegimeSignal, SignalLevel, HorizonReading,
)


# ── Helpers ──────────────────────────────────────────────


def _make_regime(state: RegimeState) -> RegimeAssessment:
    """Create a minimal RegimeAssessment."""
    return RegimeAssessment(
        state=state,
        signals=[],
        hostile_count=1 if state == RegimeState.HOSTILE else 0,
        fragile_count=1 if state == RegimeState.FRAGILE else 0,
        normal_count=1 if state == RegimeState.NORMAL else 0,
        timestamp="2026-03-18T12:00:00",
        explanation="Test regime",
    )


def _make_state(ticker: str, name: str, state: AnalysisState,
                confidence: int = 60) -> StateClassification:
    """Create a minimal StateClassification."""
    return StateClassification(
        ticker=ticker,
        name=name,
        state=state,
        confidence=confidence,
        sessions_in_state=5,
        transition_pressure=TransitionPressure.STABLE,
        prior_state=None,
        state_changed=False,
        explanation="Test state",
    )


def _make_pump(ticker: str, name: str, pump_delta: float = 0.0,
               pump_score: float = 0.5) -> PumpScoreReading:
    """Create a minimal PumpScoreReading."""
    return PumpScoreReading(
        ticker=ticker,
        name=name,
        rs_pillar=0.3,
        participation_pillar=0.3,
        flow_pillar=0.3,
        pump_score=pump_score,
        pump_delta=pump_delta,
        pump_delta_5d_avg=pump_delta,
    )


def _make_reversal(ticker: str, name: str, percentile: float = 50.0,
                   above_75th: bool = False) -> ReversalScoreReading:
    """Create a minimal ReversalScoreReading."""
    return ReversalScoreReading(
        ticker=ticker,
        name=name,
        breadth_det_pillar=0.3,
        price_break_pillar=0.3,
        crowding_pillar=0.3,
        reversal_score=percentile / 100.0,
        sub_signals={},
        reversal_percentile=percentile,
        above_75th=above_75th,
    )


def _make_tsa(ticker: str, name: str, analysis: AnalysisState,
              trade: TradeState) -> TradeStateAssignment:
    """Create a minimal TradeStateAssignment."""
    return TradeStateAssignment(
        ticker=ticker,
        name=name,
        analysis_state=analysis,
        trade_state=trade,
        confidence=50,
        entry_trigger="—",
        invalidation="—",
        size_class="—",
        catalyst_note="—",
        explanation="—",
    )


def _make_horizon(ticker: str, name: str,
                  pattern: HorizonPattern) -> HorizonReading:
    """Create a minimal HorizonReading."""
    return HorizonReading(
        ticker=ticker,
        name=name,
        pattern=pattern,
        rs_5d=0.01,
        rs_20d=0.01,
        rs_60d=0.01,
        rs_5d_sign="+",
        rs_20d_sign="+",
        rs_60d_sign="+",
        conviction=80,
        description="Test",
        is_rotation_signal=pattern in (HorizonPattern.ROTATION_IN, HorizonPattern.ROTATION_OUT),
        is_trap=pattern == HorizonPattern.DEAD_CAT,
        is_entry_zone=pattern == HorizonPattern.HEALTHY_DIP,
    )


_SECTOR_TICKERS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU",
                    "XLRE", "XLC", "XLY", "XLP", "XLB"]

_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLB": "Materials",
}


def _base_result(regime_state=RegimeState.NORMAL,
                 crisis_types=None,
                 default_analysis=AnalysisState.AMBIGUOUS,
                 default_trade=TradeState.NO_TRADE,
                 default_horizon=HorizonPattern.NO_PATTERN,
                 default_pump_delta=0.0,
                 overrides=None):
    """
    Build a minimal result dict with all 11 sectors.
    overrides: dict of ticker -> dict of field overrides
    """
    if crisis_types is None:
        crisis_types = [CrisisType.NONE]
    overrides = overrides or {}

    states = {}
    trade_states = {}
    horizon_readings = {}
    pumps = {}
    reversal_map = {}

    for t in _SECTOR_TICKERS:
        name = _SECTOR_NAMES[t]
        ov = overrides.get(t, {})

        a_state = ov.get("analysis", default_analysis)
        t_state = ov.get("trade", default_trade)
        h_pattern = ov.get("horizon", default_horizon)
        p_delta = ov.get("pump_delta", default_pump_delta)
        conf = ov.get("confidence", 60)

        states[t] = _make_state(t, name, a_state, confidence=conf)
        trade_states[t] = _make_tsa(t, name, a_state, t_state)
        horizon_readings[t] = _make_horizon(t, name, h_pattern)
        pumps[t] = _make_pump(t, name, pump_delta=p_delta)
        reversal_map[t] = _make_reversal(t, name)

    return {
        "regime": _make_regime(regime_state),
        "states": states,
        "trade_states": trade_states,
        "horizon_readings": horizon_readings,
        "pumps": pumps,
        "reversal_map": reversal_map,
        "crisis_types": crisis_types,
    }


# ── Tests: Regime Override Bullish ────────────────────────


class TestRegimeOverrideBullish:
    def test_broadening_hostile_detected(self):
        """Broadening + HOSTILE -> HIGH contradiction."""
        result = _base_result(
            regime_state=RegimeState.HOSTILE,
            overrides={"XLK": {"analysis": AnalysisState.BROADENING}},
        )
        contradictions = detect_contradictions(result)
        regime_overrides = [c for c in contradictions
                           if c["type"] == "REGIME_OVERRIDE_BULLISH"
                           and c["ticker"] == "XLK"]
        assert len(regime_overrides) == 1
        assert regime_overrides[0]["severity"] == "HIGH"

    def test_ambiguous_hostile_not_flagged(self):
        """Ambiguous + HOSTILE -> NOT flagged (Ambiguous is correct to block)."""
        result = _base_result(
            regime_state=RegimeState.HOSTILE,
            default_analysis=AnalysisState.AMBIGUOUS,
        )
        contradictions = detect_contradictions(result)
        regime_overrides = [c for c in contradictions
                           if c["type"] == "REGIME_OVERRIDE_BULLISH"]
        assert len(regime_overrides) == 0


# ── Tests: Crisis Beneficiary Blocked ─────────────────────


class TestCrisisBeneficiaryBlocked:
    def test_xle_oil_shock_flagged(self):
        """XLE Broadening + Oil Shock crisis -> CRISIS_BENEFICIARY_BLOCKED."""
        result = _base_result(
            regime_state=RegimeState.HOSTILE,
            crisis_types=[CrisisType.OIL_SHOCK],
            overrides={"XLE": {"analysis": AnalysisState.BROADENING}},
        )
        contradictions = detect_contradictions(result)
        crisis_blocked = [c for c in contradictions
                         if c["type"] == "CRISIS_BENEFICIARY_BLOCKED"
                         and c["ticker"] == "XLE"]
        assert len(crisis_blocked) == 1
        assert "long_modifier" in crisis_blocked[0]["detail"]

    def test_xlk_oil_shock_not_beneficiary(self):
        """XLK (victim in oil shock, long_mod=0.5) -> NOT flagged as beneficiary."""
        result = _base_result(
            regime_state=RegimeState.HOSTILE,
            crisis_types=[CrisisType.OIL_SHOCK],
            overrides={"XLK": {"analysis": AnalysisState.BROADENING}},
        )
        contradictions = detect_contradictions(result)
        crisis_blocked = [c for c in contradictions
                         if c["type"] == "CRISIS_BENEFICIARY_BLOCKED"
                         and c["ticker"] == "XLK"]
        assert len(crisis_blocked) == 0


# ── Tests: Horizon Trade Conflict ─────────────────────────


class TestHorizonTradeConflict:
    def test_full_confirm_hedge_flagged(self):
        """Full Confirm + trade=Hedge -> HIGH conflict."""
        result = _base_result(
            overrides={"XLK": {
                "horizon": HorizonPattern.FULL_CONFIRM,
                "trade": TradeState.HEDGE,
            }},
        )
        contradictions = detect_contradictions(result)
        horizon_conflicts = [c for c in contradictions
                            if c["type"] == "HORIZON_TRADE_CONFLICT"
                            and c["ticker"] == "XLK"]
        assert len(horizon_conflicts) == 1
        assert horizon_conflicts[0]["severity"] == "HIGH"

    def test_full_reject_hedge_not_flagged(self):
        """Full Reject + trade=Hedge -> consistent, no conflict."""
        result = _base_result(
            overrides={"XLK": {
                "horizon": HorizonPattern.FULL_REJECT,
                "trade": TradeState.HEDGE,
            }},
        )
        contradictions = detect_contradictions(result)
        horizon_conflicts = [c for c in contradictions
                            if c["type"] == "HORIZON_TRADE_CONFLICT"
                            and c["ticker"] == "XLK"]
        assert len(horizon_conflicts) == 0


# ── Tests: Delta State Mismatch ──────────────────────────


class TestDeltaStateMismatch:
    def test_strong_delta_ambiguous_flagged(self):
        """Delta +0.05 + Ambiguous -> MEDIUM mismatch."""
        result = _base_result(
            default_analysis=AnalysisState.AMBIGUOUS,
            overrides={"XLK": {"pump_delta": 0.05}},
        )
        contradictions = detect_contradictions(result)
        delta_mismatches = [c for c in contradictions
                          if c["type"] == "DELTA_STATE_MISMATCH"
                          and c["ticker"] == "XLK"]
        assert len(delta_mismatches) == 1
        assert delta_mismatches[0]["severity"] == "MEDIUM"

    def test_weak_delta_not_flagged(self):
        """Delta +0.01 + Ambiguous -> NOT flagged (below 0.03 threshold)."""
        result = _base_result(
            default_analysis=AnalysisState.AMBIGUOUS,
            overrides={"XLK": {"pump_delta": 0.01}},
        )
        contradictions = detect_contradictions(result)
        delta_mismatches = [c for c in contradictions
                          if c["type"] == "DELTA_STATE_MISMATCH"
                          and c["ticker"] == "XLK"]
        assert len(delta_mismatches) == 0


# ── Tests: System Level ──────────────────────────────────


class TestSystemLevel:
    def test_high_ambiguous_coverage_flagged(self):
        """8 of 11 Ambiguous -> COVERAGE_GAP."""
        # Default is all Ambiguous; override 3 to non-ambiguous
        overrides = {
            "XLK": {"analysis": AnalysisState.BROADENING},
            "XLV": {"analysis": AnalysisState.ACCUMULATION},
            "XLF": {"analysis": AnalysisState.DISTRIBUTION},
        }
        result = _base_result(
            default_analysis=AnalysisState.AMBIGUOUS,
            overrides=overrides,
        )
        contradictions = detect_contradictions(result)
        coverage = [c for c in contradictions if c["type"] == "COVERAGE_GAP"]
        assert len(coverage) == 1
        assert coverage[0]["severity"] == "HIGH"
        assert "8 of 11" in coverage[0]["detail"]

    def test_all_same_trade_flagged(self):
        """All 11 = Hedge -> ALL_SAME_TRADE."""
        result = _base_result(default_trade=TradeState.HEDGE)
        contradictions = detect_contradictions(result)
        all_same = [c for c in contradictions if c["type"] == "ALL_SAME_TRADE"]
        assert len(all_same) == 1
        assert all_same[0]["severity"] == "HIGH"
        assert "Hedge" in all_same[0]["detail"]

    def test_normal_not_flagged(self):
        """Mixed states, no > 50% Ambiguous -> no system contradiction."""
        overrides = {
            "XLK": {"analysis": AnalysisState.BROADENING, "trade": TradeState.LONG_ENTRY},
            "XLV": {"analysis": AnalysisState.ACCUMULATION, "trade": TradeState.SELECTIVE_ADD},
            "XLF": {"analysis": AnalysisState.DISTRIBUTION, "trade": TradeState.HEDGE},
            "XLE": {"analysis": AnalysisState.EXHAUSTION, "trade": TradeState.REDUCE},
            "XLI": {"analysis": AnalysisState.AMBIGUOUS, "trade": TradeState.NO_TRADE},
            "XLU": {"analysis": AnalysisState.OVERT_PUMP, "trade": TradeState.HOLD},
            "XLRE": {"analysis": AnalysisState.AMBIGUOUS, "trade": TradeState.WATCHLIST},
            "XLC": {"analysis": AnalysisState.OVERT_DUMP, "trade": TradeState.HEDGE},
            "XLY": {"analysis": AnalysisState.BROADENING, "trade": TradeState.LONG_ENTRY},
            "XLP": {"analysis": AnalysisState.ACCUMULATION, "trade": TradeState.SELECTIVE_ADD},
            "XLB": {"analysis": AnalysisState.DISTRIBUTION, "trade": TradeState.REDUCE},
        }
        result = _base_result(overrides=overrides)
        contradictions = detect_contradictions(result)
        system_contras = [c for c in contradictions
                         if c["type"] in ("COVERAGE_GAP", "ALL_SAME_TRADE")]
        assert len(system_contras) == 0
