"""
Schema unit tests — verify enums, dataclass creation, field access.
"""
import pytest
from engine.schemas import (
    RegimeState, SignalLevel, AnalysisState, TransitionPressure, BreadthSignal,
    RegimeSignal, RegimeAssessment, RSReading, BreadthReading,
    PumpScoreReading, StateClassification, DailySnapshot,
)


class TestEnums:
    def test_regime_state_values(self):
        assert RegimeState.NORMAL.value == "NORMAL"
        assert RegimeState.FRAGILE.value == "FRAGILE"
        assert RegimeState.HOSTILE.value == "HOSTILE"
        assert len(RegimeState) == 3

    def test_signal_level_values(self):
        assert SignalLevel.NORMAL.value == "NORMAL"
        assert SignalLevel.FRAGILE.value == "FRAGILE"
        assert SignalLevel.HOSTILE.value == "HOSTILE"
        assert len(SignalLevel) == 3

    def test_analysis_state_values(self):
        assert AnalysisState.OVERT_DUMP.value == "Overt Dump"
        assert AnalysisState.DISTRIBUTION.value == "Distribution"
        assert AnalysisState.AMBIGUOUS.value == "Ambiguous"
        assert AnalysisState.ACCUMULATION.value == "Accumulation"
        assert AnalysisState.OVERT_PUMP.value == "Overt Pump"
        assert len(AnalysisState) == 5

    def test_transition_pressure_values(self):
        assert TransitionPressure.UP.value == "Up"
        assert TransitionPressure.STABLE.value == "Stable"
        assert TransitionPressure.DOWN.value == "Down"
        assert TransitionPressure.BREAK.value == "Break"
        assert len(TransitionPressure) == 4

    def test_breadth_signal_values(self):
        assert BreadthSignal.HEALTHY.value == "HEALTHY"
        assert BreadthSignal.NARROWING.value == "NARROWING"
        assert BreadthSignal.DIVERGING.value == "DIVERGING"
        assert len(BreadthSignal) == 3


class TestRegimeSignal:
    def test_creation(self):
        sig = RegimeSignal(name="VIX", raw_value=15.0, level=SignalLevel.NORMAL, description="Low VIX")
        assert sig.name == "VIX"
        assert sig.raw_value == 15.0
        assert sig.level == SignalLevel.NORMAL
        assert sig.description == "Low VIX"


class TestRegimeAssessment:
    def test_creation(self):
        sig = RegimeSignal(name="VIX", raw_value=15.0, level=SignalLevel.NORMAL, description="ok")
        ra = RegimeAssessment(
            state=RegimeState.NORMAL,
            signals=[sig],
            hostile_count=0,
            fragile_count=0,
            normal_count=1,
            timestamp="2026-03-18T10:00:00",
            explanation="All clear",
        )
        assert ra.state == RegimeState.NORMAL
        assert len(ra.signals) == 1
        assert ra.hostile_count == 0


class TestRSReading:
    def test_creation(self):
        rs = RSReading(
            ticker="XLK", name="Technology",
            rs_5d=1.2, rs_20d=2.5, rs_60d=4.0,
            rs_slope=0.3, rs_rank=1, rs_rank_change=2, rs_composite=72.5,
        )
        assert rs.ticker == "XLK"
        assert rs.rs_rank == 1
        assert rs.rs_composite == 72.5


class TestBreadthReading:
    def test_creation(self):
        br = BreadthReading(
            rsp_spy_ratio=1.05,
            rsp_spy_ratio_20d_change=0.02,
            rsp_spy_ratio_zscore=0.5,
            signal=BreadthSignal.HEALTHY,
            explanation="Broad participation",
        )
        assert br.signal == BreadthSignal.HEALTHY


class TestPumpScoreReading:
    def test_creation(self):
        ps = PumpScoreReading(
            ticker="XLE", name="Energy",
            rs_pillar=80.0, participation_pillar=65.0, flow_pillar=70.0,
            pump_score=0.72, pump_delta=0.03, pump_delta_5d_avg=0.02,
        )
        assert ps.pump_score == 0.72
        assert ps.ticker == "XLE"


class TestStateClassification:
    def test_creation(self):
        sc = StateClassification(
            ticker="XLK", name="Technology",
            state=AnalysisState.ACCUMULATION,
            confidence=72, sessions_in_state=6,
            transition_pressure=TransitionPressure.UP,
            prior_state=AnalysisState.ACCUMULATION,
            state_changed=False,
            explanation="Pump delta positive 6 sessions",
        )
        assert sc.state == AnalysisState.ACCUMULATION
        assert sc.confidence == 72
        assert sc.prior_state == AnalysisState.ACCUMULATION

    def test_creation_no_prior_state(self):
        sc = StateClassification(
            ticker="XLK", name="Technology",
            state=AnalysisState.AMBIGUOUS,
            confidence=30, sessions_in_state=1,
            transition_pressure=TransitionPressure.STABLE,
            prior_state=None,
            state_changed=True,
            explanation="First classification",
        )
        assert sc.prior_state is None
        assert sc.state_changed is True


class TestDailySnapshot:
    def test_creation(self):
        sig = RegimeSignal(name="VIX", raw_value=15.0, level=SignalLevel.NORMAL, description="ok")
        regime = RegimeAssessment(
            state=RegimeState.NORMAL, signals=[sig],
            hostile_count=0, fragile_count=0, normal_count=1,
            timestamp="2026-03-18T10:00:00", explanation="Clear",
        )
        breadth = BreadthReading(
            rsp_spy_ratio=1.0, rsp_spy_ratio_20d_change=0.0,
            rsp_spy_ratio_zscore=0.0, signal=BreadthSignal.NARROWING,
            explanation="Flat",
        )
        snap = DailySnapshot(
            date="2026-03-18", timestamp="2026-03-18T10:00:00",
            regime=regime, sectors=[], breadth=breadth,
            pump_scores=[], states=[],
        )
        assert snap.date == "2026-03-18"
        assert snap.regime.state == RegimeState.NORMAL
