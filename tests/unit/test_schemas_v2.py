"""
Schema v2 unit tests — Phase 2 additions.
"""
import pytest
from engine.schemas import (
    GroupType, IndustryRSReading, ReversalScoreReading,
    TurnoverCheck, PumpMapRow, DailySnapshot,
    RegimeState, RegimeAssessment, SignalLevel, RegimeSignal,
    RSReading, BreadthReading, BreadthSignal,
    PumpScoreReading, StateClassification, AnalysisState, TransitionPressure,
)


class TestGroupTypeEnum:
    def test_values(self):
        assert GroupType.SECTOR.value == "sector"
        assert GroupType.INDUSTRY.value == "industry"
        assert len(GroupType) == 2


class TestIndustryRSReading:
    def test_creation(self):
        r = IndustryRSReading(
            ticker="SMH", name="Semiconductors", parent_sector="XLK",
            group_type=GroupType.INDUSTRY,
            rs_5d=1.5, rs_20d=3.0, rs_60d=5.0, rs_slope=0.2, rs_composite=80.0,
            rs_5d_vs_parent=0.5, rs_20d_vs_parent=1.0, rs_60d_vs_parent=2.0,
            rs_slope_vs_parent=0.1, rs_composite_vs_parent=65.0,
            industry_composite=75.5,
            rs_rank=1, rs_rank_change=2, rs_rank_within_sector=1,
        )
        assert r.ticker == "SMH"
        assert r.parent_sector == "XLK"
        assert r.group_type == GroupType.INDUSTRY
        assert r.industry_composite == 75.5
        assert r.rs_rank_within_sector == 1

    def test_vs_parent_fields_independent(self):
        r = IndustryRSReading(
            ticker="XBI", name="Biotech", parent_sector="XLV",
            group_type=GroupType.INDUSTRY,
            rs_5d=2.0, rs_20d=1.0, rs_60d=-1.0, rs_slope=-0.1, rs_composite=40.0,
            rs_5d_vs_parent=-0.5, rs_20d_vs_parent=-1.5, rs_60d_vs_parent=-3.0,
            rs_slope_vs_parent=-0.2, rs_composite_vs_parent=20.0,
            industry_composite=34.0,
            rs_rank=15, rs_rank_change=-3, rs_rank_within_sector=2,
        )
        assert r.rs_20d_vs_parent == -1.5
        assert r.rs_composite != r.rs_composite_vs_parent


class TestReversalScoreReading:
    def test_creation(self):
        r = ReversalScoreReading(
            ticker="XLK", name="Technology",
            breadth_det_pillar=75.0, price_break_pillar=62.0, crowding_pillar=58.0,
            reversal_score=0.65, sub_signals={"rs_slope": -0.05, "clv_trend": -0.3},
            reversal_percentile=82.0, above_75th=True,
        )
        assert r.reversal_score == 0.65
        assert r.above_75th is True
        assert "rs_slope" in r.sub_signals

    def test_low_reversal(self):
        r = ReversalScoreReading(
            ticker="XLF", name="Financials",
            breadth_det_pillar=15.0, price_break_pillar=20.0, crowding_pillar=10.0,
            reversal_score=0.15, sub_signals={},
            reversal_percentile=12.0, above_75th=False,
        )
        assert r.above_75th is False
        assert r.reversal_percentile == 12.0


class TestTurnoverCheck:
    def test_pass(self):
        tc = TurnoverCheck(
            candidate_ticker="XLV", current_ticker="XLK",
            delta_advantage=0.12, persistence_sessions=5,
            current_state_exempt=False, passes_filter=True,
            reason="Clear advantage",
        )
        assert tc.passes_filter is True
        assert tc.delta_advantage == 0.12

    def test_fail(self):
        tc = TurnoverCheck(
            candidate_ticker="XLI", current_ticker="XLF",
            delta_advantage=0.03, persistence_sessions=5,
            current_state_exempt=False, passes_filter=False,
            reason="Marginal improvement",
        )
        assert tc.passes_filter is False

    def test_exempt(self):
        tc = TurnoverCheck(
            candidate_ticker="XLV", current_ticker="XLK",
            delta_advantage=0.03, persistence_sessions=1,
            current_state_exempt=True, passes_filter=True,
            reason="Current in Exhaustion — exempt",
        )
        assert tc.current_state_exempt is True
        assert tc.passes_filter is True


class TestPumpMapRow:
    def test_sector_row(self):
        row = PumpMapRow(
            ticker="XLK", name="Technology", group_type=GroupType.SECTOR,
            parent_sector=None, tier="T1",
            regime_state=RegimeState.NORMAL,
            pump_score=0.72, pump_delta=0.03, pump_delta_5d_avg=0.02,
            reversal_score=0.25, reversal_percentile=30.0,
            analysis_state=AnalysisState.ACCUMULATION,
            transition_pressure=TransitionPressure.UP, confidence=72,
            rs_composite=80.0, rs_rank=1, rs_rank_change=2,
            rs_vs_parent=None,
        )
        assert row.parent_sector is None
        assert row.group_type == GroupType.SECTOR

    def test_industry_row(self):
        row = PumpMapRow(
            ticker="SMH", name="Semiconductors", group_type=GroupType.INDUSTRY,
            parent_sector="XLK", tier="T1",
            regime_state=RegimeState.NORMAL,
            pump_score=0.80, pump_delta=0.05, pump_delta_5d_avg=0.04,
            reversal_score=0.35, reversal_percentile=42.0,
            analysis_state=AnalysisState.OVERT_PUMP,
            transition_pressure=TransitionPressure.UP, confidence=80,
            rs_composite=90.0, rs_rank=1, rs_rank_change=1,
            rs_vs_parent=65.0,
        )
        assert row.parent_sector == "XLK"
        assert row.rs_vs_parent == 65.0


class TestDailySnapshotV2:
    def test_backward_compat_defaults(self):
        """Phase 1 snapshot creation still works — new fields default to empty."""
        sig = RegimeSignal(name="vix", raw_value=15.0, level=SignalLevel.NORMAL, description="ok")
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
        assert snap.industry_rs == []
        assert snap.reversal_scores == []
        assert snap.pump_map == []

    def test_with_phase2_fields(self):
        sig = RegimeSignal(name="vix", raw_value=15.0, level=SignalLevel.NORMAL, description="ok")
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
        ind = IndustryRSReading(
            ticker="SMH", name="Semiconductors", parent_sector="XLK",
            group_type=GroupType.INDUSTRY,
            rs_5d=1.0, rs_20d=2.0, rs_60d=3.0, rs_slope=0.1, rs_composite=70.0,
            rs_5d_vs_parent=0.5, rs_20d_vs_parent=1.0, rs_60d_vs_parent=1.5,
            rs_slope_vs_parent=0.05, rs_composite_vs_parent=60.0,
            industry_composite=67.0, rs_rank=1, rs_rank_change=0, rs_rank_within_sector=1,
        )
        rev = ReversalScoreReading(
            ticker="XLK", name="Technology",
            breadth_det_pillar=30.0, price_break_pillar=25.0, crowding_pillar=20.0,
            reversal_score=0.25, sub_signals={}, reversal_percentile=30.0, above_75th=False,
        )
        snap = DailySnapshot(
            date="2026-03-18", timestamp="2026-03-18T10:00:00",
            regime=regime, sectors=[], breadth=breadth,
            pump_scores=[], states=[],
            industry_rs=[ind], reversal_scores=[rev], pump_map=[],
        )
        assert len(snap.industry_rs) == 1
        assert snap.industry_rs[0].ticker == "SMH"
        assert len(snap.reversal_scores) == 1
