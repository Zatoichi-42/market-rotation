"""
Industry state classification tests — based on multi-timeframe RS pattern.
"""
import pytest
from engine.schemas import (
    AnalysisState, GroupType, IndustryRSReading, ReversalScoreReading, RegimeState,
)
from engine.industry_state import classify_industry_state, classify_all_industries


def _ir(ticker="SMH", rs_5d=0.02, rs_20d=0.03, rs_60d=0.05, slope=0.01,
        composite=80.0, rank=1):
    return IndustryRSReading(
        ticker=ticker, name=ticker, parent_sector="XLK", group_type=GroupType.INDUSTRY,
        rs_5d=rs_5d, rs_20d=rs_20d, rs_60d=rs_60d, rs_slope=slope, rs_composite=composite,
        rs_5d_vs_parent=0.01, rs_20d_vs_parent=0.01, rs_60d_vs_parent=0.02,
        rs_slope_vs_parent=0.005, rs_composite_vs_parent=60.0,
        industry_composite=composite, rs_rank=rank, rs_rank_change=0, rs_rank_within_sector=1,
    )


class TestIndustryStateFromRS:

    def test_overt_pump(self):
        """Strong RS at all timeframes, top rank, positive slope → Overt Pump."""
        ir = _ir(rs_5d=0.03, rs_20d=0.05, rs_60d=0.08, slope=0.015, composite=85.0, rank=2)
        result = classify_industry_state(ir)
        assert result.state == AnalysisState.OVERT_PUMP

    def test_accumulation(self):
        """Positive slope, building momentum → Accumulation."""
        ir = _ir(rs_5d=0.01, rs_20d=-0.005, rs_60d=-0.02, slope=0.005, composite=45.0, rank=10)
        result = classify_industry_state(ir)
        assert result.state == AnalysisState.ACCUMULATION

    def test_ambiguous(self):
        """Mixed timeframes, no clear direction → Ambiguous."""
        ir = _ir(rs_5d=-0.0005, rs_20d=-0.01, rs_60d=0.005, slope=-0.0001, composite=50.0, rank=12)
        result = classify_industry_state(ir)
        assert result.state == AnalysisState.AMBIGUOUS

    def test_exhaustion(self):
        """Was strong (60d positive) but 5d weakening → Exhaustion."""
        ir = _ir(rs_5d=-0.02, rs_20d=-0.01, rs_60d=0.06, slope=-0.008, composite=60.0, rank=6)
        result = classify_industry_state(ir)
        assert result.state == AnalysisState.EXHAUSTION

    def test_overt_dump(self):
        """RS negative everywhere, bottom rank → Overt Dump."""
        ir = _ir(rs_5d=-0.03, rs_20d=-0.04, rs_60d=-0.06, slope=-0.01, composite=10.0, rank=20)
        result = classify_industry_state(ir)
        assert result.state == AnalysisState.OVERT_DUMP

    def test_exhaustion_to_dump_with_reversal(self):
        """Exhaustion + high reversal score → Overt Dump."""
        ir = _ir(rs_5d=-0.01, rs_20d=-0.02, rs_60d=0.04, slope=-0.005, composite=40.0, rank=8)
        rev = ReversalScoreReading(
            ticker="SMH", name="SMH", breadth_det_pillar=70, price_break_pillar=60,
            crowding_pillar=50, reversal_score=0.65, sub_signals={},
            reversal_percentile=85.0, above_75th=True,
        )
        result = classify_industry_state(ir, reversal_score=rev)
        assert result.state == AnalysisState.OVERT_DUMP

    def test_hostile_regime_reduces_confidence(self):
        ir = _ir()
        normal = classify_industry_state(ir, regime=RegimeState.NORMAL)
        hostile = classify_industry_state(ir, regime=RegimeState.HOSTILE)
        assert hostile.confidence < normal.confidence

    def test_all_timeframes_aligned_high_confidence(self):
        """All 3 timeframes positive + positive slope → high confidence."""
        ir = _ir(rs_5d=0.02, rs_20d=0.03, rs_60d=0.04, slope=0.01, composite=80.0, rank=3)
        result = classify_industry_state(ir)
        assert result.confidence >= 70

    def test_mixed_timeframes_lower_confidence(self):
        ir = _ir(rs_5d=0.02, rs_20d=-0.01, rs_60d=0.03, slope=0.0, composite=50.0, rank=10)
        result = classify_industry_state(ir)
        assert result.confidence <= 55


class TestClassifyAllIndustries:

    def test_returns_dict(self):
        irs = [_ir("SMH", rank=1), _ir("XBI", rs_5d=-0.02, rs_20d=-0.03, slope=-0.01, rank=15)]
        results = classify_all_industries(irs)
        assert "SMH" in results
        assert "XBI" in results

    def test_different_states(self):
        irs = [
            _ir("SMH", rs_5d=0.03, rs_20d=0.05, rs_60d=0.08, slope=0.015, composite=85, rank=1),
            _ir("XBI", rs_5d=-0.03, rs_20d=-0.04, rs_60d=-0.06, slope=-0.01, composite=10, rank=20),
        ]
        results = classify_all_industries(irs)
        assert results["SMH"].state == AnalysisState.OVERT_PUMP
        assert results["XBI"].state == AnalysisState.OVERT_DUMP
