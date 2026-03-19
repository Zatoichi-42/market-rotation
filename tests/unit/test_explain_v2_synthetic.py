"""
Explain v2 unit tests — reversal, turnover, industry RS explanations.
"""
import pytest
from engine.schemas import (
    ReversalScoreReading, TurnoverCheck, IndustryRSReading,
    GroupType, RegimeState, AnalysisState,
)
from engine.explain import explain_reversal, explain_turnover, explain_industry_rs


def _rev(ticker="XLK", score=0.65, pct=82.0, above=True,
         bd=75.0, pb=62.0, cr=58.0):
    return ReversalScoreReading(
        ticker=ticker, name=ticker,
        breadth_det_pillar=bd, price_break_pillar=pb, crowding_pillar=cr,
        reversal_score=score, sub_signals={"rs_slope": -0.05, "clv_trend": -0.3},
        reversal_percentile=pct, above_75th=above,
    )


def _ind(ticker="SMH", parent="XLK", rs20=0.052, rs20p=0.021, rank=3, rank_chg=2):
    return IndustryRSReading(
        ticker=ticker, name="Semiconductors", parent_sector=parent,
        group_type=GroupType.INDUSTRY,
        rs_5d=0.02, rs_20d=rs20, rs_60d=0.08, rs_slope=0.01, rs_composite=80.0,
        rs_5d_vs_parent=0.01, rs_20d_vs_parent=rs20p, rs_60d_vs_parent=0.03,
        rs_slope_vs_parent=0.005, rs_composite_vs_parent=65.0,
        industry_composite=75.0, rs_rank=rank, rs_rank_change=rank_chg,
        rs_rank_within_sector=1,
    )


class TestExplainReversal:

    def test_elevated_reversal(self):
        result = explain_reversal(_rev(score=0.68, pct=82.0, above=True), RegimeState.NORMAL)
        assert "XLK" in result
        assert "0.68" in result or "68" in result
        assert "82" in result or "percentile" in result.lower()

    def test_low_reversal(self):
        result = explain_reversal(_rev(score=0.15, pct=12.0, above=False), RegimeState.NORMAL)
        assert "low" in result.lower() or "15" in result

    def test_hostile_regime_noted(self):
        result = explain_reversal(_rev(), RegimeState.HOSTILE)
        assert "HOSTILE" in result

    def test_pillar_values_included(self):
        result = explain_reversal(_rev(bd=75.0, pb=62.0, cr=58.0), RegimeState.NORMAL)
        assert "75" in result
        assert "62" in result
        assert "58" in result

    def test_returns_string(self):
        assert isinstance(explain_reversal(_rev(), RegimeState.NORMAL), str)
        assert len(explain_reversal(_rev(), RegimeState.NORMAL)) > 20


class TestExplainTurnover:

    def test_pass_explanation(self):
        tc = TurnoverCheck(
            candidate_ticker="XLV", current_ticker="XLK",
            delta_advantage=0.12, persistence_sessions=5,
            current_state_exempt=False, passes_filter=True,
            reason="PASS: clear advantage",
        )
        result = explain_turnover(tc)
        assert "PASS" in result
        assert "XLV" in result
        assert "XLK" in result

    def test_fail_explanation(self):
        tc = TurnoverCheck(
            candidate_ticker="XLI", current_ticker="XLF",
            delta_advantage=0.03, persistence_sessions=5,
            current_state_exempt=False, passes_filter=False,
            reason="FAIL: marginal",
        )
        result = explain_turnover(tc)
        assert "FAIL" in result or "marginal" in result.lower() or "do not" in result.lower()

    def test_exempt_explanation(self):
        tc = TurnoverCheck(
            candidate_ticker="XLV", current_ticker="XLK",
            delta_advantage=0.03, persistence_sessions=1,
            current_state_exempt=True, passes_filter=True,
            reason="PASS: exempt",
        )
        result = explain_turnover(tc)
        assert "exempt" in result.lower()

    def test_returns_string(self):
        tc = TurnoverCheck("A", "B", 0.1, 3, False, True, "ok")
        assert isinstance(explain_turnover(tc), str)


class TestExplainIndustryRS:

    def test_outperforming_parent(self):
        result = explain_industry_rs(_ind(rs20p=0.021))
        assert "SMH" in result
        assert "XLK" in result
        assert "outperforming" in result.lower() or "+2.1" in result or "driving" in result.lower()

    def test_underperforming_parent(self):
        ind = _ind(ticker="XBI", parent="XLV", rs20p=-0.015)
        ind = IndustryRSReading(
            ticker="XBI", name="Biotech", parent_sector="XLV",
            group_type=GroupType.INDUSTRY,
            rs_5d=0.0, rs_20d=-0.01, rs_60d=-0.03, rs_slope=-0.01, rs_composite=30.0,
            rs_5d_vs_parent=-0.005, rs_20d_vs_parent=-0.015, rs_60d_vs_parent=-0.02,
            rs_slope_vs_parent=-0.005, rs_composite_vs_parent=25.0,
            industry_composite=28.5, rs_rank=15, rs_rank_change=-2,
            rs_rank_within_sector=2,
        )
        result = explain_industry_rs(ind)
        assert "XBI" in result
        assert "lagging" in result.lower() or "underperforming" in result.lower()

    def test_includes_rank(self):
        result = explain_industry_rs(_ind(rank=3, rank_chg=2))
        assert "#3" in result or "rank 3" in result.lower()

    def test_returns_string(self):
        assert isinstance(explain_industry_rs(_ind()), str)
        assert len(explain_industry_rs(_ind())) > 20
