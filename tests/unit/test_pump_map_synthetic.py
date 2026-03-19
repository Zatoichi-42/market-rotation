"""
Pump Map assembly unit tests.
"""
import pytest
from engine.schemas import (
    PumpMapRow, GroupType, RegimeState, RegimeAssessment, SignalLevel, RegimeSignal,
    RSReading, IndustryRSReading, PumpScoreReading, ReversalScoreReading,
    StateClassification, AnalysisState, TransitionPressure,
)
from engine.pump_map import build_pump_map


def _regime():
    sig = RegimeSignal(name="vix", raw_value=15.0, level=SignalLevel.NORMAL, description="ok")
    return RegimeAssessment(state=RegimeState.NORMAL, signals=[sig],
                            hostile_count=0, fragile_count=0, normal_count=1,
                            timestamp="t", explanation="ok")


def _sector(ticker="XLK", rank=1, composite=80.0):
    return RSReading(ticker=ticker, name=ticker, rs_5d=0.01, rs_20d=0.02, rs_60d=0.03,
                     rs_slope=0.01, rs_rank=rank, rs_rank_change=0, rs_composite=composite)


def _industry(ticker="SMH", parent="XLK", rank=1, ind_comp=75.0):
    return IndustryRSReading(
        ticker=ticker, name=ticker, parent_sector=parent, group_type=GroupType.INDUSTRY,
        rs_5d=0.01, rs_20d=0.03, rs_60d=0.05, rs_slope=0.01, rs_composite=80.0,
        rs_5d_vs_parent=0.005, rs_20d_vs_parent=0.01, rs_60d_vs_parent=0.02,
        rs_slope_vs_parent=0.005, rs_composite_vs_parent=65.0,
        industry_composite=ind_comp, rs_rank=rank, rs_rank_change=0, rs_rank_within_sector=1,
    )


def _pump(ticker="XLK", score=0.70, delta=0.02):
    return PumpScoreReading(ticker=ticker, name=ticker, rs_pillar=70.0,
                            participation_pillar=60.0, flow_pillar=65.0,
                            pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta)


def _rev(ticker="XLK", score=0.25, pct=30.0, above=False):
    return ReversalScoreReading(ticker=ticker, name=ticker,
                                breadth_det_pillar=30.0, price_break_pillar=25.0, crowding_pillar=20.0,
                                reversal_score=score, sub_signals={}, reversal_percentile=pct, above_75th=above)


def _state(ticker="XLK", state=AnalysisState.BROADENING, conf=72):
    return StateClassification(ticker=ticker, name=ticker, state=state, confidence=conf,
                               sessions_in_state=5, transition_pressure=TransitionPressure.UP,
                               prior_state=None, state_changed=False, explanation="")


UNIVERSE = {
    "sectors": [{"ticker": "XLK", "tier": "T1"}, {"ticker": "XLE", "tier": "T1"}],
    "industries": [{"ticker": "SMH", "parent_sector": "XLK", "tier": "T1"}],
}


class TestPumpMapAssembly:

    def test_all_sectors_present(self):
        rows = build_pump_map(
            regime=_regime(),
            sector_rs=[_sector("XLK", 1), _sector("XLE", 2)],
            industry_rs=[],
            pump_scores=[_pump("XLK"), _pump("XLE", 0.50, -0.01)],
            reversal_scores=[_rev("XLK"), _rev("XLE")],
            states=[_state("XLK"), _state("XLE", AnalysisState.ACCUMULATION, 50)],
            universe=UNIVERSE,
        )
        tickers = {r.ticker for r in rows}
        assert "XLK" in tickers
        assert "XLE" in tickers

    def test_industries_included(self):
        rows = build_pump_map(
            regime=_regime(),
            sector_rs=[_sector("XLK")],
            industry_rs=[_industry("SMH")],
            pump_scores=[_pump("XLK"), _pump("SMH", 0.80, 0.05)],
            reversal_scores=[_rev("XLK"), _rev("SMH")],
            states=[_state("XLK"), _state("SMH", AnalysisState.OVERT_PUMP, 80)],
            universe=UNIVERSE,
        )
        tickers = {r.ticker for r in rows}
        assert "SMH" in tickers

    def test_parent_sector_populated_for_industries(self):
        rows = build_pump_map(
            regime=_regime(),
            sector_rs=[_sector("XLK")],
            industry_rs=[_industry("SMH", "XLK")],
            pump_scores=[_pump("XLK"), _pump("SMH")],
            reversal_scores=[_rev("XLK"), _rev("SMH")],
            states=[_state("XLK"), _state("SMH")],
            universe=UNIVERSE,
        )
        smh = next(r for r in rows if r.ticker == "SMH")
        assert smh.parent_sector == "XLK"
        assert smh.group_type == GroupType.INDUSTRY

    def test_parent_sector_none_for_sectors(self):
        rows = build_pump_map(
            regime=_regime(),
            sector_rs=[_sector("XLK")],
            industry_rs=[],
            pump_scores=[_pump("XLK")],
            reversal_scores=[_rev("XLK")],
            states=[_state("XLK")],
            universe=UNIVERSE,
        )
        xlk = next(r for r in rows if r.ticker == "XLK")
        assert xlk.parent_sector is None
        assert xlk.group_type == GroupType.SECTOR

    def test_reversal_warning_flagged(self):
        rows = build_pump_map(
            regime=_regime(),
            sector_rs=[_sector("XLK")],
            industry_rs=[],
            pump_scores=[_pump("XLK")],
            reversal_scores=[_rev("XLK", score=0.80, pct=90.0, above=True)],
            states=[_state("XLK")],
            universe=UNIVERSE,
        )
        xlk = next(r for r in rows if r.ticker == "XLK")
        assert xlk.reversal_percentile == 90.0

    def test_returns_pump_map_row(self):
        rows = build_pump_map(
            regime=_regime(),
            sector_rs=[_sector("XLK")],
            industry_rs=[],
            pump_scores=[_pump("XLK")],
            reversal_scores=[_rev("XLK")],
            states=[_state("XLK")],
            universe=UNIVERSE,
        )
        assert isinstance(rows[0], PumpMapRow)
