"""Trade State Mapper — tests every mapping path."""
import pytest
from engine.schemas import (
    AnalysisState, RegimeState, TransitionPressure, TradeState,
    CatalystAction, CatalystAssessment, ShockType,
    StateClassification, PumpScoreReading, ReversalScoreReading,
)
from engine.trade_state_mapper import map_trade_state


def _state(ticker="XLE", state=AnalysisState.OVERT_PUMP, confidence=60):
    return StateClassification(
        ticker=ticker, name="Test", state=state, confidence=confidence,
        sessions_in_state=5, transition_pressure=TransitionPressure.STABLE,
        prior_state=AnalysisState.ACCUMULATION, state_changed=False, explanation="",
    )

def _pump(ticker="XLE", score=0.70, delta=0.05, delta_5d=0.02):
    return PumpScoreReading(ticker=ticker, name="Test", rs_pillar=80.0,
        participation_pillar=60.0, flow_pillar=55.0,
        pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta_5d)

def _catalyst(action=CatalystAction.CLEAR):
    return CatalystAssessment(action=action, scheduled_catalyst=None,
        shock_detected=ShockType.NONE, shock_magnitude=0.0,
        affected_sectors=[], confidence_modifier=0, explanation="", multi_sector_count=5)

def _reversal(above=True, pctl=90.0):
    return ReversalScoreReading(ticker="XLE", name="Test",
        breadth_det_pillar=30.0, price_break_pillar=50.0, crowding_pillar=40.0,
        reversal_score=0.40, sub_signals={}, reversal_percentile=pctl, above_75th=above)


class TestRegimeOverride:
    def test_hostile_forces_hedge_on_overt_pump(self):
        r = map_trade_state(_state(), _pump(), RegimeState.HOSTILE, _catalyst(), rs_rank=1)
        assert r.trade_state == TradeState.HEDGE

    def test_hostile_forces_hedge_on_accumulation(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.08),
            RegimeState.HOSTILE, _catalyst(), rs_rank=1)
        assert r.trade_state == TradeState.HEDGE

    def test_hostile_forces_hedge_on_exhaustion(self):
        r = map_trade_state(_state(state=AnalysisState.EXHAUSTION), _pump(delta=-0.03),
            RegimeState.HOSTILE, _catalyst(), rs_rank=10)
        assert r.trade_state == TradeState.HEDGE


class TestCatalystOverride:
    def test_embargo_blocks_accumulation_entry(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.05, delta_5d=0.03),
            RegimeState.NORMAL, _catalyst(CatalystAction.EMBARGO), rs_rank=2)
        assert r.trade_state == TradeState.NO_TRADE

    def test_embargo_allows_hold_on_overt_pump(self):
        r = map_trade_state(_state(), _pump(), RegimeState.NORMAL,
            _catalyst(CatalystAction.EMBARGO), rs_rank=1)
        assert r.trade_state == TradeState.HOLD

    def test_embargo_note_in_output(self):
        r = map_trade_state(_state(), _pump(), RegimeState.NORMAL,
            _catalyst(CatalystAction.EMBARGO), rs_rank=1)
        assert "Embargo" in r.catalyst_note or "EMBARGO" in r.catalyst_note


class TestOvertPump:
    def test_top_rank_positive_delta_selective_add(self):
        r = map_trade_state(_state(), _pump(delta=0.05), RegimeState.NORMAL, _catalyst(), rs_rank=1)
        assert r.trade_state == TradeState.SELECTIVE_ADD

    def test_top_rank_flat_delta_hold(self):
        r = map_trade_state(_state(), _pump(delta=0.002), RegimeState.NORMAL, _catalyst(), rs_rank=2)
        assert r.trade_state == TradeState.HOLD

    def test_mid_rank_hold(self):
        r = map_trade_state(_state(), _pump(delta=0.05), RegimeState.NORMAL, _catalyst(), rs_rank=6)
        assert r.trade_state == TradeState.HOLD

    def test_has_entry_trigger(self):
        r = map_trade_state(_state(), _pump(delta=0.05), RegimeState.NORMAL, _catalyst(), rs_rank=1)
        assert r.entry_trigger != "—"

    def test_has_invalidation(self):
        r = map_trade_state(_state(), _pump(delta=0.05), RegimeState.NORMAL, _catalyst(), rs_rank=1)
        assert r.invalidation != "—"


class TestAccumulation:
    def test_good_rank_positive_delta_long_entry(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.04, delta_5d=0.02),
            RegimeState.NORMAL, _catalyst(), rs_rank=3)
        assert r.trade_state == TradeState.LONG_ENTRY

    def test_bad_rank_watchlist(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.04, delta_5d=0.02),
            RegimeState.NORMAL, _catalyst(), rs_rank=8)
        assert r.trade_state == TradeState.WATCHLIST

    def test_weak_delta_5d_watchlist(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.003, delta_5d=0.002),
            RegimeState.NORMAL, _catalyst(), rs_rank=2)
        assert r.trade_state == TradeState.WATCHLIST

    def test_fragile_regime_reduces_size(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.05, delta_5d=0.03),
            RegimeState.FRAGILE, _catalyst(), rs_rank=3)
        assert r.trade_state == TradeState.LONG_ENTRY
        assert "0.5" in r.size_class

    def test_long_entry_has_entry_trigger(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.05, delta_5d=0.03),
            RegimeState.NORMAL, _catalyst(), rs_rank=3)
        assert r.entry_trigger != "—"


class TestExhaustion:
    def test_reversal_above_75th_pair_candidate(self):
        r = map_trade_state(_state(state=AnalysisState.EXHAUSTION), _pump(delta=-0.02),
            RegimeState.NORMAL, _catalyst(), rs_rank=9, reversal=_reversal(above=True))
        assert r.trade_state == TradeState.PAIR_CANDIDATE

    def test_reversal_below_75th_reduce(self):
        r = map_trade_state(_state(state=AnalysisState.EXHAUSTION), _pump(delta=-0.02),
            RegimeState.NORMAL, _catalyst(), rs_rank=9, reversal=_reversal(above=False, pctl=50.0))
        assert r.trade_state == TradeState.REDUCE

    def test_no_reversal_data_reduce(self):
        r = map_trade_state(_state(state=AnalysisState.EXHAUSTION), _pump(delta=-0.02),
            RegimeState.NORMAL, _catalyst(), rs_rank=9)
        assert r.trade_state == TradeState.REDUCE


class TestAmbiguous:
    def test_ambiguous_always_no_trade(self):
        r = map_trade_state(_state(state=AnalysisState.AMBIGUOUS), _pump(delta=0.08),
            RegimeState.NORMAL, _catalyst(), rs_rank=1)
        assert r.trade_state == TradeState.NO_TRADE


class TestOvertDump:
    def test_overt_dump_pair_candidate(self):
        r = map_trade_state(_state(state=AnalysisState.OVERT_DUMP), _pump(delta=-0.05),
            RegimeState.NORMAL, _catalyst(), rs_rank=11)
        assert r.trade_state == TradeState.PAIR_CANDIDATE


class TestSignalHierarchy:
    def test_regime_beats_catalyst(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.05, delta_5d=0.03),
            RegimeState.HOSTILE, _catalyst(CatalystAction.CLEAR), rs_rank=1)
        assert r.trade_state == TradeState.HEDGE

    def test_catalyst_beats_analysis(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.05, delta_5d=0.03),
            RegimeState.NORMAL, _catalyst(CatalystAction.EMBARGO), rs_rank=2)
        assert r.trade_state == TradeState.NO_TRADE

    def test_caution_does_not_block_entry(self):
        r = map_trade_state(_state(state=AnalysisState.ACCUMULATION), _pump(delta=0.05, delta_5d=0.03),
            RegimeState.NORMAL, _catalyst(CatalystAction.CAUTION), rs_rank=3)
        assert r.trade_state == TradeState.LONG_ENTRY
