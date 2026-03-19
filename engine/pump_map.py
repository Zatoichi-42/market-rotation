"""
Pump Map Builder — Dual-engine output combining continuation + fragility.
"""
from engine.schemas import (
    PumpMapRow, GroupType, RegimeState, RegimeAssessment,
    RSReading, IndustryRSReading, PumpScoreReading,
    ReversalScoreReading, StateClassification, AnalysisState, TransitionPressure,
)


def build_pump_map(
    regime: RegimeAssessment,
    sector_rs: list[RSReading],
    industry_rs: list[IndustryRSReading],
    pump_scores: list[PumpScoreReading],
    reversal_scores: list[ReversalScoreReading],
    states: list[StateClassification],
    universe: dict,
) -> list[PumpMapRow]:
    """
    Assemble the dual-engine Pump Map.
    Combines all engine outputs into a single ranked table.
    """
    pump_map = {}
    for p in pump_scores:
        pump_map[p.ticker] = p
    rev_map = {}
    for r in reversal_scores:
        rev_map[r.ticker] = r
    state_map = {}
    for s in states:
        state_map[s.ticker] = s

    # Build tier lookup
    tier_map = {}
    for sec in universe.get("sectors", []):
        tier_map[sec["ticker"]] = sec.get("tier", "T1")
    for ind in universe.get("industries", []):
        tier_map[ind["ticker"]] = ind.get("tier", "T1")

    # Industry parent lookup
    ind_parent_map = {}
    for ind in universe.get("industries", []):
        ind_parent_map[ind["ticker"]] = ind["parent_sector"]

    rows = []

    # Sectors
    for rs in sector_rs:
        pump = pump_map.get(rs.ticker)
        rev = rev_map.get(rs.ticker)
        state = state_map.get(rs.ticker)
        rows.append(PumpMapRow(
            ticker=rs.ticker,
            name=rs.name,
            group_type=GroupType.SECTOR,
            parent_sector=None,
            tier=tier_map.get(rs.ticker, "T1"),
            regime_state=regime.state,
            pump_score=pump.pump_score if pump else 0.0,
            pump_delta=pump.pump_delta if pump else 0.0,
            pump_delta_5d_avg=pump.pump_delta_5d_avg if pump else 0.0,
            reversal_score=rev.reversal_score if rev else 0.0,
            reversal_percentile=rev.reversal_percentile if rev else 50.0,
            analysis_state=state.state if state else AnalysisState.AMBIGUOUS,
            transition_pressure=state.transition_pressure if state else TransitionPressure.STABLE,
            confidence=state.confidence if state else 50,
            rs_composite=rs.rs_composite,
            rs_rank=rs.rs_rank,
            rs_rank_change=rs.rs_rank_change,
            rs_vs_parent=None,
        ))

    # Industries
    for irs in industry_rs:
        pump = pump_map.get(irs.ticker)
        rev = rev_map.get(irs.ticker)
        state = state_map.get(irs.ticker)
        rows.append(PumpMapRow(
            ticker=irs.ticker,
            name=irs.name,
            group_type=GroupType.INDUSTRY,
            parent_sector=irs.parent_sector,
            tier=tier_map.get(irs.ticker, "T1"),
            regime_state=regime.state,
            pump_score=pump.pump_score if pump else 0.0,
            pump_delta=pump.pump_delta if pump else 0.0,
            pump_delta_5d_avg=pump.pump_delta_5d_avg if pump else 0.0,
            reversal_score=rev.reversal_score if rev else 0.0,
            reversal_percentile=rev.reversal_percentile if rev else 50.0,
            analysis_state=state.state if state else AnalysisState.AMBIGUOUS,
            transition_pressure=state.transition_pressure if state else TransitionPressure.STABLE,
            confidence=state.confidence if state else 50,
            rs_composite=irs.industry_composite,
            rs_rank=irs.rs_rank,
            rs_rank_change=irs.rs_rank_change,
            rs_vs_parent=irs.rs_composite_vs_parent,
        ))

    # Sort: state priority, then pump_delta descending
    state_priority = {
        AnalysisState.OVERT_PUMP: 0, AnalysisState.BROADENING: 1,
        AnalysisState.ACCUMULATION: 2, AnalysisState.EXHAUSTION: 3,
        AnalysisState.ROTATION: 4, AnalysisState.AMBIGUOUS: 5,
    }
    rows.sort(key=lambda r: (state_priority.get(r.analysis_state, 9), -r.pump_delta))

    return rows
