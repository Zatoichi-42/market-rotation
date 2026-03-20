"""
Trade State Mapper — Layer 4. Pure deterministic function.
f(AnalysisState, Regime, Catalyst, Reversal, Rank, Delta) → TradeState

Signal hierarchy (no lower level overrides higher):
1. Regime Gate — HOSTILE → Hedge
2. Catalyst Gate — EMBARGO blocks new entries
3. Analysis State — determines action space
4. Reversal Score — gates short/pair entries
5. RS Rank + Delta — determines entry quality and sizing
"""
from engine.schemas import (
    AnalysisState, RegimeState, TradeState, TradeStateAssignment,
    CatalystAction, CatalystAssessment,
    PumpScoreReading, ReversalScoreReading, StateClassification,
)


def map_trade_state(
    state: StateClassification,
    pump: PumpScoreReading,
    regime: RegimeState,
    catalyst: CatalystAssessment,
    reversal: ReversalScoreReading | None = None,
    concentration=None,
    rs_rank: int = 6,
) -> TradeStateAssignment:
    """Map one sector's signals to a trade state. First match wins."""
    ticker = state.ticker
    name = state.name
    analysis = state.state
    confidence = state.confidence
    delta = pump.pump_delta
    delta_5d = pump.pump_delta_5d_avg
    catalyst_note = catalyst.action.value if catalyst else "Clear"

    # ── 1. HOSTILE → Hedge ──
    if regime == RegimeState.HOSTILE:
        return _build(ticker, name, analysis, TradeState.HEDGE, min(confidence, 30),
                      "Same session at close", "Regime returns to Normal",
                      "Per hostile limits", catalyst_note,
                      "HOSTILE regime overrides all signals. Hedge or go flat.")

    # ── 2. EMBARGO → block entries ──
    if catalyst and catalyst.action == CatalystAction.EMBARGO:
        if analysis == AnalysisState.OVERT_PUMP:
            return _build(ticker, name, analysis, TradeState.HOLD, confidence,
                          "—", "—", "—", catalyst_note,
                          f"EMBARGO ({catalyst.scheduled_catalyst}). Hold existing, no new entries.")
        return _build(ticker, name, analysis, TradeState.NO_TRADE, confidence,
                      "—", "—", "—", catalyst_note,
                      f"EMBARGO ({catalyst.scheduled_catalyst}). No entries until event passes.")

    # ── Size multiplier for FRAGILE ──
    regime_size = "0.5x (FRAGILE)" if regime == RegimeState.FRAGILE else "1.0x"

    # ── 3. Ambiguous → No Trade ──
    if analysis == AnalysisState.AMBIGUOUS:
        return _build(ticker, name, analysis, TradeState.NO_TRADE, confidence,
                      "—", "—", "—", catalyst_note,
                      "Ambiguous — conflicting signals, no edge. Wait for clarity.")

    # ── 4. Broadening → Long Entry ──
    if analysis == AnalysisState.BROADENING:
        if regime == RegimeState.FRAGILE:
            return _build(ticker, name, analysis, TradeState.WATCHLIST, confidence,
                          "—", "Regime → NORMAL AND delta still positive",
                          "0.5x (FRAGILE)", catalyst_note,
                          f"Broadening but FRAGILE regime. Small long or wait.")
        return _build(ticker, name, analysis, TradeState.LONG_ENTRY, confidence,
                      "Close today OR pullback to 20d MA",
                      "Delta negative 3 sessions OR regime → HOSTILE",
                      regime_size, catalyst_note,
                      f"Broadening — participation expanding. Best long entry zone.")

    # ── 5. Distribution → Reduce ──
    if analysis == AnalysisState.DISTRIBUTION:
        if regime == RegimeState.FRAGILE:
            return _build(ticker, name, analysis, TradeState.REDUCE, confidence,
                          "—", "Delta positive 5+ sessions",
                          "—", catalyst_note,
                          "Distribution + FRAGILE. Reduce faster, tighten stops.")
        return _build(ticker, name, analysis, TradeState.REDUCE, confidence,
                      "—", "Delta positive 5+ sessions",
                      "—", catalyst_note,
                      "Distribution — smart money exiting. Reduce, tighten stops. Do not short yet.")

    # ── 6-8. Overt Pump ──
    if analysis == AnalysisState.OVERT_PUMP:
        if delta > 0.01 and rs_rank <= 3:
            return _build(ticker, name, analysis, TradeState.SELECTIVE_ADD, confidence,
                          "Intraday pullback ≥1% that recovers by close",
                          "Delta negative 3 sessions OR regime → HOSTILE",
                          regime_size, catalyst_note,
                          f"Overt Pump, rank #{rs_rank}, delta {delta:+.3f}. Add on pullback only.")
        if rs_rank <= 3:
            return _build(ticker, name, analysis, TradeState.HOLD, confidence,
                          "—", "Delta negative 5 sessions OR reversal > 75th pctl",
                          regime_size, catalyst_note,
                          f"Overt Pump (mature), rank #{rs_rank}. Hold, stop adding.")
        return _build(ticker, name, analysis, TradeState.HOLD, confidence,
                      "—", "Falls out of top 5", regime_size, catalyst_note,
                      f"Overt Pump but rank #{rs_rank}. Not primary leader.")

    # ── 7-8. Accumulation ──
    if analysis == AnalysisState.ACCUMULATION:
        if rs_rank <= 5 and delta_5d > 0.005:
            if catalyst and catalyst.action == CatalystAction.SHOCK_PAUSE:
                return _build(ticker, name, analysis, TradeState.WATCHLIST, confidence,
                              "—", "—", "—", catalyst_note,
                              "Accumulation with momentum, but shock pause. Wait.")
            if confidence >= 50:
                conf_size = regime_size
            elif confidence >= 30:
                conf_size = "0.75x" if regime == RegimeState.NORMAL else "0.5x (FRAGILE)"
            else:
                conf_size = "0.5x" if regime == RegimeState.NORMAL else "0.5x (FRAGILE)"
            return _build(ticker, name, analysis, TradeState.LONG_ENTRY, confidence,
                          "Close today OR pullback ≥1.5% holding 20d MA within 5 sessions",
                          "Delta negative 5 sessions OR regime → HOSTILE",
                          conf_size, catalyst_note,
                          f"Accumulation, rank #{rs_rank}, 5d avg delta {delta_5d:+.3f}. Entry zone.")
        return _build(ticker, name, analysis, TradeState.WATCHLIST, confidence,
                      "—", "Delta positive 5+ sessions AND rank ≤5",
                      "—", catalyst_note,
                      f"Accumulation but rank #{rs_rank}, weak delta. Watch, don't enter.")

    # ── 9-10. Exhaustion ──
    if analysis == AnalysisState.EXHAUSTION:
        if reversal and reversal.above_75th:
            return _build(ticker, name, analysis, TradeState.PAIR_CANDIDATE, confidence,
                          "Day after confirmation, at close or failed rally",
                          "Reversal < 50th pctl OR pump delta turns positive",
                          "Defined-risk (put spread preferred)", catalyst_note,
                          f"Exhaustion + reversal {reversal.reversal_percentile:.0f}th pctl. Pair candidate.")
        return _build(ticker, name, analysis, TradeState.REDUCE, confidence,
                      "—", "Delta positive 3+ sessions",
                      "—", catalyst_note,
                      "Exhaustion, no reversal confirmation. Reduce, don't short yet.")

    # ── 11. Overt Dump ──
    if analysis == AnalysisState.OVERT_DUMP:
        return _build(ticker, name, analysis, TradeState.PAIR_CANDIDATE, confidence,
                      "Day after confirmation, at close or failed rally",
                      "Pump delta turns positive OR regime shift",
                      "Defined-risk only", catalyst_note,
                      "Overt Dump — active rotation out. Pair trade short leg.")

    # ── 12. Default ──
    return _build(ticker, name, analysis, TradeState.WATCHLIST, confidence,
                  "—", "—", "—", catalyst_note,
                  "No clear signal. Monitor for state change.")


def map_all_trade_states(
    states: dict[str, StateClassification],
    pumps: dict[str, PumpScoreReading],
    regime: RegimeState,
    catalyst: CatalystAssessment,
    rs_ranks: dict[str, int],
    reversal_scores: dict | None = None,
    concentrations: dict | None = None,
) -> dict[str, TradeStateAssignment]:
    """Map all sectors/industries to trade states."""
    results = {}
    for ticker, st in states.items():
        pump = pumps.get(ticker)
        if not pump:
            continue
        rev = reversal_scores.get(ticker) if reversal_scores else None
        conc = concentrations.get(ticker) if concentrations else None
        results[ticker] = map_trade_state(
            state=st, pump=pump, regime=regime, catalyst=catalyst,
            reversal=rev, concentration=conc, rs_rank=rs_ranks.get(ticker, 6),
        )
    return results


def _build(ticker, name, analysis, trade, confidence,
           entry, inval, size, catalyst_note, explanation):
    return TradeStateAssignment(
        ticker=ticker, name=name, analysis_state=analysis,
        trade_state=trade, confidence=confidence,
        entry_trigger=entry, invalidation=inval,
        size_class=size, catalyst_note=catalyst_note,
        explanation=explanation,
    )
