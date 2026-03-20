"""
Human-readable explanation generator for the Pump Rotation System.

Generates plain-English summaries for:
- Regime classifications
- State assignments
- Breadth readings

Data source convention:
  live data (alt data — source/disclaimer) [lagging data — source/disclaimer]
"""
import math

from engine.schemas import (
    RegimeState, SignalLevel, RegimeAssessment,
    AnalysisState, TransitionPressure, StateClassification,
    BreadthSignal, BreadthReading, PumpScoreReading,
    ReversalScoreReading, TurnoverCheck, IndustryRSReading,
)


# ── Source tags ────────────────────────────────────────

_SRC_VIX = "^VIX via yfinance — live"
_SRC_VIX3M = "^VIX3M via yfinance — live"
_SRC_HYG_LQD = "HYG/LQD via yfinance — live"
_SRC_RSP_SPY = "RSP/SPY via yfinance — live"
_SRC_FRED_OAS = "BAMLH0A0HYM2 via FRED — 1–2 business day lag"

_SIGNAL_SOURCES = {
    "vix": _SRC_VIX,
    "term_structure": f"{_SRC_VIX} / {_SRC_VIX3M}",
    "breadth": _SRC_RSP_SPY,
    "credit": _SRC_HYG_LQD,
    "oil": "CL=F via yfinance — live",
}


def explain_regime(assessment: RegimeAssessment, fred_hy_oas_value: float | None = None) -> str:
    """
    Generate human-readable explanation for regime classification.
    Cites all data sources with lag disclaimers.
    """
    state = assessment.state
    signals = assessment.signals

    if not signals:
        return f"{state.value}: No signals available — classification based on defaults."

    # Build per-signal detail with source tags
    sig_details = []
    for s in signals:
        src = _SIGNAL_SOURCES.get(s.name, "unknown source")
        sig_details.append(f"{s.name}={s.raw_value:.2f} [{s.level.value}] ({src})")
    details_str = "; ".join(sig_details)

    # FRED supplement line
    fred_line = ""
    if fred_hy_oas_value is not None and not math.isnan(fred_hy_oas_value):
        fred_line = (
            f" [FRED HY OAS: {fred_hy_oas_value:.0f} bps ({_SRC_FRED_OAS})]"
        )

    if state == RegimeState.NORMAL:
        return (
            f"NORMAL: All {len(signals)} signals clear. "
            f"{details_str}.{fred_line} "
            f"Full operation — standard position sizing."
        )
    elif state == RegimeState.HOSTILE:
        hostile_sigs = [s for s in signals if s.level == SignalLevel.HOSTILE]
        hostile_names = ", ".join(s.name for s in hostile_sigs)
        return (
            f"HOSTILE: {assessment.hostile_count} hostile signal(s) triggered "
            f"({hostile_names}). {details_str}.{fred_line} "
            f"Action: No new momentum longs. Hedge/cash only."
        )
    else:  # FRAGILE
        concerning = [s for s in signals if s.level in (SignalLevel.HOSTILE, SignalLevel.FRAGILE)]
        concerning_names = ", ".join(s.name for s in concerning)
        return (
            f"FRAGILE: {assessment.hostile_count} hostile, "
            f"{assessment.fragile_count} fragile signal(s) "
            f"({concerning_names}). {details_str}.{fred_line} "
            f"Action: Reduce position sizes, tighten stops, caution on new entries."
        )


def explain_state(
    classification: StateClassification,
    pump: PumpScoreReading,
    regime: RegimeState,
) -> str:
    """Generate human-readable explanation for state assignment."""
    ticker = classification.ticker
    name = classification.name
    state = classification.state
    conf = classification.confidence
    sessions = classification.sessions_in_state
    pressure = classification.transition_pressure
    score = pump.pump_score
    delta = pump.pump_delta

    parts = [
        f"{ticker} ({name}) classified as {state.value} "
        f"(confidence: {conf}, {sessions} session(s) in state)."
    ]

    # State-specific context
    if state == AnalysisState.OVERT_PUMP:
        parts.append(
            f"Pump score {score:.2f} (top quartile) with delta {delta:+.3f} — "
            f"strong institutional flow, clear sector leadership."
        )
    elif state == AnalysisState.ACCUMULATION:
        parts.append(
            f"Pump score {score:.2f} with delta {delta:+.3f} — "
            f"momentum building, participation expanding."
        )
    elif state == AnalysisState.EXHAUSTION:
        parts.append(
            f"Pump score {score:.2f} with delta {delta:+.3f} — "
            f"momentum fading, watch for rotation signal."
        )
    elif state == AnalysisState.OVERT_DUMP:
        parts.append(
            f"Pump score {score:.2f} declining (delta {delta:+.3f}) — "
            f"active rotation out, capital leaving."
        )
    elif state == AnalysisState.AMBIGUOUS:
        parts.append(
            f"Pump score {score:.2f}, delta {delta:+.3f} — "
            f"conflicting signals, no clear direction."
        )

    parts.append(f"Transition pressure: {pressure.value}.")

    # Regime overlay
    if regime == RegimeState.HOSTILE:
        parts.append("Regime is HOSTILE — all trade signals overridden, no new longs.")
    elif regime == RegimeState.FRAGILE:
        parts.append("Regime is FRAGILE — reduced confidence, tighten risk management.")
    else:
        parts.append("Regime is NORMAL — full operation allowed.")

    return " ".join(parts)


def explain_breadth(reading: BreadthReading) -> str:
    """Generate human-readable breadth explanation with source citations."""
    signal = reading.signal
    ratio = reading.rsp_spy_ratio
    change = reading.rsp_spy_ratio_20d_change
    zscore = reading.rsp_spy_ratio_zscore

    z_str = f"{zscore:.2f}" if not math.isnan(zscore) else "N/A (insufficient history)"
    src = f"({_SRC_RSP_SPY})"

    if signal == BreadthSignal.HEALTHY:
        return (
            f"HEALTHY breadth. RSP/SPY ratio: {ratio:.4f}, "
            f"20d change: {change:+.4f}, z-score: {z_str} {src}. "
            f"Broad market participation confirmed — equal-weight keeping pace."
        )
    elif signal == BreadthSignal.NARROWING:
        return (
            f"NARROWING breadth. RSP/SPY ratio: {ratio:.4f}, "
            f"20d change: {change:+.4f}, z-score: {z_str} {src}. "
            f"Equal-weight underperforming cap-weight — participation weakening."
        )
    else:
        return (
            f"DIVERGING breadth. RSP/SPY ratio: {ratio:.4f}, "
            f"20d change: {change:+.4f}, z-score: {z_str} {src}. "
            f"Significant breadth divergence — narrow leadership, caution warranted."
        )


# ═══════════════════════════════════════════════════════
# PHASE 2: REVERSAL, TURNOVER, INDUSTRY
# ═══════════════════════════════════════════════════════

def explain_reversal(reading: ReversalScoreReading, regime: RegimeState) -> str:
    """Human-readable explanation for reversal score."""
    score = reading.reversal_score
    pct = reading.reversal_percentile
    bd = reading.breadth_det_pillar
    pb = reading.price_break_pillar
    cr = reading.crowding_pillar

    if reading.above_75th:
        level = "ELEVATED fragility"
    elif pct > 50:
        level = "moderate fragility"
    else:
        level = "low fragility"

    parts = [
        f"{reading.ticker} Reversal Score: {score:.2f} "
        f"({pct:.0f}th percentile — {level})."
    ]
    parts.append(
        f"Breadth Det: {bd:.0f}/100, "
        f"Price Break: {pb:.0f}/100, "
        f"Crowding: {cr:.0f}/100."
    )

    if regime == RegimeState.HOSTILE:
        parts.append("Regime is HOSTILE — reversal risk compounded by macro stress.")
    elif regime == RegimeState.FRAGILE:
        parts.append("Regime is FRAGILE — elevated reversal scores warrant extra caution.")

    return " ".join(parts)


def explain_turnover(check: TurnoverCheck) -> str:
    """Human-readable explanation for turnover filter decision."""
    if check.current_state_exempt:
        return (
            f"Turnover filter: PASS (exempt). {check.current_ticker} is in an exempt state. "
            f"Candidate {check.candidate_ticker} accepted with delta advantage "
            f"{check.delta_advantage:+.3f}."
        )
    elif check.passes_filter:
        return (
            f"Turnover filter: PASS. {check.candidate_ticker} Pump delta exceeds "
            f"{check.current_ticker} by {check.delta_advantage:+.3f} for "
            f"{check.persistence_sessions} consecutive sessions. Rotation justified."
        )
    else:
        return (
            f"Turnover filter: FAIL. {check.candidate_ticker} delta advantage "
            f"{check.delta_advantage:+.3f} over {check.current_ticker} — "
            f"marginal improvement, do not rotate."
        )


def explain_industry_rs(reading: IndustryRSReading) -> str:
    """Human-readable explanation for industry RS."""
    ticker = reading.ticker
    name = reading.name
    parent = reading.parent_sector
    rs20 = reading.rs_20d
    rs20p = reading.rs_20d_vs_parent
    rank = reading.rs_rank
    rank_chg = reading.rs_rank_change

    rank_dir = ""
    if rank_chg > 0:
        rank_dir = f", up from #{rank + rank_chg}"
    elif rank_chg < 0:
        rank_dir = f", down from #{rank + rank_chg}"

    if rs20p > 0.001:
        vs_parent_desc = f"outperforming own sector by {rs20p:+.1%} — driving the sector"
    elif rs20p < -0.001:
        vs_parent_desc = f"lagging own sector by {rs20p:+.1%} — underperforming within sector"
    else:
        vs_parent_desc = "tracking sector closely — neutral vs parent"

    return (
        f"{ticker} ({name}, parent: {parent}): "
        f"vs SPY: 20d RS = {rs20:+.1%} (rank #{rank} of industries{rank_dir}). "
        f"vs {parent}: 20d RS = {rs20p:+.1%} — {vs_parent_desc}."
    )
