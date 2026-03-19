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
    if state == AnalysisState.BROADENING:
        parts.append(
            f"Pump score {score:.2f} with delta {delta:+.3f} — "
            f"momentum building, participation expanding."
        )
    elif state == AnalysisState.OVERT_PUMP:
        parts.append(
            f"Pump score {score:.2f} (top quartile) with delta {delta:+.3f} — "
            f"strong institutional flow, clear sector leadership."
        )
    elif state == AnalysisState.EXHAUSTION:
        parts.append(
            f"Pump score {score:.2f} with delta {delta:+.3f} — "
            f"momentum fading, watch for rotation signal."
        )
    elif state == AnalysisState.ACCUMULATION:
        parts.append(
            f"Pump score {score:.2f} with delta {delta:+.3f} — "
            f"early-stage positioning, not yet confirmed."
        )
    elif state == AnalysisState.ROTATION:
        parts.append(
            f"Pump score {score:.2f} declining (delta {delta:+.3f}) — "
            f"capital rotating out, look for receiving sector."
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
