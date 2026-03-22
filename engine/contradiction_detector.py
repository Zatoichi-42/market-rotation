"""
Contradiction Detector — finds internal conflicts in system output.
Each contradiction has type, severity, ticker, detail, and a question for LLM evaluation.
"""
from engine.schemas import AnalysisState, RegimeState, HorizonPattern, CrisisType

_BULLISH_STATES = {AnalysisState.ACCUMULATION, AnalysisState.BROADENING, AnalysisState.OVERT_PUMP}
_BEARISH_STATES = {AnalysisState.DISTRIBUTION, AnalysisState.EXHAUSTION, AnalysisState.OVERT_DUMP}
_BULLISH_HORIZONS = {HorizonPattern.FULL_CONFIRM, HorizonPattern.ROTATION_IN, HorizonPattern.HEALTHY_DIP}

def detect_contradictions(result: dict) -> list[dict]:
    """
    Run all contradiction checks against the pipeline result.
    Returns list of dicts with keys: type, severity, ticker, detail, question
    """
    contradictions = []

    states = result.get("states", {})
    trade_states = result.get("trade_states", {})
    horizon_readings = result.get("horizon_readings", {})
    regime = result.get("regime")
    pumps = result.get("pumps", {})
    reversal_map = result.get("reversal_map", {})
    crisis_types = result.get("crisis_types", [])

    regime_state = regime.state if regime else RegimeState.NORMAL

    for ticker, sc in states.items():
        state = sc.state
        ts = trade_states.get(ticker)
        hr = horizon_readings.get(ticker)
        pump = pumps.get(ticker)
        rev = reversal_map.get(ticker)

        trade_val = ts.trade_state.value if ts else "?"
        horizon_pattern = hr.pattern if hr else HorizonPattern.NO_PATTERN
        name = sc.name

        # 1. REGIME_OVERRIDE_BULLISH
        if state in _BULLISH_STATES and regime_state == RegimeState.HOSTILE:
            contradictions.append({
                "type": "REGIME_OVERRIDE_BULLISH",
                "severity": "HIGH",
                "ticker": ticker,
                "detail": f"{ticker} ({name}) is {state.value} (bullish) but regime HOSTILE blocks all longs.",
                "question": f"Is the HOSTILE driver actually bullish for {ticker} ({name})? Check crisis alignment.",
            })

        # 2. CRISIS_BENEFICIARY_BLOCKED (uses crisis alignment matrix)
        if state in _BULLISH_STATES and regime_state in (RegimeState.HOSTILE, RegimeState.FRAGILE):
            from engine.crisis_alignment import CRISIS_ALIGNMENT, CrisisType as CT, DEFAULT_ALIGNMENT
            for ct in crisis_types:
                if ct in (CT.NONE, CT.MULTI_CRISIS):
                    continue
                alignment = CRISIS_ALIGNMENT.get(ct, {})
                long_mod, _ = alignment.get(ticker, DEFAULT_ALIGNMENT)
                if long_mod > 1.0:
                    contradictions.append({
                        "type": "CRISIS_BENEFICIARY_BLOCKED",
                        "severity": "HIGH",
                        "ticker": ticker,
                        "detail": (f"{ticker} ({name}) has long_modifier {long_mod} for {ct.value} — "
                                   f"it BENEFITS from the crisis that triggered {regime_state.value}."),
                        "question": (f"Should {ticker} ({name}) get a regime exemption during {ct.value}? "
                                     f"It's the strongest bullish signal blocked by its own tailwind."),
                    })
                    break  # One crisis type is enough

        # 3. HORIZON_TRADE_CONFLICT
        if horizon_pattern in _BULLISH_HORIZONS and trade_val in ("Hedge", "No Trade", "Reduce"):
            sev = "HIGH" if horizon_pattern == HorizonPattern.FULL_CONFIRM else "MEDIUM"
            contradictions.append({
                "type": "HORIZON_TRADE_CONFLICT",
                "severity": sev,
                "ticker": ticker,
                "detail": f"{ticker} ({name}) horizon is {horizon_pattern.value} (bullish) but trade state is {trade_val}.",
                "question": f"When regime clears, is {ticker} ({name}) the first entry candidate?",
            })

        # 4. DELTA_STATE_MISMATCH
        if pump and pump.pump_delta > 0.03 and state in (_BEARISH_STATES | {AnalysisState.AMBIGUOUS}):
            contradictions.append({
                "type": "DELTA_STATE_MISMATCH",
                "severity": "MEDIUM",
                "ticker": ticker,
                "detail": (f"{ticker} ({name}) has pump delta {pump.pump_delta:+.3f} (strong positive) "
                           f"but state is {state.value}."),
                "question": (f"Is {ticker} ({name})'s delta surge a genuine reversal or mechanical bounce? "
                             + (f"Horizon is Dead Cat — system thinks it's a trap." if horizon_pattern == HorizonPattern.DEAD_CAT else "")),
            })

        # 5. LOW_CONFIDENCE_DIRECTIONAL
        if state != AnalysisState.AMBIGUOUS and sc.confidence <= 20:
            contradictions.append({
                "type": "LOW_CONFIDENCE_DIRECTIONAL",
                "severity": "LOW",
                "ticker": ticker,
                "detail": f"{ticker} ({name}) is {state.value} but confidence only {sc.confidence}%.",
                "question": f"Is this barely-above-threshold or genuine edge?",
            })

        # 6. REVERSAL_BULLISH_CONFLICT
        if rev and rev.above_75th and state in _BULLISH_STATES:
            contradictions.append({
                "type": "REVERSAL_BULLISH_CONFLICT",
                "severity": "MEDIUM",
                "ticker": ticker,
                "detail": f"{ticker} ({name}) is {state.value} but reversal score is {rev.reversal_percentile:.0f}th percentile (above 75th).",
                "question": f"Is {ticker} ({name})'s bullish classification safe with high fragility?",
            })

    # System-level checks
    # 7. COVERAGE_GAP
    ambiguous_count = sum(1 for sc in states.values() if sc.state == AnalysisState.AMBIGUOUS)
    total = len(states) if states else 1
    if total > 0 and ambiguous_count / total > 0.5:
        sector_count = sum(1 for t in states if len(t) <= 4)  # rough sector filter
        contradictions.append({
            "type": "COVERAGE_GAP",
            "severity": "HIGH",
            "ticker": "SYSTEM",
            "detail": f"{ambiguous_count} of {total} groups are Ambiguous ({ambiguous_count/total:.0%}).",
            "question": "Genuine uncertainty or classifier over-penalization?",
        })

    # 8. ALL_SAME_TRADE
    if trade_states:
        trade_vals = {ts.trade_state.value for ts in trade_states.values()}
        if len(trade_vals) == 1:
            contradictions.append({
                "type": "ALL_SAME_TRADE",
                "severity": "HIGH",
                "ticker": "SYSTEM",
                "detail": f"All sectors mapped to {list(trade_vals)[0]}. Zero differentiation.",
                "question": "Is the gate over-triggered? The system provides no sector-level value.",
            })

    return contradictions
