"""
Industry state classification based on multi-timeframe RS pattern.

Uses 5d, 20d, 60d RS + slope to determine state directly from the RS trajectory,
rather than requiring pump score history.

Pattern logic:
  Overt Pump:   RS positive at 5d+20d, top rank, slope positive
  Accumulation: RS improving (slope positive or 5d > 0)
  Ambiguous:    Mixed signals (some timeframes positive, some negative)
  Exhaustion:   RS was strong (60d positive) but 5d/20d weakening (slope negative)
  Overt Dump:   RS negative at 5d+20d+60d, slope negative, bottom rank
"""
from engine.schemas import (
    AnalysisState, TransitionPressure, StateClassification,
    IndustryRSReading, ReversalScoreReading, RegimeState,
)


def classify_industry_state(
    ir: IndustryRSReading,
    regime: RegimeState = RegimeState.NORMAL,
    reversal_score: ReversalScoreReading | None = None,
    prior: StateClassification | None = None,
) -> StateClassification:
    """
    Classify an industry's state from its multi-timeframe RS pattern.
    """
    rs_5d = ir.rs_5d
    rs_20d = ir.rs_20d
    rs_60d = ir.rs_60d
    slope = ir.rs_slope
    rank = ir.rs_rank
    composite = ir.industry_composite

    # Count positive/negative timeframes
    tf_pos = sum(1 for v in [rs_5d, rs_20d, rs_60d] if v > 0.001)
    tf_neg = sum(1 for v in [rs_5d, rs_20d, rs_60d] if v < -0.001)
    all_rs_neg = tf_neg == 3

    # Check vs-parent: if underperforming parent by >5% on 20d, not truly driving sector
    vs_parent_ok = ir.rs_20d_vs_parent > -0.05

    # Determine state from RS pattern (7-state)
    if tf_pos >= 2 and slope > 0.001 and rank <= 5 and composite >= 70 and vs_parent_ok:
        state = AnalysisState.OVERT_PUMP
    elif tf_pos >= 2 and slope > 0.001 and not all_rs_neg:
        state = AnalysisState.BROADENING
    elif (slope > 0.001 or rs_5d > 0.001) and not all_rs_neg:
        state = AnalysisState.ACCUMULATION
    elif rs_60d > 0.005 and (slope < -0.001 or rs_5d < -0.001):
        # Was strong (60d positive) but short-term weakening
        state = AnalysisState.EXHAUSTION
    elif all_rs_neg and slope < -0.001 and rank >= 15:
        state = AnalysisState.OVERT_DUMP
    elif tf_neg >= 2 and slope < -0.001 and rs_60d > 0.001:
        state = AnalysisState.EXHAUSTION
    elif tf_neg >= 2 and slope < -0.001:
        state = AnalysisState.DISTRIBUTION
    elif all_rs_neg and slope > 0.001:
        # All RS negative but slope turning — bottom fishing or dead cat bounce
        state = AnalysisState.AMBIGUOUS
    elif tf_pos >= 1 and tf_neg >= 1:
        state = AnalysisState.AMBIGUOUS
    else:
        state = AnalysisState.AMBIGUOUS

    # Reversal score can push Exhaustion → Overt Dump
    if (state == AnalysisState.EXHAUSTION
            and reversal_score is not None
            and reversal_score.above_75th):
        state = AnalysisState.OVERT_DUMP

    # Track sessions
    if prior and state == prior.state:
        sessions = prior.sessions_in_state + 1
        changed = False
    else:
        sessions = 1
        changed = True

    # Transition pressure from slope
    if changed:
        pressure = TransitionPressure.BREAK
    elif slope > 0.002:
        pressure = TransitionPressure.UP
    elif slope < -0.002:
        pressure = TransitionPressure.DOWN
    else:
        pressure = TransitionPressure.STABLE

    # Confidence
    confidence = 60
    if tf_pos == 3 and slope > 0:
        confidence += 20  # All timeframes aligned positive
    elif tf_neg == 3 and slope < 0:
        confidence += 15  # All aligned negative
    elif tf_pos >= 1 and tf_neg >= 1:
        confidence -= 15  # Mixed
    if regime == RegimeState.HOSTILE:
        confidence -= 30
    elif regime == RegimeState.FRAGILE:
        confidence -= 20
    confidence = max(10, min(95, confidence))

    # Explanation
    explanation = (
        f"{ir.ticker} ({ir.name}): {state.value} "
        f"(RS 5d={rs_5d:+.2%}, 20d={rs_20d:+.2%}, 60d={rs_60d:+.2%}, "
        f"slope={slope:+.4f}, rank #{rank}, composite={composite:.0f})"
    )

    return StateClassification(
        ticker=ir.ticker, name=ir.name, state=state,
        confidence=confidence, sessions_in_state=sessions,
        transition_pressure=pressure,
        prior_state=prior.state if prior else None,
        state_changed=changed, explanation=explanation,
    )


def classify_all_industries(
    industry_rs: list[IndustryRSReading],
    regime: RegimeState = RegimeState.NORMAL,
    reversal_scores: dict | None = None,
    priors: dict | None = None,
) -> dict[str, StateClassification]:
    """Classify all industries. Returns {ticker: StateClassification}."""
    results = {}
    for ir in industry_rs:
        rev = reversal_scores.get(ir.ticker) if reversal_scores else None
        prior = priors.get(ir.ticker) if priors else None
        results[ir.ticker] = classify_industry_state(ir, regime, rev, prior)
    return results
