"""
State Classifier — 7-state symmetric architecture.

3 bullish: Accumulation → Broadening → Overt Pump
3 bearish: Distribution → Exhaustion → Overt Dump
1 neutral: Ambiguous

Classification order (critical):
  1. VETO checks (force Ambiguous)
  2. Overt Dump (strongest bearish)
  3. Overt Pump (strongest bullish)
  4. Exhaustion (mid bearish)
  5. Distribution (weak bearish)
  6. Broadening (mid bullish)
  7. Accumulation (weak bullish)
  8. Default → Ambiguous

Confidence floors enforce minimum confidence per state.
"""
from engine.schemas import (
    AnalysisState, TransitionPressure, RegimeState,
    StateClassification, PumpScoreReading, ReversalScoreReading,
    HorizonPattern,
)
from engine.explain import explain_state


_DELTA_NEAR_ZERO = 0.005

# Confidence floors — if computed confidence < floor, downgrade the state
_CONFIDENCE_FLOORS = {
    AnalysisState.OVERT_PUMP: 40,
    AnalysisState.OVERT_DUMP: 40,
    AnalysisState.BROADENING: 35,
    AnalysisState.EXHAUSTION: 35,
    AnalysisState.ACCUMULATION: 25,
    AnalysisState.DISTRIBUTION: 25,
    AnalysisState.AMBIGUOUS: 10,
}

# Downgrade chain when confidence too low
_DOWNGRADE = {
    AnalysisState.OVERT_PUMP: AnalysisState.BROADENING,
    AnalysisState.BROADENING: AnalysisState.ACCUMULATION,
    AnalysisState.ACCUMULATION: AnalysisState.AMBIGUOUS,
    AnalysisState.OVERT_DUMP: AnalysisState.EXHAUSTION,
    AnalysisState.EXHAUSTION: AnalysisState.DISTRIBUTION,
    AnalysisState.DISTRIBUTION: AnalysisState.AMBIGUOUS,
}


def classify_state(
    pump: PumpScoreReading,
    prior: StateClassification | None,
    regime: RegimeState,
    rs_rank: int,
    pump_percentile: float,
    delta_history: list[float],
    settings: dict,
    reversal_score: ReversalScoreReading | None = None,
    concentration=None,
    catalyst_confidence_modifier: int = 0,
    total_groups: int = 11,
    rs_5d: float = 0.0,
    rs_20d: float = 0.0,
    rs_60d: float = 0.0,
    horizon_pattern: HorizonPattern | None = None,
) -> StateClassification:
    """Classify a single sector/group's analysis state using 7-state architecture."""
    ticker = pump.ticker
    name = pump.name

    # Determine raw state
    state = _determine_state(pump, prior, rs_rank, pump_percentile, delta_history,
                             settings, reversal_score=reversal_score,
                             concentration=concentration, total_groups=total_groups,
                             rs_5d=rs_5d, rs_20d=rs_20d, rs_60d=rs_60d,
                             horizon_pattern=horizon_pattern)

    # Track sessions
    if prior and state == prior.state:
        sessions_in_state = prior.sessions_in_state + 1
    else:
        sessions_in_state = 1

    state_changed = prior is None or state != prior.state
    pressure = _compute_pressure(delta_history, state_changed)

    # Compute confidence
    confidence = _compute_confidence(pump, regime, rs_rank, pump_percentile,
                                      delta_history, state,
                                      reversal_score=reversal_score,
                                      concentration=concentration,
                                      catalyst_confidence_modifier=catalyst_confidence_modifier,
                                      horizon_pattern=horizon_pattern,
                                      rs_5d=rs_5d, rs_20d=rs_20d, rs_60d=rs_60d)

    # Apply confidence floors — downgrade if below minimum
    state, confidence = _apply_confidence_floors(state, confidence)

    prior_state = prior.state if prior else None
    classification = StateClassification(
        ticker=ticker, name=name, state=state, confidence=confidence,
        sessions_in_state=sessions_in_state, transition_pressure=pressure,
        prior_state=prior_state, state_changed=state_changed, explanation="",
    )
    classification.explanation = explain_state(classification, pump, regime)
    return classification


def classify_all_sectors(
    pumps: dict[str, PumpScoreReading],
    priors: dict[str, StateClassification],
    regime: RegimeState,
    rs_ranks: dict[str, int],
    pump_percentiles: dict[str, float],
    delta_histories: dict[str, list[float]],
    settings: dict,
    reversal_scores: dict[str, ReversalScoreReading] | None = None,
    concentrations: dict | None = None,
    catalyst_confidence_modifier: int = 0,
    rs_values: dict[str, tuple[float, float, float]] | None = None,
    horizon_patterns: dict[str, HorizonPattern] | None = None,
) -> dict[str, StateClassification]:
    """Classify all sectors. Returns dict[ticker, StateClassification]."""
    results = {}
    for ticker, pump in pumps.items():
        prior = priors.get(ticker)
        rank = rs_ranks.get(ticker, 6)
        pctl = pump_percentiles.get(ticker, 50.0)
        hist = delta_histories.get(ticker, [])
        rev = reversal_scores.get(ticker) if reversal_scores else None
        conc = concentrations.get(ticker) if concentrations else None

        rs_5d_val, rs_20d_val, rs_60d_val = (0.0, 0.0, 0.0)
        if rs_values and ticker in rs_values:
            rs_5d_val, rs_20d_val, rs_60d_val = rs_values[ticker]

        hp = None
        if horizon_patterns and ticker in horizon_patterns:
            hp = horizon_patterns[ticker]

        results[ticker] = classify_state(
            pump=pump, prior=prior, regime=regime,
            rs_rank=rank, pump_percentile=pctl,
            delta_history=hist, settings=settings,
            reversal_score=rev, concentration=conc,
            catalyst_confidence_modifier=catalyst_confidence_modifier,
            rs_5d=rs_5d_val, rs_20d=rs_20d_val, rs_60d=rs_60d_val,
            horizon_pattern=hp,
        )
    return results


# ═══════════════════════════════════════════════════════
# INTERNAL
# ═══════════════════════════════════════════════════════

def _determine_state(
    pump: PumpScoreReading,
    prior: StateClassification | None,
    rs_rank: int,
    pump_percentile: float,
    delta_history: list[float],
    settings: dict,
    reversal_score: ReversalScoreReading | None = None,
    concentration=None,
    total_groups: int = 11,
    rs_5d: float = 0.0,
    rs_20d: float = 0.0,
    rs_60d: float = 0.0,
    horizon_pattern: HorizonPattern | None = None,
) -> AnalysisState:
    """Core 7-state classification. Order is critical."""
    score = pump.pump_score
    delta = pump.pump_delta
    delta_5d = pump.pump_delta_5d_avg

    prior_state = prior.state if prior else None
    prior_sessions = prior.sessions_in_state if prior else 0

    rev_pctl = reversal_score.reversal_percentile if reversal_score else 0.0
    all_rs_negative = rs_5d < -0.001 and rs_20d < -0.001 and rs_60d < -0.001

    # Count consecutive negative deltas at tail
    consec_negative = 0
    for d in reversed(delta_history):
        if d < -_DELTA_NEAR_ZERO:
            consec_negative += 1
        else:
            break

    # Count consecutive positive deltas at tail
    consec_positive = 0
    for d in reversed(delta_history):
        if d > _DELTA_NEAR_ZERO:
            consec_positive += 1
        else:
            break

    # Recent delta analysis
    recent = delta_history[-5:] if len(delta_history) >= 5 else delta_history
    n_positive = sum(1 for d in recent if d > _DELTA_NEAR_ZERO)
    n_negative = sum(1 for d in recent if d < -_DELTA_NEAR_ZERO)
    is_mixed = n_positive >= 2 and n_negative >= 2

    max_ambiguous = settings.get("ambiguous", {}).get("max_duration", 15)
    bottom_quartile_rank = max(1, int(total_groups * 0.75)) + 1

    # ── STEP 1: VETO CHECKS (force Ambiguous) ──
    if rev_pctl > 90 and all_rs_negative:
        return AnalysisState.AMBIGUOUS

    if score > 0.60 and all_rs_negative:
        return AnalysisState.AMBIGUOUS

    if rev_pctl > 75 and delta > _DELTA_NEAR_ZERO and delta_5d < -_DELTA_NEAR_ZERO:
        return AnalysisState.AMBIGUOUS

    # ── VETO: Horizon pattern overrides ──
    # DEAD_CAT (↑↓↓) and FULL_REJECT (↓↓↓) block all bullish states
    _bullish_blocked = (horizon_pattern is not None and horizon_pattern in (
        HorizonPattern.DEAD_CAT, HorizonPattern.FULL_REJECT))

    # ── Forced exit from Ambiguous at max duration ──
    if prior_state == AnalysisState.AMBIGUOUS and prior_sessions >= max_ambiguous:
        pass  # Fall through to reclassify
    elif prior_state == AnalysisState.AMBIGUOUS and is_mixed:
        return AnalysisState.AMBIGUOUS

    # ── STEP 2: OVERT DUMP ──
    if (all_rs_negative
            and consec_negative >= 5
            and rev_pctl > 75
            and rs_rank >= bottom_quartile_rank):
        return AnalysisState.OVERT_DUMP

    # Continued Overt Dump
    if (prior_state in (AnalysisState.EXHAUSTION, AnalysisState.OVERT_DUMP,
                        AnalysisState.DISTRIBUTION)
            and delta < -_DELTA_NEAR_ZERO
            and rs_rank >= bottom_quartile_rank
            and reversal_score is not None and reversal_score.above_75th):
        return AnalysisState.OVERT_DUMP

    # ── STEP 3: OVERT PUMP ──
    min_overt_pctl = settings.get("overt_pump", {}).get("min_pump_percentile", 75)
    if (pump_percentile >= min_overt_pctl
            and rs_rank <= 3
            and delta > _DELTA_NEAR_ZERO
            and rev_pctl < 75
            and rs_60d > -0.005):
        return AnalysisState.OVERT_PUMP

    # Mature hold: top rank + high percentile + flat delta
    if (pump_percentile >= 70
            and rs_rank <= 3
            and abs(delta) <= _DELTA_NEAR_ZERO
            and consec_negative < 3
            and rev_pctl < 75):
        return AnalysisState.OVERT_PUMP

    # ── STEP 4: EXHAUSTION ──
    # Requires evidence of prior high
    had_prior_high = (
        prior_state in (AnalysisState.OVERT_PUMP, AnalysisState.BROADENING,
                        AnalysisState.ACCUMULATION)
        or rs_60d > 0.005
    )
    min_exh_sessions = settings.get("exhaustion", {}).get("pump_delta_nonpositive_sessions", 3)

    # Sustained leader exemption: rank #1 + Full Confirm + strong 60d RS
    # gets a much higher bar for Exhaustion (normal pullbacks don't trigger it)
    sl_cfg = settings.get("sustained_leader", {})
    _is_sustained_leader = (
        rs_rank == 1
        and rs_60d > sl_cfg.get("min_rs_60d", 0.15)
        and horizon_pattern == HorizonPattern.FULL_CONFIRM
        and not all_rs_negative
    )
    _exh_consec = min_exh_sessions + sl_cfg.get("extra_exh_sessions", 3) if _is_sustained_leader else min_exh_sessions
    _exh_rev_bar = sl_cfg.get("exh_rev_bar", 80) if _is_sustained_leader else 50

    if (had_prior_high
            and consec_negative >= _exh_consec
            and rev_pctl > _exh_rev_bar):
        if concentration is not None and hasattr(concentration, 'regime'):
            from engine.schemas import ConcentrationRegime
            if concentration.regime == ConcentrationRegime.CONCENTRATED_HEALTHY:
                if reversal_score is not None and reversal_score.above_75th:
                    return AnalysisState.EXHAUSTION
                else:
                    pass  # Suppress — fall through
            else:
                return AnalysisState.EXHAUSTION
        else:
            return AnalysisState.EXHAUSTION

    # ── STEP 5: DISTRIBUTION ──
    # Sustained leaders skip Distribution on moderate pullbacks (fall through to Broadening)
    min_dist_sessions = settings.get("distribution", {}).get("pump_delta_negative_sessions", 3)
    if (consec_negative >= min_dist_sessions
            and 0.35 <= score <= 0.65
            and not all_rs_negative
            and not _is_sustained_leader):
        return AnalysisState.DISTRIBUTION

    # Distribution from prior Overt Pump/Broadening with negative delta
    if (prior_state in (AnalysisState.OVERT_PUMP, AnalysisState.BROADENING)
            and consec_negative >= min_dist_sessions
            and score > 0.50
            and not _is_sustained_leader):
        return AnalysisState.DISTRIBUTION

    # ── STEP 6: BROADENING ──
    min_broad_sessions = settings.get("broadening", {}).get("rs_delta_positive_sessions", 5)
    if (consec_positive >= min_broad_sessions
            and pump_percentile > 50
            and rev_pctl < 75
            and not _bullish_blocked):
        return AnalysisState.BROADENING

    # ── STEP 6b: SUSTAINED LEADER ──
    # Rank #1 with strong 60d RS gets a wider reversal leash.
    _sustained_rev_threshold = 75
    if rs_rank == 1 and rs_60d > 0.20:
        _sustained_rev_threshold = 85
    _sl_max_neg = sl_cfg.get("broadening_max_consec_neg", 6)

    if (rs_rank == 1
            and pump_percentile >= 70  # Lowered from 80 — sustained leaders can dip
            and rs_60d > 0.10
            and rev_pctl < _sustained_rev_threshold
            and consec_negative < _sl_max_neg
            and not _bullish_blocked):
        return AnalysisState.BROADENING

    # ── STEP 7: ACCUMULATION ──
    if (delta > _DELTA_NEAR_ZERO
            and delta_5d > 0
            and rev_pctl < 75
            and not all_rs_negative
            and not _bullish_blocked):
        return AnalysisState.ACCUMULATION

    # Simple positive delta (weaker signal)
    if (delta > _DELTA_NEAR_ZERO and rev_pctl < 75
            and not all_rs_negative and not _bullish_blocked):
        return AnalysisState.ACCUMULATION

    # Negative delta without hitting Distribution thresholds
    if delta < -_DELTA_NEAR_ZERO and consec_negative >= min_dist_sessions:
        if had_prior_high:
            return AnalysisState.DISTRIBUTION
        return AnalysisState.DISTRIBUTION

    # ── STEP 8: AMBIGUOUS ──
    if is_mixed:
        return AnalysisState.AMBIGUOUS

    if prior is None:
        return AnalysisState.AMBIGUOUS

    # Stay in prior if non-ambiguous
    if prior_state and prior_state != AnalysisState.AMBIGUOUS:
        return prior_state

    return AnalysisState.AMBIGUOUS


def _apply_confidence_floors(state: AnalysisState, confidence: int) -> tuple:
    """Apply confidence floors. Downgrade state if confidence is below minimum."""
    while state in _CONFIDENCE_FLOORS:
        floor = _CONFIDENCE_FLOORS[state]
        if confidence >= floor:
            break
        # Downgrade to next state
        next_state = _DOWNGRADE.get(state)
        if next_state is None:
            break
        state = next_state
    return state, confidence


def _compute_pressure(delta_history: list[float], state_changed: bool) -> TransitionPressure:
    if state_changed:
        return TransitionPressure.BREAK
    if len(delta_history) < 3:
        return TransitionPressure.STABLE
    recent_3 = delta_history[-3:]
    if all(d > _DELTA_NEAR_ZERO for d in recent_3):
        return TransitionPressure.UP
    elif all(d < -_DELTA_NEAR_ZERO for d in recent_3):
        return TransitionPressure.DOWN
    return TransitionPressure.STABLE


def _compute_confidence(
    pump: PumpScoreReading,
    regime: RegimeState,
    rs_rank: int,
    pump_percentile: float,
    delta_history: list[float],
    state: AnalysisState,
    reversal_score: ReversalScoreReading | None = None,
    concentration=None,
    catalyst_confidence_modifier: int = 0,
    horizon_pattern: HorizonPattern | None = None,
    rs_5d: float = 0.0,
    rs_20d: float = 0.0,
    rs_60d: float = 0.0,
) -> int:
    confidence = 60

    if pump_percentile > 75 and rs_rank <= 3:
        confidence += 15
    elif pump_percentile > 50 and rs_rank <= 5:
        confidence += 10

    # All pillars aligned
    pillar_spread = (max(pump.rs_pillar, pump.participation_pillar, pump.flow_pillar) -
                     min(pump.rs_pillar, pump.participation_pillar, pump.flow_pillar))
    if pillar_spread < 15:
        confidence += 10
    elif pillar_spread > 50:
        confidence -= 15
    elif pillar_spread > 30:
        confidence -= 5

    if len(delta_history) >= 3:
        recent = delta_history[-3:]
        if all(d > _DELTA_NEAR_ZERO for d in recent):
            confidence += 10
        elif all(d < -_DELTA_NEAR_ZERO for d in recent):
            confidence += 5

    # Delta 5d avg confirms state direction
    delta_5d = pump.pump_delta_5d_avg
    bullish = state in (AnalysisState.ACCUMULATION, AnalysisState.BROADENING, AnalysisState.OVERT_PUMP)
    bearish = state in (AnalysisState.DISTRIBUTION, AnalysisState.EXHAUSTION, AnalysisState.OVERT_DUMP)
    if bullish and delta_5d > _DELTA_NEAR_ZERO:
        confidence += 5
    elif bullish and delta_5d < -_DELTA_NEAR_ZERO:
        confidence -= 15
    elif bearish and delta_5d < -_DELTA_NEAR_ZERO:
        confidence += 5
    elif bearish and delta_5d > _DELTA_NEAR_ZERO:
        confidence -= 15

    if state == AnalysisState.AMBIGUOUS:
        confidence -= 15

    if reversal_score is not None:
        if reversal_score.above_75th and pump.pump_delta > _DELTA_NEAR_ZERO:
            confidence -= 20  # Reversal > 75th in bullish = conflicting
        elif reversal_score.above_75th and pump.pump_delta < -_DELTA_NEAR_ZERO:
            confidence += 10  # Converging bearish evidence

    if concentration is not None and hasattr(concentration, 'participation_modifier'):
        confidence += concentration.participation_modifier

    confidence += catalyst_confidence_modifier

    # Horizon pattern confidence modifiers
    if horizon_pattern is not None:
        bullish = state in (AnalysisState.ACCUMULATION, AnalysisState.BROADENING, AnalysisState.OVERT_PUMP)
        bearish = state in (AnalysisState.DISTRIBUTION, AnalysisState.EXHAUSTION, AnalysisState.OVERT_DUMP)
        if horizon_pattern == HorizonPattern.FULL_CONFIRM and bullish:
            confidence += 10
        elif horizon_pattern == HorizonPattern.FULL_REJECT and bearish:
            confidence += 10
        elif horizon_pattern == HorizonPattern.ROTATION_IN and bullish:
            confidence += 5
        elif horizon_pattern == HorizonPattern.ROTATION_OUT and bearish:
            confidence += 5

    if regime == RegimeState.HOSTILE:
        confidence -= 25
    elif regime == RegimeState.FRAGILE:
        confidence -= 10  # Calibration seed v9.1 — reduced from -15 to avoid Ambiguous overload

    # ── Rank-1 confidence floor ──
    # The market's dominant sector with all horizons confirming cannot
    # drop below confidence 55 regardless of regime penalties.
    if (rs_rank == 1
            and horizon_pattern == HorizonPattern.FULL_CONFIRM
            and rs_5d > 0 and rs_20d > 0 and rs_60d > 0):
        confidence = max(confidence, 55)

    return max(10, min(95, confidence))
