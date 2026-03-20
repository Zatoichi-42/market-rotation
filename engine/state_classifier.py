"""
State Classifier — Assigns preliminary analysis state labels to each sector.

States:
- Accumulation: Low-to-mid score, positive delta, early stage
- Broadening: Delta positive 5+ sessions, score crossing above 50th pctl
- Overt Pump: Top quartile score, top-3 rank, delta positive
- Exhaustion: Was top quartile, delta negative 3+ sessions
- Rotation/Reversal: Score declining, rank dropping
- Ambiguous: Conflicting signals, no clear direction

Confidence is adjusted by:
- Regime overlay (HOSTILE -30, FRAGILE -20)
- Pillar alignment (conflicting pillars reduce confidence)
- Clamped to [10, 95]
"""
from engine.schemas import (
    AnalysisState, TransitionPressure, RegimeState,
    StateClassification, PumpScoreReading, ReversalScoreReading,
)
from engine.explain import explain_state


# Threshold for "near zero" delta
_DELTA_NEAR_ZERO = 0.005


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
) -> StateClassification:
    """
    Classify a single sector's analysis state.

    Args:
        pump: Current session's PumpScoreReading
        prior: Previous session's StateClassification (None if first)
        regime: Current regime state
        rs_rank: Current RS rank (1 = strongest)
        pump_percentile: Pump score percentile (0-100) across all sectors
        delta_history: Recent pump delta values (newest last)
        settings: State classifier settings from config
        reversal_score: Optional ReversalScoreReading (Phase 2).
        concentration: Optional ConcentrationReading. When CONCENTRATED_HEALTHY,
                       require reversal_score.above_75th before Exhaustion transition.
        catalyst_confidence_modifier: Catalyst gate penalty (negative int).
    """
    ticker = pump.ticker
    name = pump.name

    # Determine new state
    state = _determine_state(pump, prior, rs_rank, pump_percentile, delta_history, settings,
                             reversal_score=reversal_score, concentration=concentration)

    # Track sessions in state
    if prior and state == prior.state:
        sessions_in_state = prior.sessions_in_state + 1
    else:
        sessions_in_state = 1

    state_changed = prior is None or state != prior.state

    # Compute transition pressure
    pressure = _compute_pressure(delta_history, state_changed)

    # Compute confidence
    confidence = _compute_confidence(pump, regime, rs_rank, pump_percentile, delta_history, state,
                                     reversal_score=reversal_score,
                                     concentration=concentration,
                                     catalyst_confidence_modifier=catalyst_confidence_modifier)

    # Prior state reference
    prior_state = prior.state if prior else None

    # Build classification (explanation filled below)
    classification = StateClassification(
        ticker=ticker, name=name, state=state, confidence=confidence,
        sessions_in_state=sessions_in_state, transition_pressure=pressure,
        prior_state=prior_state, state_changed=state_changed, explanation="",
    )

    # Generate explanation
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

        results[ticker] = classify_state(
            pump=pump, prior=prior, regime=regime,
            rs_rank=rank, pump_percentile=pctl,
            delta_history=hist, settings=settings,
            reversal_score=rev,
            concentration=conc,
            catalyst_confidence_modifier=catalyst_confidence_modifier,
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
) -> AnalysisState:
    """Core state determination logic."""
    score = pump.pump_score
    delta = pump.pump_delta

    prior_state = prior.state if prior else None
    prior_sessions = prior.sessions_in_state if prior else 0

    # Count recent positive / negative / mixed deltas
    recent = delta_history[-5:] if len(delta_history) >= 5 else delta_history
    n_positive = sum(1 for d in recent if d > _DELTA_NEAR_ZERO)
    n_negative = sum(1 for d in recent if d < -_DELTA_NEAR_ZERO)
    n_near_zero = len(recent) - n_positive - n_negative
    is_mixed = n_positive >= 2 and n_negative >= 2

    # Count consecutive nonpositive deltas at tail
    consec_nonpositive = 0
    for d in reversed(delta_history):
        if d <= _DELTA_NEAR_ZERO:
            consec_nonpositive += 1
        else:
            break

    # Count consecutive positive deltas at tail
    consec_positive = 0
    for d in reversed(delta_history):
        if d > _DELTA_NEAR_ZERO:
            consec_positive += 1
        else:
            break

    min_overt_pctl = settings.get("overt_pump", {}).get("min_pump_percentile", 75)
    min_dist_nonpos = settings.get("exhaustion", {}).get("pump_delta_nonpositive_sessions", 3)
    max_ambiguous = settings.get("ambiguous", {}).get("max_duration", 15)

    # ── Forced exit from Ambiguous at max duration ──
    if prior_state == AnalysisState.AMBIGUOUS and prior_sessions >= max_ambiguous:
        pass  # Fall through to reclassify
    elif prior_state == AnalysisState.AMBIGUOUS and is_mixed:
        return AnalysisState.AMBIGUOUS

    # ── OVERT PUMP: top quartile + top 3 rank + delta positive ──
    if pump_percentile >= min_overt_pctl and rs_rank <= 3 and delta > _DELTA_NEAR_ZERO:
        return AnalysisState.OVERT_PUMP

    # ── OVERT PUMP (MATURE): high score + top rank + flat delta ──
    # Prevents misclassifying a mature leader as Exhaustion.
    # A sector at rank ≤3 with pump_percentile ≥70 and delta near zero is holding, not exhausting.
    if (pump_percentile >= 70
            and rs_rank <= 3
            and abs(delta) <= _DELTA_NEAR_ZERO
            and consec_nonpositive < min_dist_nonpos + 2):
        return AnalysisState.OVERT_PUMP

    # ── OVERT DUMP: via reversal score (Exhaustion + high reversal) ──
    if (prior_state == AnalysisState.EXHAUSTION
            and reversal_score is not None
            and reversal_score.above_75th):
        return AnalysisState.OVERT_DUMP

    # ── OVERT DUMP: continued decline + bottom rank ──
    if (prior_state in (AnalysisState.EXHAUSTION, AnalysisState.OVERT_DUMP)
            and delta < -_DELTA_NEAR_ZERO and rs_rank >= 7):
        return AnalysisState.OVERT_DUMP

    # ── EXHAUSTION: was strong, now DECLINING (requires negative deltas, not just flat) ──
    consec_negative = 0
    for d in reversed(delta_history):
        if d < -_DELTA_NEAR_ZERO:
            consec_negative += 1
        else:
            break

    if (prior_state in (AnalysisState.OVERT_PUMP, AnalysisState.ACCUMULATION)
            and consec_negative >= min_dist_nonpos):
        if concentration is not None and hasattr(concentration, 'regime'):
            from engine.schemas import ConcentrationRegime
            if concentration.regime == ConcentrationRegime.CONCENTRATED_HEALTHY:
                # Only allow Exhaustion if reversal score also confirms
                if reversal_score is not None and reversal_score.above_75th:
                    return AnalysisState.EXHAUSTION
                else:
                    return prior_state  # Suppress — leaders still healthy
        return AnalysisState.EXHAUSTION

    # ── ACCUMULATION: positive delta or building momentum ──
    if delta > _DELTA_NEAR_ZERO:
        return AnalysisState.ACCUMULATION

    # ── AMBIGUOUS: mixed signals ──
    if is_mixed:
        return AnalysisState.AMBIGUOUS

    # ── First classification ──
    if prior is None:
        return AnalysisState.AMBIGUOUS

    # ── Default: stay in prior state ──
    if prior_state and prior_state != AnalysisState.AMBIGUOUS:
        return prior_state

    return AnalysisState.AMBIGUOUS


def _compute_pressure(delta_history: list[float], state_changed: bool) -> TransitionPressure:
    """Compute transition pressure from recent delta history."""
    if state_changed:
        return TransitionPressure.BREAK

    if len(delta_history) < 3:
        return TransitionPressure.STABLE

    recent_3 = delta_history[-3:]

    if all(d > _DELTA_NEAR_ZERO for d in recent_3):
        return TransitionPressure.UP
    elif all(d < -_DELTA_NEAR_ZERO for d in recent_3):
        return TransitionPressure.DOWN
    else:
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
) -> int:
    """Compute confidence score (10-95)."""
    confidence = 60  # Base

    # Strong aligned signals boost confidence
    if pump_percentile > 75 and rs_rank <= 3:
        confidence += 15
    elif pump_percentile > 50 and rs_rank <= 5:
        confidence += 10

    # Consistent delta direction boosts confidence
    if len(delta_history) >= 3:
        recent = delta_history[-3:]
        if all(d > _DELTA_NEAR_ZERO for d in recent):
            confidence += 10
        elif all(d < -_DELTA_NEAR_ZERO for d in recent):
            confidence += 5  # Consistent decline is still a signal

    # Pillar alignment — check if pillars are telling the same story
    pillar_spread = max(pump.rs_pillar, pump.participation_pillar, pump.flow_pillar) - \
                    min(pump.rs_pillar, pump.participation_pillar, pump.flow_pillar)
    if pillar_spread > 50:
        confidence -= 15  # Conflicting pillars
    elif pillar_spread > 30:
        confidence -= 5

    # Ambiguous state = low confidence
    if state == AnalysisState.AMBIGUOUS:
        confidence -= 15

    # Reversal score integration (Phase 2)
    if reversal_score is not None:
        pump_delta = pump.pump_delta
        if reversal_score.above_75th and pump_delta < -_DELTA_NEAR_ZERO:
            # Converging evidence of exhaustion → boost confidence
            confidence += 10
        elif reversal_score.above_75th and pump_delta > _DELTA_NEAR_ZERO:
            # Conflicting: pump rising but reversal elevated → reduce confidence
            confidence -= 15
        elif not reversal_score.above_75th and pump_delta > _DELTA_NEAR_ZERO:
            # Confirming: pump rising, low reversal risk → boost
            confidence += 5

    # Concentration modifier
    if concentration is not None and hasattr(concentration, 'participation_modifier'):
        confidence += concentration.participation_modifier

    # Catalyst modifier
    confidence += catalyst_confidence_modifier

    # Regime overlay
    if regime == RegimeState.HOSTILE:
        confidence -= 30
    elif regime == RegimeState.FRAGILE:
        confidence -= 20

    # Clamp
    return max(10, min(95, confidence))
