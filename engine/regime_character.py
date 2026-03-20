"""
Regime Character Classifier -- Phase 4, Step 4.

Converts raw regime signals into one of six market character labels.
The character tells the operator *what kind* of market this is, which
downstream modules use to adjust sizing, entry triggers, and exit urgency.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from engine.schemas import RegimeState, RegimeCharacter, RegimeCharacterReading


def classify_regime_character(
    spy_20d_return: float,
    vix_level: float,
    vix_20d_change: float,
    breadth_zscore: float,
    breadth_zscore_change_5d: float,
    cross_sector_dispersion: float,
    correlation_zscore: float,
    credit_zscore: float,
    gold_divergence_active: bool,
    gate_level: RegimeState,
    prior_character: Optional[RegimeCharacter] = None,
    sessions_in_prior: int = 0,
    dispersion_history: list[float] | None = None,
) -> RegimeCharacterReading:
    """
    Classify the current market into one of six regime characters.

    Classification uses first-match-wins priority ordering.
    A persistence filter requires 3 sessions to confirm a character
    change (except transitions into CRISIS, which are immediate).
    """
    # ----- raw classification (first match wins) -----
    raw_character = _classify_raw(
        spy_20d_return=spy_20d_return,
        vix_level=vix_level,
        vix_20d_change=vix_20d_change,
        breadth_zscore=breadth_zscore,
        breadth_zscore_change_5d=breadth_zscore_change_5d,
        cross_sector_dispersion=cross_sector_dispersion,
        correlation_zscore=correlation_zscore,
        gold_divergence_active=gold_divergence_active,
        gate_level=gate_level,
        prior_character=prior_character,
        dispersion_history=dispersion_history,
    )

    # ----- persistence filter -----
    character, sessions_in_character = _apply_persistence(
        raw_character=raw_character,
        prior_character=prior_character,
        sessions_in_prior=sessions_in_prior,
    )

    # ----- derived trends -----
    breadth_trend = _breadth_trend(breadth_zscore_change_5d)
    vix_trend = _vix_trend(vix_20d_change)

    # ----- confidence -----
    confidence = _compute_confidence(
        character=character,
        gate_level=gate_level,
        breadth_trend=breadth_trend,
        vix_trend=vix_trend,
    )

    # ----- description -----
    description = _build_description(character, gate_level, breadth_trend, vix_trend)

    return RegimeCharacterReading(
        character=character,
        gate_level=gate_level,
        confidence=confidence,
        spy_20d_return=spy_20d_return,
        cross_sector_dispersion=cross_sector_dispersion,
        breadth_trend=breadth_trend,
        vix_trend=vix_trend,
        prior_character=prior_character,
        sessions_in_character=sessions_in_character,
        description=description,
    )


# ── Private helpers ──────────────────────────────────────────


def _classify_raw(
    spy_20d_return: float,
    vix_level: float,
    vix_20d_change: float,
    breadth_zscore: float,
    breadth_zscore_change_5d: float,
    cross_sector_dispersion: float,
    correlation_zscore: float,
    gold_divergence_active: bool,
    gate_level: RegimeState,
    prior_character: Optional[RegimeCharacter],
    dispersion_history: list[float] | None,
) -> RegimeCharacter:
    """First-match-wins classification."""

    # 1. CRISIS
    if (
        gate_level == RegimeState.HOSTILE
        or (vix_level > 35 and correlation_zscore > 1.5)
        or gold_divergence_active
    ):
        return RegimeCharacter.CRISIS

    # 2. RECOVERY
    if (
        prior_character in (RegimeCharacter.CRISIS, RegimeCharacter.TRENDING_BEAR)
        and vix_20d_change < -3
        and breadth_zscore_change_5d > 0
        and spy_20d_return > 0
    ):
        return RegimeCharacter.RECOVERY

    # 3. TRENDING_BULL
    if (
        spy_20d_return > 0.02
        and correlation_zscore < 0.5
        and breadth_zscore > -0.3
        and vix_level < 22
    ):
        return RegimeCharacter.TRENDING_BULL

    # 4. TRENDING_BEAR
    if spy_20d_return < -0.02 and vix_level > 18 and breadth_zscore < 0:
        return RegimeCharacter.TRENDING_BEAR

    # 5. ROTATION
    if abs(spy_20d_return) < 0.02 and _dispersion_is_high(
        cross_sector_dispersion, dispersion_history
    ):
        return RegimeCharacter.ROTATION

    # 6. CHOPPY (everything else)
    return RegimeCharacter.CHOPPY


def _dispersion_is_high(
    current: float, history: list[float] | None
) -> bool:
    """Check whether current dispersion exceeds the 75th percentile of history."""
    if history and len(history) >= 1:
        threshold = float(np.percentile(history, 75))
        return current > threshold
    # No history available -- use static fallback
    return current > 0.03


def _apply_persistence(
    raw_character: RegimeCharacter,
    prior_character: Optional[RegimeCharacter],
    sessions_in_prior: int,
) -> tuple[RegimeCharacter, int]:
    """
    Require 3 sessions to confirm a character change.
    CRISIS always takes effect immediately.
    """
    if raw_character == RegimeCharacter.CRISIS:
        # CRISIS is never suppressed by persistence
        if prior_character == RegimeCharacter.CRISIS:
            return raw_character, sessions_in_prior + 1
        return raw_character, 1

    if prior_character is not None and raw_character != prior_character:
        if sessions_in_prior < 3:
            # Not enough sessions -- keep the prior character
            return prior_character, sessions_in_prior + 1
        # Enough sessions in prior -- allow transition
        return raw_character, 1

    if prior_character is not None and raw_character == prior_character:
        return raw_character, sessions_in_prior + 1

    # No prior -- first classification
    return raw_character, 1


def _breadth_trend(change_5d: float) -> str:
    if change_5d > 0.1:
        return "improving"
    if change_5d < -0.1:
        return "deteriorating"
    return "stable"


def _vix_trend(vix_20d_change: float) -> str:
    if vix_20d_change > 2:
        return "rising"
    if vix_20d_change < -2:
        return "declining"
    return "stable"


def _compute_confidence(
    character: RegimeCharacter,
    gate_level: RegimeState,
    breadth_trend: str,
    vix_trend: str,
) -> int:
    """
    Start at 60.
    +15 if gate matches character direction, -15 if it contradicts.
    +10 if VIX and breadth trends agree with character.
    Clamp to 0-100.
    """
    confidence = 60

    # Gate alignment
    if _gate_matches_character(character, gate_level):
        confidence += 15
    elif _gate_contradicts_character(character, gate_level):
        confidence -= 15

    # Trend alignment
    if _trends_agree_with_character(character, breadth_trend, vix_trend):
        confidence += 10

    return max(0, min(100, confidence))


def _gate_matches_character(character: RegimeCharacter, gate: RegimeState) -> bool:
    """Gate supports the character direction."""
    bullish = {RegimeCharacter.TRENDING_BULL, RegimeCharacter.RECOVERY}
    bearish = {RegimeCharacter.CRISIS, RegimeCharacter.TRENDING_BEAR}

    if character in bullish and gate == RegimeState.NORMAL:
        return True
    if character in bearish and gate == RegimeState.HOSTILE:
        return True
    return False


def _gate_contradicts_character(character: RegimeCharacter, gate: RegimeState) -> bool:
    """Gate works against the character direction."""
    bullish = {RegimeCharacter.TRENDING_BULL, RegimeCharacter.RECOVERY}
    bearish = {RegimeCharacter.CRISIS, RegimeCharacter.TRENDING_BEAR}

    if character in bullish and gate == RegimeState.HOSTILE:
        return True
    if character in bearish and gate == RegimeState.NORMAL:
        return True
    # Neutral characters (CHOPPY, ROTATION) with FRAGILE gate → mild contradiction
    if character in {RegimeCharacter.CHOPPY, RegimeCharacter.ROTATION} and gate == RegimeState.FRAGILE:
        return True
    return False


def _trends_agree_with_character(
    character: RegimeCharacter, breadth_trend: str, vix_trend: str
) -> bool:
    """VIX and breadth trends both support the character."""
    if character == RegimeCharacter.TRENDING_BULL:
        return breadth_trend == "improving" and vix_trend == "declining"
    if character == RegimeCharacter.TRENDING_BEAR:
        return breadth_trend == "deteriorating" and vix_trend == "rising"
    if character == RegimeCharacter.CRISIS:
        return breadth_trend == "deteriorating" and vix_trend == "rising"
    if character == RegimeCharacter.RECOVERY:
        return breadth_trend == "improving" and vix_trend == "declining"
    return False


def _build_description(
    character: RegimeCharacter,
    gate: RegimeState,
    breadth_trend: str,
    vix_trend: str,
) -> str:
    labels = {
        RegimeCharacter.TRENDING_BULL: "Trending bull market",
        RegimeCharacter.TRENDING_BEAR: "Trending bear market",
        RegimeCharacter.CHOPPY: "Choppy / range-bound market",
        RegimeCharacter.CRISIS: "Crisis conditions",
        RegimeCharacter.RECOVERY: "Recovery from stress",
        RegimeCharacter.ROTATION: "Sector rotation regime",
    }
    base = labels.get(character, character.value)
    return f"{base} (gate={gate.value}, breadth {breadth_trend}, VIX {vix_trend})."
