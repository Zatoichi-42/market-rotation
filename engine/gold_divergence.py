"""
Gold/VIX Divergence Detector — regime modifier.

Detects margin-call / liquidity crisis regime when gold sells off
WITH equities while VIX is elevated. This is a special-condition
detector that can only tighten the regime gate.
"""
import pandas as pd

from engine.schemas import SignalLevel, RegimeState, GoldDivergenceReading


def compute_gold_vix_divergence(
    gold_prices: pd.Series,
    spy_prices: pd.Series,
    vix_level: float,
    gold_decline_threshold: float = -0.02,
    spy_decline_threshold: float = -0.02,
    vix_threshold: float = 25.0,
) -> GoldDivergenceReading | None:
    """
    Detect margin-call / liquidity crisis regime.

    HOSTILE: Gold 5d return < -2% AND SPY 5d return < -2% AND VIX > 25
    FRAGILE: Gold flat/down while SPY down AND VIX > 20
    NORMAL:  Otherwise

    Returns None if insufficient data.
    """
    if gold_prices is None or spy_prices is None:
        return None
    if len(gold_prices) < 6 or len(spy_prices) < 6:
        return None

    gold_5d = (gold_prices.iloc[-1] / gold_prices.iloc[-6]) - 1
    spy_5d = (spy_prices.iloc[-1] / spy_prices.iloc[-6]) - 1

    is_margin_call = (
        gold_5d < gold_decline_threshold
        and spy_5d < spy_decline_threshold
        and vix_level > vix_threshold
    )

    if is_margin_call:
        level = SignalLevel.HOSTILE
        desc = (
            f"MARGIN CALL REGIME: Gold {gold_5d:+.1%} (5d) selling with equities "
            f"(SPY {spy_5d:+.1%}) while VIX at {vix_level:.1f}. "
            f"Forced liquidation — ALL assets at risk including safe havens."
        )
    elif gold_5d < 0 and spy_5d < spy_decline_threshold and vix_level > 20:
        level = SignalLevel.FRAGILE
        desc = (
            f"Gold/VIX watch: Gold {gold_5d:+.1%} (5d) while SPY {spy_5d:+.1%} "
            f"and VIX at {vix_level:.1f}. Approaching margin-call signature."
        )
    else:
        level = SignalLevel.NORMAL
        desc = f"Gold/VIX normal. Gold {gold_5d:+.1%}, SPY {spy_5d:+.1%}, VIX {vix_level:.1f}."

    return GoldDivergenceReading(
        gold_5d_return=float(gold_5d),
        spy_5d_return=float(spy_5d),
        vix_level=float(vix_level),
        is_margin_call_regime=bool(is_margin_call),
        level=level,
        description=desc,
    )


def apply_gold_divergence_modifier(
    current_state: RegimeState,
    gd_reading: GoldDivergenceReading | None,
) -> tuple:
    """
    Apply gold/VIX divergence as regime modifier. Can only tighten.

    Returns (new_state, modifier_explanation).
    """
    if gd_reading is None:
        return current_state, ""

    if gd_reading.is_margin_call_regime:
        explanation = (
            "MARGIN CALL REGIME: Gold selling with equities while VIX elevated. "
            "Forced liquidation / liquidity demand, not normal risk-off. "
            "ALL assets at risk including traditional safe havens."
        )
        if current_state == RegimeState.NORMAL:
            return RegimeState.FRAGILE, explanation
        return current_state, explanation

    return current_state, ""
