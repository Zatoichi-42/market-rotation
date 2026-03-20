"""
Gold/Silver Ratio — regime modifier.

Computes GLD/SLV ratio z-score against 504-day history.
Can only tighten the regime gate (NORMAL→FRAGILE), never loosen it.
Interacts with gold/VIX divergence as dual precious metals stress signal.
"""
import math
import numpy as np
import pandas as pd

from engine.schemas import SignalLevel, GoldSilverRatioReading


def compute_gold_silver_ratio(
    gld_prices: pd.Series,
    slv_prices: pd.Series,
    window: int = 504,
    gold_vix_divergence_active: bool = False,
) -> GoldSilverRatioReading | None:
    """
    Compute gold/silver ratio regime signal.

    Method:
        1. ratio = GLD / SLV (daily)
        2. z-score ratio vs `window`-day history
        3. Classify level
        4. Check for margin-call amplification

    Thresholds:
        z-score < 1.0   → NORMAL
        1.0 ≤ z < 2.0   → FRAGILE
        z-score ≥ 2.0    → HOSTILE

    Boundary rule: exact threshold goes to the worse bucket.

    Returns None if insufficient data for computation.
    """
    if gld_prices is None or slv_prices is None:
        return None
    if len(gld_prices) < 2 or len(slv_prices) < 2:
        return None

    # Align series
    combined = pd.DataFrame({"gld": gld_prices, "slv": slv_prices}).dropna()
    if len(combined) < 2:
        return None

    ratio_series = combined["gld"] / combined["slv"]
    ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio_series) < 2:
        return None

    current_ratio = ratio_series.iloc[-1]

    # Z-score against history
    hist = ratio_series.tail(window)
    mean = hist.mean()
    std = hist.std()
    if std == 0 or math.isnan(std):
        zscore = 0.0
    else:
        zscore = (current_ratio - mean) / std

    # Classify level (exact boundary → worse bucket)
    if zscore >= 2.0:
        level = SignalLevel.HOSTILE
    elif zscore >= 1.0:
        level = SignalLevel.FRAGILE
    else:
        level = SignalLevel.NORMAL

    # 5d returns
    gold_5d = _safe_return(combined["gld"], 5)
    silver_5d = _safe_return(combined["slv"], 5)
    silver_underperforming = bool(silver_5d < gold_5d)

    # Margin-call amplifier: requires BOTH gold/VIX divergence AND silver weaker
    margin_call_amplifier = bool(gold_vix_divergence_active and silver_underperforming)

    # Description
    if margin_call_amplifier:
        desc = (
            f"Gold/silver ratio {current_ratio:.2f} ({zscore:+.2f}σ). "
            f"Silver falling harder than gold (silver {silver_5d:+.1%} vs gold {gold_5d:+.1%}) "
            f"confirms forced liquidation (speculative positioning unwinding)."
        )
    elif level == SignalLevel.HOSTILE:
        desc = (
            f"Gold/silver ratio {current_ratio:.2f} ({zscore:+.2f}σ) — EXTREME. "
            f"Industrial demand collapsing. Silver {silver_5d:+.1%} vs Gold {gold_5d:+.1%}."
        )
    elif level == SignalLevel.FRAGILE:
        if silver_underperforming:
            desc = (
                f"Gold/silver ratio {current_ratio:.2f} ({zscore:+.2f}σ) — elevated. "
                f"Silver underperforming gold — industrial demand weakening. "
                f"Silver {silver_5d:+.1%} vs Gold {gold_5d:+.1%}."
            )
        else:
            desc = (
                f"Gold/silver ratio {current_ratio:.2f} ({zscore:+.2f}σ) — elevated. "
                f"Silver {silver_5d:+.1%} vs Gold {gold_5d:+.1%}."
            )
    else:
        desc = f"Gold/silver ratio {current_ratio:.2f} ({zscore:+.2f}σ) — normal range."

    return GoldSilverRatioReading(
        ratio=current_ratio,
        ratio_zscore=zscore,
        level=level,
        gold_5d_return=gold_5d,
        silver_5d_return=silver_5d,
        silver_underperforming=silver_underperforming,
        margin_call_amplifier=margin_call_amplifier,
        description=desc,
    )


def apply_gold_silver_modifier(
    current_state,
    gs_reading: GoldSilverRatioReading | None,
    gold_vix_divergence_active: bool = False,
) -> tuple:
    """
    Apply gold/silver ratio as regime modifier. Can only tighten, never loosen.

    Returns (new_state, modifier_explanation).
    """
    from engine.schemas import RegimeState

    if gs_reading is None:
        return current_state, ""

    explanations = []

    # Dual modifier check
    if gold_vix_divergence_active and gs_reading.level in (SignalLevel.FRAGILE, SignalLevel.HOSTILE):
        explanations.append(
            "DUAL PRECIOUS METALS STRESS: Gold selling with equities (margin-call signal) "
            "AND silver underperforming gold (speculative liquidation). "
            "This is the strongest liquidity-crisis signature the system can detect."
        )
        if current_state == RegimeState.NORMAL:
            current_state = RegimeState.FRAGILE

    # Gold/silver only (without gold/VIX) → industrial slowdown, not margin call
    elif gs_reading.level in (SignalLevel.FRAGILE, SignalLevel.HOSTILE):
        if gs_reading.level == SignalLevel.HOSTILE:
            explanations.append(
                f"Gold/silver ratio extreme ({gs_reading.ratio_zscore:+.2f}σ) — "
                f"industrial demand collapse signal."
            )
        else:
            explanations.append(
                f"Gold/silver ratio elevated ({gs_reading.ratio_zscore:+.2f}σ) — "
                f"industrial slowdown signal."
            )
        # Can only tighten: NORMAL → FRAGILE. Does NOT force HOSTILE on its own.
        if current_state == RegimeState.NORMAL:
            current_state = RegimeState.FRAGILE

    return current_state, " ".join(explanations)


def _safe_return(series: pd.Series, periods: int) -> float:
    """Compute return over N periods, return 0.0 if insufficient data."""
    if len(series) <= periods:
        return 0.0
    return (series.iloc[-1] / series.iloc[-periods - 1]) - 1
