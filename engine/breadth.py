"""
Breadth Divergence — RSP/SPY ratio analysis.

Computes:
- RSP/SPY price ratio (equal-weight vs cap-weight proxy)
- 20-day change in the ratio
- Rolling z-score of the ratio (504-day window)
- Classification: HEALTHY / NARROWING / DIVERGING

Boundary convention: exact boundary goes to the WORSE bucket.
- z-score > 0.0 → HEALTHY
- z-score <= 0.0 and > -1.0 → NARROWING
- z-score <= -1.0 → DIVERGING

Conservative default: when data is insufficient, defaults to NARROWING.
"""
import math
import numpy as np
import pandas as pd

from engine.schemas import BreadthSignal, BreadthReading
from engine.normalizer import compute_zscore


# Minimum history needed for a meaningful z-score
_MIN_ZSCORE_HISTORY = 252


def _classify_breadth_signal(zscore: float, ratio_20d_change: float) -> BreadthSignal:
    """
    Classify breadth signal from z-score.

    Boundary convention: exact boundary goes to worse case.
    """
    if math.isnan(zscore):
        return BreadthSignal.NARROWING  # Conservative default

    if zscore <= -1.0:
        return BreadthSignal.DIVERGING
    elif zscore <= 0.0:
        return BreadthSignal.NARROWING
    else:
        return BreadthSignal.HEALTHY


def compute_breadth(
    prices: pd.DataFrame,
    zscore_window: int = 504,
    spy_col: str = "SPY",
    rsp_col: str = "RSP",
) -> BreadthReading:
    """
    Compute breadth reading from SPY and RSP price data.

    Returns BreadthReading with conservative defaults when data is missing.
    """
    # Handle missing columns
    if spy_col not in prices.columns or rsp_col not in prices.columns:
        return BreadthReading(
            rsp_spy_ratio=float("nan"),
            rsp_spy_ratio_20d_change=float("nan"),
            rsp_spy_ratio_zscore=float("nan"),
            signal=BreadthSignal.NARROWING,
            explanation="Breadth unavailable: missing RSP or SPY data. Defaulting to NARROWING.",
        )

    # Forward-fill small gaps, then compute ratio
    spy = prices[spy_col].ffill()
    rsp = prices[rsp_col].ffill()
    ratio = rsp / spy

    # Current ratio value
    current_ratio = ratio.iloc[-1]
    if np.isnan(current_ratio):
        return BreadthReading(
            rsp_spy_ratio=float("nan"),
            rsp_spy_ratio_20d_change=float("nan"),
            rsp_spy_ratio_zscore=float("nan"),
            signal=BreadthSignal.NARROWING,
            explanation="Breadth unavailable: current ratio is NaN. Defaulting to NARROWING.",
        )

    # 20-day change in ratio
    if len(ratio) >= 20:
        ratio_20d_ago = ratio.iloc[-20]
        if not np.isnan(ratio_20d_ago) and ratio_20d_ago != 0:
            change_20d = (current_ratio - ratio_20d_ago) / ratio_20d_ago
        else:
            change_20d = 0.0
    else:
        change_20d = 0.0

    # Z-score of current ratio against history
    valid_ratio = ratio.dropna()
    if len(valid_ratio) >= _MIN_ZSCORE_HISTORY:
        history = valid_ratio.iloc[-zscore_window:] if len(valid_ratio) >= zscore_window else valid_ratio
        zscore = compute_zscore(current_ratio, history)
    else:
        zscore = float("nan")

    # Classify
    signal = _classify_breadth_signal(zscore, change_20d)

    # Explanation
    explanation = _build_explanation(signal, current_ratio, change_20d, zscore)

    return BreadthReading(
        rsp_spy_ratio=current_ratio,
        rsp_spy_ratio_20d_change=change_20d,
        rsp_spy_ratio_zscore=zscore,
        signal=signal,
        explanation=explanation,
    )


def _build_explanation(signal: BreadthSignal, ratio: float, change_20d: float, zscore: float) -> str:
    """Build human-readable breadth explanation."""
    z_str = f"{zscore:.2f}" if not math.isnan(zscore) else "N/A (insufficient history)"

    if signal == BreadthSignal.HEALTHY:
        return (
            f"HEALTHY breadth. RSP/SPY ratio: {ratio:.4f}, "
            f"20d change: {change_20d:+.4f}, z-score: {z_str}. "
            f"Broad market participation confirmed."
        )
    elif signal == BreadthSignal.NARROWING:
        return (
            f"NARROWING breadth. RSP/SPY ratio: {ratio:.4f}, "
            f"20d change: {change_20d:+.4f}, z-score: {z_str}. "
            f"Equal-weight underperforming cap-weight — participation weakening."
        )
    else:
        return (
            f"DIVERGING breadth. RSP/SPY ratio: {ratio:.4f}, "
            f"20d change: {change_20d:+.4f}, z-score: {z_str}. "
            f"Significant breadth divergence — narrow leadership, caution warranted."
        )
