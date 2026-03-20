"""
Cross-Sector Correlation Monitor — regime signal #6.

Computes average pairwise correlation across sector ETFs.
High correlation = diversification breaking down = regime stress.
"""
import math
import numpy as np
import pandas as pd

from engine.schemas import SignalLevel, CorrelationReading


_SECTOR_TICKERS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]


def compute_cross_sector_correlation(
    prices: pd.DataFrame,
    window: int = 21,
    zscore_window: int = 504,
    fragile_zscore: float = 0.5,
    hostile_zscore: float = 1.5,
    absolute_hostile: float = 0.80,
) -> CorrelationReading | None:
    """
    Compute average pairwise correlation across all sector ETFs.

    Returns None if insufficient data.

    Thresholds:
        z-score < fragile_zscore → NORMAL
        fragile_zscore ≤ z < hostile_zscore → FRAGILE
        z-score ≥ hostile_zscore → HOSTILE
        OR avg_correlation > absolute_hostile → HOSTILE
    """
    available = [t for t in _SECTOR_TICKERS if t in prices.columns]
    if len(available) < 3:
        return None

    returns = prices[available].pct_change().dropna()
    if len(returns) < window + 1:
        return None

    # Rolling average pairwise correlation
    avg_corr_series = _rolling_avg_correlation(returns, window)
    if avg_corr_series is None or len(avg_corr_series) < 2:
        return None

    current_avg = avg_corr_series.iloc[-1]
    if math.isnan(current_avg):
        return None

    # Z-score against history
    hist = avg_corr_series.tail(zscore_window)
    mean = hist.mean()
    std = hist.std()
    if std == 0 or math.isnan(std):
        zscore = 0.0
    else:
        zscore = (current_avg - mean) / std

    # Find max and min correlated pairs from current correlation matrix
    recent_returns = returns.tail(window)
    corr_matrix = recent_returns.corr()
    max_pair, min_pair = _find_extreme_pairs(corr_matrix, available)

    # Classify
    if current_avg > absolute_hostile or zscore >= hostile_zscore:
        level = SignalLevel.HOSTILE
        desc = (f"Avg sector correlation {current_avg:.2f} ({zscore:+.2f}σ) — "
                f"sectors moving in lockstep. Rotation signals unreliable.")
    elif zscore >= fragile_zscore:
        level = SignalLevel.FRAGILE
        desc = (f"Avg sector correlation {current_avg:.2f} ({zscore:+.2f}σ) — "
                f"correlation rising, diversification weakening.")
    else:
        level = SignalLevel.NORMAL
        desc = (f"Avg sector correlation {current_avg:.2f} ({zscore:+.2f}σ) — "
                f"healthy dispersion between sectors.")

    return CorrelationReading(
        avg_correlation=float(current_avg),
        avg_corr_zscore=float(zscore),
        level=level,
        max_corr_pair=max_pair,
        min_corr_pair=min_pair,
        description=desc,
    )


def _rolling_avg_correlation(returns: pd.DataFrame, window: int) -> pd.Series | None:
    """Compute rolling average pairwise correlation across all columns."""
    n_cols = len(returns.columns)
    if n_cols < 2:
        return None

    n_pairs = n_cols * (n_cols - 1) // 2
    result = []
    for i in range(window, len(returns) + 1):
        chunk = returns.iloc[i - window:i]
        corr = chunk.corr()
        # Extract upper triangle (exclude diagonal)
        upper = []
        for r in range(n_cols):
            for c in range(r + 1, n_cols):
                val = corr.iloc[r, c]
                if not math.isnan(val):
                    upper.append(val)
        avg = np.mean(upper) if upper else float("nan")
        result.append(avg)

    return pd.Series(result, index=returns.index[window - 1:])


def _find_extreme_pairs(corr_matrix: pd.DataFrame, tickers: list) -> tuple:
    """Find most and least correlated pairs from correlation matrix."""
    max_val, min_val = -2.0, 2.0
    max_pair = (tickers[0], tickers[1]) if len(tickers) >= 2 else ("", "")
    min_pair = max_pair

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            val = corr_matrix.iloc[i, j]
            if math.isnan(val):
                continue
            if val > max_val:
                max_val = val
                max_pair = (tickers[i], tickers[j])
            if val < min_val:
                min_val = val
                min_pair = (tickers[i], tickers[j])

    return max_pair, min_pair


def compute_cross_sector_dispersion(sector_returns_20d: dict[str, float]) -> float:
    """
    Return std dev of 20-day returns across sectors.

    If fewer than 3 values are provided, returns 0.0 — there is not enough
    data for a meaningful dispersion measure.
    """
    values = list(sector_returns_20d.values())
    if len(values) < 3:
        return 0.0
    return float(np.std(values))
