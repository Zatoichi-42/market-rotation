"""
Normalization utilities for the Pump Rotation System.

Provides:
- Z-score computation (single value and rolling series)
- Cross-sectional percentile ranking
- Blended normalization (z-score percentile + cross-section percentile)
- Signal decay for stale/unchanged signals
"""
import math
import numpy as np
import pandas as pd


def compute_zscore(value: float, series: pd.Series) -> float:
    """
    Compute the z-score of a single value against a historical series.

    Returns NaN if series has fewer than 2 data points.
    Returns 0.0 if standard deviation is zero (all identical values).
    """
    if len(series) < 2:
        return float("nan")

    mean = series.mean()
    std = series.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return (value - mean) / std


def compute_zscore_series(values: pd.Series, window: int = 504) -> pd.Series:
    """
    Compute rolling z-scores for an entire series.

    Each point's z-score is calculated against the preceding `window` values.
    Points with insufficient history (< window) return NaN.
    """
    rolling_mean = values.rolling(window=window, min_periods=window).mean()
    rolling_std = values.rolling(window=window, min_periods=window).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (values - rolling_mean) / rolling_std
    # Where std was zero, set z-score to 0
    zscore = zscore.fillna(0.0)
    # But keep NaN where we didn't have enough data
    zscore.iloc[:window - 1] = np.nan

    return zscore


def percentile_rank(values: pd.Series) -> pd.Series:
    """
    Cross-sectional percentile rank.

    Ranks values from 0 to 100 where 100 = best (highest value).
    Uses average method for ties, then scales to [0, 100].
    """
    n = len(values)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series(50.0, index=values.index)

    # rank: 1 = smallest, n = largest (average for ties)
    ranks = values.rank(method="average")
    # Scale to 0-100: percentile = (rank - 1) / (n - 1) * 100
    pcts = (ranks - 1) / (n - 1) * 100.0
    return pcts


def blend_normalize(zscore_pct: float, xsection_pct: float) -> float:
    """
    Blend z-score percentile with cross-section percentile.

    Simple average of the two, clamped to [0, 100].
    """
    blended = (zscore_pct + xsection_pct) / 2.0
    return max(0.0, min(100.0, blended))


def apply_decay(signal: float, unchanged_sessions: int,
                halflife: int = 10, start_after: int = 10) -> float:
    """
    Apply exponential decay to a signal that hasn't changed.

    - No decay if unchanged_sessions < start_after
    - After start_after, decay with the given halflife
    - signal * 0.5^((unchanged - start_after) / halflife)

    Never returns below 0.
    """
    if unchanged_sessions < start_after:
        return signal

    decay_sessions = unchanged_sessions - start_after
    decay_factor = math.pow(0.5, decay_sessions / halflife)
    return max(0.0, signal * decay_factor)
