"""
Flow quality pillar — per-sector volume/price quality metrics from OHLCV.

Measures:
1. RVOL: relative volume vs 20d average, percentile-ranked vs 60d history
2. CLV: Close Location Value, 5d avg percentile-ranked vs 60d history
3. Up-volume ratio: volume on up days / total volume
"""
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


def compute_flow_pillar(
    prices: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    volumes: pd.DataFrame,
    ticker: str,
    window: int = 20,
) -> float:
    """
    Flow quality pillar (0-100).
    Returns 50.0 when data is missing or insufficient.
    """
    for df in [prices, highs, lows, volumes]:
        if ticker not in df.columns:
            return 50.0

    close = prices[ticker].dropna()
    high = highs[ticker].dropna()
    low = lows[ticker].dropna()
    vol = volumes[ticker].dropna()

    min_len = min(len(close), len(high), len(low), len(vol))
    if min_len < window:
        return 50.0

    # Align lengths
    close = close.iloc[-min_len:]
    high = high.iloc[-min_len:]
    low = low.iloc[-min_len:]
    vol = vol.iloc[-min_len:]

    scores = []

    # 1. RVOL: current volume / 20d average, percentile-ranked vs 60d
    vol_ma = vol.rolling(window).mean()
    rvol_series = (vol / vol_ma).dropna()
    if len(rvol_series) >= window:
        current_rvol = rvol_series.iloc[-1]
        history = rvol_series.iloc[-min(60, len(rvol_series)):]
        rvol_pct = percentileofscore(history.values, current_rvol, kind="rank")
        scores.append(rvol_pct)
    else:
        scores.append(50.0)

    # 2. CLV: (close - low) / (high - low), 5d avg, percentile-ranked
    h_range = high - low
    h_range = h_range.replace(0, np.nan)
    clv = ((close - low) / h_range).fillna(0.5)
    clv_5d = clv.rolling(5).mean().dropna()
    if len(clv_5d) >= window:
        current_clv = clv_5d.iloc[-1]
        history = clv_5d.iloc[-min(60, len(clv_5d)):]
        clv_pct = percentileofscore(history.values, current_clv, kind="rank")
        scores.append(clv_pct)
    else:
        scores.append(50.0)

    # 3. Up-volume ratio: volume on up days / total volume over window
    daily_ret = close.pct_change()
    if len(daily_ret) >= window:
        recent_ret = daily_ret.iloc[-window:]
        recent_vol = vol.iloc[-window:]
        up_mask = recent_ret > 0
        up_vol = recent_vol[up_mask].sum()
        total_vol = recent_vol.sum()
        if total_vol > 0:
            up_ratio = up_vol / total_vol
            # Map: 0.6+ = strong (75+), 0.4-0.6 = neutral, <0.4 = weak
            up_score = max(0, min(100, up_ratio * 100))
            scores.append(up_score)
        else:
            scores.append(50.0)
    else:
        scores.append(50.0)

    result = np.mean(scores)
    return max(0.0, min(100.0, float(result)))
