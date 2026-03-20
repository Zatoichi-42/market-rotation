"""
Participation pillar — per-sector breadth using industry children as proxy.

Measures:
1. Breadth: fraction of child industries outperforming the sector
2. EW vs CW spread: average child return vs sector return
3. Trend: is participation improving (5d vs 20d comparison)
"""
import numpy as np
import pandas as pd


def compute_participation_pillar(
    prices: pd.DataFrame,
    sector_ticker: str,
    child_tickers: list[str],
    window: int = 20,
) -> float:
    """
    Participation pillar (0-100) for a sector.
    Returns 50.0 (neutral) when data is insufficient or no children available.
    """
    if not child_tickers or sector_ticker not in prices.columns:
        return 50.0

    # Filter to children actually in prices
    available = [t for t in child_tickers if t in prices.columns]
    if not available:
        return 50.0

    if len(prices) < window:
        return 50.0

    sector = prices[sector_ticker]

    # 1. Breadth: fraction of children outperforming sector over window
    sector_ret = sector.iloc[-1] / sector.iloc[-window] - 1
    children_above = 0
    for t in available:
        child = prices[t]
        if len(child.dropna()) >= window:
            child_ret = child.iloc[-1] / child.iloc[-window] - 1
            if child_ret > sector_ret:
                children_above += 1
    breadth_score = (children_above / len(available)) * 100

    # 2. EW vs CW spread: average child return vs sector return
    child_rets = []
    for t in available:
        child = prices[t]
        if len(child.dropna()) >= window:
            child_rets.append(child.iloc[-1] / child.iloc[-window] - 1)
    if child_rets:
        avg_child_ret = np.mean(child_rets)
        spread = avg_child_ret - sector_ret
        # Normalize: spread of +2% → ~75, spread of -2% → ~25, 0 → 50
        spread_score = max(0, min(100, 50 + spread * 2500))
    else:
        spread_score = 50.0

    # 3. Trend: compare 5d breadth vs 20d breadth
    trend_score = 50.0
    if len(prices) >= window and len(available) > 0:
        short_window = min(5, window)
        sector_ret_short = sector.iloc[-1] / sector.iloc[-short_window] - 1
        short_above = 0
        for t in available:
            child = prices[t]
            if len(child.dropna()) >= short_window:
                cr = child.iloc[-1] / child.iloc[-short_window] - 1
                if cr > sector_ret_short:
                    short_above += 1
        short_breadth = short_above / len(available)
        long_breadth = children_above / len(available)
        # Improving if short-term breadth > long-term breadth
        if short_breadth > long_breadth + 0.05:
            trend_score = 70.0
        elif short_breadth < long_breadth - 0.05:
            trend_score = 30.0

    # Average of three sub-signals
    result = (breadth_score + spread_score + trend_score) / 3.0
    return max(0.0, min(100.0, result))
