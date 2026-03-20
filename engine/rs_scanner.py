"""
RS Scanner — Relative Strength computation for sector ETFs.

Computes:
- RS at multiple windows (5d, 20d, 60d): sector return minus SPY return
- RS slope: rate of change of 20d RS over slope_window
- Cross-sectional ranking (rank 1 = strongest)
- Composite score: weighted blend of percentile-ranked RS at each window
"""
import numpy as np
import pandas as pd

from engine.schemas import RSReading
from engine.normalizer import percentile_rank


def compute_rs(prices: pd.DataFrame, ticker: str, window: int,
               benchmark: str = "SPY") -> pd.Series:
    """
    Compute relative strength of a ticker vs benchmark over a rolling window.

    RS = (sector_return_over_window) - (benchmark_return_over_window)
    Both returns are simple percentage returns over the window.

    Returns a pd.Series with same index as prices. Early values are NaN
    where insufficient history exists for the window.
    """
    if ticker not in prices.columns or benchmark not in prices.columns:
        return pd.Series(np.nan, index=prices.index)

    sector_ret = prices[ticker].pct_change(periods=window)
    bench_ret = prices[benchmark].pct_change(periods=window)
    return sector_ret - bench_ret


def compute_rs_all(prices: pd.DataFrame, tickers: list[str], window: int,
                   benchmark: str = "SPY") -> pd.DataFrame:
    """
    Compute RS for all tickers at a given window.
    Returns DataFrame: date × ticker with RS values.
    """
    rs_data = {}
    for ticker in tickers:
        rs_data[ticker] = compute_rs(prices, ticker, window, benchmark)
    return pd.DataFrame(rs_data, index=prices.index)


def compute_rs_readings(
    prices: pd.DataFrame,
    sector_names: dict[str, str],
    windows: list[int] = [5, 20, 60],
    slope_window: int = 5,
    composite_weights: dict[int, float] = {5: 0.2, 20: 0.5, 60: 0.3},
    prior_ranks: dict[str, int] | None = None,
    benchmark: str = "SPY",
) -> list[RSReading]:
    """
    Compute full RS readings for all sectors.

    Returns a list of RSReading dataclasses, one per sector, ranked by 20d RS.
    """
    tickers = list(sector_names.keys())

    # Compute RS at each window
    rs_by_window: dict[int, pd.DataFrame] = {}
    for w in windows:
        rs_by_window[w] = compute_rs_all(prices, tickers, w, benchmark)

    # Get the latest RS value for each ticker at each window
    latest_rs: dict[int, dict[str, float]] = {}
    for w in windows:
        latest = rs_by_window[w].iloc[-1]
        latest_rs[w] = latest.to_dict()

    # Compute RS slope: rate of change of 20d RS over slope_window
    rs_20d_series = rs_by_window[20] if 20 in rs_by_window else rs_by_window[windows[1]]
    slopes = {}
    for ticker in tickers:
        rs_series = rs_20d_series[ticker].dropna()
        if len(rs_series) >= slope_window:
            # Slope = (current - slope_window_ago) / slope_window
            slopes[ticker] = rs_series.iloc[-1] - rs_series.iloc[-slope_window]
        else:
            slopes[ticker] = 0.0

    # Rank by 20d RS (rank 1 = strongest = highest RS)
    rs_20d_values = pd.Series({t: latest_rs[20].get(t, np.nan) for t in tickers})
    # Drop NaN before ranking, then fill back with worst rank
    valid_rs = rs_20d_values.dropna()
    ranked = valid_rs.rank(ascending=False, method="first").astype(int)
    # Assign worst ranks to NaN tickers
    next_rank = len(valid_rs) + 1
    current_ranks = {}
    for t in tickers:
        if t in ranked:
            current_ranks[t] = int(ranked[t])
        else:
            current_ranks[t] = next_rank
            next_rank += 1

    # Compute rank changes
    rank_changes = {}
    for ticker in tickers:
        if prior_ranks and ticker in prior_ranks:
            # Positive = improved (lower rank number = better)
            rank_changes[ticker] = prior_ranks[ticker] - current_ranks[ticker]
        else:
            rank_changes[ticker] = 0

    # Compute composite score: weighted percentile rank across COMPOSITE windows only
    # Only windows with composite weights contribute (5d/20d/60d)
    composite_scores = {}
    pct_by_window = {}
    composite_windows = list(composite_weights.keys())
    for w in composite_windows:
        if w in latest_rs:
            rs_vals = pd.Series({t: latest_rs[w].get(t, np.nan) for t in tickers})
            pct_by_window[w] = percentile_rank(rs_vals.dropna())

    for ticker in tickers:
        weighted_sum = 0.0
        weight_sum = 0.0
        for w in composite_windows:
            if w in pct_by_window and ticker in pct_by_window[w] and not np.isnan(pct_by_window[w][ticker]):
                weighted_sum += composite_weights[w] * pct_by_window[w][ticker]
                weight_sum += composite_weights[w]
        if weight_sum > 0:
            composite_scores[ticker] = weighted_sum / weight_sum
        else:
            composite_scores[ticker] = 50.0  # Default mid-range

    # Build RSReading objects
    readings = []
    for ticker in tickers:
        rs_vals = {}
        for w in windows:
            v = latest_rs.get(w, {}).get(ticker, np.nan)
            rs_vals[w] = 0.0 if np.isnan(v) else v

        # Extended windows: use NaN if insufficient data, else 0.0
        rs_2d = rs_vals.get(2, float('nan') if 2 in windows and len(prices) < 3 else rs_vals.get(2, 0.0))
        rs_10d = rs_vals.get(10, float('nan') if 10 in windows and len(prices) < 11 else rs_vals.get(10, 0.0))
        rs_120d_raw = latest_rs.get(120, {}).get(ticker, np.nan)
        rs_120d = float('nan') if (120 in windows and np.isnan(rs_120d_raw)) else (0.0 if 120 not in windows else rs_120d_raw)

        readings.append(RSReading(
            ticker=ticker,
            name=sector_names[ticker],
            rs_5d=rs_vals.get(5, 0.0),
            rs_20d=rs_vals.get(20, 0.0),
            rs_60d=rs_vals.get(60, 0.0),
            rs_slope=slopes[ticker],
            rs_rank=current_ranks[ticker],
            rs_rank_change=rank_changes[ticker],
            rs_composite=composite_scores[ticker],
            rs_2d=rs_2d if not np.isnan(rs_2d) else 0.0,
            rs_10d=rs_10d if not np.isnan(rs_10d) else 0.0,
            rs_120d=rs_120d if not np.isnan(rs_120d) else 0.0,
        ))

    return readings
