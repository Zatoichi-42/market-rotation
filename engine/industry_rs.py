"""
Industry RS Scanner — RS vs SPY and RS vs parent sector.

Two RS dimensions per industry:
1. RS vs SPY (absolute strength in market context)
2. RS vs parent sector ETF (strength WITHIN its sector)

industry_composite = vs_spy_weight * rs_composite + vs_parent_weight * rs_composite_vs_parent
"""
import numpy as np
import pandas as pd

from engine.schemas import IndustryRSReading, GroupType
from engine.rs_scanner import compute_rs
from engine.normalizer import percentile_rank


def compute_industry_rs(
    prices: pd.DataFrame,
    industries: list[dict],
    benchmark: str = "SPY",
    windows: list[int] = [5, 20, 60],
    slope_window: int = 5,
    composite_weights: dict = None,
    vs_parent_weight: float = 0.30,
    prior_rankings: dict = None,
) -> list[IndustryRSReading]:
    """
    Compute RS for each industry at 5/20/60d vs SPY and vs parent sector.
    """
    if composite_weights is None:
        composite_weights = {5: 0.20, 20: 0.50, 60: 0.30}

    tickers = [ind["ticker"] for ind in industries]
    ticker_to_config = {ind["ticker"]: ind for ind in industries}

    # Compute RS vs SPY for each window
    rs_vs_spy = {}
    for w in windows:
        rs_vs_spy[w] = {}
        for t in tickers:
            rs_series = compute_rs(prices, t, w, benchmark)
            rs_vs_spy[w][t] = rs_series.iloc[-1] if not rs_series.empty and not np.isnan(rs_series.iloc[-1]) else 0.0

    # Compute RS vs parent for each window
    rs_vs_parent = {}
    for w in windows:
        rs_vs_parent[w] = {}
        for t in tickers:
            parent = ticker_to_config[t]["parent_sector"]
            rs_series = compute_rs(prices, t, w, parent)
            rs_vs_parent[w][t] = rs_series.iloc[-1] if not rs_series.empty and not np.isnan(rs_series.iloc[-1]) else 0.0

    # Compute slopes (5-session RoC of 20d RS)
    slopes_spy = {}
    slopes_parent = {}
    for t in tickers:
        # vs SPY
        rs20_series = compute_rs(prices, t, 20, benchmark)
        rs20_clean = rs20_series.dropna()
        if len(rs20_clean) >= slope_window:
            slopes_spy[t] = rs20_clean.iloc[-1] - rs20_clean.iloc[-slope_window]
        else:
            slopes_spy[t] = 0.0

        # vs parent
        parent = ticker_to_config[t]["parent_sector"]
        rs20p_series = compute_rs(prices, t, 20, parent)
        rs20p_clean = rs20p_series.dropna()
        if len(rs20p_clean) >= slope_window:
            slopes_parent[t] = rs20p_clean.iloc[-1] - rs20p_clean.iloc[-slope_window]
        else:
            slopes_parent[t] = 0.0

    # Percentile rank within all industries for each window (vs SPY)
    pct_spy = {}
    for w in windows:
        vals = pd.Series({t: rs_vs_spy[w][t] for t in tickers})
        valid = vals.dropna()
        if not valid.empty:
            pct_spy[w] = percentile_rank(valid)
        else:
            pct_spy[w] = pd.Series(50.0, index=vals.index)

    # Percentile rank vs parent (across all industries, not within sector)
    pct_parent = {}
    for w in windows:
        vals = pd.Series({t: rs_vs_parent[w][t] for t in tickers})
        valid = vals.dropna()
        if not valid.empty:
            pct_parent[w] = percentile_rank(valid)
        else:
            pct_parent[w] = pd.Series(50.0, index=vals.index)

    # Composite scores
    composites_spy = {}
    composites_parent = {}
    for t in tickers:
        ws_sum, ws_wt = 0.0, 0.0
        wp_sum, wp_wt = 0.0, 0.0
        for w in windows:
            spy_val = pct_spy[w].get(t, 50.0)
            par_val = pct_parent[w].get(t, 50.0)
            if not np.isnan(spy_val):
                ws_sum += composite_weights[w] * spy_val
                ws_wt += composite_weights[w]
            if not np.isnan(par_val):
                wp_sum += composite_weights[w] * par_val
                wp_wt += composite_weights[w]
        composites_spy[t] = ws_sum / ws_wt if ws_wt > 0 else 50.0
        composites_parent[t] = wp_sum / wp_wt if wp_wt > 0 else 50.0

    # Industry composite (blend)
    vs_spy_weight = 1.0 - vs_parent_weight
    industry_composites = {}
    for t in tickers:
        industry_composites[t] = vs_spy_weight * composites_spy[t] + vs_parent_weight * composites_parent[t]

    # Rank across all industries by 20d RS vs SPY
    rs20_vals = pd.Series({t: rs_vs_spy[20].get(t, np.nan) for t in tickers})
    valid_rs = rs20_vals.dropna()
    ranked = valid_rs.rank(ascending=False, method="first").astype(int) if not valid_rs.empty else pd.Series(dtype=int)
    current_ranks = {}
    next_rank = len(valid_rs) + 1
    for t in tickers:
        if t in ranked.index:
            current_ranks[t] = int(ranked[t])
        else:
            current_ranks[t] = next_rank
            next_rank += 1

    # Rank within sector
    sector_groups = {}
    for t in tickers:
        parent = ticker_to_config[t]["parent_sector"]
        sector_groups.setdefault(parent, []).append(t)

    within_sector_ranks = {}
    for parent, children in sector_groups.items():
        child_rs = pd.Series({t: rs_vs_spy[20].get(t, np.nan) for t in children}).dropna()
        if not child_rs.empty:
            child_ranked = child_rs.rank(ascending=False, method="first").astype(int)
            for t in children:
                within_sector_ranks[t] = int(child_ranked[t]) if t in child_ranked.index else len(children)
        else:
            for t in children:
                within_sector_ranks[t] = 1

    # Rank changes
    rank_changes = {}
    for t in tickers:
        if prior_rankings and t in prior_rankings:
            rank_changes[t] = prior_rankings[t] - current_ranks[t]
        else:
            rank_changes[t] = 0

    # Build readings
    readings = []
    for t in tickers:
        cfg = ticker_to_config[t]
        readings.append(IndustryRSReading(
            ticker=t,
            name=cfg["name"],
            parent_sector=cfg["parent_sector"],
            group_type=GroupType.INDUSTRY,
            rs_5d=rs_vs_spy[5].get(t, 0.0),
            rs_20d=rs_vs_spy[20].get(t, 0.0),
            rs_60d=rs_vs_spy[60].get(t, 0.0),
            rs_slope=slopes_spy[t],
            rs_composite=composites_spy[t],
            rs_5d_vs_parent=rs_vs_parent[5].get(t, 0.0),
            rs_20d_vs_parent=rs_vs_parent[20].get(t, 0.0),
            rs_60d_vs_parent=rs_vs_parent[60].get(t, 0.0),
            rs_slope_vs_parent=slopes_parent[t],
            rs_composite_vs_parent=composites_parent[t],
            industry_composite=industry_composites[t],
            rs_rank=current_ranks[t],
            rs_rank_change=rank_changes[t],
            rs_rank_within_sector=within_sector_ranks[t],
            rs_2d=rs_vs_spy.get(2, {}).get(t, 0.0),
            rs_10d=rs_vs_spy.get(10, {}).get(t, 0.0),
            rs_120d=rs_vs_spy.get(120, {}).get(t, 0.0),
            rs_2d_vs_parent=rs_vs_parent.get(2, {}).get(t, 0.0),
            rs_10d_vs_parent=rs_vs_parent.get(10, {}).get(t, 0.0),
            rs_120d_vs_parent=rs_vs_parent.get(120, {}).get(t, 0.0),
        ))

    return readings
