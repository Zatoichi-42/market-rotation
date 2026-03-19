"""
Reversal Score Engine — Fragility detection from OHLCV data.

Three orthogonal pillars:
1. Breadth Deterioration (40%): RS slope reversal, participation decay
2. Price/RS Break Quality (30%): Failed breakouts, gap fades, CLV trend, follow-through
3. Crowding/Stretch (30%): Distance from MA, relative volume, price acceleration
"""
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from engine.schemas import ReversalScoreReading
from engine.normalizer import compute_zscore


def compute_reversal_score(
    prices: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    volumes: pd.DataFrame,
    ticker: str,
    benchmark: str = "SPY",
    settings: dict = None,
    weights: dict = None,
    history_scores: pd.Series = None,
) -> ReversalScoreReading:
    """Compute the Reversal Score for a single group."""
    if settings is None:
        settings = {}
    if weights is None:
        weights = {"breadth_det_weight": 0.40, "price_break_weight": 0.30, "crowding_weight": 0.30}

    if ticker not in prices.columns:
        return _empty_reading(ticker)

    bd_pillar, bd_subs = _compute_breadth_deterioration(prices, ticker, benchmark, settings)
    pb_pillar, pb_subs = _compute_price_break_quality(prices, highs, lows, ticker, settings)
    cr_pillar, cr_subs = _compute_crowding_stretch(prices, volumes, ticker, settings)

    # Weighted composite → 0-1
    composite = (
        weights["breadth_det_weight"] * bd_pillar
        + weights["price_break_weight"] * pb_pillar
        + weights["crowding_weight"] * cr_pillar
    ) / 100.0

    composite = max(0.0, min(1.0, composite))

    # All sub-signals
    all_subs = {**bd_subs, **pb_subs, **cr_subs}

    # Percentile in own history
    if history_scores is not None and len(history_scores) > 0:
        rev_pct = percentileofscore(history_scores.values, composite, kind="rank")
    else:
        rev_pct = 50.0  # No history → assume mid-range

    above_75th = rev_pct >= 75.0

    return ReversalScoreReading(
        ticker=ticker,
        name=ticker,
        breadth_det_pillar=bd_pillar,
        price_break_pillar=pb_pillar,
        crowding_pillar=cr_pillar,
        reversal_score=composite,
        sub_signals=all_subs,
        reversal_percentile=rev_pct,
        above_75th=above_75th,
    )


def compute_reversal_scores_batch(
    prices: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    volumes: pd.DataFrame,
    tickers: list[str],
    benchmark: str = "SPY",
    settings: dict = None,
    weights: dict = None,
    history_scores: dict = None,
) -> list[ReversalScoreReading]:
    """Batch compute reversal scores for multiple groups."""
    results = []
    for t in tickers:
        hist = history_scores.get(t) if history_scores else None
        results.append(compute_reversal_score(
            prices, highs, lows, volumes, t,
            benchmark=benchmark, settings=settings, weights=weights,
            history_scores=hist,
        ))
    return results


# ═══════════════════════════════════════════════════════
# PILLAR 1: BREADTH DETERIORATION (40%)
# ═══════════════════════════════════════════════════════

def _compute_breadth_deterioration(
    prices: pd.DataFrame, ticker: str, benchmark: str, settings: dict,
) -> tuple[float, dict]:
    """
    Measures whether participation/momentum is narrowing.
    Sub-signals: rs_slope_reversal, participation_decay.
    """
    subs = {}
    scores = []

    close = prices[ticker]
    bench = prices[benchmark] if benchmark in prices.columns else close

    # 1. RS slope reversal: was positive → now negative?
    rs = (close.pct_change(periods=20) - bench.pct_change(periods=20)).dropna()
    lookback = settings.get("rs_slope_lookback", 5)
    if len(rs) >= lookback * 2:
        recent_slope = rs.iloc[-1] - rs.iloc[-lookback]
        prior_slope = rs.iloc[-lookback] - rs.iloc[-lookback * 2]
        # If prior was positive and recent is negative → reversal
        if prior_slope > 0 and recent_slope < 0:
            reversal_magnitude = min(abs(recent_slope) / max(abs(prior_slope), 0.001), 2.0)
            subs["rs_slope_reversal"] = reversal_magnitude
            scores.append(min(reversal_magnitude * 50, 100))
        else:
            subs["rs_slope_reversal"] = 0.0
            scores.append(0.0)
    else:
        subs["rs_slope_reversal"] = 0.0
        scores.append(0.0)

    # 2. Participation decay: % of days outperforming drops
    decay_window = settings.get("participation_decay_window", 20)
    if len(close) >= decay_window * 2:
        sector_rets = close.pct_change().dropna()
        bench_rets = bench.pct_change().dropna()
        aligned = pd.DataFrame({"s": sector_rets, "b": bench_rets}).dropna()

        if len(aligned) >= decay_window * 2:
            recent_win = (aligned["s"].iloc[-decay_window:] > aligned["b"].iloc[-decay_window:]).mean()
            prior_win = (aligned["s"].iloc[-decay_window*2:-decay_window] > aligned["b"].iloc[-decay_window*2:-decay_window]).mean()
            decay = prior_win - recent_win  # Positive = deterioration
            subs["participation_decay"] = decay
            scores.append(min(max(decay * 200, 0), 100))  # 0.5 drop → 100
        else:
            subs["participation_decay"] = 0.0
            scores.append(0.0)
    else:
        subs["participation_decay"] = 0.0
        scores.append(0.0)

    pillar = np.mean(scores) if scores else 50.0
    return max(0, min(100, pillar)), subs


# ═══════════════════════════════════════════════════════
# PILLAR 2: PRICE/RS BREAK QUALITY (30%)
# ═══════════════════════════════════════════════════════

def _compute_price_break_quality(
    prices: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame,
    ticker: str, settings: dict,
) -> tuple[float, dict]:
    """
    Measures whether breakouts are failing.
    Sub-signals: failed_breakout_rate, gap_fade_rate, clv_trend, follow_through.
    """
    subs = {}
    scores = []

    close = prices[ticker]
    high = highs[ticker] if ticker in highs.columns else close * 1.005
    low = lows[ticker] if ticker in lows.columns else close * 0.995

    n = len(close)
    fb_lookback = settings.get("failed_breakout_lookback", 20)
    fb_reversal = settings.get("failed_breakout_reversal_days", 3)

    # 1. Failed breakout rate
    new_highs = 0
    failed = 0
    if n > fb_lookback + fb_reversal:
        rolling_max = close.rolling(fb_lookback).max()
        for i in range(fb_lookback, n - fb_reversal):
            if close.iloc[i] >= rolling_max.iloc[i]:
                new_highs += 1
                # Check if reversed within fb_reversal days
                breakout_close = close.iloc[i]
                reversed_any = any(close.iloc[i+j] < breakout_close for j in range(1, fb_reversal + 1) if i + j < n)
                if reversed_any:
                    failed += 1
    fb_rate = failed / max(new_highs, 1)
    subs["failed_breakout_rate"] = fb_rate
    scores.append(min(fb_rate * 100, 100))

    # 2. Gap fade rate
    gap_threshold = settings.get("gap_threshold_pct", 0.005)
    gap_lookback = settings.get("gap_fade_lookback", 20)
    gaps_total = 0
    gaps_faded = 0
    if n > gap_lookback:
        for i in range(max(1, n - gap_lookback), n):
            prior_close = close.iloc[i - 1]
            # Approximate open as midpoint of high/low
            approx_open = (high.iloc[i] + low.iloc[i]) / 2
            if approx_open > prior_close * (1 + gap_threshold):
                gaps_total += 1
                if close.iloc[i] < approx_open:
                    gaps_faded += 1
    gf_rate = gaps_faded / max(gaps_total, 1) if gaps_total > 0 else 0.5
    subs["gap_fade_rate"] = gf_rate
    scores.append(min(gf_rate * 100, 100))

    # 3. CLV trend (close location value)
    clv_short = settings.get("clv_short_window", 5)
    clv_long = settings.get("clv_long_window", 20)
    h_range = high - low
    h_range = h_range.replace(0, np.nan)
    clv = (close - low) / h_range
    clv = clv.fillna(0.5)

    if n >= clv_long:
        clv_short_avg = clv.iloc[-clv_short:].mean()
        clv_long_avg = clv.iloc[-clv_long:].mean()
        clv_diff = clv_long_avg - clv_short_avg  # Positive = deterioration
        subs["clv_trend"] = clv_diff
        scores.append(min(max(clv_diff * 200, 0), 100))
    else:
        subs["clv_trend"] = 0.0
        scores.append(0.0)

    # 4. Follow-through rate
    ft_window = settings.get("follow_through_window", 10)
    if n > ft_window + 1:
        daily_rets = close.pct_change().dropna()
        recent = daily_rets.iloc[-ft_window:]
        up_days = recent > 0
        if up_days.sum() > 0:
            # Count up days followed by another up day
            follow = sum(1 for i in range(len(up_days) - 1) if up_days.iloc[i] and up_days.iloc[i + 1])
            ft_rate = follow / max(up_days.sum(), 1)
        else:
            ft_rate = 0.0
        subs["follow_through"] = ft_rate
        # Low follow-through = higher reversal signal
        scores.append(max(0, (1 - ft_rate) * 80))
    else:
        subs["follow_through"] = 0.5
        scores.append(25.0)

    pillar = np.mean(scores) if scores else 50.0
    return max(0, min(100, pillar)), subs


# ═══════════════════════════════════════════════════════
# PILLAR 3: CROWDING / STRETCH (30%)
# ═══════════════════════════════════════════════════════

def _compute_crowding_stretch(
    prices: pd.DataFrame, volumes: pd.DataFrame,
    ticker: str, settings: dict,
) -> tuple[float, dict]:
    """
    Measures whether the move is over-extended.
    Sub-signals: distance_from_ma, rvol, price_acceleration.
    """
    subs = {}
    scores = []

    close = prices[ticker]
    n = len(close)
    ma_period = settings.get("distance_ma_period", 20)

    # 1. Distance from 20d MA (z-scored)
    if n >= ma_period:
        ma = close.rolling(ma_period).mean()
        std = close.rolling(ma_period).std()
        std = std.replace(0, np.nan)
        dist = ((close - ma) / std).dropna()
        if len(dist) > 0:
            current_dist = dist.iloc[-1]
            subs["distance_from_ma"] = current_dist
            # Map: 0σ → 0, 2σ → 50, 4σ → 100
            scores.append(min(max(abs(current_dist) * 25, 0), 100))
        else:
            subs["distance_from_ma"] = 0.0
            scores.append(0.0)
    else:
        subs["distance_from_ma"] = 0.0
        scores.append(0.0)

    # 2. Relative volume (RVOL)
    rvol_lookback = settings.get("rvol_lookback", 20)
    if ticker in volumes.columns and n >= rvol_lookback:
        vol = volumes[ticker]
        vol_ma = vol.rolling(rvol_lookback).mean()
        if vol_ma.iloc[-1] > 0:
            rvol = vol.iloc[-1] / vol_ma.iloc[-1]
            subs["rvol"] = rvol
            # Map: 1x → 0, 2x → 50, 3x+ → 100
            scores.append(min(max((rvol - 1.0) * 50, 0), 100))
        else:
            subs["rvol"] = 1.0
            scores.append(0.0)
    else:
        subs["rvol"] = 1.0
        scores.append(0.0)

    # 3. Price acceleration (ROC of ROC)
    fast = settings.get("price_accel_fast", 5)
    slow = settings.get("price_accel_slow", 20)
    if n >= slow + fast:
        roc_slow = close.pct_change(periods=slow)
        roc_fast = roc_slow.diff(periods=fast)  # Acceleration
        if not roc_fast.empty and not np.isnan(roc_fast.iloc[-1]):
            accel = roc_fast.iloc[-1]
            subs["price_acceleration"] = accel
            # Positive acceleration = parabolic. Map via abs value.
            scores.append(min(max(abs(accel) * 500, 0), 100))
        else:
            subs["price_acceleration"] = 0.0
            scores.append(0.0)
    else:
        subs["price_acceleration"] = 0.0
        scores.append(0.0)

    pillar = np.mean(scores) if scores else 50.0
    return max(0, min(100, pillar)), subs


def _empty_reading(ticker: str) -> ReversalScoreReading:
    return ReversalScoreReading(
        ticker=ticker, name=ticker,
        breadth_det_pillar=0.0, price_break_pillar=0.0, crowding_pillar=0.0,
        reversal_score=0.0, sub_signals={},
        reversal_percentile=50.0, above_75th=False,
    )
