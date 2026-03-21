"""
Treasury Context — bond/equity regime classification for defensive vehicle selection.

Reads TLT, SHY, SPY prices plus optional FRED data (10y yield, 3m T-bill, MOVE index)
to classify the Treasury environment into Supportive / Mixed / Adverse, detect shock type,
and recommend a defensive vehicle.

Constitution v9 — Layer added atop Regime Gate.
"""
from __future__ import annotations

import pandas as pd

from engine.schemas import (
    RegimeState,
    SignalLevel,
    RegimeSignal,
    TreasuryFit,
    TreasuryShockType,
    DefensiveVehicle,
    TreasuryContextReading,
)


# ── Default thresholds used by gate_watch ────────────────────
_GATE_WATCH_THRESHOLDS = {
    "vix": {"normal_max": 20, "fragile_max": 30},
    "breadth": {"normal_min_zscore": 0.0, "fragile_min_zscore": -1.0},
    "credit": {"normal_min_zscore": -0.5, "fragile_min_zscore": -1.5},
}


def _default_reading(cash_hurdle: float = 0.0) -> TreasuryContextReading:
    """Return a safe default reading when data is insufficient."""
    return TreasuryContextReading(
        treasury_fit=TreasuryFit.MIXED,
        cash_hurdle=cash_hurdle,
        shock_type=TreasuryShockType.NONE,
        sb_correlation=0.0,
        move_level=100.0,
        tlt_vs_shy_20d=0.0,
        yield_10y_20d_change=0.0,
        defensive_vehicle=DefensiveVehicle.SHY,
        gate_watch=False,
        description="Treasury context unavailable — defaulting to Mixed/SHY.",
    )


# ─────────────────────────────────────────────────────────────
# 1. Stock-Bond Correlation
# ─────────────────────────────────────────────────────────────

def compute_sb_correlation(
    tlt_prices: pd.Series,
    spy_prices: pd.Series,
    window: int = 21,
) -> float:
    """
    Compute 21-day rolling correlation between TLT and SPY daily returns.
    Returns the latest value.  Returns 0.0 if insufficient data.
    """
    if len(tlt_prices) < 2 or len(spy_prices) < 2:
        return 0.0

    tlt_ret = tlt_prices.pct_change().dropna()
    spy_ret = spy_prices.pct_change().dropna()

    # Align on common index
    combined = pd.concat([tlt_ret.rename("tlt"), spy_ret.rename("spy")], axis=1).dropna()
    if len(combined) < window:
        return 0.0

    rolling_corr = combined["tlt"].rolling(window).corr(combined["spy"])
    latest = rolling_corr.iloc[-1]
    if pd.isna(latest):
        return 0.0
    return float(latest)


# ─────────────────────────────────────────────────────────────
# 2. Treasury Fit Classification
# ─────────────────────────────────────────────────────────────

def classify_treasury_fit(
    sb_corr: float,
    tlt_vs_shy_20d: float,
    yield_10y_20d_change: float,
) -> TreasuryFit:
    """
    Coarse 3-bucket classification.

    Supportive: sb_corr < -0.15 AND (tlt_vs_shy_20d > 0 OR yield_10y_20d_change < -0.1)
    Adverse:    sb_corr >  0.15 AND (tlt_vs_shy_20d < 0 OR yield_10y_20d_change >  0.1)
    Mixed:      everything else
    """
    if sb_corr < -0.15 and (tlt_vs_shy_20d > 0 or yield_10y_20d_change < -0.1):
        return TreasuryFit.SUPPORTIVE
    if sb_corr > 0.15 and (tlt_vs_shy_20d < 0 or yield_10y_20d_change > 0.1):
        return TreasuryFit.ADVERSE
    return TreasuryFit.MIXED


# ─────────────────────────────────────────────────────────────
# 3. Shock Type Classification
# ─────────────────────────────────────────────────────────────

def classify_shock_type(
    yield_10y_20d_change: float,
    tlt_vs_shy_20d: float,
    move_level: float,
) -> TreasuryShockType:
    """
    Coarse shock typing for defensive vehicle selection.

    Growth Scare:              yield_10y_20d_change < -0.2 AND tlt_vs_shy_20d > 0.01
    Inflation/Term Premium:    yield_10y_20d_change >  0.2 AND tlt_vs_shy_20d < -0.01
    None:                      otherwise
    """
    if yield_10y_20d_change < -0.2 and tlt_vs_shy_20d > 0.01:
        return TreasuryShockType.GROWTH_SCARE
    if yield_10y_20d_change > 0.2 and tlt_vs_shy_20d < -0.01:
        return TreasuryShockType.INFLATION_TERM_PREMIUM
    return TreasuryShockType.NONE


# ─────────────────────────────────────────────────────────────
# 4. Defensive Vehicle Selection
# ─────────────────────────────────────────────────────────────

def select_defensive_vehicle(
    treasury_fit: TreasuryFit,
    shock_type: TreasuryShockType,
    tip_outperforming: bool = False,
) -> DefensiveVehicle:
    """
    Select defensive vehicle based on Treasury Fit and shock type.

    Supportive + Growth Scare            → TLT
    Supportive + other                   → IEF
    Mixed + any                          → SHY
    Adverse + inflation + TIP outperf.   → TIP
    Adverse + other                      → BIL
    """
    if treasury_fit == TreasuryFit.SUPPORTIVE:
        if shock_type == TreasuryShockType.GROWTH_SCARE:
            return DefensiveVehicle.TLT
        return DefensiveVehicle.IEF

    if treasury_fit == TreasuryFit.MIXED:
        return DefensiveVehicle.SHY

    # Adverse
    if shock_type == TreasuryShockType.INFLATION_TERM_PREMIUM and tip_outperforming:
        return DefensiveVehicle.TIP
    return DefensiveVehicle.BIL


# ─────────────────────────────────────────────────────────────
# 5. Gate Watch
# ─────────────────────────────────────────────────────────────

def compute_gate_watch(
    regime_signals: list,
    settings: dict,
) -> bool:
    """
    Binary early warning.  True when >50% of signals are within 1 reading
    of their next-worse threshold.

    For each signal, check distance to its fragile/hostile threshold.
    If signal is NORMAL and close to FRAGILE threshold → near-flip.
    If signal is FRAGILE and close to HOSTILE threshold → near-flip.

    "Close" = within 20% of the threshold gap.

    gate_watch = True if count(near_flip) > count(signals) / 2
    """
    if not regime_signals:
        return False

    regime_cfg = settings.get("regime", _GATE_WATCH_THRESHOLDS)

    near_flip_count = 0

    for sig in regime_signals:
        name = sig.name
        value = sig.raw_value
        level = sig.level

        if name == "vix":
            cfg = regime_cfg.get("vix", _GATE_WATCH_THRESHOLDS["vix"])
            normal_max = cfg["normal_max"]
            fragile_max = cfg["fragile_max"]

            if level == SignalLevel.NORMAL:
                # Gap = normal_max (threshold), "close" = within 20% of it
                gap = normal_max
                close_dist = gap * 0.20
                if value >= normal_max - close_dist:
                    near_flip_count += 1
            elif level == SignalLevel.FRAGILE:
                gap = fragile_max - normal_max
                close_dist = gap * 0.20
                if value >= fragile_max - close_dist:
                    near_flip_count += 1

        elif name == "breadth":
            cfg = regime_cfg.get("breadth", _GATE_WATCH_THRESHOLDS["breadth"])
            normal_min = cfg["normal_min_zscore"]
            fragile_min = cfg["fragile_min_zscore"]

            if level == SignalLevel.NORMAL:
                # Breadth: lower is worse. "Close" to FRAGILE = near normal_min
                gap = abs(normal_min - fragile_min)  # gap between buckets
                close_dist = gap * 0.20
                if value <= normal_min + close_dist:
                    near_flip_count += 1
            elif level == SignalLevel.FRAGILE:
                gap = abs(normal_min - fragile_min)
                close_dist = gap * 0.20
                if value <= fragile_min + close_dist:
                    near_flip_count += 1

        elif name == "credit":
            cfg = regime_cfg.get("credit", _GATE_WATCH_THRESHOLDS["credit"])
            normal_min = cfg["normal_min_zscore"]
            fragile_min = cfg["fragile_min_zscore"]

            if level == SignalLevel.NORMAL:
                gap = abs(normal_min - fragile_min)
                close_dist = gap * 0.20
                if value <= normal_min + close_dist:
                    near_flip_count += 1
            elif level == SignalLevel.FRAGILE:
                gap = abs(normal_min - fragile_min)
                close_dist = gap * 0.20
                if value <= fragile_min + close_dist:
                    near_flip_count += 1

        # Other signal types (term_structure, oil, correlation) — not tracked by gate_watch

    return near_flip_count > len(regime_signals) / 2


# ─────────────────────────────────────────────────────────────
# 6. Main Entry Point
# ─────────────────────────────────────────────────────────────

def compute_treasury_context(
    prices: pd.DataFrame,
    regime_signals: list,
    regime_state: RegimeState,
    settings: dict,
    move_level: float = 100.0,
    yield_10y: pd.Series | None = None,
    tbill_3m: float | None = None,
) -> TreasuryContextReading:
    """
    Main entry point.  Computes all treasury context in one call.

    Steps:
    1. Compute SB correlation (TLT vs SPY 21d)
    2. Compute TLT vs SHY 20d return difference
    3. Compute 10y yield 20d change (from yield_10y series or 0.0)
    4. Classify Treasury Fit
    5. Classify Shock Type
    6. Select Defensive Vehicle
    7. Compute Gate Watch
    8. Cash hurdle = tbill_3m or 0.0
    9. Build description string
    """
    cash_hurdle = tbill_3m if tbill_3m is not None else 0.0

    # Guard: need TLT, SHY, SPY columns
    required = {"TLT", "SHY", "SPY"}
    if not required.issubset(set(prices.columns)):
        return _default_reading(cash_hurdle)

    tlt = prices["TLT"].dropna()
    shy = prices["SHY"].dropna()
    spy = prices["SPY"].dropna()

    if len(tlt) < 2 or len(shy) < 2 or len(spy) < 2:
        return _default_reading(cash_hurdle)

    # 1. SB correlation
    sb_corr = compute_sb_correlation(tlt, spy, window=21)

    # 2. TLT vs SHY 20d return difference
    lookback = min(20, len(tlt) - 1, len(shy) - 1)
    if lookback > 0:
        tlt_20d_ret = (tlt.iloc[-1] / tlt.iloc[-1 - lookback]) - 1.0
        shy_20d_ret = (shy.iloc[-1] / shy.iloc[-1 - lookback]) - 1.0
        tlt_vs_shy_20d = tlt_20d_ret - shy_20d_ret
    else:
        tlt_vs_shy_20d = 0.0

    # 3. 10y yield 20d change
    yield_10y_20d_change = 0.0
    if yield_10y is not None and len(yield_10y.dropna()) >= 2:
        y = yield_10y.dropna()
        y_lookback = min(20, len(y) - 1)
        if y_lookback > 0:
            yield_10y_20d_change = float(y.iloc[-1] - y.iloc[-1 - y_lookback])

    # 4. Classify Treasury Fit
    treasury_fit = classify_treasury_fit(sb_corr, tlt_vs_shy_20d, yield_10y_20d_change)

    # 5. Classify Shock Type
    shock_type = classify_shock_type(yield_10y_20d_change, tlt_vs_shy_20d, move_level)

    # 6. Select Defensive Vehicle
    defensive_vehicle = select_defensive_vehicle(treasury_fit, shock_type)

    # 7. Gate Watch
    gate_watch = compute_gate_watch(regime_signals, settings)

    # 8. Cash hurdle already computed above

    # 9. Description
    parts = [
        f"Treasury Fit: {treasury_fit.value}",
        f"SB corr: {sb_corr:+.2f}",
        f"TLT-SHY 20d: {tlt_vs_shy_20d:+.3f}",
        f"10y chg: {yield_10y_20d_change:+.2f}",
        f"MOVE: {move_level:.0f}",
        f"Shock: {shock_type.value}",
        f"Vehicle: {defensive_vehicle.value}",
    ]
    if gate_watch:
        parts.append("GATE WATCH active")
    description = " | ".join(parts)

    return TreasuryContextReading(
        treasury_fit=treasury_fit,
        cash_hurdle=cash_hurdle,
        shock_type=shock_type,
        sb_correlation=sb_corr,
        move_level=move_level,
        tlt_vs_shy_20d=tlt_vs_shy_20d,
        yield_10y_20d_change=yield_10y_20d_change,
        defensive_vehicle=defensive_vehicle,
        gate_watch=gate_watch,
        description=description,
    )
