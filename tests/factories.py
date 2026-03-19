"""
Synthetic data factories for deterministic testing.

Every market scenario the system should handle has a factory function here.
These produce pandas DataFrames with known values so test assertions are exact.

NAMING CONVENTION:
    make_{scenario}() → dict with keys: 'prices', 'vix', 'vix3m', 'volumes'

SCENARIOS COVERED:
    - normal_market: VIX low, breadth healthy, credit stable, clear sector leadership
    - hostile_market: VIX spiking, breadth collapsed, credit stress
    - fragile_market: mixed signals, borderline thresholds
    - edge_case_thresholds: values exactly ON threshold boundaries
    - missing_data: gaps, NaN, partial tickers
    - sector_rotation: one sector decaying while another accelerating
    - breadth_divergence: SPY up, RSP down (narrow market)
    - momentum_crash: sharp reversal in leaders
    - flat_market: everything choppy, no clear direction → Ambiguous
    - single_sector_pump: one sector dramatically outperforming
    - all_sectors_same: no differentiation → tests rank stability
"""

import pandas as pd
import numpy as np


SECTOR_TICKERS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]
ALL_TICKERS = ["SPY", "RSP", "HYG", "LQD", "QQQ", "IWM", "DIA"] + SECTOR_TICKERS


def _make_date_index(n_days: int, end_date: str = "2026-03-18") -> pd.DatetimeIndex:
    """Generate n trading days ending at end_date (skip weekends)."""
    end = pd.Timestamp(end_date)
    dates = pd.bdate_range(end=end, periods=n_days)
    return dates


def _make_prices_from_returns(tickers: list[str], daily_returns: dict[str, list[float]],
                              start_price: float = 100.0,
                              n_days: int = None) -> pd.DataFrame:
    """
    Build a price DataFrame from specified daily returns per ticker.
    daily_returns: {ticker: [r1, r2, ...]} where r is decimal return (0.01 = +1%)
    If n_days specified and > len(returns), pad with near-zero returns at the front.
    """
    if n_days is None:
        n_days = max(len(v) for v in daily_returns.values())

    dates = _make_date_index(n_days)
    data = {}
    for ticker in tickers:
        returns = daily_returns.get(ticker, [0.001] * n_days)
        if len(returns) < n_days:
            returns = [0.001] * (n_days - len(returns)) + returns
        returns = returns[:n_days]
        prices = [start_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        data[ticker] = prices[1:]  # Drop seed price

    return pd.DataFrame(data, index=dates[:n_days])


def _constant_vix(n_days: int, vix_level: float, vix3m_level: float) -> tuple[pd.Series, pd.Series]:
    """Generate constant VIX and VIX3M series."""
    dates = _make_date_index(n_days)
    vix = pd.Series(vix_level, index=dates, name="vix")
    vix3m = pd.Series(vix3m_level, index=dates, name="vix3m")
    return vix, vix3m


# ═══════════════════════════════════════════════════════
# SCENARIO FACTORIES
# ═══════════════════════════════════════════════════════

def make_normal_market(n_days: int = 120) -> dict:
    """
    NORMAL regime scenario:
    - VIX: steady around 15, VIX3M ~17 (contango, ratio ~0.88)
    - SPY: steady uptrend (+0.05%/day)
    - RSP: slightly outperforming SPY (+0.06%/day, healthy breadth)
    - HYG: stable, slightly positive (+0.02%/day)
    - LQD: stable (+0.01%/day)
    - XLK: strongest sector (+0.10%/day)
    - XLE: weakest sector (-0.03%/day)
    - Others: distributed between
    """
    np.random.seed(42)

    # Define daily returns per ticker
    sector_rates = {
        "XLK": 0.0010,   # Strongest
        "XLC": 0.0008,
        "XLY": 0.0007,
        "XLF": 0.0006,
        "XLI": 0.0005,
        "XLV": 0.0004,
        "XLP": 0.0003,
        "XLU": 0.0002,
        "XLRE": 0.0001,
        "XLB": 0.0000,
        "XLE": -0.0003,  # Weakest
    }

    daily_returns = {}
    for ticker, rate in sector_rates.items():
        noise = np.random.normal(0, 0.002, n_days)
        daily_returns[ticker] = list(rate + noise)

    # Market-level
    daily_returns["SPY"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["RSP"] = list(0.0006 + np.random.normal(0, 0.002, n_days))  # Slightly outperforms
    daily_returns["HYG"] = list(0.0002 + np.random.normal(0, 0.001, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["QQQ"] = list(0.0006 + np.random.normal(0, 0.003, n_days))
    daily_returns["IWM"] = list(0.0004 + np.random.normal(0, 0.003, n_days))
    daily_returns["DIA"] = list(0.0004 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)

    # VIX steady around 15, VIX3M ~17 → ratio ~0.88
    vix, vix3m = _constant_vix(n_days, 15.0, 17.0)
    # Add small noise
    vix = vix + pd.Series(np.random.normal(0, 0.5, n_days), index=vix.index)
    vix3m = vix3m + pd.Series(np.random.normal(0, 0.3, n_days), index=vix3m.index)

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_hostile_market(n_days: int = 120) -> dict:
    """
    HOSTILE regime scenario:
    - VIX: spiking above 35, VIX3M ~30 (backwardation, ratio ~1.17)
    - SPY: sharp decline (-0.5%/day recent)
    - RSP: declining faster than SPY (breadth collapsing)
    - HYG: declining sharply (credit stress)
    - LQD: stable or rising (flight to quality)
    - All sectors negative, high correlation
    """
    np.random.seed(43)

    daily_returns = {}
    # All sectors negative, correlated
    for ticker in SECTOR_TICKERS:
        daily_returns[ticker] = list(-0.005 + np.random.normal(0, 0.003, n_days))

    daily_returns["SPY"] = list(-0.005 + np.random.normal(0, 0.003, n_days))
    daily_returns["RSP"] = list(-0.007 + np.random.normal(0, 0.003, n_days))  # Worse than SPY
    daily_returns["HYG"] = list(-0.004 + np.random.normal(0, 0.002, n_days))  # Credit stress
    daily_returns["LQD"] = list(0.001 + np.random.normal(0, 0.001, n_days))   # Flight to quality
    daily_returns["QQQ"] = list(-0.006 + np.random.normal(0, 0.004, n_days))
    daily_returns["IWM"] = list(-0.007 + np.random.normal(0, 0.004, n_days))
    daily_returns["DIA"] = list(-0.004 + np.random.normal(0, 0.003, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)

    # VIX spiking, backwardation
    vix, vix3m = _constant_vix(n_days, 35.0, 30.0)
    vix = vix + pd.Series(np.random.normal(0, 2.0, n_days), index=vix.index)
    vix3m = vix3m + pd.Series(np.random.normal(0, 1.0, n_days), index=vix3m.index)

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="climax")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_fragile_market(n_days: int = 120) -> dict:
    """
    FRAGILE regime scenario:
    - VIX: 24 (between 20 and 30)
    - Term structure: flat (VIX/VIX3M ~ 1.00)
    - Mixed breadth: RSP slightly underperforming SPY
    - Credit: slightly stressed but not hostile
    """
    np.random.seed(44)

    sector_rates = {
        "XLK": 0.0004,
        "XLC": 0.0003,
        "XLY": 0.0002,
        "XLF": 0.0001,
        "XLI": 0.0000,
        "XLV": -0.0001,
        "XLP": -0.0001,
        "XLU": -0.0002,
        "XLRE": -0.0003,
        "XLB": -0.0003,
        "XLE": -0.0004,
    }

    daily_returns = {}
    for ticker, rate in sector_rates.items():
        daily_returns[ticker] = list(rate + np.random.normal(0, 0.003, n_days))

    daily_returns["SPY"] = list(0.0001 + np.random.normal(0, 0.003, n_days))
    daily_returns["RSP"] = list(-0.0001 + np.random.normal(0, 0.003, n_days))
    daily_returns["HYG"] = list(-0.001 + np.random.normal(0, 0.002, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["QQQ"] = list(0.0002 + np.random.normal(0, 0.004, n_days))
    daily_returns["IWM"] = list(-0.0002 + np.random.normal(0, 0.004, n_days))
    daily_returns["DIA"] = list(0.0001 + np.random.normal(0, 0.003, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)

    # VIX ~24, flat term structure
    vix, vix3m = _constant_vix(n_days, 24.0, 24.0)
    vix = vix + pd.Series(np.random.normal(0, 1.0, n_days), index=vix.index)
    vix3m = vix3m + pd.Series(np.random.normal(0, 0.5, n_days), index=vix3m.index)

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_edge_case_thresholds(n_days: int = 120) -> dict:
    """
    VALUES EXACTLY ON THRESHOLD BOUNDARIES.
    Returns a dict with boundary-specific sub-scenarios accessible by key.
    The main 'prices' are normal-like but VIX/VIX3M are set to boundary values.
    """
    np.random.seed(45)

    daily_returns = {}
    for ticker in ALL_TICKERS:
        daily_returns[ticker] = list(0.0003 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)

    # VIX exactly 20.0 (Normal/Fragile boundary)
    dates = _make_date_index(n_days)
    vix_20 = pd.Series(20.0, index=dates, name="vix")
    vix3m_20 = pd.Series(21.0, index=dates, name="vix3m")  # ratio = 0.952 → FRAGILE

    # VIX exactly 30.0 (Fragile/Hostile boundary)
    vix_30 = pd.Series(30.0, index=dates, name="vix")
    vix3m_30 = pd.Series(28.0, index=dates, name="vix3m")

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix_20,
        "vix3m": vix3m_20,
        "vix_30": vix_30,
        "vix3m_30": vix3m_30,
        "volumes": volumes,
    }


def make_sector_rotation(n_days: int = 60) -> dict:
    """
    BATON PASS scenario:
    - XLK: strong first half (+0.15%/day), decaying second half (-0.05%/day)
    - XLV: weak first half (-0.05%/day), accelerating second half (+0.15%/day)
    - Other sectors: mild uptrend throughout
    - Market regime stays NORMAL
    """
    np.random.seed(46)
    half = n_days // 2

    daily_returns = {}
    # XLK: strong then weak
    xlk_returns = list(0.0015 + np.random.normal(0, 0.002, half)) + \
                  list(-0.0005 + np.random.normal(0, 0.002, n_days - half))
    daily_returns["XLK"] = xlk_returns

    # XLV: weak then strong
    xlv_returns = list(-0.0005 + np.random.normal(0, 0.002, half)) + \
                  list(0.0015 + np.random.normal(0, 0.002, n_days - half))
    daily_returns["XLV"] = xlv_returns

    # Others: mild uptrend
    for ticker in SECTOR_TICKERS:
        if ticker not in ("XLK", "XLV"):
            daily_returns[ticker] = list(0.0003 + np.random.normal(0, 0.002, n_days))

    daily_returns["SPY"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["RSP"] = list(0.0006 + np.random.normal(0, 0.002, n_days))
    daily_returns["HYG"] = list(0.0002 + np.random.normal(0, 0.001, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["QQQ"] = list(0.0005 + np.random.normal(0, 0.003, n_days))
    daily_returns["IWM"] = list(0.0004 + np.random.normal(0, 0.003, n_days))
    daily_returns["DIA"] = list(0.0004 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)
    vix, vix3m = _constant_vix(n_days, 15.0, 17.0)

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_breadth_divergence(n_days: int = 60) -> dict:
    """
    NARROW MARKET scenario:
    - SPY rising (driven by mega-caps, +0.08%/day)
    - RSP flat or declining (-0.02%/day, most stocks not participating)
    - RSP/SPY ratio falling → z-score going negative
    - Breadth signal should be DIVERGING
    """
    np.random.seed(47)

    daily_returns = {}
    daily_returns["SPY"] = list(0.0008 + np.random.normal(0, 0.002, n_days))
    daily_returns["RSP"] = list(-0.0002 + np.random.normal(0, 0.002, n_days))
    daily_returns["HYG"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["QQQ"] = list(0.0010 + np.random.normal(0, 0.003, n_days))
    daily_returns["IWM"] = list(-0.0003 + np.random.normal(0, 0.003, n_days))
    daily_returns["DIA"] = list(0.0005 + np.random.normal(0, 0.002, n_days))

    # XLK drives narrow rally, most sectors flat/negative
    daily_returns["XLK"] = list(0.0015 + np.random.normal(0, 0.002, n_days))
    daily_returns["XLC"] = list(0.0010 + np.random.normal(0, 0.002, n_days))
    for ticker in SECTOR_TICKERS:
        if ticker not in daily_returns:
            daily_returns[ticker] = list(-0.0001 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)
    vix, vix3m = _constant_vix(n_days, 16.0, 18.0)

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_flat_choppy_market(n_days: int = 60) -> dict:
    """
    AMBIGUOUS scenario:
    - All sectors flip-flopping, no clear leader
    - RS deltas alternate positive/negative every 2-3 days
    - VIX moderate (18-22)
    - No persistent signal in any direction
    """
    np.random.seed(48)

    daily_returns = {}
    for ticker in ALL_TICKERS:
        # High noise, near-zero drift → choppy
        daily_returns[ticker] = list(np.random.normal(0, 0.005, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)

    dates = _make_date_index(n_days)
    vix = pd.Series(20.0 + np.random.normal(0, 1.5, n_days), index=dates, name="vix")
    vix3m = pd.Series(20.5 + np.random.normal(0, 1.0, n_days), index=dates, name="vix3m")

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_single_sector_pump(n_days: int = 60, pumping_sector: str = "XLE") -> dict:
    """
    ONE SECTOR OVERT PUMP:
    - pumping_sector: +0.3%/day consistently
    - All others: flat to mildly positive (+0.03%/day)
    """
    np.random.seed(49)

    daily_returns = {}
    for ticker in SECTOR_TICKERS:
        if ticker == pumping_sector:
            daily_returns[ticker] = list(0.003 + np.random.normal(0, 0.001, n_days))
        else:
            daily_returns[ticker] = list(0.0003 + np.random.normal(0, 0.002, n_days))

    daily_returns["SPY"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["RSP"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["HYG"] = list(0.0002 + np.random.normal(0, 0.001, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["QQQ"] = list(0.0005 + np.random.normal(0, 0.003, n_days))
    daily_returns["IWM"] = list(0.0004 + np.random.normal(0, 0.003, n_days))
    daily_returns["DIA"] = list(0.0004 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)
    vix, vix3m = _constant_vix(n_days, 14.0, 16.0)

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_missing_data(n_days: int = 60) -> dict:
    """
    DATA QUALITY scenario:
    - VIX3M has 5 consecutive NaN days (indices 20-24)
    - XLRE has 3 missing days (indices 15-17)
    - HYG has a gap (indices 30-32) but LQD doesn't
    """
    np.random.seed(50)

    daily_returns = {}
    for ticker in ALL_TICKERS:
        daily_returns[ticker] = list(0.0003 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)

    # Introduce NaN gaps
    prices.iloc[15:18, prices.columns.get_loc("XLRE")] = np.nan
    prices.iloc[30:33, prices.columns.get_loc("HYG")] = np.nan

    dates = _make_date_index(n_days)
    vix = pd.Series(16.0 + np.random.normal(0, 0.5, n_days), index=dates, name="vix")
    vix3m = pd.Series(18.0 + np.random.normal(0, 0.3, n_days), index=dates, name="vix3m")
    # VIX3M has 5 consecutive NaN
    vix3m.iloc[20:25] = np.nan

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_momentum_crash(n_days: int = 60) -> dict:
    """
    MOMENTUM CRASH scenario:
    - First 40 days: XLK, XLY clear leaders (+0.12%/day)
    - Days 40-45: sharp reversal — leaders drop 3%/day, laggards rally 1%/day
    - Days 45-60: continued weakness in leaders
    - VIX spikes from 15 to 32
    - Credit proxy deteriorates
    """
    np.random.seed(51)

    daily_returns = {}
    # Phase 1 (0-39): leaders strong
    # Phase 2 (40-44): crash
    # Phase 3 (45-59): continued weakness

    for ticker in SECTOR_TICKERS:
        if ticker in ("XLK", "XLY"):
            phase1 = list(0.0012 + np.random.normal(0, 0.002, 40))
            phase2 = list(-0.030 + np.random.normal(0, 0.005, 5))
            phase3 = list(-0.003 + np.random.normal(0, 0.003, n_days - 45))
        elif ticker in ("XLU", "XLP"):  # Defensive sectors rally
            phase1 = list(0.0001 + np.random.normal(0, 0.002, 40))
            phase2 = list(0.010 + np.random.normal(0, 0.003, 5))
            phase3 = list(0.002 + np.random.normal(0, 0.002, n_days - 45))
        else:
            phase1 = list(0.0003 + np.random.normal(0, 0.002, 40))
            phase2 = list(-0.010 + np.random.normal(0, 0.005, 5))
            phase3 = list(-0.001 + np.random.normal(0, 0.003, n_days - 45))
        daily_returns[ticker] = phase1 + phase2 + phase3

    # SPY crashes
    daily_returns["SPY"] = list(0.0005 + np.random.normal(0, 0.002, 40)) + \
                           list(-0.020 + np.random.normal(0, 0.005, 5)) + \
                           list(-0.002 + np.random.normal(0, 0.003, n_days - 45))
    daily_returns["RSP"] = list(0.0004 + np.random.normal(0, 0.002, 40)) + \
                           list(-0.025 + np.random.normal(0, 0.005, 5)) + \
                           list(-0.003 + np.random.normal(0, 0.003, n_days - 45))
    daily_returns["HYG"] = list(0.0002 + np.random.normal(0, 0.001, 40)) + \
                           list(-0.015 + np.random.normal(0, 0.003, 5)) + \
                           list(-0.002 + np.random.normal(0, 0.002, n_days - 45))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, 40)) + \
                           list(0.002 + np.random.normal(0, 0.001, 5)) + \
                           list(0.001 + np.random.normal(0, 0.001, n_days - 45))
    daily_returns["QQQ"] = list(0.0006 + np.random.normal(0, 0.003, 40)) + \
                           list(-0.025 + np.random.normal(0, 0.005, 5)) + \
                           list(-0.003 + np.random.normal(0, 0.003, n_days - 45))
    daily_returns["IWM"] = list(0.0003 + np.random.normal(0, 0.003, 40)) + \
                           list(-0.020 + np.random.normal(0, 0.005, 5)) + \
                           list(-0.003 + np.random.normal(0, 0.003, n_days - 45))
    daily_returns["DIA"] = list(0.0004 + np.random.normal(0, 0.002, 40)) + \
                           list(-0.018 + np.random.normal(0, 0.004, 5)) + \
                           list(-0.002 + np.random.normal(0, 0.003, n_days - 45))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)

    # VIX: calm then spike
    dates = _make_date_index(n_days)
    vix_vals = np.concatenate([
        15.0 + np.random.normal(0, 0.5, 40),
        np.linspace(15, 32, 5) + np.random.normal(0, 1.0, 5),
        32.0 + np.random.normal(0, 2.0, n_days - 45),
    ])
    vix = pd.Series(vix_vals, index=dates, name="vix")
    vix3m_vals = np.concatenate([
        17.0 + np.random.normal(0, 0.3, 40),
        np.linspace(17, 27, 5) + np.random.normal(0, 0.5, 5),
        27.0 + np.random.normal(0, 1.0, n_days - 45),
    ])
    vix3m = pd.Series(vix3m_vals, index=dates, name="vix3m")

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="climax")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


def make_all_sectors_identical(n_days: int = 60) -> dict:
    """
    NO DIFFERENTIATION scenario:
    - All 11 sectors have identical returns (+0.05%/day)
    - RS values all near zero
    """
    np.random.seed(52)

    daily_returns = {}
    base_returns = list(0.0005 + np.random.normal(0, 0.0001, n_days))
    for ticker in SECTOR_TICKERS:
        daily_returns[ticker] = base_returns.copy()

    daily_returns["SPY"] = base_returns.copy()
    daily_returns["RSP"] = base_returns.copy()
    daily_returns["HYG"] = list(0.0002 + np.random.normal(0, 0.0001, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.0001, n_days))
    daily_returns["QQQ"] = base_returns.copy()
    daily_returns["IWM"] = base_returns.copy()
    daily_returns["DIA"] = base_returns.copy()

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)
    vix, vix3m = _constant_vix(n_days, 15.0, 17.0)

    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="normal")

    return {
        "prices": prices,
        "vix": vix,
        "vix3m": vix3m,
        "volumes": volumes,
    }


# ═══════════════════════════════════════════════════════
# HELPERS FOR VIX
# ═══════════════════════════════════════════════════════

def make_vix_series(n_days: int, pattern: str = "normal") -> pd.DataFrame:
    """
    Generate VIX and VIX3M series.
    Patterns: "normal" (~15), "fragile" (~25), "hostile" (~35), "spike" (15→35 in 5 days)
    Returns DataFrame with columns: ['vix', 'vix3m']
    """
    np.random.seed(53)
    dates = _make_date_index(n_days)

    if pattern == "normal":
        vix_vals = 15.0 + np.random.normal(0, 0.5, n_days)
        vix3m_vals = 17.0 + np.random.normal(0, 0.3, n_days)
    elif pattern == "fragile":
        vix_vals = 25.0 + np.random.normal(0, 1.0, n_days)
        vix3m_vals = 25.0 + np.random.normal(0, 0.5, n_days)
    elif pattern == "hostile":
        vix_vals = 35.0 + np.random.normal(0, 2.0, n_days)
        vix3m_vals = 30.0 + np.random.normal(0, 1.0, n_days)
    elif pattern == "spike":
        # Normal for most, spike at end
        spike_start = max(0, n_days - 10)
        vix_vals = np.concatenate([
            15.0 + np.random.normal(0, 0.5, spike_start),
            np.linspace(15, 35, min(5, n_days - spike_start)),
            35.0 + np.random.normal(0, 2.0, max(0, n_days - spike_start - 5)),
        ])[:n_days]
        vix3m_vals = np.concatenate([
            17.0 + np.random.normal(0, 0.3, spike_start),
            np.linspace(17, 28, min(5, n_days - spike_start)),
            28.0 + np.random.normal(0, 1.0, max(0, n_days - spike_start - 5)),
        ])[:n_days]
    else:
        raise ValueError(f"Unknown VIX pattern: {pattern}")

    return pd.DataFrame({"vix": vix_vals, "vix3m": vix3m_vals}, index=dates)


# ═══════════════════════════════════════════════════════
# VOLUME FACTORIES
# ═══════════════════════════════════════════════════════

def make_volume_data(n_days: int, tickers: list[str],
                     base_volume: int = 10_000_000,
                     pattern: str = "normal") -> pd.DataFrame:
    """
    Generate volume data.
    Patterns: "normal", "climax" (spike at end), "drying_up" (declining), "surge" (increasing)
    """
    np.random.seed(54)
    dates = _make_date_index(n_days)
    data = {}

    for ticker in tickers:
        if pattern == "normal":
            vols = base_volume + np.random.normal(0, base_volume * 0.1, n_days)
        elif pattern == "climax":
            # Normal then spike in last 10 days
            normal_part = base_volume + np.random.normal(0, base_volume * 0.1, max(0, n_days - 10))
            spike_part = base_volume * 3 + np.random.normal(0, base_volume * 0.3, min(10, n_days))
            vols = np.concatenate([normal_part, spike_part])
        elif pattern == "drying_up":
            scale = np.linspace(1.0, 0.3, n_days)
            vols = base_volume * scale + np.random.normal(0, base_volume * 0.05, n_days)
        elif pattern == "surge":
            scale = np.linspace(0.5, 2.0, n_days)
            vols = base_volume * scale + np.random.normal(0, base_volume * 0.05, n_days)
        else:
            raise ValueError(f"Unknown volume pattern: {pattern}")

        data[ticker] = np.maximum(vols, 1000).astype(int)  # No negative volumes

    return pd.DataFrame(data, index=dates)


# ═══════════════════════════════════════════════════════
# PHASE 2: INDUSTRY + REVERSAL FACTORIES
# ═══════════════════════════════════════════════════════

INDUSTRY_TICKERS = [
    "SMH", "IGV", "HACK", "SOXX", "XBI", "IHI", "KRE", "IAI", "KIE",
    "XOP", "OIH", "URA", "ITA", "XAR", "XHB", "ITB", "XRT", "IBUY",
    "XME", "GDX", "VNQ", "TAN", "NLR",
]

INDUSTRY_PARENT_MAP = {
    "SMH": "XLK", "IGV": "XLK", "HACK": "XLK", "SOXX": "XLK",
    "XBI": "XLV", "IHI": "XLV",
    "KRE": "XLF", "IAI": "XLF", "KIE": "XLF",
    "XOP": "XLE", "OIH": "XLE", "URA": "XLE",
    "ITA": "XLI", "XAR": "XLI",
    "XHB": "XLY", "ITB": "XLY", "XRT": "XLY", "IBUY": "XLY",
    "XME": "XLB", "GDX": "XLB",
    "VNQ": "XLRE",
    "TAN": "XLU",
    "NLR": "XLU",
    "SOXX": "XLK",
    "URA": "XLE",
}

ALL_TICKERS_V2 = ALL_TICKERS + INDUSTRY_TICKERS


def _add_industry_prices(base: dict, n_days: int, industry_returns: dict,
                         seed: int = 200) -> dict:
    """Add industry price columns to an existing scenario dict."""
    np.random.seed(seed)
    dates = base["prices"].index
    for ticker in INDUSTRY_TICKERS:
        if ticker in industry_returns:
            rets = industry_returns[ticker]
        else:
            # Default: track parent sector + small noise
            parent = INDUSTRY_PARENT_MAP[ticker]
            if parent in base["prices"].columns:
                parent_rets = base["prices"][parent].pct_change().fillna(0).values
                rets = list(parent_rets + np.random.normal(0, 0.001, len(parent_rets)))
            else:
                rets = list(np.random.normal(0.0003, 0.002, n_days))
        # Build price column
        prices_list = [100.0]
        for r in rets[:n_days]:
            prices_list.append(prices_list[-1] * (1 + r))
        base["prices"][ticker] = prices_list[1:n_days+1]

    # Add highs/lows for reversal score computation
    base["highs"] = base["prices"] * (1 + np.abs(np.random.normal(0, 0.005, base["prices"].shape)))
    base["lows"] = base["prices"] * (1 - np.abs(np.random.normal(0, 0.005, base["prices"].shape)))
    # Ensure high >= close >= low
    base["highs"] = base["highs"].clip(lower=base["prices"])
    base["lows"] = base["lows"].clip(upper=base["prices"])

    # Extend volumes for industry tickers
    for ticker in INDUSTRY_TICKERS:
        if ticker not in base["volumes"].columns:
            base["volumes"][ticker] = np.random.randint(1_000_000, 20_000_000, n_days)

    return base


def make_industry_normal_market(n_days: int = 120) -> dict:
    """
    Normal market WITH industry ETFs.
    - SMH strongly outperforming XLK (driving the sector)
    - XBI lagging XLV (not driving its sector)
    - XOP roughly tracking XLE (neutral vs parent)
    """
    base = make_normal_market(n_days)
    industry_returns = {
        "SMH": list(0.0015 + np.random.normal(0, 0.002, n_days)),  # Strong vs XLK (+0.15%/day)
        "XBI": list(-0.0005 + np.random.normal(0, 0.002, n_days)),  # Lagging XLV (-0.05%/day)
    }
    return _add_industry_prices(base, n_days, industry_returns, seed=201)


def make_industry_rotation(n_days: int = 60) -> dict:
    """
    INDUSTRY-LEVEL BATON PASS:
    - SMH was leading (first 30 days), now decaying
    - XBI was lagging (first 30 days), now accelerating
    - Industry signal leads sector signal
    """
    base = make_sector_rotation(n_days)
    half = n_days // 2
    industry_returns = {
        "SMH": list(0.002 + np.random.normal(0, 0.002, half)) +
               list(-0.001 + np.random.normal(0, 0.002, n_days - half)),
        "XBI": list(-0.001 + np.random.normal(0, 0.002, half)) +
               list(0.002 + np.random.normal(0, 0.002, n_days - half)),
    }
    return _add_industry_prices(base, n_days, industry_returns, seed=202)


def make_reversal_exhaustion(n_days: int = 60) -> dict:
    """
    EXHAUSTION REVERSAL scenario:
    - XLE strong run (days 1-40): +0.25%/day, rising volume
    - Days 41-60: price near highs but RS slope negative, CLV declining,
      failed breakout, volume spiking (climax)
    """
    np.random.seed(203)
    daily_returns = {}
    for ticker in SECTOR_TICKERS:
        if ticker == "XLE":
            phase1 = list(0.0025 + np.random.normal(0, 0.002, 40))
            phase2 = list(-0.0005 + np.random.normal(0, 0.003, n_days - 40))
            daily_returns[ticker] = phase1 + phase2
        else:
            daily_returns[ticker] = list(0.0003 + np.random.normal(0, 0.002, n_days))

    daily_returns["SPY"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["RSP"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["HYG"] = list(0.0002 + np.random.normal(0, 0.001, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["QQQ"] = list(0.0005 + np.random.normal(0, 0.003, n_days))
    daily_returns["IWM"] = list(0.0004 + np.random.normal(0, 0.003, n_days))
    daily_returns["DIA"] = list(0.0004 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)
    vix, vix3m = _constant_vix(n_days, 16.0, 18.0)
    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="climax")

    result = {"prices": prices, "vix": vix, "vix3m": vix3m, "volumes": volumes}
    return _add_industry_prices(result, n_days, {}, seed=204)


def make_reversal_crowding(n_days: int = 60) -> dict:
    """
    CROWDING / STRETCH scenario:
    - XLK parabolic last 20 days (+0.5%/day accelerating)
    - Price acceleration extreme, distance from MA > 3σ, volume 2x+
    """
    np.random.seed(205)
    daily_returns = {}
    for ticker in SECTOR_TICKERS:
        if ticker == "XLK":
            normal = list(0.001 + np.random.normal(0, 0.002, n_days - 20))
            parabolic = list(np.linspace(0.003, 0.008, 20) + np.random.normal(0, 0.001, 20))
            daily_returns[ticker] = normal + parabolic
        else:
            daily_returns[ticker] = list(0.0003 + np.random.normal(0, 0.002, n_days))

    daily_returns["SPY"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["RSP"] = list(0.0005 + np.random.normal(0, 0.002, n_days))
    daily_returns["HYG"] = list(0.0002 + np.random.normal(0, 0.001, n_days))
    daily_returns["LQD"] = list(0.0001 + np.random.normal(0, 0.001, n_days))
    daily_returns["QQQ"] = list(0.0006 + np.random.normal(0, 0.003, n_days))
    daily_returns["IWM"] = list(0.0004 + np.random.normal(0, 0.003, n_days))
    daily_returns["DIA"] = list(0.0004 + np.random.normal(0, 0.002, n_days))

    prices = _make_prices_from_returns(ALL_TICKERS, daily_returns, n_days=n_days)
    vix, vix3m = _constant_vix(n_days, 15.0, 17.0)
    volumes = make_volume_data(n_days, ALL_TICKERS, pattern="surge")

    result = {"prices": prices, "vix": vix, "vix3m": vix3m, "volumes": volumes}
    return _add_industry_prices(result, n_days, {}, seed=206)


def make_turnover_marginal(n_days: int = 30) -> dict:
    """Current holding Pump Δ = +0.02, candidate Δ = +0.05. Gap = 0.03 < 0.08 threshold."""
    return {
        "current_deltas": [0.02] * n_days,
        "candidate_deltas": [0.05] * n_days,
    }


def make_turnover_clear(n_days: int = 30) -> dict:
    """Current holding Pump Δ = -0.01, candidate Δ = +0.10. Gap = 0.11 > 0.08."""
    return {
        "current_deltas": [-0.01] * n_days,
        "candidate_deltas": [0.10] * n_days,
    }


def make_turnover_exempt(n_days: int = 30) -> dict:
    """Current in Exhaustion, advantage only 0.04 — exempt from threshold."""
    return {
        "current_deltas": [0.01] * n_days,
        "candidate_deltas": [0.05] * n_days,
        "current_state": "Exhaustion",
    }
