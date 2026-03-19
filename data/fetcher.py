"""
Data fetcher — yfinance + FRED data retrieval.

Fetches:
- Price data (OHLCV) for all tickers via yfinance bulk download
- VIX and VIX3M via yfinance
- HY OAS spread via FRED (optional, graceful fallback)
"""
import os
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Sector + market tickers for bulk download
_SECTOR_TICKERS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]
_MARKET_TICKERS = ["SPY", "RSP", "HYG", "LQD", "QQQ", "IWM", "DIA"]
_INDUSTRY_TICKERS = [
    "SMH", "IGV", "HACK", "SOXX", "XBI", "IHI", "KRE", "IAI", "KIE",
    "XOP", "OIH", "URA", "ITA", "XAR", "XHB", "ITB", "XRT", "IBUY",
    "XME", "GDX", "VNQ", "TAN", "NLR",
]
_VIX_TICKERS = ["^VIX", "^VIX3M"]


def fetch_all(config: dict, force_refresh: bool = False) -> dict:
    """
    Master fetch. Returns dict:
    {
        'prices': DataFrame (date × ticker, Close prices),
        'volumes': DataFrame (date × ticker, Volume),
        'highs': DataFrame (date × ticker, High),
        'lows': DataFrame (date × ticker, Low),
        'vix': Series (date, vix close),
        'vix3m': Series (date, vix3m close),
        'credit': DataFrame (date, [hyg_close, lqd_close]),
        'fred_hy_oas': DataFrame or None,
        'metadata': dict,
    }
    """
    period = config.get("data", {}).get("fetch_period", "2y")
    # Include industry tickers from config if present, else use defaults
    industry_tickers = [ind["ticker"] for ind in config.get("industries", [])] or _INDUSTRY_TICKERS
    all_tickers = _MARKET_TICKERS + _SECTOR_TICKERS + industry_tickers
    # Deduplicate while preserving order
    seen = set()
    all_tickers = [t for t in all_tickers if not (t in seen or seen.add(t))]
    warnings = []
    errors = []

    # Fetch equity prices (daily bars — may be stale during market hours)
    prices, volumes, highs, lows = _fetch_yfinance_equities(all_tickers, period, errors, warnings)

    # Overlay live intraday quotes on today's row
    prices, highs, lows = _overlay_live_quotes(prices, highs, lows, all_tickers, warnings)

    # Fetch VIX data + overlay live VIX quote
    vix, vix3m = _fetch_vix(period, errors, warnings)
    vix, vix3m = _overlay_live_vix(vix, vix3m, warnings)

    # Credit data from equities
    credit = pd.DataFrame(index=prices.index)
    if "HYG" in prices.columns:
        credit["hyg_close"] = prices["HYG"]
    if "LQD" in prices.columns:
        credit["lqd_close"] = prices["LQD"]

    # FRED data (optional)
    fred_data = _fetch_fred_safe(config)

    metadata = {
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
        "tickers": list(prices.columns),
        "rows": len(prices),
        "last_price_date": str(prices.index[-1].date()) if not prices.empty else "N/A",
        "live_overlay": True,
        "errors": errors,
        "warnings": warnings,
    }

    return {
        "prices": prices,
        "volumes": volumes,
        "highs": highs,
        "lows": lows,
        "vix": vix,
        "vix3m": vix3m,
        "credit": credit,
        "fred_hy_oas": fred_data,
        "metadata": metadata,
    }


def _fetch_yfinance_equities(
    tickers: list[str], period: str,
    errors: list, warnings: list,
    max_retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV for equity tickers. Retry up to max_retries. Forward-fill 3 days max."""
    raw = None
    for attempt in range(max_retries):
        try:
            raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            logger.warning(f"yfinance attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                errors.append(f"yfinance failed after {max_retries} attempts: {e}")

    if raw is None or raw.empty:
        empty_idx = pd.DatetimeIndex([])
        return (
            pd.DataFrame(index=empty_idx),
            pd.DataFrame(index=empty_idx),
            pd.DataFrame(index=empty_idx),
            pd.DataFrame(index=empty_idx),
        )

    # Handle MultiIndex columns from bulk download
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
        volumes = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
        highs = raw["High"] if "High" in raw.columns.get_level_values(0) else pd.DataFrame()
        lows = raw["Low"] if "Low" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        # Single ticker case
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in raw.columns else pd.DataFrame()
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]}) if "Volume" in raw.columns else pd.DataFrame()
        highs = raw[["High"]].rename(columns={"High": tickers[0]}) if "High" in raw.columns else pd.DataFrame()
        lows = raw[["Low"]].rename(columns={"Low": tickers[0]}) if "Low" in raw.columns else pd.DataFrame()

    # Forward-fill gaps (max 3 days)
    closes = closes.ffill(limit=3)
    volumes = volumes.ffill(limit=3)
    highs = highs.ffill(limit=3)
    lows = lows.ffill(limit=3)

    # Check for missing tickers
    for t in tickers:
        if t not in closes.columns:
            warnings.append(f"Ticker {t} missing from download")

    return closes, volumes, highs, lows


def _fetch_vix(
    period: str, errors: list, warnings: list,
) -> tuple[pd.Series, pd.Series]:
    """Fetch VIX and VIX3M separately."""
    vix = pd.Series(dtype=float, name="vix")
    vix3m = pd.Series(dtype=float, name="vix3m")

    for ticker, name, target in [("^VIX", "VIX", "vix"), ("^VIX3M", "VIX3M", "vix3m")]:
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                series = raw["Close"].squeeze()
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                series.name = target
                series = series.ffill(limit=3)
                if target == "vix":
                    vix = series
                else:
                    vix3m = series
            else:
                warnings.append(f"{name} download returned empty")
        except Exception as e:
            warnings.append(f"{name} fetch failed: {e}")

    return vix, vix3m


def _overlay_live_quotes(
    prices: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame,
    tickers: list[str], warnings: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patch today's row in prices/highs/lows with live intraday quotes.
    yf.download daily bars may be stale (yesterday's close) during market hours.
    yf.Ticker.info['regularMarketPrice'] gives the actual live quote.
    """
    if prices.empty:
        return prices, highs, lows

    today = prices.index[-1]
    updated = 0

    for t in tickers:
        if t not in prices.columns:
            continue
        try:
            info = yf.Ticker(t).info
            live_price = info.get("regularMarketPrice")
            if live_price and live_price > 0:
                current = prices.at[today, t]
                if abs(live_price - current) / max(current, 0.01) > 0.0001:
                    prices.at[today, t] = live_price
                    # Update high/low if live price exceeds
                    if t in highs.columns:
                        highs.at[today, t] = max(highs.at[today, t], live_price)
                    if t in lows.columns:
                        lows.at[today, t] = min(lows.at[today, t], live_price)
                    updated += 1
        except Exception:
            pass  # Silent — live overlay is best-effort

    if updated > 0:
        logger.info(f"Live quote overlay: updated {updated} tickers")

    return prices, highs, lows


def _overlay_live_vix(
    vix: pd.Series, vix3m: pd.Series, warnings: list,
) -> tuple[pd.Series, pd.Series]:
    """Patch VIX/VIX3M with live quotes during market hours."""
    for ticker, series, name in [("^VIX", vix, "vix"), ("^VIX3M", vix3m, "vix3m")]:
        if series.empty:
            continue
        try:
            info = yf.Ticker(ticker).info
            live = info.get("regularMarketPrice")
            if live and live > 0:
                series.iloc[-1] = live
        except Exception:
            pass
    return vix, vix3m


def _fetch_fred_safe(config: dict) -> pd.DataFrame | None:
    """Fetch FRED HY OAS data. Returns None on any failure."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_key_here":
        logger.info("FRED API key not configured — skipping FRED data")
        return None

    fred_config = config.get("fred", {})
    series_id = fred_config.get("hy_oas", "BAMLH0A0HYM2")

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)
        if data is not None and not data.empty:
            return data.to_frame(name="hy_oas")
    except Exception as e:
        logger.warning(f"FRED fetch failed: {e}")

    return None
