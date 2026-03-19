"""
Backfill historical data and compute snapshots for replay + z-score distributions.

Usage:
    python scripts/backfill.py                    # Default: 3 years
    python scripts/backfill.py --years 5          # 5 years
    python scripts/backfill.py --start 2020-01-01 # Specific start date
"""
import argparse
import sys
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.schemas import (
    DailySnapshot, PumpScoreReading, RegimeState,
)
from engine.regime_gate import classify_regime_from_data
from engine.rs_scanner import compute_rs_readings
from engine.breadth import compute_breadth
from engine.normalizer import compute_zscore, percentile_rank
from engine.pump_score import compute_pump_score
from engine.state_classifier import classify_all_sectors
from data.snapshots import save_snapshot, list_snapshots


SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}
SECTOR_TICKERS = list(SECTOR_NAMES.keys())
MARKET_TICKERS = ["SPY", "RSP", "HYG", "LQD", "QQQ", "IWM", "DIA"]


def fetch_full_history(years: int = 3, start_date: str = None):
    """Download full history for all tickers."""
    import yfinance as yf

    all_tickers = MARKET_TICKERS + SECTOR_TICKERS
    period = f"{years}y" if start_date is None else None
    start = start_date

    print(f"Downloading {len(all_tickers)} equity tickers...")
    if start:
        equity_data = yf.download(all_tickers, start=start, auto_adjust=True, progress=True)
    else:
        equity_data = yf.download(all_tickers, period=period, auto_adjust=True, progress=True)

    if isinstance(equity_data.columns, pd.MultiIndex):
        prices = equity_data["Close"].ffill(limit=3)
        volumes = equity_data["Volume"].ffill(limit=3)
    else:
        prices = equity_data[["Close"]].ffill(limit=3)
        volumes = equity_data[["Volume"]].ffill(limit=3)

    print(f"Downloading VIX and VIX3M...")
    vix_data = yf.download(["^VIX", "^VIX3M"], period=period, start=start,
                           auto_adjust=True, progress=True)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_closes = vix_data["Close"].ffill(limit=3)
        vix = vix_closes["^VIX"] if "^VIX" in vix_closes.columns else pd.Series(dtype=float)
        vix3m = vix_closes["^VIX3M"] if "^VIX3M" in vix_closes.columns else pd.Series(dtype=float)
    else:
        vix = vix_data["Close"].ffill(limit=3)
        vix3m = pd.Series(dtype=float)

    # Save raw prices
    os.makedirs("data/store/history", exist_ok=True)
    prices.to_parquet("data/store/history/prices.parquet")
    volumes.to_parquet("data/store/history/volumes.parquet")
    print(f"Saved prices: {prices.shape}, volumes: {volumes.shape}")

    return prices, volumes, vix, vix3m


def compute_snapshots(prices, volumes, vix, vix3m, settings, min_warmup=60):
    """Compute a snapshot for each trading day after warmup period."""
    dates = prices.index[min_warmup:]
    total = len(dates)
    print(f"Computing {total} daily snapshots (warmup={min_warmup} days)...")

    prior_states = {}
    prior_ranks = {}
    delta_histories = {t: [] for t in SECTOR_TICKERS}
    prior_scores = {t: [] for t in SECTOR_TICKERS}
    saved = 0
    gaps = 0

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        # Slice up to this date
        p = prices.loc[:date]

        if len(p) < min_warmup:
            continue

        try:
            # Regime
            vix_val = vix.loc[:date].iloc[-1] if date in vix.index or len(vix.loc[:date]) > 0 else 20.0
            vix3m_val = vix3m.loc[:date].iloc[-1] if len(vix3m.loc[:date]) > 0 else 20.0
            if pd.isna(vix_val):
                vix_val = 20.0
            if pd.isna(vix3m_val):
                vix3m_val = 20.0

            breadth_reading = compute_breadth(p)
            bz = breadth_reading.rsp_spy_ratio_zscore
            if np.isnan(bz):
                bz = 0.0

            # Credit z-score
            if "HYG" in p.columns and "LQD" in p.columns:
                cr = p["HYG"] / p["LQD"]
                cr_clean = cr.dropna()
                credit_z = compute_zscore(cr_clean.iloc[-1], cr_clean) if len(cr_clean) > 2 else 0.0
            else:
                credit_z = 0.0

            regime = classify_regime_from_data(
                vix_val, vix3m_val, bz, credit_z, settings["regime"]
            )

            # RS
            rs_cfg = settings["rs"]
            rs_readings = compute_rs_readings(
                p, SECTOR_NAMES,
                windows=rs_cfg["windows"],
                slope_window=rs_cfg["slope_window"],
                composite_weights=rs_cfg["composite_weights"],
                prior_ranks=prior_ranks if prior_ranks else None,
            )

            # Pump Scores
            pump_weights = settings["pump_score"]
            pumps = {}
            for r in rs_readings:
                score = compute_pump_score(r.rs_composite, 50.0, 50.0, pump_weights)
                prior_scores[r.ticker].append(score)
                scores = prior_scores[r.ticker]
                delta = scores[-1] - scores[-2] if len(scores) >= 2 else 0.0
                delta_histories[r.ticker].append(delta)
                dh = delta_histories[r.ticker]
                d5 = sum(dh[-5:]) / len(dh[-5:]) if dh else 0.0

                pumps[r.ticker] = PumpScoreReading(
                    ticker=r.ticker, name=r.name,
                    rs_pillar=r.rs_composite, participation_pillar=50.0, flow_pillar=50.0,
                    pump_score=score, pump_delta=delta, pump_delta_5d_avg=d5,
                )

            # State Classifier
            rs_ranks = {r.ticker: r.rs_rank for r in rs_readings}
            pump_pcts = percentile_rank(pd.Series({t: p.pump_score for t, p in pumps.items()}))
            states = classify_all_sectors(
                pumps=pumps, priors=prior_states, regime=regime.state,
                rs_ranks=rs_ranks, pump_percentiles=pump_pcts.to_dict(),
                delta_histories={t: delta_histories[t][-10:] for t in SECTOR_TICKERS},
                settings=settings["state"],
            )

            # Save snapshot
            snapshot = DailySnapshot(
                date=date_str,
                timestamp=datetime.now(timezone.utc).isoformat(),
                regime=regime,
                sectors=rs_readings,
                breadth=breadth_reading,
                pump_scores=list(pumps.values()),
                states=list(states.values()),
            )
            save_snapshot(snapshot)
            saved += 1

            # Update priors
            prior_states = states
            prior_ranks = rs_ranks

        except Exception as e:
            gaps += 1
            if gaps <= 5:
                print(f"  Warning: {date_str} failed: {e}")

        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"  {i+1}/{total} days processed, {saved} snapshots saved")

    return saved, gaps


def main():
    parser = argparse.ArgumentParser(description="Backfill historical snapshots")
    parser.add_argument("--years", type=int, default=3, help="Years of history (default: 3)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)

    start_time = time.time()

    prices, volumes, vix, vix3m = fetch_full_history(years=args.years, start_date=args.start)
    saved, gaps = compute_snapshots(prices, volumes, vix, vix3m, settings)

    elapsed = time.time() - start_time
    available = list_snapshots()

    print(f"\n{'═' * 50}")
    print(f"BACKFILL COMPLETE")
    print(f"  Snapshots saved: {saved}")
    print(f"  Gaps/errors:     {gaps}")
    print(f"  Date range:      {available[0] if available else 'N/A'} → {available[-1] if available else 'N/A'}")
    print(f"  Total available:  {len(available)}")
    print(f"  Time elapsed:    {elapsed:.1f}s")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
