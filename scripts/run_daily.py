"""
One-click daily run:
1. Load config
2. Check cache; fetch if stale
3. Run engines: regime → RS → breadth → pump scores → states
4. Save snapshot
5. Print summary
6. Launch Streamlit dashboard

Usage:
    python scripts/run_daily.py              # Normal run
    python scripts/run_daily.py --refresh    # Force data refresh
    python scripts/run_daily.py --no-dash    # Compute only, no dashboard
    python scripts/run_daily.py --test       # Run pytest first, abort if failures
"""
import argparse
import subprocess
import sys
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_all
from data.cache import get_connection, log_fetch, is_cache_stale
from data.snapshots import save_snapshot
from engine.schemas import DailySnapshot, PumpScoreReading
from engine.regime_gate import classify_regime_from_data
from engine.rs_scanner import compute_rs_readings
from engine.breadth import compute_breadth
from engine.normalizer import compute_zscore, percentile_rank
from engine.pump_score import compute_pump_score
from engine.state_classifier import classify_all_sectors

SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


def run_tests():
    """Run unit tests. Return True if all pass."""
    print("Running unit tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
        capture_output=True, text=True,
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print("TESTS FAILED — aborting daily run.")
        print(result.stderr[-500:] if result.stderr else "")
        return False
    print("All tests passed.\n")
    return True


def run_pipeline(settings, universe, force_refresh=False):
    """Run the full compute pipeline."""
    config = {**settings, **universe}

    # Check cache
    conn = get_connection()
    expiry = settings.get("data", {}).get("cache_expiry_hours", 18)
    if not force_refresh and not is_cache_stale(conn, expiry):
        print("Cache is fresh — skipping fetch (use --refresh to override)")
    else:
        print("Fetching market data...")

    start = time.time()
    data = fetch_all(config, force_refresh=force_refresh)
    fetch_time = time.time() - start
    print(f"  Fetch: {fetch_time:.1f}s, {data['metadata']['rows']} rows, {len(data['metadata']['tickers'])} tickers")

    if data["metadata"]["errors"]:
        for e in data["metadata"]["errors"]:
            print(f"  ERROR: {e}")

    log_fetch(conn, "daily", data["metadata"])
    prices = data["prices"]

    # Regime
    vix_val = data["vix"].iloc[-1] if len(data["vix"]) > 0 else 20.0
    vix3m_val = data["vix3m"].iloc[-1] if len(data["vix3m"]) > 0 else 20.0
    breadth_reading = compute_breadth(prices)
    bz = breadth_reading.rsp_spy_ratio_zscore
    if np.isnan(bz):
        bz = 0.0

    credit = data["credit"]
    if "hyg_close" in credit.columns and "lqd_close" in credit.columns:
        cr = credit["hyg_close"] / credit["lqd_close"]
        cr_clean = cr.dropna()
        credit_z = compute_zscore(cr_clean.iloc[-1], cr_clean) if len(cr_clean) > 2 else 0.0
    else:
        credit_z = 0.0

    fred_hy_oas = data.get("fred_hy_oas")
    fred_oas_bps = None
    if fred_hy_oas is not None and not fred_hy_oas.empty:
        fred_oas_bps = fred_hy_oas["hy_oas"].dropna().iloc[-1] * 100  # % → bps

    regime = classify_regime_from_data(vix_val, vix3m_val, bz, credit_z, settings["regime"],
                                       fred_hy_oas_value=fred_oas_bps)

    # RS
    rs_cfg = settings["rs"]
    rs_readings = compute_rs_readings(
        prices, SECTOR_NAMES,
        windows=rs_cfg["windows"],
        slope_window=rs_cfg["slope_window"],
        composite_weights=rs_cfg["composite_weights"],
    )

    # Compute pump scores across recent history for real deltas
    pump_weights = settings["pump_score"]
    lookback = 20
    score_history = {t: [] for t in SECTOR_NAMES}

    for i in range(max(0, len(prices) - lookback), len(prices)):
        day_prices = prices.iloc[:i+1]
        if len(day_prices) < 20:
            continue
        day_rs = compute_rs_readings(
            day_prices, SECTOR_NAMES,
            windows=rs_cfg["windows"],
            slope_window=rs_cfg["slope_window"],
            composite_weights=rs_cfg["composite_weights"],
        )
        for r in day_rs:
            sc = compute_pump_score(r.rs_composite, 50.0, 50.0, pump_weights)
            score_history[r.ticker].append(sc)

    pumps = {}
    delta_histories = {}
    for r in rs_readings:
        hist = score_history[r.ticker]
        current_score = hist[-1] if hist else compute_pump_score(r.rs_composite, 50.0, 50.0, pump_weights)

        deltas = []
        for j in range(1, len(hist)):
            deltas.append(hist[j] - hist[j-1])
        delta_histories[r.ticker] = deltas

        delta = deltas[-1] if deltas else 0.0
        d5_window = deltas[-5:] if deltas else [0.0]
        delta_5d = sum(d5_window) / len(d5_window)

        pumps[r.ticker] = PumpScoreReading(
            ticker=r.ticker, name=r.name,
            rs_pillar=r.rs_composite, participation_pillar=50.0, flow_pillar=50.0,
            pump_score=current_score, pump_delta=delta, pump_delta_5d_avg=delta_5d,
        )

    # Load prior states from most recent snapshot
    from data.snapshots import list_snapshots as _ls, load_snapshot as _load
    prior_states = {}
    _available = _ls()
    if _available:
        try:
            _last = _load(_available[-1])
            prior_states = {s.ticker: s for s in _last.states}
        except Exception:
            pass

    # State Classifier
    rs_ranks = {r.ticker: r.rs_rank for r in rs_readings}
    pump_pcts = percentile_rank(pd.Series({t: p.pump_score for t, p in pumps.items()}))
    states = classify_all_sectors(
        pumps=pumps, priors=prior_states, regime=regime.state,
        rs_ranks=rs_ranks, pump_percentiles=pump_pcts.to_dict(),
        delta_histories=delta_histories,
        settings=settings["state"],
    )

    # Save snapshot
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot = DailySnapshot(
        date=now_str,
        timestamp=datetime.now(timezone.utc).isoformat(),
        regime=regime,
        sectors=rs_readings,
        breadth=breadth_reading,
        pump_scores=list(pumps.values()),
        states=list(states.values()),
    )
    save_snapshot(snapshot)

    return regime, rs_readings, breadth_reading, pumps, states


def print_summary(regime, rs_readings, breadth, pumps, states):
    """Print console summary."""
    print(f"\n{'═' * 60}")
    print(f"  PUMP ROTATION SYSTEM — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═' * 60}")

    # Regime
    print(f"\n  REGIME: {regime.state.value}")
    for s in regime.signals:
        print(f"    {s.name:20s} = {s.raw_value:8.3f}  [{s.level.value}]")

    # Breadth
    print(f"\n  BREADTH: {breadth.signal.value} (z: {breadth.rsp_spy_ratio_zscore:.2f})" if not np.isnan(breadth.rsp_spy_ratio_zscore) else f"\n  BREADTH: {breadth.signal.value}")

    # Rankings
    print(f"\n  {'Rank':>4} {'Ticker':<6} {'Sector':<16} {'RS 20d':>8} {'Comp':>5} {'Pump':>5} {'Delta':>7} {'State':<14} {'Conf':>4}")
    print(f"  {'-'*78}")
    for r in sorted(rs_readings, key=lambda x: x.rs_rank):
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        print(f"  {r.rs_rank:4d} {r.ticker:<6} {r.name:<16} {r.rs_20d:+.3%} {r.rs_composite:5.1f} "
              f"{pump.pump_score:.2f}  {pump.pump_delta:+.3f} {state.state.value:<14} {state.confidence:3d}%")

    print(f"\n{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Daily pipeline run")
    parser.add_argument("--refresh", action="store_true", help="Force data refresh")
    parser.add_argument("--no-dash", action="store_true", help="Skip dashboard launch")
    parser.add_argument("--test", action="store_true", help="Run tests first, abort on failure")
    args = parser.parse_args()

    if args.test:
        if not run_tests():
            sys.exit(1)

    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)

    regime, rs_readings, breadth, pumps, states = run_pipeline(settings, universe, args.refresh)
    print_summary(regime, rs_readings, breadth, pumps, states)

    if not args.no_dash:
        print("Launching dashboard...")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
                          "--server.headless", "true"])
        print("Dashboard running at http://localhost:8501")


if __name__ == "__main__":
    main()
