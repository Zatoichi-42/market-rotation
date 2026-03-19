"""
Streamlit dashboard — main entry point.
6 tabs: Regime, Sector Table, Industries, Breadth, Replay, Debug.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from data.fetcher import fetch_all
from data.snapshots import list_snapshots, load_snapshot
from engine.regime_gate import classify_regime_from_data
from engine.rs_scanner import compute_rs_readings, compute_rs_all
from engine.breadth import compute_breadth
from engine.normalizer import compute_zscore, percentile_rank
from engine.industry_rs import compute_industry_rs
from engine.reversal_score import compute_reversal_scores_batch
from engine.pump_score import compute_pump_score
from engine.state_classifier import classify_all_sectors
from engine.schemas import (
    DailySnapshot, PumpScoreReading, RegimeState,
)

st.set_page_config(page_title="Pump Rotation System", layout="wide", page_icon="📊")

SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


@st.cache_data(ttl=3600)
def load_config():
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)
    return settings, universe


@st.cache_data(ttl=1800)
def run_pipeline():
    """Run the full pipeline and return current snapshot + raw data."""
    settings, universe = load_config()
    config = {**settings, **universe}
    data = fetch_all(config)
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

    # Extract latest FRED HY OAS if available (FRED reports %, convert to bps)
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

    # RS history for sparklines + pump score history
    rs_20d_history = compute_rs_all(prices, list(SECTOR_NAMES.keys()), window=20)

    # Compute pump scores across recent history to get real deltas
    pump_weights = settings["pump_score"]
    lookback = 20  # sessions of pump history for deltas
    score_history = {t: [] for t in SECTOR_NAMES}
    sector_tickers = list(SECTOR_NAMES.keys())

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

    # Build current pump scores with real deltas
    pumps = {}
    delta_histories = {}
    for r in rs_readings:
        hist = score_history[r.ticker]
        current_score = hist[-1] if hist else compute_pump_score(r.rs_composite, 50.0, 50.0, pump_weights)

        # Compute delta sequence
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

    # Load prior states from most recent snapshot if available
    from data.snapshots import list_snapshots, load_snapshot
    prior_states = {}
    available_snaps = list_snapshots()
    if available_snaps:
        try:
            last_snap = load_snapshot(available_snaps[-1])
            prior_states = {s.ticker: s for s in last_snap.states}
        except Exception:
            pass

    # Industry RS (Phase 2)
    industries_cfg = universe.get("industries", [])
    available_industries = [i for i in industries_cfg if i["ticker"] in prices.columns]
    industry_rs_readings = compute_industry_rs(prices, available_industries) if available_industries else []

    # Reversal Scores (Phase 2) — with rolling history for real percentiles
    from engine.reversal_score import compute_reversal_score
    rev_settings = settings.get("reversal", {})
    rev_weights = settings.get("reversal_score", {"breadth_det_weight": 0.40, "price_break_weight": 0.30, "crowding_weight": 0.30})
    all_tickers_for_rev = [r.ticker for r in rs_readings]

    # Build score history: compute reversal score at 20 historical points
    rev_history = {t: [] for t in all_tickers_for_rev}
    hist_step = max(1, (len(prices) - 60) // 20)
    for i in range(60, len(prices), hist_step):
        p_slice = prices.iloc[:i+1]
        h_slice = data["highs"].iloc[:i+1]
        l_slice = data["lows"].iloc[:i+1]
        v_slice = data["volumes"].iloc[:i+1]
        for t in all_tickers_for_rev:
            if t in p_slice.columns:
                r = compute_reversal_score(p_slice, h_slice, l_slice, v_slice, t,
                                           settings=rev_settings, weights=rev_weights)
                rev_history[t].append(r.reversal_score)

    history_series = {t: pd.Series(scores) for t, scores in rev_history.items() if scores}

    reversal_readings = compute_reversal_scores_batch(
        prices, data["highs"], data["lows"], data["volumes"],
        all_tickers_for_rev, settings=rev_settings, weights=rev_weights,
        history_scores=history_series,
    )
    reversal_map = {r.ticker: r for r in reversal_readings}

    # Add industry pump scores (use industry_composite as RS pillar)
    for ir in industry_rs_readings:
        if ir.ticker not in pumps:
            ind_score = compute_pump_score(ir.industry_composite, 50.0, 50.0, pump_weights)
            pumps[ir.ticker] = PumpScoreReading(
                ticker=ir.ticker, name=ir.name,
                rs_pillar=ir.industry_composite, participation_pillar=50.0, flow_pillar=50.0,
                pump_score=ind_score, pump_delta=0.0, pump_delta_5d_avg=0.0,
            )
            delta_histories[ir.ticker] = [0.0]

    # State Classifier (sectors + industries, with reversal scores)
    all_ranks = {r.ticker: r.rs_rank for r in rs_readings}
    for ir in industry_rs_readings:
        all_ranks[ir.ticker] = ir.rs_rank
    pump_pcts = percentile_rank(pd.Series({t: p.pump_score for t, p in pumps.items()}))
    states = classify_all_sectors(
        pumps=pumps, priors=prior_states, regime=regime.state,
        rs_ranks=all_ranks, pump_percentiles=pump_pcts.to_dict(),
        delta_histories=delta_histories,
        settings=settings["state"],
        reversal_scores=reversal_map,
    )

    return {
        "regime": regime,
        "rs_readings": rs_readings,
        "breadth": breadth_reading,
        "pumps": pumps,
        "states": states,
        "prices": prices,
        "vix": data["vix"],
        "vix3m": data["vix3m"],
        "rs_history": rs_20d_history,
        "vix_val": vix_val,
        "vix3m_val": vix3m_val,
        "credit_z": credit_z,
        "credit": credit,
        "fred_hy_oas": data.get("fred_hy_oas"),
        "industry_rs": industry_rs_readings,
        "reversal_scores": reversal_readings,
        "reversal_map": reversal_map,
    }


def main():
    st.title("Pump Rotation System")
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    result = run_pipeline()

    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Regime Gate", "Sector Rankings", "Industries", "Breadth", "Replay", "Debug"
    ])

    with tab1:
        from dashboard.components.regime_panel import render_regime_panel
        render_regime_panel(result)
        # Baton pass alerts + reversal diagnostics on page 1
        from dashboard.components.baton_pass_alert import render_baton_pass_alerts
        render_baton_pass_alerts(result)
        from dashboard.components.reversal_diagnostics import render_reversal_diagnostics
        render_reversal_diagnostics(result)

    with tab2:
        from dashboard.components.sector_table import render_sector_table
        render_sector_table(result)

    with tab3:
        from dashboard.components.industry_panel import render_industry_panel
        render_industry_panel(result)

    with tab4:
        from dashboard.components.breadth_chart import render_breadth_chart
        render_breadth_chart(result)

    with tab5:
        from dashboard.components.replay_panel import render_replay_panel
        render_replay_panel(result)

    with tab6:
        from dashboard.components.debug_panel import render_debug_panel
        render_debug_panel(result)


if __name__ == "__main__":
    main()
