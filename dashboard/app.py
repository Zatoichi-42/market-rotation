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
from engine.industry_state import classify_all_industries
from engine.reversal_score import compute_reversal_scores_batch
from engine.pump_score import compute_pump_score
from engine.participation import compute_participation_pillar
from engine.flow_quality import compute_flow_pillar
from engine.state_classifier import classify_all_sectors
from engine.catalyst_gate import load_catalyst_calendar, assess_catalyst
from engine.concentration_monitor import compute_concentration_all
from engine.schemas import (
    DailySnapshot, PumpScoreReading, RegimeState, RegimeCharacter,
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


@st.cache_data(ttl=300)  # 5 min cache — ensures fresh intraday data
def run_pipeline():
    """Run the full pipeline and return current snapshot + raw data.
    Uses latest available prices (intraday during market hours)."""
    settings, universe = load_config()
    config = {**settings, **universe}
    data = fetch_all(config, force_refresh=True)  # Always get latest prices
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

    # Oil regime signal
    oil_ticker = universe.get("commodities", {}).get("oil_wti", "CL=F")
    oil_z = float("nan")
    if oil_ticker in prices.columns:
        oil_series = prices[oil_ticker].dropna()
        if len(oil_series) > 60:
            oil_z = compute_zscore(oil_series.iloc[-1], oil_series)

    # Cross-sector correlation (signal #6)
    from engine.correlation import compute_cross_sector_correlation
    corr_settings = settings.get("correlation", {})
    corr_reading = compute_cross_sector_correlation(
        prices,
        window=corr_settings.get("window", 21),
        zscore_window=corr_settings.get("zscore_window", 504),
        fragile_zscore=corr_settings.get("fragile_zscore", 0.5),
        hostile_zscore=corr_settings.get("hostile_zscore", 1.5),
        absolute_hostile=corr_settings.get("absolute_hostile", 0.80),
    )
    corr_z = corr_reading.avg_corr_zscore if corr_reading else float("nan")

    # Gold/VIX divergence modifier
    from engine.gold_divergence import compute_gold_vix_divergence
    modifiers = universe.get("market_modifiers", {})
    gld_ticker = modifiers.get("gold", "GLD")
    slv_ticker = modifiers.get("silver", "SLV")
    gd_settings = settings.get("gold_divergence", {})
    gd_reading = None
    if gld_ticker in prices.columns and "SPY" in prices.columns:
        gd_reading = compute_gold_vix_divergence(
            prices[gld_ticker], prices["SPY"], vix_val,
            gold_decline_threshold=gd_settings.get("gold_decline_threshold", -0.02),
            spy_decline_threshold=gd_settings.get("spy_decline_threshold", -0.02),
            vix_threshold=gd_settings.get("vix_threshold", 25),
        )

    # Gold/Silver ratio modifier
    from engine.gold_silver_ratio import compute_gold_silver_ratio
    gs_reading = None
    if gld_ticker in prices.columns and slv_ticker in prices.columns:
        gs_window = settings.get("gold_silver_ratio", {}).get("zscore_window", 504)
        gs_reading = compute_gold_silver_ratio(
            prices[gld_ticker], prices[slv_ticker],
            window=gs_window,
            gold_vix_divergence_active=(gd_reading.is_margin_call_regime if gd_reading else False),
        )

    regime = classify_regime_from_data(vix_val, vix3m_val, bz, credit_z, settings["regime"],
                                       fred_hy_oas_value=fred_oas_bps, oil_zscore=oil_z,
                                       correlation_zscore=corr_z,
                                       gold_silver_reading=gs_reading,
                                       gold_divergence_reading=gd_reading)

    # Regime Character (Phase 4)
    from engine.regime_character import classify_regime_character
    from engine.correlation import compute_cross_sector_dispersion
    # SPY 20d return
    spy_20d_return = 0.0
    if "SPY" in prices.columns and len(prices) >= 21:
        spy_vals = prices["SPY"].dropna()
        if len(spy_vals) >= 21:
            spy_20d_return = float(spy_vals.iloc[-1] / spy_vals.iloc[-21] - 1)

    # VIX 20d change
    vix_20d_change = 0.0
    if len(data["vix"]) >= 21:
        vix_20d_change = float(data["vix"].iloc[-1] - data["vix"].iloc[-21])

    # Breadth 5d change (z-score delta)
    breadth_zscore_change_5d = 0.0  # Approximation — use RSP/SPY change
    if "RSP" in prices.columns and "SPY" in prices.columns and len(prices) >= 6:
        rsp = prices["RSP"].dropna()
        spy_s = prices["SPY"].dropna()
        if len(rsp) >= 6 and len(spy_s) >= 6:
            ratio_now = rsp.iloc[-1] / spy_s.iloc[-1]
            ratio_5d = rsp.iloc[-6] / spy_s.iloc[-6]
            breadth_zscore_change_5d = float(ratio_now - ratio_5d) * 10  # scale

    regime_char_reading = classify_regime_character(
        spy_20d_return=spy_20d_return,
        vix_level=vix_val,
        vix_20d_change=vix_20d_change,
        breadth_zscore=bz,
        breadth_zscore_change_5d=breadth_zscore_change_5d,
        cross_sector_dispersion=0.0,  # Updated after RS computed
        correlation_zscore=corr_z if not np.isnan(corr_z) else 0.0,
        credit_zscore=credit_z,
        gold_divergence_active=(gd_reading.is_margin_call_regime if gd_reading else False),
        gate_level=regime.state,
    )

    # Catalyst Gate (between regime and state classifier)
    catalysts = load_catalyst_calendar()
    today_str = prices.index[-1].strftime("%Y-%m-%d")
    catalyst_assessment = assess_catalyst(
        today_str, prices, catalysts, data["vix"],
        catalyst_settings=settings.get("catalyst"),
        shock_settings=settings.get("catalyst"),
    )

    # Concentration Monitor
    sector_leaders = universe.get("sector_leaders", {})
    concentrations = compute_concentration_all(
        prices, sector_leaders, ew_cw_zscore=bz,
        settings=settings.get("concentration"),
    )
    concentration_map = {c.sector_ticker: c for c in concentrations}

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

    def _get_sector_children(sector_ticker, univ):
        return [ind["ticker"] for ind in univ.get("industries", [])
                if ind.get("parent_sector") == sector_ticker]

    for i in range(max(0, len(prices) - lookback), len(prices)):
        day_prices = prices.iloc[:i+1]
        day_highs = data["highs"].iloc[:i+1]
        day_lows = data["lows"].iloc[:i+1]
        day_volumes = data["volumes"].iloc[:i+1]
        if len(day_prices) < 20:
            continue
        day_rs = compute_rs_readings(
            day_prices, SECTOR_NAMES,
            windows=rs_cfg["windows"],
            slope_window=rs_cfg["slope_window"],
            composite_weights=rs_cfg["composite_weights"],
        )
        for r in day_rs:
            children = _get_sector_children(r.ticker, universe)
            participation = compute_participation_pillar(day_prices, r.ticker, children)
            flow = compute_flow_pillar(day_prices, day_highs, day_lows, day_volumes, r.ticker)
            sc = compute_pump_score(r.rs_composite, participation, flow, pump_weights)
            score_history[r.ticker].append(sc)

    # Build current pump scores with real deltas
    pumps = {}
    delta_histories = {}
    for r in rs_readings:
        hist = score_history[r.ticker]
        children = _get_sector_children(r.ticker, universe)
        part = compute_participation_pillar(prices, r.ticker, children)
        flow = compute_flow_pillar(prices, data["highs"], data["lows"], data["volumes"], r.ticker)
        current_score = hist[-1] if hist else compute_pump_score(r.rs_composite, part, flow, pump_weights)

        deltas = []
        for j in range(1, len(hist)):
            deltas.append(hist[j] - hist[j-1])
        delta_histories[r.ticker] = deltas

        delta = deltas[-1] if deltas else 0.0
        d5_window = deltas[-5:] if deltas else [0.0]
        delta_5d = sum(d5_window) / len(d5_window)

        pumps[r.ticker] = PumpScoreReading(
            ticker=r.ticker, name=r.name,
            rs_pillar=r.rs_composite, participation_pillar=part, flow_pillar=flow,
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

    # Horizon Patterns (Phase 4) — after all RS computed
    from engine.horizon_patterns import classify_all_horizon_patterns
    horizon_readings = classify_all_horizon_patterns(
        rs_readings, industry_rs_readings,
        near_zero_threshold=settings.get("horizon", {}).get("near_zero_threshold", 0.003),
    )
    # Extract just the pattern enums for the state classifier
    horizon_pattern_map = {t: hr.pattern for t, hr in horizon_readings.items()}

    # Update cross-sector dispersion now that RS is available
    sector_20d_returns = {r.ticker: r.rs_20d for r in rs_readings}
    dispersion = compute_cross_sector_dispersion(sector_20d_returns)
    # Re-classify regime character with actual dispersion
    regime_char_reading = classify_regime_character(
        spy_20d_return=spy_20d_return,
        vix_level=vix_val,
        vix_20d_change=vix_20d_change,
        breadth_zscore=bz,
        breadth_zscore_change_5d=breadth_zscore_change_5d,
        cross_sector_dispersion=dispersion,
        correlation_zscore=corr_z if not np.isnan(corr_z) else 0.0,
        credit_zscore=credit_z,
        gold_divergence_active=(gd_reading.is_margin_call_regime if gd_reading else False),
        gate_level=regime.state,
    )

    # Compute industry pump score history for real deltas (same pattern as sectors)
    ind_score_history = {ir.ticker: [] for ir in industry_rs_readings}
    for i in range(max(0, len(prices) - lookback), len(prices)):
        day_prices = prices.iloc[:i+1]
        if len(day_prices) < 20:
            continue
        day_ind_rs = compute_industry_rs(day_prices, available_industries) if available_industries else []
        for ir in day_ind_rs:
            sc = compute_pump_score(ir.industry_composite, 50.0, 50.0, pump_weights)
            ind_score_history[ir.ticker].append(sc)

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

    # Add industry pump scores with real deltas from history
    for ir in industry_rs_readings:
        if ir.ticker not in pumps:
            hist = ind_score_history.get(ir.ticker, [])
            current_score = hist[-1] if hist else compute_pump_score(ir.industry_composite, 50.0, 50.0, pump_weights)
            deltas = [hist[j] - hist[j-1] for j in range(1, len(hist))] if len(hist) >= 2 else []
            delta = deltas[-1] if deltas else 0.0
            d5 = sum(deltas[-5:]) / len(deltas[-5:]) if deltas else 0.0
            delta_histories[ir.ticker] = deltas if deltas else [0.0]
            pumps[ir.ticker] = PumpScoreReading(
                ticker=ir.ticker, name=ir.name,
                rs_pillar=ir.industry_composite, participation_pillar=50.0, flow_pillar=50.0,
                pump_score=current_score, pump_delta=delta, pump_delta_5d_avg=d5,
            )

    # State Classifier (sectors + industries, with reversal scores)
    all_ranks = {r.ticker: r.rs_rank for r in rs_readings}
    for ir in industry_rs_readings:
        all_ranks[ir.ticker] = ir.rs_rank
    pump_pcts = percentile_rank(pd.Series({t: p.pump_score for t, p in pumps.items()}))

    # Build RS values dict for state classifier veto checks
    rs_vals = {}
    for r in rs_readings:
        rs_vals[r.ticker] = (r.rs_5d, r.rs_20d, r.rs_60d)
    for ir in industry_rs_readings:
        rs_vals[ir.ticker] = (ir.rs_5d, ir.rs_20d, ir.rs_60d)

    states = classify_all_sectors(
        pumps=pumps, priors=prior_states, regime=regime.state,
        rs_ranks=all_ranks, pump_percentiles=pump_pcts.to_dict(),
        delta_histories=delta_histories,
        settings=settings["state"],
        reversal_scores=reversal_map,
        concentrations=concentration_map,
        catalyst_confidence_modifier=catalyst_assessment.confidence_modifier,
        rs_values=rs_vals,
        horizon_patterns=horizon_pattern_map,
    )

    # Industry states from multi-timeframe RS pattern
    industry_states = classify_all_industries(
        industry_rs_readings, regime=regime.state, reversal_scores=reversal_map,
        horizon_patterns=horizon_pattern_map,
    )
    # Merge industry states into the main states dict
    states.update(industry_states)

    # Trade State Mapper (Layer 4)
    from engine.trade_state_mapper import map_all_trade_states
    all_ranks_for_trade = {r.ticker: r.rs_rank for r in rs_readings}
    for ir in industry_rs_readings:
        all_ranks_for_trade[ir.ticker] = ir.rs_rank
    trade_states = map_all_trade_states(
        states=states, pumps=pumps, regime=regime.state,
        catalyst=catalyst_assessment, rs_ranks=all_ranks_for_trade,
        reversal_scores=reversal_map, concentrations=concentration_map,
        regime_character=regime_char_reading.character,
    )

    return {
        "regime": regime,
        "rs_readings": rs_readings,
        "breadth": breadth_reading,
        "pumps": pumps,
        "states": states,
        "trade_states": trade_states,
        "prices": prices,
        "vix": data["vix"],
        "vix3m": data["vix3m"],
        "rs_history": rs_20d_history,
        "vix_val": vix_val,
        "vix3m_val": vix3m_val,
        "credit_z": credit_z,
        "credit": credit,
        "fred_hy_oas": data.get("fred_hy_oas"),
        "catalyst": catalyst_assessment,
        "concentrations": concentration_map,
        "industry_rs": industry_rs_readings,
        "industry_states": industry_states,
        "reversal_scores": reversal_readings,
        "reversal_map": reversal_map,
        "gold_silver_reading": gs_reading,
        "gold_divergence_reading": gd_reading,
        "correlation_reading": corr_reading,
        "horizon_readings": horizon_readings,
        "regime_character": regime_char_reading,
    }


def main():
    st.title("Pump Rotation System")

    result = run_pipeline()

    # Export button in sidebar
    from dashboard.components.export import render_export_button
    render_export_button(result)

    prices_last = result["prices"].index[-1].strftime("%Y-%m-%d")
    st.caption(
        f"Pipeline: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Price data through: **{prices_last}** (today's latest — refreshes every 5 min)"
    )

    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Regime Gate", "Sector Rankings", "Industries", "Breadth",
        "Today", "Signal Reliability", "Replay", "Debug",
    ])

    with tab1:
        from dashboard.components.regime_panel import render_regime_panel
        render_regime_panel(result)
        # Baton pass alerts + reversal diagnostics on page 1
        from dashboard.components.baton_pass_alert import render_baton_pass_alerts
        render_baton_pass_alerts(result)
        _render_industry_divergence_alerts(result)
        from dashboard.components.reversal_diagnostics import render_reversal_diagnostics
        render_reversal_diagnostics(result)

    with tab2:
        from dashboard.components.sector_table import render_sector_table
        render_sector_table(result)
        from dashboard.components.performance_spectrum import render_sector_spectrum
        render_sector_spectrum(result)

    with tab3:
        from dashboard.components.industry_panel import render_industry_panel
        render_industry_panel(result)
        from dashboard.components.performance_spectrum import render_industry_spectrum
        render_industry_spectrum(result)

    with tab4:
        from dashboard.components.breadth_chart import render_breadth_chart
        render_breadth_chart(result)

    with tab5:
        from dashboard.components.interpretation_panel import render_interpretation_panel
        render_interpretation_panel(result)

    with tab6:
        from dashboard.components.signal_reliability import render_signal_reliability
        render_signal_reliability(result)

    with tab7:
        from dashboard.components.replay_panel import render_replay_panel
        render_replay_panel(result)

    with tab8:
        from dashboard.components.debug_panel import render_debug_panel
        render_debug_panel(result)


def _render_industry_divergence_alerts(result):
    """Surface cases where industry state contradicts parent sector state."""
    from engine.schemas import AnalysisState as AS
    industry_rs = result.get("industry_rs", [])
    states = result.get("states", {})
    bullish = {AS.OVERT_PUMP, AS.ACCUMULATION}
    bearish = {AS.EXHAUSTION, AS.OVERT_DUMP}

    alerts = []
    for ir in industry_rs:
        ind_st = states.get(ir.ticker)
        par_st = states.get(ir.parent_sector)
        if not ind_st or not par_st:
            continue
        if ind_st.state in bullish and par_st.state in bearish:
            alerts.append(("LEADING", ir, ind_st, par_st))
        elif ind_st.state in bearish and par_st.state in bullish:
            alerts.append(("LAGGING", ir, ind_st, par_st))

    if alerts:
        st.subheader("Industry Divergence Alerts")
        for typ, ir, ind_st, par_st in alerts[:5]:
            if typ == "LEADING":
                st.warning(
                    f"**{ir.ticker} ({ir.name})** is {ind_st.state.value} "
                    f"while parent **{ir.parent_sector}** is {par_st.state.value}. "
                    f"RS vs parent: {ir.rs_20d_vs_parent:+.2%}. "
                    f"Industry may be the better expression."
                )
            else:
                st.error(
                    f"**{ir.ticker} ({ir.name})** is {ind_st.state.value} "
                    f"while parent **{ir.parent_sector}** is {par_st.state.value}. "
                    f"Industry-specific weakness."
                )


if __name__ == "__main__":
    main()
