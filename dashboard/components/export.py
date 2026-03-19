"""
Export — generates exhaustive system report in multiple formats.

Includes EVERY feature: regime, signals, sectors, industries, breadth, credit,
reversal diagnostics, heatmaps (text-described), sparkline trends (text-described),
baton pass alerts, 1d/5d/20d/60d metrics, signal reliability, valuations.
"""
import io
import json
import csv
import math
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np

from engine.schemas import AnalysisState


_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


def render_export_button(result: dict):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Report")
    fmt = st.sidebar.selectbox("Format", ["Claude XML", "Markdown", "CSV", "JSON"], key="export_fmt")

    if st.sidebar.button("Generate Export", key="export_btn"):
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        if fmt == "Claude XML":
            content = _build_claude_xml(result)
            st.sidebar.download_button("Download XML", content, f"pump_report_{now}.xml",
                                        mime="application/xml", key="dl_xml")
        elif fmt == "Markdown":
            content = _build_markdown(result)
            st.sidebar.download_button("Download MD", content, f"pump_report_{now}.md",
                                        mime="text/markdown", key="dl_md")
        elif fmt == "CSV":
            content = _build_csv(result)
            st.sidebar.download_button("Download CSV", content, f"pump_report_{now}.csv",
                                        mime="text/csv", key="dl_csv")
        elif fmt == "JSON":
            content = _build_json(result)
            st.sidebar.download_button("Download JSON", content, f"pump_report_{now}.json",
                                        mime="application/json", key="dl_json")


# ═══════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════

def _rs_color_word(val):
    """Describe RS value as a color word for heatmap text."""
    pct = val * 100
    if pct > 3:
        return "BRIGHT GREEN"
    elif pct > 1:
        return "GREEN"
    elif pct > 0.2:
        return "LIGHT GREEN"
    elif pct > -0.2:
        return "NEUTRAL"
    elif pct > -1:
        return "LIGHT RED"
    elif pct > -3:
        return "RED"
    else:
        return "BRIGHT RED"


def _trend_description(prices, ticker, benchmark="SPY"):
    """Describe the 20d RS trend in words."""
    if ticker not in prices.columns or benchmark not in prices.columns:
        return "no data"
    rs = prices[ticker].pct_change(20) - prices[benchmark].pct_change(20)
    rs_clean = rs.dropna().tail(60)
    if len(rs_clean) < 10:
        return "insufficient history"
    start, mid, end = rs_clean.iloc[0], rs_clean.iloc[len(rs_clean)//2], rs_clean.iloc[-1]
    if end > mid > start:
        return "steadily rising"
    elif end > start and end > mid:
        return "accelerating upward"
    elif end < mid < start:
        return "steadily declining"
    elif end < start and end < mid:
        return "accelerating downward"
    elif abs(end - start) < 0.005:
        return "flat/choppy"
    elif end > start:
        return "net positive with volatility"
    else:
        return "net negative with volatility"


def _get_1d_moves(result):
    """Get 1d moves for all sectors."""
    prices = result["prices"]
    spy_1d = prices["SPY"].pct_change().iloc[-1] if "SPY" in prices.columns else 0
    moves = []
    for r in sorted(result.get("rs_readings", []), key=lambda x: x.rs_rank):
        if r.ticker in prices.columns:
            sec_1d = prices[r.ticker].pct_change().iloc[-1]
            moves.append({"ticker": r.ticker, "name": r.name, "return_1d": sec_1d,
                          "rs_1d": sec_1d - spy_1d})
    return spy_1d, moves


def _get_rolling_moves(result):
    """Get 1d/5d/20d/60d RS leaders."""
    prices = result["prices"]
    leaders = {}
    for window, label in [(1, "1d"), (5, "5d"), (20, "20d"), (60, "60d")]:
        if len(prices) <= window or "SPY" not in prices.columns:
            continue
        spy_w = prices["SPY"].pct_change(window).iloc[-1]
        best_t, best_v = None, -999
        for r in result.get("rs_readings", []):
            if r.ticker in prices.columns:
                sec_w = prices[r.ticker].pct_change(window).iloc[-1]
                rs_w = sec_w - spy_w
                if rs_w > best_v:
                    best_t, best_v = r.ticker, rs_w
        if best_t:
            leaders[label] = (best_t, _SECTOR_NAMES.get(best_t, best_t), best_v)
    return leaders


def _get_baton_passes(result):
    """Detect baton pass alerts."""
    pumps = result.get("pumps", {})
    states = result.get("states", {})
    alerts = []
    sector_data = []
    for t in _SECTOR_NAMES:
        p = pumps.get(t)
        s = states.get(t)
        if p:
            sector_data.append({"ticker": t, "name": p.name, "delta": p.pump_delta,
                                "state": s.state.value if s else "—"})
    rising = [d for d in sector_data if d["delta"] > 0.005]
    declining = [d for d in sector_data if d["delta"] < -0.005]
    for r in rising:
        for d in declining:
            diff = r["delta"] - d["delta"]
            if diff >= 0.04:
                alerts.append(f"{r['ticker']} ({r['name']}) overtaking {d['ticker']} ({d['name']}): "
                              f"delta diff {diff:+.3f}")
    return alerts


def _get_signal_reliability(result):
    """Compute signal reliability for 1d/5d/20d/60d lookbacks."""
    from data.snapshots import list_snapshots, load_snapshot
    prices = result["prices"]
    available = list_snapshots()
    _BULLISH = {"Overt Pump", "Accumulation"}
    _BEARISH = {"Exhaustion", "Overt Dump"}

    reliability = {}
    for days, label in [(1, "1d"), (5, "5d"), (20, "20d"), (60, "60d")]:
        if len(available) <= days + 1:
            continue
        snap_date = available[-(days + 1)]
        try:
            snap = load_snapshot(snap_date)
        except Exception:
            continue
        state_map = {s.ticker: s for s in snap.states} if snap.states else {}
        if snap_date not in prices.index.strftime("%Y-%m-%d").values:
            continue
        dloc = prices.index.get_indexer(pd.to_datetime([snap_date]), method="nearest")[0]
        now_loc = len(prices) - 1
        if now_loc <= dloc:
            continue
        spy_fwd = (prices["SPY"].iloc[now_loc] / prices["SPY"].iloc[dloc]) - 1 if "SPY" in prices.columns else 0

        hits, total = 0, 0
        details = []
        for ticker in _SECTOR_NAMES:
            if ticker not in prices.columns:
                continue
            snap_state = state_map.get(ticker)
            if not snap_state:
                continue
            sv = snap_state.state.value
            sec_fwd = (prices[ticker].iloc[now_loc] / prices[ticker].iloc[dloc]) - 1
            fwd_rs = sec_fwd - spy_fwd
            if sv in _BULLISH:
                hit = fwd_rs > 0
            elif sv in _BEARISH:
                hit = fwd_rs < 0
            else:
                hit = abs(fwd_rs) < 0.02
            total += 1
            if hit:
                hits += 1
            details.append({"ticker": ticker, "state_then": sv, "fwd_rs": fwd_rs,
                            "hit": hit, "verdict": "HIT" if hit else "MISS"})

        reliability[label] = {
            "snap_date": snap_date, "hits": hits, "total": total,
            "hit_rate": hits / total if total > 0 else 0,
            "details": details,
        }
    return reliability


# ═══════════════════════════════════════════════════════
# CLAUDE XML
# ═══════════════════════════════════════════════════════

def _build_claude_xml(result: dict) -> str:
    regime = result["regime"]
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    breadth = result.get("breadth")
    industry_rs = sorted(result.get("industry_rs", []), key=lambda x: x.rs_rank)
    reversal_scores = result.get("reversal_scores", [])
    prices = result["prices"]
    last_date = prices.index[-1].strftime("%Y-%m-%d")
    vix = result.get("vix")
    vix3m = result.get("vix3m")
    credit = result.get("credit")

    rev_map = {rv.ticker: rv for rv in reversal_scores}
    spy_1d, moves_1d = _get_1d_moves(result)
    rolling_leaders = _get_rolling_moves(result)
    baton_alerts = _get_baton_passes(result)

    X = _xml_escape
    L = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append(f'<pump_rotation_report date="{last_date}" '
             f'generated="{datetime.now(timezone.utc).isoformat()}" '
             f'live_quotes="true">')

    # ── Regime ──
    L.append(f'  <regime state="{regime.state.value}" hostile="{regime.hostile_count}" fragile="{regime.fragile_count}">')
    for sig in regime.signals:
        L.append(f'    <signal name="{sig.name}" value="{sig.raw_value:.4f}" level="{sig.level.value}" '
                 f'description="{X(sig.description)}"/>')
    L.append(f'    <explanation>{X(regime.explanation)}</explanation>')
    L.append('  </regime>')

    # ── VIX Detail ──
    if vix is not None and len(vix) >= 2:
        vix_1d = vix.iloc[-1] - vix.iloc[-2]
        ratio = vix.iloc[-1] / vix3m.iloc[-1] if vix3m is not None and len(vix3m) > 0 and vix3m.iloc[-1] > 0 else 0
        L.append(f'  <vix current="{vix.iloc[-1]:.2f}" change_1d="{vix_1d:+.2f}" '
                 f'vix3m="{vix3m.iloc[-1]:.2f}" ratio="{ratio:.3f}" '
                 f'description="VIX term structure chart: 60-day trend of VIX vs VIX3M. '
                 f'Threshold lines at 20 (fragile) and 30 (hostile)."/>')

    # ── Breadth ──
    if breadth:
        z = f"{breadth.rsp_spy_ratio_zscore:.4f}" if not math.isnan(breadth.rsp_spy_ratio_zscore) else "NaN"
        L.append(f'  <breadth signal="{breadth.signal.value}" ratio="{breadth.rsp_spy_ratio:.6f}" '
                 f'zscore="{z}" change_20d="{breadth.rsp_spy_ratio_20d_change:.6f}" '
                 f'description="RSP/SPY ratio chart: 60-day trend with 20d MA. '
                 f'Rising = broad participation, falling = narrow leadership."/>')

    # ── Credit ──
    if credit is not None and "hyg_close" in credit.columns and "lqd_close" in credit.columns:
        cr = (credit["hyg_close"] / credit["lqd_close"]).dropna()
        if len(cr) >= 2:
            cr_1d = cr.iloc[-1] - cr.iloc[-2]
            L.append(f'  <credit hyg_lqd_ratio="{cr.iloc[-1]:.6f}" change_1d="{cr_1d:+.6f}" '
                     f'description="HYG/LQD credit ratio chart: falling = flight to quality (risk-off)."/>')

    fred = result.get("fred_hy_oas")
    if fred is not None and not fred.empty:
        oas = fred["hy_oas"].dropna().iloc[-1] * 100
        L.append(f'  <fred_hy_oas bps="{oas:.0f}" '
                 f'note="FRED data delayed 1-2 business days"/>')

    # ── Sector Heatmap (text) ──
    L.append('  <sector_heatmap description="RS Heat Map: rows=sectors sorted by rank, '
             'columns=5d/20d/60d RS. Color: green=outperforming SPY, red=underperforming, '
             'centered on 0. Pattern reading: green-green-green=sustained leader, '
             'red-red-green=former leader decaying (rotation OUT), '
             'green-green-red=new leader emerging (rotation IN).">')
    for r in rs_readings:
        state = states.get(r.ticker)
        sv = state.state.value if state else "—"
        L.append(f'    <row ticker="{r.ticker}" name="{r.name}" '
                 f'rs_5d="{r.rs_5d*100:+.1f}%" color_5d="{_rs_color_word(r.rs_5d)}" '
                 f'rs_20d="{r.rs_20d*100:+.1f}%" color_20d="{_rs_color_word(r.rs_20d)}" '
                 f'rs_60d="{r.rs_60d*100:+.1f}%" color_60d="{_rs_color_word(r.rs_60d)}" '
                 f'state="{sv}"/>')
    L.append('  </sector_heatmap>')

    # ── Sectors ──
    L.append('  <sectors>')
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        rev = rev_map.get(r.ticker)
        trend = _trend_description(prices, r.ticker)
        L.append(f'    <sector ticker="{r.ticker}" name="{r.name}" rank="{r.rs_rank}">')
        L.append(f'      <rs rs_5d="{r.rs_5d:.6f}" rs_20d="{r.rs_20d:.6f}" rs_60d="{r.rs_60d:.6f}" '
                 f'slope="{r.rs_slope:.6f}" composite="{r.rs_composite:.2f}"/>')
        L.append(f'      <sparkline_trend description="20d RS sparkline over 60 trading days: {trend}"/>')
        if pump:
            L.append(f'      <pump score="{pump.pump_score:.4f}" delta="{pump.pump_delta:.4f}" '
                     f'delta_5d_avg="{pump.pump_delta_5d_avg:.4f}"/>')
        if rev:
            L.append(f'      <reversal score="{rev.reversal_score:.4f}" percentile="{rev.reversal_percentile:.1f}" '
                     f'above_75th="{rev.above_75th}" breadth_det="{rev.breadth_det_pillar:.1f}" '
                     f'price_break="{rev.price_break_pillar:.1f}" crowding="{rev.crowding_pillar:.1f}">')
            for k, v in rev.sub_signals.items():
                L.append(f'        <sub_signal name="{k}" value="{v:.4f}"/>')
            L.append(f'      </reversal>')
        if state:
            L.append(f'      <state value="{state.state.value}" confidence="{state.confidence}" '
                     f'sessions="{state.sessions_in_state}" pressure="{state.transition_pressure.value}"/>')
            L.append(f'      <explanation>{X(state.explanation)}</explanation>')
        L.append('    </sector>')
    L.append('  </sectors>')

    # ── Industry Heatmap (text) ──
    if industry_rs:
        L.append('  <industry_heatmap description="Industry RS Heat Map: same as sector heatmap but includes '
                 'vs-Parent column showing RS relative to parent sector ETF.">')
        for ir in industry_rs:
            state = states.get(ir.ticker)
            sv = state.state.value if state else "—"
            parent_state = states.get(ir.parent_sector)
            psv = parent_state.state.value if parent_state else "—"
            L.append(f'    <row ticker="{ir.ticker}" name="{ir.name}" parent="{ir.parent_sector}" '
                     f'rs_5d="{ir.rs_5d*100:+.1f}%" color_5d="{_rs_color_word(ir.rs_5d)}" '
                     f'rs_20d="{ir.rs_20d*100:+.1f}%" color_20d="{_rs_color_word(ir.rs_20d)}" '
                     f'rs_60d="{ir.rs_60d*100:+.1f}%" color_60d="{_rs_color_word(ir.rs_60d)}" '
                     f'vs_parent_20d="{ir.rs_20d_vs_parent*100:+.1f}%" '
                     f'color_vs_parent="{_rs_color_word(ir.rs_20d_vs_parent)}" '
                     f'state="{sv}" parent_state="{psv}"/>')
        L.append('  </industry_heatmap>')

    # ── Industries ──
    if industry_rs:
        L.append('  <industries>')
        for ir in industry_rs:
            state = states.get(ir.ticker)
            parent_state = states.get(ir.parent_sector)
            trend = _trend_description(prices, ir.ticker)
            L.append(f'    <industry ticker="{ir.ticker}" name="{ir.name}" parent="{ir.parent_sector}" '
                     f'rank="{ir.rs_rank}" rank_within_sector="{ir.rs_rank_within_sector}">')
            L.append(f'      <rs rs_5d="{ir.rs_5d:.6f}" rs_20d="{ir.rs_20d:.6f}" rs_60d="{ir.rs_60d:.6f}" '
                     f'slope="{ir.rs_slope:.6f}" composite="{ir.industry_composite:.2f}"/>')
            L.append(f'      <rs_vs_parent rs_5d="{ir.rs_5d_vs_parent:.6f}" rs_20d="{ir.rs_20d_vs_parent:.6f}" '
                     f'rs_60d="{ir.rs_60d_vs_parent:.6f}"/>')
            L.append(f'      <sparkline_trend description="20d RS sparkline: {trend}"/>')
            if state:
                L.append(f'      <state value="{state.state.value}" confidence="{state.confidence}"/>')
            if parent_state:
                L.append(f'      <parent_state value="{parent_state.state.value}"/>')
            L.append('    </industry>')
        L.append('  </industries>')

    # ── Composite Score Rankings (text description of bar chart) ──
    L.append('  <composite_rankings description="Horizontal bar chart sorted best-to-worst. '
             'Bar color by state: deep green=Overt Pump, light green=Accumulation, '
             'gray=Ambiguous, light red=Exhaustion, deep red=Overt Dump.">')
    for r in sorted(rs_readings, key=lambda x: x.rs_composite, reverse=True):
        state = states.get(r.ticker)
        sv = state.state.value if state else "—"
        L.append(f'    <bar ticker="{r.ticker}" composite="{r.rs_composite:.1f}" state="{sv}"/>')
    L.append('  </composite_rankings>')

    # ── 1d Moves ──
    L.append(f'  <today_moves spy_1d="{spy_1d:.6f}">')
    for m in sorted(moves_1d, key=lambda x: x["rs_1d"], reverse=True):
        L.append(f'    <move ticker="{m["ticker"]}" name="{m["name"]}" '
                 f'return_1d="{m["return_1d"]:.6f}" rs_1d="{m["rs_1d"]:.6f}"/>')
    L.append('  </today_moves>')

    # ── Rolling RS Leaders ──
    L.append('  <rolling_leaders>')
    for label, (t, name, v) in rolling_leaders.items():
        L.append(f'    <leader period="{label}" ticker="{t}" name="{name}" rs="{v:.6f}"/>')
    L.append('  </rolling_leaders>')

    # ── Baton Pass Alerts ──
    if baton_alerts:
        L.append('  <baton_pass_alerts>')
        for a in baton_alerts:
            L.append(f'    <alert>{X(a)}</alert>')
        L.append('  </baton_pass_alerts>')

    # ── Reversal Diagnostics (top 5) ──
    rev_sorted = sorted(reversal_scores, key=lambda x: x.reversal_score, reverse=True)[:5]
    if rev_sorted:
        L.append('  <reversal_diagnostics description="Top 5 most fragile groups with sub-signal detail.">')
        for rv in rev_sorted:
            state = states.get(rv.ticker)
            sv = state.state.value if state else "—"
            subs = rv.sub_signals
            L.append(f'    <group ticker="{rv.ticker}" reversal_score="{rv.reversal_score:.4f}" '
                     f'percentile="{rv.reversal_percentile:.0f}" '
                     f'breadth_det="{rv.breadth_det_pillar:.0f}" price_break="{rv.price_break_pillar:.0f}" '
                     f'crowding="{rv.crowding_pillar:.0f}" '
                     f'clv="{subs.get("clv_trend", 0):.3f}" '
                     f'gap_fade="{subs.get("gap_fade_rate", 0):.3f}" '
                     f'follow_through="{subs.get("follow_through", 0):.3f}" '
                     f'distance_ma="{subs.get("distance_from_ma", 0):.2f}" '
                     f'rvol="{subs.get("rvol", 1):.2f}" '
                     f'state="{sv}"/>')
        L.append('  </reversal_diagnostics>')

    L.append('</pump_rotation_report>')
    return "\n".join(L)


# ═══════════════════════════════════════════════════════
# MARKDOWN
# ═══════════════════════════════════════════════════════

def _build_markdown(result: dict) -> str:
    regime = result["regime"]
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    breadth = result.get("breadth")
    industry_rs = sorted(result.get("industry_rs", []), key=lambda x: x.rs_rank)
    reversal_scores = result.get("reversal_scores", [])
    prices = result["prices"]
    last_date = prices.index[-1].strftime("%Y-%m-%d")
    rev_map = {rv.ticker: rv for rv in reversal_scores}

    spy_1d, moves_1d = _get_1d_moves(result)
    rolling_leaders = _get_rolling_moves(result)
    baton_alerts = _get_baton_passes(result)

    L = [f"# Pump Rotation System Report — {last_date}", "",
         f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Live quotes: Yes*", ""]

    # Regime
    L.append(f"## Regime: **{regime.state.value}**")
    L.append(f"> {regime.explanation}")
    L.append("")
    L.append("| Signal | Value | Level |")
    L.append("|--------|-------|-------|")
    for sig in regime.signals:
        L.append(f"| {sig.name} | {sig.raw_value:.2f} | {sig.level.value} |")
    L.append("")

    # VIX
    vix = result.get("vix")
    vix3m = result.get("vix3m")
    if vix is not None and len(vix) >= 2:
        L.append(f"**VIX**: {vix.iloc[-1]:.1f} (1d change: {vix.iloc[-1]-vix.iloc[-2]:+.1f})")
        if vix3m is not None and len(vix3m) > 0:
            L.append(f"**VIX/VIX3M ratio**: {vix.iloc[-1]/vix3m.iloc[-1]:.3f}")
        L.append(f"*VIX term structure chart: 60-day VIX vs VIX3M with threshold lines at 20 and 30.*")
        L.append("")

    # Breadth
    if breadth:
        z = f"{breadth.rsp_spy_ratio_zscore:.2f}" if not math.isnan(breadth.rsp_spy_ratio_zscore) else "N/A"
        L.append(f"## Breadth: **{breadth.signal.value}** (z: {z})")
        L.append(f"RSP/SPY ratio: {breadth.rsp_spy_ratio:.4f}, 20d change: {breadth.rsp_spy_ratio_20d_change:+.4f}")
        L.append(f"*RSP/SPY ratio chart: 60-day trend with 20d MA. Rising = broad participation.*")
        L.append("")

    # Sector Heatmap
    L.append("## Sector RS Heat Map")
    L.append("*Color: GREEN=outperforming SPY, RED=underperforming. Pattern: GGG=sustained leader, "
             "RRG=decaying (rotation OUT), GGR=emerging (rotation IN)*")
    L.append("")
    L.append("| Sector | 5d RS | Color | 20d RS | Color | 60d RS | Color | State |")
    L.append("|--------|-------|-------|--------|-------|--------|-------|-------|")
    for r in rs_readings:
        state = states.get(r.ticker)
        sv = state.state.value if state else "—"
        L.append(f"| {r.ticker} ({r.name}) | {r.rs_5d*100:+.1f}% | {_rs_color_word(r.rs_5d)} | "
                 f"{r.rs_20d*100:+.1f}% | {_rs_color_word(r.rs_20d)} | "
                 f"{r.rs_60d*100:+.1f}% | {_rs_color_word(r.rs_60d)} | {sv} |")
    L.append("")

    # Sector Rankings
    L.append("## Sector Rankings")
    L.append("")
    L.append("| Rank | Sector | 20d Trend | RS 5d | RS 20d | RS 60d | Slope | Comp | Pump | Delta | Rev | State | Conf |")
    L.append("|------|--------|-----------|-------|--------|--------|-------|------|------|-------|-----|-------|------|")
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        rev = rev_map.get(r.ticker)
        trend = _trend_description(prices, r.ticker)
        sv = state.state.value if state else "—"
        sc = f"{state.confidence}%" if state else "—"
        ps = f"{pump.pump_score:.2f}" if pump else "—"
        pd_v = f"{pump.pump_delta:+.3f}" if pump else "—"
        rv_s = f"{rev.reversal_score:.2f}" if rev else "—"
        L.append(f"| #{r.rs_rank} | {r.ticker} ({r.name}) | {trend} | "
                 f"{r.rs_5d:+.2%} | {r.rs_20d:+.2%} | {r.rs_60d:+.2%} | "
                 f"{r.rs_slope:+.4f} | {r.rs_composite:.0f} | {ps} | {pd_v} | {rv_s} | {sv} | {sc} |")
    L.append("")

    # Industries
    if industry_rs:
        L.append("## Industry Rankings")
        L.append("")
        L.append("| Rank | Industry | Parent | 20d Trend | RS 5d | RS 20d | RS 60d | vs Parent | Comp | State | Parent State |")
        L.append("|------|----------|--------|-----------|-------|--------|--------|-----------|------|-------|--------------|")
        for ir in industry_rs:
            state = states.get(ir.ticker)
            parent_state = states.get(ir.parent_sector)
            trend = _trend_description(prices, ir.ticker)
            sv = state.state.value if state else "—"
            psv = parent_state.state.value if parent_state else "—"
            L.append(f"| #{ir.rs_rank} | {ir.ticker} ({ir.name}) | {ir.parent_sector} | {trend} | "
                     f"{ir.rs_5d:+.2%} | {ir.rs_20d:+.2%} | {ir.rs_60d:+.2%} | "
                     f"{ir.rs_20d_vs_parent:+.2%} | {ir.industry_composite:.0f} | {sv} | {psv} |")
        L.append("")

    # 1d Moves
    L.append(f"## Today's Moves (SPY: {spy_1d:+.2%})")
    L.append("*1d RS waterfall chart: horizontal bars colored by state on Overt Dump→Overt Pump spectrum.*")
    L.append("")
    for m in sorted(moves_1d, key=lambda x: x["rs_1d"], reverse=True):
        arrow = "▲" if m["rs_1d"] > 0.001 else "▼" if m["rs_1d"] < -0.001 else "—"
        L.append(f"- {arrow} {m['ticker']} ({m['name']}): {m['return_1d']:+.2%} (RS: {m['rs_1d']:+.2%})")
    L.append("")

    # Rolling leaders
    if rolling_leaders:
        L.append("## Rolling RS Leaders")
        for label, (t, name, v) in rolling_leaders.items():
            L.append(f"- **{label}**: {t} ({name}) RS {v:+.2%}")
        L.append("")

    # Baton passes
    if baton_alerts:
        L.append("## Baton Pass Alerts")
        for a in baton_alerts:
            L.append(f"- {a}")
        L.append("")

    # Reversal diagnostics
    rev_sorted = sorted(reversal_scores, key=lambda x: x.reversal_score, reverse=True)[:5]
    if rev_sorted:
        L.append("## Reversal Diagnostics (top 5 fragile)")
        L.append("")
        L.append("| Group | Rev Score | %ile | BreadthDet | PriceBreak | Crowding | CLV | RVOL | State |")
        L.append("|-------|-----------|------|------------|------------|----------|-----|------|-------|")
        for rv in rev_sorted:
            state = states.get(rv.ticker)
            sv = state.state.value if state else "—"
            subs = rv.sub_signals
            L.append(f"| {rv.ticker} | {rv.reversal_score:.2f} | {rv.reversal_percentile:.0f}% | "
                     f"{rv.breadth_det_pillar:.0f} | {rv.price_break_pillar:.0f} | "
                     f"{rv.crowding_pillar:.0f} | {subs.get('clv_trend',0):.2f} | "
                     f"{subs.get('rvol',1):.1f}x | {sv} |")
        L.append("")

    # Composite ranking description
    L.append("## Composite Score Ranking")
    L.append("*Horizontal bar chart sorted best-to-worst. Bar color by state.*")
    L.append("")
    for r in sorted(rs_readings, key=lambda x: x.rs_composite, reverse=True):
        state = states.get(r.ticker)
        sv = state.state.value if state else "—"
        bar = "█" * max(1, int(r.rs_composite / 5))
        L.append(f"- {r.ticker} ({r.name}): {bar} {r.rs_composite:.0f} [{sv}]")
    L.append("")

    return "\n".join(L)


# ═══════════════════════════════════════════════════════
# CSV (sector + industry rows)
# ═══════════════════════════════════════════════════════

def _build_csv(result: dict) -> str:
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    industry_rs = sorted(result.get("industry_rs", []), key=lambda x: x.rs_rank)
    reversal_scores = result.get("reversal_scores", [])
    prices = result["prices"]
    spy_1d = prices["SPY"].pct_change().iloc[-1] if "SPY" in prices.columns else 0
    rev_map = {rv.ticker: rv for rv in reversal_scores}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Type", "Rank", "Ticker", "Name", "Parent",
        "RS_5d", "RS_20d", "RS_60d", "Slope", "Composite",
        "RS_vs_Parent_20d",
        "Pump_Score", "Pump_Delta",
        "Reversal_Score", "Reversal_Pctile",
        "State", "Confidence", "Parent_State",
        "Return_1d", "RS_1d", "20d_Trend",
    ])
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        rev = rev_map.get(r.ticker)
        sec_1d = prices[r.ticker].pct_change().iloc[-1] if r.ticker in prices.columns else 0
        trend = _trend_description(prices, r.ticker)
        writer.writerow([
            "Sector", r.rs_rank, r.ticker, r.name, "",
            f"{r.rs_5d:.6f}", f"{r.rs_20d:.6f}", f"{r.rs_60d:.6f}", f"{r.rs_slope:.6f}", f"{r.rs_composite:.2f}",
            "",
            f"{pump.pump_score:.4f}" if pump else "", f"{pump.pump_delta:.4f}" if pump else "",
            f"{rev.reversal_score:.4f}" if rev else "", f"{rev.reversal_percentile:.1f}" if rev else "",
            state.state.value if state else "", state.confidence if state else "", "",
            f"{sec_1d:.6f}", f"{sec_1d - spy_1d:.6f}", trend,
        ])
    for ir in industry_rs:
        state = states.get(ir.ticker)
        parent_state = states.get(ir.parent_sector)
        sec_1d = prices[ir.ticker].pct_change().iloc[-1] if ir.ticker in prices.columns else 0
        trend = _trend_description(prices, ir.ticker)
        writer.writerow([
            "Industry", ir.rs_rank, ir.ticker, ir.name, ir.parent_sector,
            f"{ir.rs_5d:.6f}", f"{ir.rs_20d:.6f}", f"{ir.rs_60d:.6f}", f"{ir.rs_slope:.6f}",
            f"{ir.industry_composite:.2f}", f"{ir.rs_20d_vs_parent:.6f}",
            "", "",
            "", "",
            state.state.value if state else "", state.confidence if state else "",
            parent_state.state.value if parent_state else "",
            f"{sec_1d:.6f}", f"{sec_1d - spy_1d:.6f}", trend,
        ])
    return output.getvalue()


# ═══════════════════════════════════════════════════════
# JSON (exhaustive)
# ═══════════════════════════════════════════════════════

def _build_json(result: dict) -> str:
    regime = result["regime"]
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    industry_rs = sorted(result.get("industry_rs", []), key=lambda x: x.rs_rank)
    reversal_scores = result.get("reversal_scores", [])
    prices = result["prices"]
    breadth = result.get("breadth")
    spy_1d, moves_1d = _get_1d_moves(result)
    rolling_leaders = _get_rolling_moves(result)
    baton_alerts = _get_baton_passes(result)
    rev_map = {rv.ticker: rv for rv in reversal_scores}

    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "price_date": prices.index[-1].strftime("%Y-%m-%d"),
        "live_quotes": True,
        "regime": {
            "state": regime.state.value,
            "hostile_count": regime.hostile_count,
            "fragile_count": regime.fragile_count,
            "signals": [{"name": s.name, "value": s.raw_value, "level": s.level.value}
                        for s in regime.signals],
            "explanation": regime.explanation,
        },
        "breadth": {
            "signal": breadth.signal.value,
            "ratio": breadth.rsp_spy_ratio,
            "zscore": breadth.rsp_spy_ratio_zscore,
            "change_20d": breadth.rsp_spy_ratio_20d_change,
        } if breadth else None,
        "sectors": [],
        "sector_heatmap": [],
        "industries": [],
        "industry_heatmap": [],
        "today_moves": {"spy_1d": spy_1d, "sectors": moves_1d},
        "rolling_leaders": {k: {"ticker": t, "name": n, "rs": v} for k, (t, n, v) in rolling_leaders.items()},
        "baton_pass_alerts": baton_alerts,
        "reversal_diagnostics": [],
    }

    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        rev = rev_map.get(r.ticker)
        trend = _trend_description(prices, r.ticker)
        entry = {
            "rank": r.rs_rank, "ticker": r.ticker, "name": r.name,
            "rs": {"5d": r.rs_5d, "20d": r.rs_20d, "60d": r.rs_60d,
                   "slope": r.rs_slope, "composite": r.rs_composite},
            "sparkline_trend": trend,
            "pump": {"score": pump.pump_score, "delta": pump.pump_delta,
                     "delta_5d_avg": pump.pump_delta_5d_avg} if pump else None,
            "reversal": {"score": rev.reversal_score, "percentile": rev.reversal_percentile,
                         "above_75th": bool(rev.above_75th),
                         "sub_signals": {k: float(v) for k, v in rev.sub_signals.items()}} if rev else None,
            "state": {"value": state.state.value, "confidence": state.confidence,
                      "sessions": state.sessions_in_state,
                      "pressure": state.transition_pressure.value} if state else None,
        }
        data["sectors"].append(entry)
        data["sector_heatmap"].append({
            "ticker": r.ticker, "name": r.name,
            "5d": {"value": r.rs_5d, "color": _rs_color_word(r.rs_5d)},
            "20d": {"value": r.rs_20d, "color": _rs_color_word(r.rs_20d)},
            "60d": {"value": r.rs_60d, "color": _rs_color_word(r.rs_60d)},
            "state": state.state.value if state else None,
        })

    for ir in industry_rs:
        state = states.get(ir.ticker)
        parent_state = states.get(ir.parent_sector)
        trend = _trend_description(prices, ir.ticker)
        data["industries"].append({
            "rank": ir.rs_rank, "ticker": ir.ticker, "name": ir.name,
            "parent": ir.parent_sector,
            "rs": {"5d": ir.rs_5d, "20d": ir.rs_20d, "60d": ir.rs_60d,
                   "vs_parent_20d": ir.rs_20d_vs_parent,
                   "composite": ir.industry_composite},
            "sparkline_trend": trend,
            "state": state.state.value if state else None,
            "parent_state": parent_state.state.value if parent_state else None,
        })
        data["industry_heatmap"].append({
            "ticker": ir.ticker, "name": ir.name, "parent": ir.parent_sector,
            "5d": {"value": ir.rs_5d, "color": _rs_color_word(ir.rs_5d)},
            "20d": {"value": ir.rs_20d, "color": _rs_color_word(ir.rs_20d)},
            "60d": {"value": ir.rs_60d, "color": _rs_color_word(ir.rs_60d)},
            "vs_parent": {"value": ir.rs_20d_vs_parent, "color": _rs_color_word(ir.rs_20d_vs_parent)},
            "state": state.state.value if state else None,
        })

    for rv in sorted(reversal_scores, key=lambda x: x.reversal_score, reverse=True)[:5]:
        state = states.get(rv.ticker)
        data["reversal_diagnostics"].append({
            "ticker": rv.ticker, "score": rv.reversal_score, "percentile": rv.reversal_percentile,
            "pillars": {"breadth_det": rv.breadth_det_pillar, "price_break": rv.price_break_pillar,
                        "crowding": rv.crowding_pillar},
            "sub_signals": {k: float(v) for k, v in rv.sub_signals.items()},
            "state": state.state.value if state else None,
        })

    return json.dumps(data, indent=2, default=str)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
