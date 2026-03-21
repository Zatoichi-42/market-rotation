"""Panel 3: Industry rotation map — view toggle, sparklines, composite chart, heat map, valuations."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from engine.schemas import IndustryRSReading
from engine.rs_scanner import compute_rs_all

_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}

_GLOSSARY = {
    "rs_vs_parent": {"title": "RS vs Parent Sector", "body": "Positive = driving the sector. Negative = lagging within sector."},
    "industry_composite": {"title": "Industry Composite", "body": "Blend of RS-vs-SPY (70%) + RS-vs-parent (30%), percentile-ranked."},
    "heatmap": {"title": "Industry RS Heat Map", "body": "Green=outperforming, Red=underperforming. 'vs Parent' column shows within-sector strength.\n- 🟩🟩🟩🟩 = Industry driving sector AND outperforming market\n- 🟥 in 'vs Parent' = freeriding on sector beta"},
}


def _parent_label(t):
    return f"{t} ({_SECTOR_NAMES.get(t, '')})" if t in _SECTOR_NAMES else t


def _popover(key):
    e = _GLOSSARY.get(key, {})
    with st.popover("ℹ️"):
        st.markdown(f"**{e.get('title', key)}**")
        st.markdown(e.get("body", ""))


def render_industry_panel(result: dict):
    industry_rs = result.get("industry_rs", [])
    if not industry_rs:
        st.warning("No industry RS data. Run the full Phase 2 pipeline.")
        return

    prices = result["prices"]
    st.subheader("Industry Rotation Map")

    # ── Controls ──────────────────────────────────────
    view_col, filter_col, g1, g2 = st.columns([2, 2, 1, 1])
    with view_col:
        view = st.selectbox("View Mode", ["Relative Strength (vs SPY)", "Absolute Performance"],
                            index=0, key="industry_view")
    with filter_col:
        parents = sorted(set(r.parent_sector for r in industry_rs))
        options = ["All"] + [_parent_label(p) for p in parents]
        parent_map = {_parent_label(p): p for p in parents}
        selected = st.selectbox("Filter by parent sector", options)
    with g1:
        _popover("rs_vs_parent")
    with g2:
        _popover("industry_composite")

    is_rs = "Relative" in view
    if selected != "All":
        pticker = parent_map.get(selected, selected)
        display = [r for r in industry_rs if r.parent_sector == pticker]
    else:
        display = industry_rs
    display_sorted = sorted(display, key=lambda r: r.rs_rank)

    # State map
    states = result.get("states", {})
    state_map = states if isinstance(states, dict) else {s.ticker: s for s in states} if states else {}

    # Pre-compute RS history for sparklines
    ind_tickers = [r.ticker for r in display_sorted if r.ticker in prices.columns]
    rs_hist = compute_rs_all(prices, ind_tickers, window=20) if ind_tickers else pd.DataFrame()

    # ── Main Table (Feature 1+2+5) ───────────────────
    from dashboard.components.style_utils import style_dataframe
    from dashboard.components.sparkline import make_sparkline_unicode
    rows = []
    for r in display_sorted:
        vs_icon = "▲" if r.rs_20d_vs_parent > 0.001 else ("▼" if r.rs_20d_vs_parent < -0.001 else "—")
        state = state_map.get(r.ticker)
        state_val = state.state.value if state else "—"
        state_conf = f"{state.confidence}%" if state else "—"

        spark = "—"
        if r.ticker in rs_hist.columns:
            s = rs_hist[r.ticker].tail(60).dropna()
            if not s.empty:
                spark = make_sparkline_unicode(s, width=12)

        # Parent sector state
        parent_state = state_map.get(r.parent_sector)
        parent_state_val = parent_state.state.value if parent_state else "—"

        # Compute 1d RS for this industry
        rs_1d_val = 0.0
        if r.ticker in prices.columns and "SPY" in prices.columns and len(prices) >= 2:
            sec_1d = prices[r.ticker].pct_change(1).iloc[-1]
            spy_1d = prices["SPY"].pct_change(1).iloc[-1]
            if pd.notna(sec_1d) and pd.notna(spy_1d):
                rs_1d_val = sec_1d - spy_1d

        if is_rs:
            row = {"Rank": r.rs_rank, "Industry": f"{r.ticker} ({r.name})",
                   "Parent": _parent_label(r.parent_sector), "20d Trend": spark,
                   "RS 1d": f"{rs_1d_val:+.2%}",
                   "RS 5d": f"{r.rs_5d:+.2%}", "RS 20d": f"{r.rs_20d:+.2%}", "RS 60d": f"{r.rs_60d:+.2%}",
                   "RS vs Parent": f"{vs_icon} {r.rs_20d_vs_parent:+.2%}",
                   "Slope": f"{r.rs_slope:+.4f}", "Composite": f"{r.industry_composite:.1f}"}
        else:
            p1 = prices[r.ticker].pct_change(1).iloc[-1] if r.ticker in prices.columns else 0
            p5 = prices[r.ticker].pct_change(5).iloc[-1] if r.ticker in prices.columns else 0
            p20 = prices[r.ticker].pct_change(20).iloc[-1] if r.ticker in prices.columns else 0
            p60 = prices[r.ticker].pct_change(60).iloc[-1] if r.ticker in prices.columns and len(prices) > 60 else 0
            row = {"Rank": r.rs_rank, "Industry": f"{r.ticker} ({r.name})",
                   "Parent": _parent_label(r.parent_sector), "20d Trend": spark,
                   "Perf 1d": f"{p1:+.2%}", "Perf 5d": f"{p5:+.2%}", "Perf 20d": f"{p20:+.2%}", "Perf 60d": f"{p60:+.2%}",
                   "Composite": f"{r.industry_composite:.1f}"}
        row.update({
            "#Sector": f"#{r.rs_rank_within_sector}",
            "State": state_val, "Conf": state_conf,
            "Parent State": parent_state_val,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    styled = style_dataframe(df)
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── Feature 3: Composite Bar Chart (STATE-colored) ─
    from dashboard.components.composite_chart import make_composite_bar_chart
    with st.expander("Industry Composite Score Ranking", expanded=True):
        groups = []
        for r in display_sorted:
            state = state_map.get(r.ticker)
            groups.append({"ticker": r.ticker, "name": r.name,
                           "composite": r.industry_composite,
                           "state": state.state.value if state else "Accumulation"})
        fig = make_composite_bar_chart(groups, title="Industry Composite Score Ranking")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Feature 4: RS Heat Map with State + vs Parent ──
    from dashboard.components.heatmap import make_rs_heatmap
    hdr, info = st.columns([6, 1])
    with hdr:
        st.subheader("Industry RS Heat Map")
    with info:
        _popover("heatmap")

    hm_groups = []
    for r in display_sorted:
        state = state_map.get(r.ticker)
        # Compute 1d RS for heatmap
        rs_1d_hm = 0.0
        if r.ticker in prices.columns and "SPY" in prices.columns and len(prices) >= 2:
            sec_1d = prices[r.ticker].pct_change(1).iloc[-1]
            spy_1d = prices["SPY"].pct_change(1).iloc[-1]
            if pd.notna(sec_1d) and pd.notna(spy_1d):
                rs_1d_hm = sec_1d - spy_1d
        hm_groups.append({
            "ticker": r.ticker, "name": r.name,
            "rs_1d": rs_1d_hm,
            "rs_5d": r.rs_5d, "rs_20d": r.rs_20d, "rs_60d": r.rs_60d,
            "rs_vs_parent": r.rs_20d_vs_parent,
            "state": state.state.value if state else "—",
        })
    fig = make_rs_heatmap(hm_groups, title="Industry RS Heat Map")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Plotly Sparklines (4 per row per spec) ────────
    if ind_tickers:
        st.subheader("20d RS Sparklines")
        for row_start in range(0, len(display_sorted), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(display_sorted):
                    break
                r = display_sorted[idx]
                with col:
                    if r.ticker in rs_hist.columns:
                        sp = rs_hist[r.ticker].tail(60).dropna()
                        if not sp.empty:
                            color = "#22c55e" if sp.iloc[-1] > 0 else "#ef4444"
                            fill = "rgba(34,197,94,0.1)" if sp.iloc[-1] > 0 else "rgba(239,68,68,0.1)"
                            fig = go.Figure(go.Scatter(x=sp.index, y=sp.values, mode="lines",
                                                       line=dict(color=color, width=2),
                                                       fill="tozeroy", fillcolor=fill))
                            fig.add_hline(y=0, line_dash="dot", line_color="#555")
                            fig.update_layout(height=110, margin=dict(t=22, b=5, l=5, r=5),
                                              title=dict(text=f"#{r.rs_rank} {r.ticker}", font=dict(size=11)),
                                              xaxis=dict(visible=False), yaxis=dict(visible=False),
                                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Sector → Industry Drilldown ───────────────────
    st.subheader("Sector → Industry Drilldown")
    pumps = result.get("pumps", {})
    for parent in parents:
        children = sorted([r for r in industry_rs if r.parent_sector == parent],
                          key=lambda r: r.rs_rank_within_sector)
        sector_pump = pumps.get(parent)
        sector_state = state_map.get(parent)
        pump_str = f"Pump: {sector_pump.pump_score:.2f}" if sector_pump else ""
        state_str = f"State: {sector_state.state.value}" if sector_state else ""
        st.markdown(f"**{_parent_label(parent)}** — {pump_str} {state_str}")
        for c in children:
            if c.rs_20d_vs_parent > 0.001:
                icon, label = "🟢", "Driving sector"
            elif c.rs_20d_vs_parent < -0.001:
                icon, label = "🔴", "Lagging"
            else:
                icon, label = "⚪", "Neutral"
            cs = state_map.get(c.ticker)
            cs_str = f" [{cs.state.value}]" if cs else ""
            st.markdown(f"&nbsp;&nbsp;&nbsp;{icon} **{c.ticker}** ({c.name}) — "
                        f"RS vs {parent}: {c.rs_20d_vs_parent:+.2%} — {label}{cs_str}")
        st.divider()

    # ── Feature 6: Valuations ─────────────────────────
    from dashboard.components.valuations import render_valuations_panel
    render_valuations_panel([r.ticker for r in display_sorted], tab_label="Industry")
