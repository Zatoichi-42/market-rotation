"""Panel 3: Industry rotation map — RS/Performance toggle, charts, heat map, valuations."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from engine.schemas import IndustryRSReading
from engine.rs_scanner import compute_rs_all

_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}

_GLOSSARY = {
    "rs_vs_parent": {
        "title": "RS vs Parent Sector",
        "body": "Positive = the industry is **driving** the sector. Negative = lagging within sector.",
    },
    "industry_composite": {
        "title": "Industry Composite Score",
        "body": "Blend of RS-vs-SPY (70%) and RS-vs-parent (30%), percentile-ranked across all industries.",
    },
    "heatmap": {
        "title": "Industry RS Heat Map",
        "body": "Green = outperforming, Red = underperforming. Rows sorted by rank, columns by timeframe.",
    },
}


def _parent_label(ticker: str) -> str:
    name = _SECTOR_NAMES.get(ticker, "")
    return f"{ticker} ({name})" if name else ticker


def _popover(key: str):
    entry = _GLOSSARY.get(key, {})
    with st.popover("ℹ️"):
        st.markdown(f"**{entry.get('title', key)}**")
        st.markdown(entry.get("body", ""))


def render_industry_panel(result: dict):
    industry_rs = result.get("industry_rs", [])

    if not industry_rs:
        st.warning("No industry RS data available. Run the full Phase 2 pipeline.")
        return

    prices = result["prices"]

    st.subheader("Industry Rotation Map")

    # ── Controls row ──────────────────────────────────
    view_col, filter_col, g1, g2 = st.columns([2, 2, 1, 1])
    with view_col:
        view = st.selectbox("View", ["Relative Strength (vs SPY)", "Absolute Performance"], key="industry_view")
    with filter_col:
        parents = sorted(set(r.parent_sector for r in industry_rs))
        filter_options = ["All"] + [_parent_label(p) for p in parents]
        parent_from_label = {_parent_label(p): p for p in parents}
        selected = st.selectbox("Filter by parent sector", filter_options)
    with g1:
        _popover("rs_vs_parent")
    with g2:
        _popover("industry_composite")

    is_rs = "Relative" in view

    if selected != "All":
        parent_ticker = parent_from_label.get(selected, selected)
        display = [r for r in industry_rs if r.parent_sector == parent_ticker]
    else:
        display = industry_rs

    display_sorted = sorted(display, key=lambda r: r.rs_rank)

    # ── Gather states ─────────────────────────────────
    states = result.get("states", {})
    state_map = states if isinstance(states, dict) else {s.ticker: s for s in states} if states else {}

    # ── Main table ────────────────────────────────────
    from dashboard.components.style_utils import style_dataframe
    from dashboard.components.sparkline import make_sparkline_unicode

    # Pre-compute RS history for sparklines
    ind_tickers = [r.ticker for r in display_sorted]
    available_for_spark = [t for t in ind_tickers if t in prices.columns]
    rs_hist = compute_rs_all(prices, available_for_spark, window=20) if available_for_spark else pd.DataFrame()

    rows = []
    for r in display_sorted:
        vs_icon = "▲" if r.rs_20d_vs_parent > 0.001 else ("▼" if r.rs_20d_vs_parent < -0.001 else "—")
        state = state_map.get(r.ticker)
        state_val = state.state.value if state else "—"
        state_conf = f"{state.confidence}%" if state else "—"

        # Inline unicode sparkline
        spark = "—"
        if r.ticker in rs_hist.columns:
            spark_series = rs_hist[r.ticker].tail(60).dropna()
            if not spark_series.empty:
                spark = make_sparkline_unicode(spark_series, width=12)

        if is_rs:
            row = {
                "Rank": r.rs_rank,
                "Industry": f"{r.ticker} ({r.name})",
                "Parent": _parent_label(r.parent_sector),
                "20d Trend": spark,
                "RS vs SPY (20d)": f"{r.rs_20d:+.2%}",
                "RS vs Parent": f"{vs_icon} {r.rs_20d_vs_parent:+.2%}",
                "Slope": f"{r.rs_slope:+.4f}",
                "Ind. Composite": f"{r.industry_composite:.1f}",
            }
        else:
            p5 = prices[r.ticker].pct_change(5).iloc[-1] if r.ticker in prices.columns else 0
            p20 = prices[r.ticker].pct_change(20).iloc[-1] if r.ticker in prices.columns else 0
            p60 = prices[r.ticker].pct_change(60).iloc[-1] if r.ticker in prices.columns and len(prices) > 60 else 0
            row = {
                "Rank": r.rs_rank,
                "Industry": f"{r.ticker} ({r.name})",
                "Parent": _parent_label(r.parent_sector),
                "20d Trend": spark,
                "Perf 5d": f"{p5:+.2%}", "Perf 20d": f"{p20:+.2%}", "Perf 60d": f"{p60:+.2%}",
                "Ind. Composite": f"{r.industry_composite:.1f}",
            }

        row.update({
            "Within Sector": f"#{r.rs_rank_within_sector}",
            "State": state_val, "Conf": state_conf,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    styled = style_dataframe(df)
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── Sparklines (20d RS, 60 trading days) ──────────
    ind_tickers = [r.ticker for r in display_sorted]
    available_tickers = [t for t in ind_tickers if t in prices.columns]
    if available_tickers:
        st.subheader("20d RS Sparklines")
        rs_hist = compute_rs_all(prices, available_tickers, window=20)
        n_cols = 4
        for row_start in range(0, len(display_sorted), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(display_sorted):
                    break
                r = display_sorted[idx]
                with col:
                    if r.ticker in rs_hist.columns:
                        spark = rs_hist[r.ticker].tail(60).dropna()
                        if not spark.empty:
                            color = "#00d4aa" if spark.iloc[-1] > 0 else "#ff4444"
                            fill = "rgba(0,212,170,0.1)" if spark.iloc[-1] > 0 else "rgba(255,68,68,0.1)"
                            fig = go.Figure(go.Scatter(x=spark.index, y=spark.values, mode="lines",
                                                       line=dict(color=color, width=2), fill="tozeroy", fillcolor=fill))
                            fig.add_hline(y=0, line_dash="dot", line_color="#555")
                            fig.update_layout(height=110, margin=dict(t=22, b=5, l=5, r=5),
                                              title=dict(text=f"#{r.rs_rank} {r.ticker}", font=dict(size=11)),
                                              xaxis=dict(visible=False), yaxis=dict(visible=False))
                            st.plotly_chart(fig, width="stretch")

    # ── Industry Composite bar chart ──────────────────
    st.subheader("Industry Composite Score")
    by_comp = sorted(display, key=lambda r: r.industry_composite, reverse=True)
    labels = [f"{r.ticker} ({r.name})" for r in by_comp]
    vals = [r.industry_composite for r in by_comp]
    colors = ["#00d4aa" if c > 60 else "#ffa500" if c > 40 else "#ff4444" for c in vals]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors))
    fig.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Composite (0-100)",
                      xaxis=dict(categoryorder="array", categoryarray=labels, tickangle=-45))
    st.plotly_chart(fig, width="stretch")

    # ── Heat Map ──────────────────────────────────────
    hdr, info = st.columns([6, 1])
    with hdr:
        st.subheader("Industry RS Heat Map")
    with info:
        _popover("heatmap")

    hm_data = []
    for r in display_sorted:
        hm_data.append({
            "Industry": f"{r.ticker} ({r.name})",
            "5d": r.rs_5d * 100, "20d": r.rs_20d * 100, "60d": r.rs_60d * 100,
            "vs Parent 20d": r.rs_20d_vs_parent * 100,
        })
    hm_df = pd.DataFrame(hm_data).set_index("Industry")
    fig = px.imshow(hm_df.values, x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
                    color_continuous_scale=["#ff4444", "#333333", "#00d4aa"],
                    color_continuous_midpoint=0, aspect="auto",
                    labels=dict(color="RS (%)"))
    fig.update_layout(height=max(400, len(hm_df) * 28), margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

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
            child_state = state_map.get(c.ticker)
            cs_str = f" [{child_state.state.value}]" if child_state else ""
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;{icon} **{c.ticker}** ({c.name}) — "
                f"RS vs {parent}: {c.rs_20d_vs_parent:+.2%} — {label}{cs_str}"
            )
        st.divider()

    # ── Valuations ────────────────────────────────────
    from dashboard.components.valuations import render_valuations_panel
    render_valuations_panel([r.ticker for r in display_sorted], tab_label="Industry")
