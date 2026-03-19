"""Panel 2: Sector table — RS/Performance toggle, sparklines, heat map, valuations."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from engine.schemas import AnalysisState

_ETF_FULL_NAMES = {
    "XLK": "Technology Select Sector SPDR", "XLV": "Health Care Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR", "XLE": "Energy Select Sector SPDR",
    "XLI": "Industrial Select Sector SPDR", "XLU": "Utilities Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR", "XLC": "Communication Services Select Sector SPDR",
    "XLY": "Consumer Discretionary Select Sector SPDR", "XLP": "Consumer Staples Select Sector SPDR",
    "XLB": "Materials Select Sector SPDR", "SPY": "SPDR S&P 500 ETF Trust",
    "RSP": "Invesco S&P 500 Equal Weight ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)", "IWM": "iShares Russell 2000 ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
}

_GLOSSARY = {
    "rank": {"title": "RS Rank (1–11)", "body": "Cross-sectional rank based on 20d RS. 1 = strongest."},
    "rs": {"title": "Relative Strength", "body": "Sector return minus SPY return over rolling window. Positive = outperforming."},
    "performance": {"title": "Absolute Performance", "body": "Raw price return over the window, not relative to any benchmark."},
    "composite": {"title": "RS Composite (0–100)", "body": "Weighted blend: 20% × 5d + 50% × 20d + 30% × 60d, percentile-ranked across sectors."},
    "pump_score": {"title": "Pump Score (0–1)", "body": "Weighted blend of RS (40%), participation (30%), flow (30%) pillars. Delta = session-over-session change."},
    "state": {"title": "Analysis State", "body": "Accumulation → Broadening → Overt Pump → Exhaustion → Rotation. Ambiguous = conflicting signals."},
    "heatmap": {"title": "RS Heat Map", "body": "Each cell shows RS at a given timeframe. Green = outperforming SPY, red = underperforming. Lets you see multi-timeframe alignment at a glance."},
}


def _popover(key: str):
    entry = _GLOSSARY.get(key, {})
    with st.popover("ℹ️"):
        st.markdown(f"**{entry.get('title', key)}**")
        st.markdown(entry.get("body", ""))


def render_sector_table(result: dict):
    rs_readings = sorted(result["rs_readings"], key=lambda r: r.rs_rank)
    states = result["states"]
    pumps = result["pumps"]
    rs_history = result["rs_history"]
    prices = result["prices"]

    st.subheader("Sector Rankings")

    # ── View toggle ───────────────────────────────────
    view_col, g1, g2, g3 = st.columns([2, 1, 1, 1])
    with view_col:
        view = st.selectbox("View", ["Relative Strength (vs SPY)", "Absolute Performance"], key="sector_view")
    with g1:
        _popover("rs" if "Relative" in view else "performance")
    with g2:
        _popover("pump_score")
    with g3:
        _popover("state")

    is_rs = "Relative" in view

    # ── Main table ────────────────────────────────────
    from dashboard.components.style_utils import style_dataframe
    from dashboard.components.sparkline import make_sparkline_unicode
    rows = []
    rev_map = result.get("reversal_map", {})
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        state_val = state.state.value if state else "N/A"
        state_conf = state.confidence if state else 0
        pump_score = pump.pump_score if pump else 0.0
        pump_delta = pump.pump_delta if pump else 0.0

        rank_arrow = ""
        if r.rs_rank_change > 0:
            rank_arrow = f" (+{r.rs_rank_change})"
        elif r.rs_rank_change < 0:
            rank_arrow = f" ({r.rs_rank_change})"

        rev = rev_map.get(r.ticker)
        rev_str = f"{rev.reversal_score:.2f}" if rev else "—"
        rev_pct = f"{rev.reversal_percentile:.0f}%" if rev else "—"
        if rev and rev.above_75th:
            rev_pct += " ⚠"

        # Inline unicode sparkline from 20d RS history
        spark = "—"
        if r.ticker in rs_history.columns:
            spark_series = rs_history[r.ticker].tail(60).dropna()
            if not spark_series.empty:
                spark = make_sparkline_unicode(spark_series, width=12)

        if is_rs:
            row = {
                "Rank": f"#{r.rs_rank}{rank_arrow}",
                "Ticker": r.ticker, "Sector": r.name,
                "20d Trend": spark,
                "RS 5d": f"{r.rs_5d:+.2%}", "RS 20d": f"{r.rs_20d:+.2%}", "RS 60d": f"{r.rs_60d:+.2%}",
                "Slope": f"{r.rs_slope:+.4f}", "Composite": f"{r.rs_composite:.1f}",
            }
        else:
            p5 = prices[r.ticker].pct_change(5).iloc[-1] if r.ticker in prices.columns else 0
            p20 = prices[r.ticker].pct_change(20).iloc[-1] if r.ticker in prices.columns else 0
            p60 = prices[r.ticker].pct_change(60).iloc[-1] if r.ticker in prices.columns and len(prices) > 60 else 0
            row = {
                "Rank": f"#{r.rs_rank}{rank_arrow}",
                "Ticker": r.ticker, "Sector": r.name,
                "20d Trend": spark,
                "Perf 5d": f"{p5:+.2%}", "Perf 20d": f"{p20:+.2%}", "Perf 60d": f"{p60:+.2%}",
                "Composite": f"{r.rs_composite:.1f}",
            }
        row.update({
            "Pump": f"{pump_score:.2f}", "Delta": f"{pump_delta:+.3f}",
            "Rev": rev_str, "Rev %ile": rev_pct,
            "State": state_val, "Conf": f"{state_conf}%",
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    styled = style_dataframe(df)
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── Sparklines ────────────────────────────────────
    st.subheader("20d RS Sparklines (60 trading days)")
    n_cols = 3
    for row_start in range(0, len(rs_readings), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(rs_readings):
                break
            r = rs_readings[idx]
            with col:
                if r.ticker in rs_history.columns:
                    spark = rs_history[r.ticker].tail(60).dropna()
                    if not spark.empty:
                        color = "#00d4aa" if spark.iloc[-1] > 0 else "#ff4444"
                        fill = "rgba(0,212,170,0.1)" if spark.iloc[-1] > 0 else "rgba(255,68,68,0.1)"
                        fig = go.Figure(go.Scatter(x=spark.index, y=spark.values, mode="lines",
                                                   line=dict(color=color, width=2), fill="tozeroy", fillcolor=fill))
                        fig.add_hline(y=0, line_dash="dot", line_color="#555")
                        fig.update_layout(height=120, margin=dict(t=25, b=5, l=5, r=5),
                                          title=dict(text=f"#{r.rs_rank} {r.ticker} — {r.name}", font=dict(size=12)),
                                          xaxis=dict(visible=False), yaxis=dict(visible=False))
                        st.plotly_chart(fig, width="stretch")

    # ── RS Composite bar chart ────────────────────────
    st.subheader("RS Composite Score")
    by_comp = sorted(result["rs_readings"], key=lambda r: r.rs_composite, reverse=True)
    labels = [f"{r.ticker} ({r.name})" for r in by_comp]
    vals = [r.rs_composite for r in by_comp]
    colors = ["#00d4aa" if c > 60 else "#ffa500" if c > 40 else "#ff4444" for c in vals]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors))
    fig.update_layout(height=300, margin=dict(t=20, b=20), yaxis_title="Composite (0-100)",
                      xaxis=dict(categoryorder="array", categoryarray=labels))
    st.plotly_chart(fig, width="stretch")

    # ── Heat Map ──────────────────────────────────────
    hdr, info = st.columns([6, 1])
    with hdr:
        st.subheader("RS Heat Map")
    with info:
        _popover("heatmap")

    hm_data = []
    for r in sorted(result["rs_readings"], key=lambda x: x.rs_rank):
        hm_data.append({"Sector": f"{r.ticker} ({r.name})", "5d": r.rs_5d * 100, "20d": r.rs_20d * 100, "60d": r.rs_60d * 100})
    hm_df = pd.DataFrame(hm_data).set_index("Sector")
    fig = px.imshow(hm_df.values, x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
                    color_continuous_scale=["#ff4444", "#333333", "#00d4aa"],
                    color_continuous_midpoint=0, aspect="auto",
                    labels=dict(color="RS (%)"))
    fig.update_layout(height=max(300, len(hm_df) * 35), margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    # ── Valuations ────────────────────────────────────
    from dashboard.components.valuations import render_valuations_panel
    render_valuations_panel([r.ticker for r in rs_readings], tab_label="Sector")
