"""Panel 2: Sector table — view toggle, inline sparklines, composite chart, heat map, valuations."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from engine.schemas import AnalysisState
from dashboard.components.horizon_col import get_horizon_label

_GLOSSARY = {
    "rank": {"title": "RS Rank (1–11)", "body": "Cross-sectional rank based on 20d RS. 1 = strongest."},
    "rs": {"title": "Relative Strength", "body": "Sector return minus SPY return over rolling window."},
    "performance": {"title": "Absolute Performance", "body": "Raw price return, not relative to any benchmark."},
    "composite": {"title": "RS Composite (0–100)", "body": "Weighted blend: 20%×5d + 50%×20d + 30%×60d, percentile-ranked."},
    "pump_score": {"title": "Pump Score (0–1)", "body": "RS (40%), participation (30%), flow (30%). Delta = session-over-session change."},
    "state": {"title": "Analysis State", "body": "Accumulation → Broadening → Overt Pump → Exhaustion → Rotation. Ambiguous = conflicting."},
    "heatmap": {"title": "RS Heat Map", "body": "Green=outperforming SPY, Red=underperforming. Pattern across columns shows rotation:\n- 🟩🟩🟩 = Sustained leader\n- 🟥🟥🟩 = Former leader decaying (rotation OUT)\n- 🟩🟩🟥 = New leader emerging (rotation IN)"},
}


def _popover(key):
    e = _GLOSSARY.get(key, {})
    with st.popover("ℹ️"):
        st.markdown(f"**{e.get('title', key)}**")
        st.markdown(e.get("body", ""))


def render_sector_table(result: dict):
    rs_readings = sorted(result["rs_readings"], key=lambda r: r.rs_rank)
    states = result["states"]
    pumps = result["pumps"]
    rs_history = result["rs_history"]
    prices = result["prices"]

    st.subheader("Sector Rankings")

    # ── Feature 1: View Toggle ────────────────────────
    view_col, g1, g2, g3 = st.columns([2, 1, 1, 1])
    with view_col:
        view = st.selectbox("View Mode", ["Relative Strength (vs SPY)", "Absolute Performance"],
                            index=0, key="sector_view")
    with g1:
        _popover("rs" if "Relative" in view else "performance")
    with g2:
        _popover("pump_score")
    with g3:
        _popover("state")

    is_rs = "Relative" in view

    # ── Main Table with inline sparklines (Feature 2) ─
    from dashboard.components.style_utils import style_dataframe
    from dashboard.components.sparkline import make_sparkline_unicode
    from engine.arrows import compute_arrow, arrow_symbol
    rev_map = result.get("reversal_map", {})
    rows = []
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

        # Inline unicode sparkline
        spark = "—"
        if r.ticker in rs_history.columns:
            s = rs_history[r.ticker].tail(60).dropna()
            if not s.empty:
                spark = make_sparkline_unicode(s, width=12)

        if is_rs:
            row = {"Rank": f"#{r.rs_rank}{rank_arrow}", "Ticker": r.ticker, "Sector": r.name,
                   "20d Trend": spark,
                   "RS 5d": f"{r.rs_5d:+.2%}", "RS 20d": f"{r.rs_20d:+.2%}", "RS 60d": f"{r.rs_60d:+.2%}",
                   "Slope": f"{r.rs_slope:+.4f}", "Composite": f"{r.rs_composite:.1f}"}
        else:
            p5 = prices[r.ticker].pct_change(5).iloc[-1] if r.ticker in prices.columns else 0
            p20 = prices[r.ticker].pct_change(20).iloc[-1] if r.ticker in prices.columns else 0
            p60 = prices[r.ticker].pct_change(60).iloc[-1] if r.ticker in prices.columns and len(prices) > 60 else 0
            row = {"Rank": f"#{r.rs_rank}{rank_arrow}", "Ticker": r.ticker, "Sector": r.name,
                   "20d Trend": spark,
                   "Perf 5d": f"{p5:+.2%}", "Perf 20d": f"{p20:+.2%}", "Perf 60d": f"{p60:+.2%}",
                   "Composite": f"{r.rs_composite:.1f}"}
        # Concentration badge
        conc_map = result.get("concentrations", {})
        conc = conc_map.get(r.ticker)
        conc_str = conc.regime.value.split()[-1] if conc else "—"  # "Healthy"/"Fragile"/"Unhealthy"

        # Trade state
        trade_states = result.get("trade_states", {})
        ts = trade_states.get(r.ticker)
        trade_val = ts.trade_state.value if ts else "—"
        size_val = ts.size_class if ts else "—"

        # Arrow indicator for delta
        pump_delta_5d = pump.pump_delta_5d_avg if pump else 0.0
        arrow = compute_arrow(pump_delta, delta_prior=pump_delta_5d, rank_change=r.rs_rank_change)
        arrow_sym = arrow_symbol(arrow)

        # Horizon pattern
        horizon = result.get("horizon_readings", {}).get(r.ticker)
        pattern_label = get_horizon_label(horizon.pattern) if horizon else "—"

        row.update({"Pump": f"{pump_score:.2f}", "Δ": f"{arrow_sym} {pump_delta:+.3f}",
                     "Rev": rev_str, "Rev %ile": rev_pct,
                     "Conc": conc_str,
                     "State": state_val, "Trade": trade_val, "Size": size_val,
                     "Conf": f"{state_conf}%", "Pattern": pattern_label})
        rows.append(row)

    df = pd.DataFrame(rows)
    styled = style_dataframe(df)
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── Feature 3: Composite Bar Chart (STATE-colored) ─
    from dashboard.components.composite_chart import make_composite_bar_chart
    with st.expander("Composite Score Ranking", expanded=True):
        groups = []
        for r in rs_readings:
            state = states.get(r.ticker)
            groups.append({"ticker": r.ticker, "name": r.name,
                           "composite": r.rs_composite,
                           "state": state.state.value if state else "Accumulation"})
        fig = make_composite_bar_chart(groups, title="Sector Composite Score Ranking")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Feature 4: RS Heat Map with State column ──────
    from dashboard.components.heatmap import make_rs_heatmap
    hdr, info = st.columns([6, 1])
    with hdr:
        st.subheader("RS Heat Map")
    with info:
        _popover("heatmap")

    hm_groups = []
    for r in sorted(result["rs_readings"], key=lambda x: x.rs_rank):
        state = states.get(r.ticker)
        # Compute 1d RS
        rs_1d_val = 0.0
        if r.ticker in prices.columns and "SPY" in prices.columns and len(prices) >= 2:
            sec_1d = prices[r.ticker].pct_change(1).iloc[-1]
            spy_1d = prices["SPY"].pct_change(1).iloc[-1]
            if pd.notna(sec_1d) and pd.notna(spy_1d):
                rs_1d_val = sec_1d - spy_1d
        hm_groups.append({
            "ticker": r.ticker, "name": r.name,
            "rs_1d": rs_1d_val,
            "rs_5d": r.rs_5d, "rs_20d": r.rs_20d, "rs_60d": r.rs_60d,
            "rs_vs_parent": None,
            "state": state.state.value if state else "—",
        })
    fig = make_rs_heatmap(hm_groups, title="Sector RS Heat Map")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Plotly Sparklines (3 per row) ─────────────────
    st.subheader("20d RS Sparklines (60 trading days)")
    for row_start in range(0, len(rs_readings), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(rs_readings):
                break
            r = rs_readings[idx]
            with col:
                if r.ticker in rs_history.columns:
                    sp = rs_history[r.ticker].tail(60).dropna()
                    if not sp.empty:
                        color = "#22c55e" if sp.iloc[-1] > 0 else "#ef4444"
                        fill = "rgba(34,197,94,0.1)" if sp.iloc[-1] > 0 else "rgba(239,68,68,0.1)"
                        fig = go.Figure(go.Scatter(x=sp.index, y=sp.values, mode="lines",
                                                   line=dict(color=color, width=2),
                                                   fill="tozeroy", fillcolor=fill))
                        fig.add_hline(y=0, line_dash="dot", line_color="#555")
                        fig.update_layout(height=120, margin=dict(t=25, b=5, l=5, r=5),
                                          title=dict(text=f"#{r.rs_rank} {r.ticker} — {r.name}", font=dict(size=12)),
                                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Feature 6: Valuations ─────────────────────────
    from dashboard.components.valuations import render_valuations_panel
    render_valuations_panel([r.ticker for r in rs_readings], tab_label="Sector")
