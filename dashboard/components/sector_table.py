"""Panel 2: Sector RS ranked table with sparklines, pump score, state."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from engine.schemas import AnalysisState


_ETF_FULL_NAMES = {
    "XLK": "Technology Select Sector SPDR",
    "XLV": "Health Care Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR",
    "XLE": "Energy Select Sector SPDR",
    "XLI": "Industrial Select Sector SPDR",
    "XLU": "Utilities Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR",
    "XLC": "Communication Services Select Sector SPDR",
    "XLY": "Consumer Discretionary Select Sector SPDR",
    "XLP": "Consumer Staples Select Sector SPDR",
    "XLB": "Materials Select Sector SPDR",
    "SPY": "SPDR S&P 500 ETF Trust",
    "RSP": "Invesco S&P 500 Equal Weight ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
    "IWM": "iShares Russell 2000 ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
}

_STATE_COLORS = {
    AnalysisState.ACCUMULATION: "#6699cc",
    AnalysisState.BROADENING: "#00d4aa",
    AnalysisState.OVERT_PUMP: "#00ff88",
    AnalysisState.EXHAUSTION: "#ffa500",
    AnalysisState.ROTATION: "#ff4444",
    AnalysisState.AMBIGUOUS: "#888888",
}

_GLOSSARY = {
    "rs": {
        "title": "Relative Strength (RS)",
        "body": (
            "Sector return **minus** SPY return over a rolling window. "
            "Positive RS = outperforming the broad market. Negative = underperforming.\n\n"
            "**RS 5d / 20d / 60d:** Short, medium, and long lookback windows. "
            "20d is the primary ranking signal.\n\n"
            "RS is *relative*, not absolute — a sector can have negative RS in a rising market "
            "if it's rising less than SPY."
        ),
    },
    "slope": {
        "title": "RS Slope",
        "body": (
            "The 5-session rate of change of 20d RS. Measures whether a sector's relative "
            "strength is **accelerating** (positive slope) or **decaying** (negative slope).\n\n"
            "A sector can be top-ranked but decaying (exhaustion signal) or mid-ranked but "
            "accelerating (broadening signal). Slope is often more actionable than the level."
        ),
    },
    "composite": {
        "title": "RS Composite Score (0–100)",
        "body": (
            "Weighted blend of percentile-ranked RS across all three windows:\n\n"
            "- **5d RS:** 20% weight (short-term momentum)\n"
            "- **20d RS:** 50% weight (primary signal)\n"
            "- **60d RS:** 30% weight (trend confirmation)\n\n"
            "A score of 100 means the sector is top-ranked at every timeframe. "
            "50 is mid-pack. 0 is worst at every timeframe."
        ),
    },
    "pump_score": {
        "title": "Pump Score (0.00–1.00)",
        "body": (
            "Composite momentum score blending three pillars:\n\n"
            "- **RS Pillar (40%):** Relative strength composite\n"
            "- **Participation Pillar (30%):** Breadth/volume confirmation\n"
            "- **Flow Pillar (30%):** Money flow proxy\n\n"
            "**Pump Delta:** Session-over-session change in pump score. "
            "Positive delta = strengthening. Negative delta = fading.\n\n"
            "The *delta* often matters more than the *level* — "
            "a mid-score sector with rising delta (broadening) is more interesting "
            "than a high-score sector with falling delta (exhaustion)."
        ),
    },
    "state": {
        "title": "Analysis State",
        "body": (
            "Each sector is classified into one of six states:\n\n"
            "- **Accumulation:** Low-to-mid score, positive delta — early positioning\n"
            "- **Broadening:** Delta positive 5+ sessions, score above 50th pctl — momentum building\n"
            "- **Overt Pump:** Top quartile score + top 3 rank + positive delta — clear leadership\n"
            "- **Exhaustion:** Was top quartile, delta negative 3+ sessions — momentum fading\n"
            "- **Rotation/Reversal:** Score declining, rank dropping — capital leaving\n"
            "- **Ambiguous:** Conflicting signals, no clear direction\n\n"
            "**Confidence (10–95%):** How strongly the signals agree. "
            "Reduced by 20% in FRAGILE regime, 30% in HOSTILE.\n\n"
            "**Transition Pressure:** UP / STABLE / DOWN / BREAK. "
            "BREAK = state just changed this session."
        ),
    },
    "rank": {
        "title": "RS Rank (1–11)",
        "body": (
            "Cross-sectional rank across all 11 GICS sector ETFs, based on 20d RS. "
            "Rank 1 = strongest relative performer. Rank 11 = weakest.\n\n"
            "**Rank Change:** How many positions the sector moved since the prior session. "
            "+2 means it improved by 2 spots. −3 means it dropped 3 spots.\n\n"
            "Rank changes are key rotation signals — watch for sectors climbing from mid-pack "
            "into top 3 (broadening) or falling from top 3 (exhaustion)."
        ),
    },
}


def _render_glossary_popover(key: str):
    entry = _GLOSSARY.get(key, {})
    with st.popover("ℹ️"):
        st.markdown(f"**{entry.get('title', key)}**")
        st.markdown(entry.get("body", "No description available."))


def render_sector_table(result: dict):
    rs_readings = sorted(result["rs_readings"], key=lambda r: r.rs_rank)
    states = result["states"]
    pumps = result["pumps"]
    rs_history = result["rs_history"]

    # Header with glossary popovers
    st.subheader("Sector Relative Strength Rankings")
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1:
        _render_glossary_popover("rank")
    with g2:
        _render_glossary_popover("rs")
    with g3:
        _render_glossary_popover("composite")
    with g4:
        _render_glossary_popover("pump_score")
    with g5:
        _render_glossary_popover("state")

    # Build table data
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

        rows.append({
            "Rank": f"#{r.rs_rank}{rank_arrow}",
            "Ticker": r.ticker,
            "Sector": r.name,
            "ETF Name": _ETF_FULL_NAMES.get(r.ticker, r.name),
            "RS 5d": f"{r.rs_5d:+.2%}",
            "RS 20d": f"{r.rs_20d:+.2%}",
            "RS 60d": f"{r.rs_60d:+.2%}",
            "Slope": f"{r.rs_slope:+.4f}",
            "Composite": f"{r.rs_composite:.1f}",
            "Pump": f"{pump_score:.2f}",
            "Delta": f"{pump_delta:+.3f}",
            "State": state_val,
            "Conf": f"{state_conf}%",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # Sparklines
    st.subheader("20d RS Sparklines (60 trading days)")
    n_cols = 3
    readings_sorted = sorted(result["rs_readings"], key=lambda r: r.rs_rank)
    for row_start in range(0, len(readings_sorted), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(readings_sorted):
                break
            r = readings_sorted[idx]
            with col:
                if r.ticker in rs_history.columns:
                    spark_data = rs_history[r.ticker].tail(60).dropna()
                    if not spark_data.empty:
                        color = "#00d4aa" if spark_data.iloc[-1] > 0 else "#ff4444"
                        fill_rgba = "rgba(0,212,170,0.1)" if spark_data.iloc[-1] > 0 else "rgba(255,68,68,0.1)"
                        fig = go.Figure(go.Scatter(
                            x=spark_data.index, y=spark_data.values,
                            mode="lines", line=dict(color=color, width=2),
                            fill="tozeroy", fillcolor=fill_rgba,
                        ))
                        fig.add_hline(y=0, line_dash="dot", line_color="#555")
                        fig.update_layout(
                            height=120, margin=dict(t=25, b=5, l=5, r=5),
                            title=dict(text=f"#{r.rs_rank} {r.ticker} — {r.name}", font=dict(size=12)),
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                        )
                        st.plotly_chart(fig, width="stretch")

    # Composite bar chart — sorted by composite score descending (most desirable first)
    st.subheader("RS Composite Score")
    by_composite = sorted(result["rs_readings"], key=lambda r: r.rs_composite, reverse=True)
    tickers = [f"{r.ticker} ({r.name})" for r in by_composite]
    composites = [r.rs_composite for r in by_composite]
    colors = ["#00d4aa" if c > 60 else "#ffa500" if c > 40 else "#ff4444" for c in composites]
    fig = go.Figure(go.Bar(x=tickers, y=composites, marker_color=colors))
    fig.update_layout(height=300, margin=dict(t=20, b=20), yaxis_title="Composite (0-100)",
                      xaxis=dict(categoryorder="array", categoryarray=tickers))
    st.plotly_chart(fig, width="stretch")
