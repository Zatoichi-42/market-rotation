"""
Panel 7: Today's Interpretation — 1-day analysis of market moves,
rotation signals, regime context, and actionable observations.

Designed for days with big moves — surfaces what the system sees
and how it interprets the data for validation.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from engine.schemas import AnalysisState

_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}

# Deep red → red → yellow → green → deep green (overt dump to overt pump)
_MOMENTUM_COLORS = {
    "Rotation/Reversal": "#7f1d1d",  # Deep red
    "Exhaustion":        "#ef4444",  # Red
    "Ambiguous":         "#eab308",  # Yellow
    "Accumulation":      "#a3e635",  # Light green
    "Broadening":        "#22c55e",  # Green
    "Overt Pump":        "#064e3b",  # Deep green
}

_STATE_ORDER = ["Rotation/Reversal", "Exhaustion", "Ambiguous", "Accumulation", "Broadening", "Overt Pump"]


def render_interpretation_panel(result: dict):
    prices = result["prices"]
    rs_readings = result.get("rs_readings", [])
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    regime = result["regime"]
    industry_rs = result.get("industry_rs", [])

    if len(prices) < 2:
        st.warning("Insufficient data for interpretation.")
        return

    st.subheader("Today's Market Interpretation")
    last_date = prices.index[-1]
    st.caption(f"Using latest prices as of **{last_date.strftime('%Y-%m-%d')}** "
               f"(intraday during market hours, close after hours)")

    # ── Market Summary ────────────────────────────────
    # 1d = today's latest vs yesterday's close. All metrics use today as endpoint.
    spy_1d = prices["SPY"].pct_change().iloc[-1] if "SPY" in prices.columns else 0
    spy_5d = prices["SPY"].pct_change(5).iloc[-1] if "SPY" in prices.columns and len(prices) > 5 else 0

    move_desc = "sharp sell-off" if spy_1d < -0.01 else "sell-off" if spy_1d < -0.005 else "decline" if spy_1d < -0.001 else "flat" if abs(spy_1d) < 0.001 else "rally" if spy_1d < 0.01 else "strong rally"

    st.markdown(
        f"<div style='font-size:1.2em; padding:12px; background:rgba(0,0,0,0.3); border-radius:8px;'>"
        f"SPY: <strong>{spy_1d:+.2%}</strong> today ({move_desc}), "
        f"<strong>{spy_5d:+.2%}</strong> over 5 days. "
        f"Regime: <strong>{regime.state.value}</strong>. "
        f"VIX: <strong>{result['vix_val']:.1f}</strong>."
        f"</div>", unsafe_allow_html=True,
    )

    # ── Sector 1d Waterfall ───────────────────────────
    st.subheader("Sector 1d RS Waterfall")
    st.caption("Green = outperformed SPY today, Red = underperformed. Size = magnitude.")

    sector_1d = []
    for r in rs_readings:
        if r.ticker in prices.columns:
            sec_1d = prices[r.ticker].pct_change().iloc[-1]
            rs_1d = sec_1d - spy_1d
            state = states.get(r.ticker)
            sector_1d.append({
                "ticker": r.ticker, "name": _SECTOR_NAMES.get(r.ticker, r.ticker),
                "return_1d": sec_1d, "rs_1d": rs_1d,
                "state": state.state.value if state else "—",
                "pump_delta": pumps[r.ticker].pump_delta if r.ticker in pumps else 0,
            })

    sector_1d.sort(key=lambda x: x["rs_1d"], reverse=True)
    labels = [f"{s['ticker']} ({s['name']})" for s in sector_1d]
    rs_vals = [s["rs_1d"] * 100 for s in sector_1d]
    colors = [_MOMENTUM_COLORS.get(s["state"], "#94a3b8") for s in sector_1d]

    fig = go.Figure(go.Bar(
        y=labels, x=rs_vals, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in rs_vals],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_color="#555", line_width=2)
    fig.update_layout(
        height=max(300, len(labels) * 32), margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(title="1d RS vs SPY (%)", showgrid=False),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Momentum Spectrum ─────────────────────────────
    st.subheader("Momentum Spectrum: Overt Dump ← → Overt Pump")
    st.caption("Deep red = rotating out hard. Deep green = strongest pump. Position on spectrum shows conviction.")

    spectrum_rows = []
    for s in sector_1d:
        state = s["state"]
        idx = _STATE_ORDER.index(state) if state in _STATE_ORDER else 2
        score = idx / (len(_STATE_ORDER) - 1)  # 0.0 = dump, 1.0 = pump
        spectrum_rows.append({
            "Sector": f"{s['ticker']} ({s['name']})",
            "State": state,
            "1d RS": f"{s['rs_1d']:+.2%}",
            "Pump Δ": f"{s['pump_delta']:+.3f}",
            "Spectrum": score,
        })

    # Gradient bar for each sector
    for row in sorted(spectrum_rows, key=lambda x: x["Spectrum"], reverse=True):
        pct = row["Spectrum"] * 100
        # Gradient from deep red to deep green
        if pct < 20:
            bar_color = "#7f1d1d"
        elif pct < 40:
            bar_color = "#ef4444"
        elif pct < 60:
            bar_color = "#eab308"
        elif pct < 80:
            bar_color = "#22c55e"
        else:
            bar_color = "#064e3b"

        st.markdown(
            f"<div style='display:flex;align-items:center;margin:2px 0;'>"
            f"<div style='width:180px;font-size:0.85em;'>{row['Sector']}</div>"
            f"<div style='flex:1;background:#1e293b;border-radius:4px;height:20px;position:relative;'>"
            f"<div style='width:{max(5, pct)}%;background:{bar_color};height:100%;border-radius:4px;'></div>"
            f"</div>"
            f"<div style='width:100px;text-align:right;font-size:0.85em;'>{row['State']}</div>"
            f"</div>", unsafe_allow_html=True,
        )

    # ── Industry Leaders/Laggards Today ───────────────
    if industry_rs:
        st.subheader("Industry 1d Leaders & Laggards")
        ind_1d = []
        for ir in industry_rs:
            if ir.ticker in prices.columns:
                sec_1d = prices[ir.ticker].pct_change().iloc[-1]
                rs_1d = sec_1d - spy_1d
                ind_1d.append({
                    "ticker": ir.ticker, "name": ir.name,
                    "parent": _SECTOR_NAMES.get(ir.parent_sector, ir.parent_sector),
                    "rs_1d": rs_1d, "return_1d": sec_1d,
                })
        ind_1d.sort(key=lambda x: x["rs_1d"], reverse=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 5 Industries (1d RS)**")
            for d in ind_1d[:5]:
                color = "#22c55e"
                st.markdown(
                    f"<span style='color:{color}'>▲</span> "
                    f"**{d['ticker']}** ({d['name']}, {d['parent']}) "
                    f"{d['rs_1d']:+.2%} RS, {d['return_1d']:+.2%} abs",
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown("**Bottom 5 Industries (1d RS)**")
            for d in ind_1d[-5:]:
                color = "#ef4444"
                st.markdown(
                    f"<span style='color:{color}'>▼</span> "
                    f"**{d['ticker']}** ({d['name']}, {d['parent']}) "
                    f"{d['rs_1d']:+.2%} RS, {d['return_1d']:+.2%} abs",
                    unsafe_allow_html=True,
                )

    # ── Key Observations ──────────────────────────────
    st.subheader("System Observations")

    observations = []

    # Regime change risk
    if regime.state.value == "FRAGILE":
        observations.append("⚠ Regime is FRAGILE — rotation signals should be acted on cautiously.")
    elif regime.state.value == "HOSTILE":
        observations.append("🔴 Regime is HOSTILE — all momentum longs overridden. Defensive only.")

    # Big single-day RS moves
    for s in sector_1d:
        if abs(s["rs_1d"]) > 0.015:
            direction = "outperformed" if s["rs_1d"] > 0 else "underperformed"
            observations.append(
                f"{'🟢' if s['rs_1d'] > 0 else '🔴'} **{s['ticker']} ({s['name']})** "
                f"{direction} SPY by {abs(s['rs_1d']):.2%} in one day — extreme RS move."
            )

    # State + delta alignment
    for s in sector_1d:
        state = s["state"]
        delta = s["pump_delta"]
        if state == "Overt Pump" and delta < -0.005:
            observations.append(
                f"⚠ **{s['ticker']}** is Overt Pump but delta turned negative ({delta:+.3f}) — watch for exhaustion."
            )
        if state == "Exhaustion" and s["rs_1d"] > 0.01:
            observations.append(
                f"👀 **{s['ticker']}** is in Exhaustion but had a strong 1d RS ({s['rs_1d']:+.2%}) — dead cat bounce or reversal?"
            )
        if state == "Accumulation" and s["rs_1d"] > 0.01:
            observations.append(
                f"🟢 **{s['ticker']}** is Accumulation with strong 1d RS ({s['rs_1d']:+.2%}) — potential broadening signal."
            )

    # Breadth signal
    breadth = result.get("breadth")
    if breadth and breadth.signal.value == "DIVERGING":
        observations.append("⚠ Breadth is DIVERGING — narrow market leadership. Rally is fragile.")

    if not observations:
        observations.append("No extreme signals detected today. Normal rotation dynamics.")

    for obs in observations:
        st.markdown(obs)
