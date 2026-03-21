"""Panel 5: Debug — raw data inspector + explanations."""
import streamlit as st
import pandas as pd
import json
from engine.schemas import RegimeState


def render_debug_panel(result: dict):
    st.subheader("Debug Inspector")

    # Regime raw signals
    with st.expander("Regime Gate — Raw Signals", expanded=True):
        regime = result["regime"]
        st.json({
            "state": regime.state.value,
            "hostile_count": regime.hostile_count,
            "fragile_count": regime.fragile_count,
            "normal_count": regime.normal_count,
            "signals": [
                {"name": s.name, "raw_value": round(s.raw_value, 4), "level": s.level.value}
                for s in regime.signals
            ],
        })
        st.markdown(
            f"<div style='font-size: 1.1em; line-height: 1.5;'>{regime.explanation}</div>",
            unsafe_allow_html=True,
        )

    # RS raw values
    with st.expander("RS Scanner — All Readings"):
        rs_data = []
        for r in sorted(result["rs_readings"], key=lambda x: x.rs_rank):
            rs_data.append({
                "rank": r.rs_rank,
                "ticker": r.ticker,
                "name": r.name,
                "rs_5d": round(r.rs_5d, 6),
                "rs_20d": round(r.rs_20d, 6),
                "rs_60d": round(r.rs_60d, 6),
                "rs_slope": round(r.rs_slope, 6),
                "rs_rank_change": r.rs_rank_change,
                "rs_composite": round(r.rs_composite, 2),
            })
        st.dataframe(pd.DataFrame(rs_data), width="stretch", hide_index=True)

    # Breadth raw
    with st.expander("Breadth — Raw Reading"):
        br = result["breadth"]
        st.json({
            "rsp_spy_ratio": round(br.rsp_spy_ratio, 6),
            "rsp_spy_ratio_20d_change": round(br.rsp_spy_ratio_20d_change, 6),
            "rsp_spy_ratio_zscore": round(br.rsp_spy_ratio_zscore, 4) if not __import__("math").isnan(br.rsp_spy_ratio_zscore) else "NaN",
            "signal": br.signal.value,
        })
        st.text(br.explanation)

    # Pump Scores
    with st.expander("Pump Scores — All Sectors"):
        pump_data = []
        for ticker in sorted(result["pumps"].keys()):
            p = result["pumps"][ticker]
            pump_data.append({
                "ticker": p.ticker,
                "name": p.name,
                "rs_pillar": round(p.rs_pillar, 2),
                "participation_pillar": round(p.participation_pillar, 2),
                "flow_pillar": round(p.flow_pillar, 2),
                "pump_score": round(p.pump_score, 4),
                "pump_delta": round(p.pump_delta, 4),
                "pump_delta_5d_avg": round(p.pump_delta_5d_avg, 4),
            })
        st.dataframe(pd.DataFrame(pump_data), width="stretch", hide_index=True)

    # State Classifications + Explanations
    with st.expander("State Classifications — With Explanations", expanded=True):
        for ticker in sorted(result["states"].keys()):
            sc = result["states"][ticker]
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{sc.ticker} ({sc.name})** — {sc.state.value}")
                st.caption(f"Confidence: {sc.confidence}% | Sessions: {sc.sessions_in_state} | Pressure: {sc.transition_pressure.value}")
            with col2:
                st.markdown(
                    f"<div style='font-size: 1.1em; line-height: 1.5;'>{sc.explanation}</div>",
                    unsafe_allow_html=True,
                )
            st.divider()

    # Snapshot status
    with st.expander("Snapshot Status"):
        from data.snapshots import list_snapshots
        available = list_snapshots()
        st.write(f"Available snapshots: **{len(available)}**")
        if available:
            st.write(f"Date range: **{available[0]}** to **{available[-1]}**")
        else:
            st.warning("No snapshots saved yet. Run `python scripts/backfill.py` first.")

    # Raw price data sample
    with st.expander("Raw Price Data (last 5 rows)"):
        st.dataframe(result["prices"].tail(), width="stretch")

    # VIX data
    with st.expander("VIX Data (last 10)"):
        vix_df = pd.DataFrame({
            "VIX": result["vix"].tail(10),
            "VIX3M": result["vix3m"].tail(10),
        })
        if not vix_df.empty:
            vix_df["Ratio"] = vix_df["VIX"] / vix_df["VIX3M"]
            st.dataframe(vix_df, width="stretch")

    # ── State & Signal Reference ──────────────────────
    with st.expander("State & Signal Reference (exhaustive)", expanded=False):
        _render_state_reference()


def _render_state_reference():
    """Exhaustive reference for every state, signal, and color."""
    import pandas as pd
    from dashboard.components.style_utils import STATE_COLORS, STATE_BAR_COLORS, color_row_by_state

    st.markdown("### 5-State Model")
    st.markdown(
        "**Spectrum:** Overt Dump (deep red) → Exhaustion (light red) → "
        "Ambiguous (none) → Accumulation (light green) → Overt Pump (deep green)"
    )

    states = [
        {
            "State": "Overt Pump",
            "Color": "Deep Green",
            "Trigger": "Pump percentile ≥ 75, RS rank ≤ 3, delta positive",
            "Meaning": "Strongest institutional inflow. Clear sector leadership. Top quartile momentum.",
            "Action": "Primary long candidate. Full position size.",
            "Example": "XLE at rank #1, pump score 0.85, delta +0.04 for 8 sessions",
        },
        {
            "State": "Accumulation",
            "Color": "Light Green",
            "Trigger": "Pump delta positive (building momentum)",
            "Meaning": "Early-stage momentum building. Participation expanding. Not yet confirmed as leader.",
            "Action": "Watch for continuation. Potential add on breakout.",
            "Example": "XLV delta turning positive, climbing from rank #6 to #4",
        },
        {
            "State": "Ambiguous",
            "Color": "None (transparent)",
            "Trigger": "Mixed deltas (2+ positive AND 2+ negative in last 5 sessions)",
            "Meaning": "Conflicting signals. No clear direction. Flip-flopping momentum.",
            "Action": "No trade. Wait for clarity. Do not force a signal.",
            "Example": "XLI delta alternating +0.02, -0.01, +0.01, -0.02, +0.005",
        },
        {
            "State": "Exhaustion",
            "Color": "Light Red",
            "Trigger": "Was Overt Pump or Accumulation, then delta nonpositive for 3+ sessions",
            "Meaning": "Former leader losing momentum. Pump fading. Watch for rotation.",
            "Action": "Tighten stops. Reduce position. Look for rotation target.",
            "Example": "XLK was rank #1, pump delta negative for 4 sessions, RS slope turning down",
        },
        {
            "State": "Overt Dump",
            "Color": "Deep Red",
            "Trigger": "Exhaustion + reversal score > 75th percentile, OR continued decline + rank ≥ 7",
            "Meaning": "Active capital rotation OUT. Failed breakouts. Institutional selling confirmed.",
            "Action": "Exit longs. Potential short candidate. Pair against Overt Pump.",
            "Example": "XLB rank #11, pump delta -0.06, reversal score 0.72 (85th percentile)",
        },
    ]

    df = pd.DataFrame(states)
    styled = df.style.apply(color_row_by_state, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Regime Gate Signals")
    signals = [
        {"Signal": "VIX Level", "NORMAL": "< 20", "FRAGILE": "20–30", "HOSTILE": "≥ 30",
         "Source": "^VIX via yfinance (live)"},
        {"Signal": "Term Structure", "NORMAL": "< 0.95 (contango)", "FRAGILE": "0.95–1.05 (flat)", "HOSTILE": "≥ 1.05 (backwardation)",
         "Source": "^VIX / ^VIX3M (live)"},
        {"Signal": "Breadth", "NORMAL": "z > 0", "FRAGILE": "0 ≥ z > -1", "HOSTILE": "z ≤ -1",
         "Source": "RSP/SPY ratio z-score (live)"},
        {"Signal": "Credit", "NORMAL": "z > -0.5", "FRAGILE": "-0.5 ≥ z > -1.5", "HOSTILE": "z ≤ -1.5",
         "Source": "HYG/LQD ratio z-score (live), FRED OAS (1-2d lag)"},
    ]
    st.dataframe(pd.DataFrame(signals), width="stretch", hide_index=True)

    st.markdown("### Gate Aggregation")
    st.markdown(
        "- **HOSTILE**: ≥ 2 hostile signals\n"
        "- **FRAGILE**: ≥ 1 hostile OR ≥ 2 fragile signals\n"
        "- **NORMAL**: everything else\n"
        "- Boundary rule: exact threshold value goes to the WORSE bucket"
    )

    st.markdown("---")
    st.markdown("### Reversal Score Sub-Signals")
    rev_signals = [
        {"Pillar": "Breadth Deterioration (40%)", "Sub-Signal": "RS Slope Reversal",
         "What It Measures": "Was RS slope positive → now negative?", "Range": "0–100"},
        {"Pillar": "Breadth Deterioration (40%)", "Sub-Signal": "Participation Decay",
         "What It Measures": "% of days outperforming SPY (declining = bad)", "Range": "0–1"},
        {"Pillar": "Price Break Quality (30%)", "Sub-Signal": "Failed Breakout Rate",
         "What It Measures": "New 20d highs that reversed within 3 sessions", "Range": "0–1"},
        {"Pillar": "Price Break Quality (30%)", "Sub-Signal": "Gap Fade Rate",
         "What It Measures": "Gap-ups that closed below open (faded)", "Range": "0–1"},
        {"Pillar": "Price Break Quality (30%)", "Sub-Signal": "CLV Trend",
         "What It Measures": "Close Location Value: (close-low)/(high-low). Declining = bad", "Range": "0–1"},
        {"Pillar": "Price Break Quality (30%)", "Sub-Signal": "Follow-Through",
         "What It Measures": "% of up-days followed by another up-day", "Range": "0–1"},
        {"Pillar": "Crowding/Stretch (30%)", "Sub-Signal": "Distance from MA",
         "What It Measures": "Standard deviations above/below 20d MA", "Range": "σ"},
        {"Pillar": "Crowding/Stretch (30%)", "Sub-Signal": "RVOL",
         "What It Measures": "Current volume / 20d average volume", "Range": "x"},
        {"Pillar": "Crowding/Stretch (30%)", "Sub-Signal": "Price Acceleration",
         "What It Measures": "5d rate of change of 20d rate of change (parabolic detection)", "Range": "ROC"},
    ]
    st.dataframe(pd.DataFrame(rev_signals), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Transition Pressures")
    pressures = [
        {"Pressure": "UP", "Condition": "Delta > 0.005 for 3+ consecutive sessions"},
        {"Pressure": "STABLE", "Condition": "Delta near zero or mixed direction"},
        {"Pressure": "DOWN", "Condition": "Delta < -0.005 for 3+ consecutive sessions"},
        {"Pressure": "BREAK", "Condition": "State changed this session"},
    ]
    st.dataframe(pd.DataFrame(pressures), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Confidence Adjustments")
    adjustments = [
        {"Factor": "Strong aligned signals (top 3 rank + top quartile pump)", "Effect": "+15"},
        {"Factor": "Consistent delta direction (3+ sessions same sign)", "Effect": "+5 to +10"},
        {"Factor": "Conflicting pillars (spread > 50)", "Effect": "-15"},
        {"Factor": "Ambiguous state", "Effect": "-15"},
        {"Factor": "Reversal high + pump rising (conflicting)", "Effect": "-15"},
        {"Factor": "Reversal low + pump rising (confirming)", "Effect": "+5"},
        {"Factor": "FRAGILE regime", "Effect": "-20"},
        {"Factor": "HOSTILE regime", "Effect": "-30"},
        {"Factor": "Clamped range", "Effect": "Always 10–95"},
    ]
    st.dataframe(pd.DataFrame(adjustments), width="stretch", hide_index=True)
