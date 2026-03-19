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
                st.markdown(f"**{sc.ticker}** — {sc.state.value}")
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
