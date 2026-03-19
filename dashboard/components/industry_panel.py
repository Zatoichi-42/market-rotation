"""Panel 6: Industry rotation map — RS vs SPY and vs parent, drilldown by sector."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from engine.schemas import IndustryRSReading


_GLOSSARY = {
    "rs_vs_parent": {
        "title": "RS vs Parent Sector",
        "body": (
            "Measures whether an industry is outperforming or underperforming its own "
            "parent sector ETF. Positive = the industry is **driving** the sector. "
            "Negative = the industry is **freeriding** on sector beta or lagging.\n\n"
            "This is the decisive Phase 2 signal: an industry that outperforms both "
            "SPY and its parent sector is a true alpha generator."
        ),
    },
    "industry_composite": {
        "title": "Industry Composite Score",
        "body": (
            "Blend of RS-vs-SPY (70% weight) and RS-vs-parent (30% weight), "
            "each computed as a percentile rank across all industries.\n\n"
            "A score of 100 means top-ranked at every timeframe against both "
            "the market and its own sector. This is the primary ranking signal."
        ),
    },
}


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

    st.subheader("Industry Rotation Map")
    g1, g2 = st.columns([1, 1])
    with g1:
        _popover("rs_vs_parent")
    with g2:
        _popover("industry_composite")

    # Filter by parent sector
    parents = sorted(set(r.parent_sector for r in industry_rs))
    filter_options = ["All"] + parents
    selected = st.selectbox("Filter by parent sector", filter_options)

    if selected != "All":
        display = [r for r in industry_rs if r.parent_sector == selected]
    else:
        display = industry_rs

    display_sorted = sorted(display, key=lambda r: r.rs_rank)

    # Table
    rows = []
    for r in display_sorted:
        vs_parent_icon = "▲" if r.rs_20d_vs_parent > 0.001 else ("▼" if r.rs_20d_vs_parent < -0.001 else "—")
        rows.append({
            "Rank": r.rs_rank,
            "Industry": f"{r.ticker} ({r.name})",
            "Parent": r.parent_sector,
            "RS vs SPY (20d)": f"{r.rs_20d:+.2%}",
            "RS vs Parent (20d)": f"{vs_parent_icon} {r.rs_20d_vs_parent:+.2%}",
            "Slope": f"{r.rs_slope:+.4f}",
            "Ind. Composite": f"{r.industry_composite:.1f}",
            "Within Sector": f"#{r.rs_rank_within_sector}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Sector → Industry drilldown
    st.subheader("Sector → Industry Drilldown")
    sector_states = {s.ticker: s for s in result.get("states", {}).values()} if isinstance(result.get("states"), dict) else {}
    pumps = result.get("pumps", {})

    for parent in parents:
        children = [r for r in industry_rs if r.parent_sector == parent]
        children_sorted = sorted(children, key=lambda r: r.rs_rank_within_sector)

        # Sector header
        sector_pump = pumps.get(parent)
        sector_state = sector_states.get(parent)
        pump_str = f"Pump: {sector_pump.pump_score:.2f}" if sector_pump else ""
        state_str = f"State: {sector_state.state.value}" if sector_state else ""
        header = f"**{parent}** — {pump_str} {state_str}"
        st.markdown(header)

        for c in children_sorted:
            if c.rs_20d_vs_parent > 0.001:
                icon = "🟢"
                label = "Driving sector"
            elif c.rs_20d_vs_parent < -0.001:
                icon = "🔴"
                label = "Lagging"
            else:
                icon = "⚪"
                label = "Neutral"
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;{icon} **{c.ticker}** ({c.name}) — "
                f"RS vs {parent}: {c.rs_20d_vs_parent:+.2%} — {label}"
            )
        st.divider()
