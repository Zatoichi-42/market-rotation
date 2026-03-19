"""Panel 5: Historical replay — browse past snapshots and see forward returns."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data.snapshots import list_snapshots, load_snapshot

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
}


def render_replay_panel(result: dict):
    available = list_snapshots()

    if not available:
        st.warning(
            "No historical snapshots found. Run the backfill first:\n\n"
            "```bash\npython scripts/backfill.py\n```"
        )
        return

    st.subheader("Historical Replay")

    # Date slider
    date_options = available
    selected_idx = st.slider(
        "Select date",
        min_value=0, max_value=len(date_options) - 1,
        value=len(date_options) - 1,
        format=f"",
    )
    selected_date = date_options[selected_idx]
    st.markdown(f"**Selected: {selected_date}**")

    try:
        snap = load_snapshot(selected_date)
    except Exception as e:
        st.error(f"Failed to load snapshot: {e}")
        return

    # Regime comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{selected_date} Regime**")
        _render_regime_badge(snap.regime.state.value)
        st.caption(snap.regime.explanation[:200])
    with col2:
        st.markdown("**Current Regime**")
        _render_regime_badge(result["regime"].state.value)
        st.caption(result["regime"].explanation[:200])

    # Sector rankings on that date
    st.subheader(f"Sector Rankings: {selected_date}")
    sectors_sorted = sorted(snap.sectors, key=lambda s: s.rs_rank)
    rows = []
    state_map = {s.ticker: s for s in snap.states} if snap.states else {}
    pump_map = {p.ticker: p for p in snap.pump_scores} if snap.pump_scores else {}
    for s in sectors_sorted:
        state = state_map.get(s.ticker)
        pump = pump_map.get(s.ticker)
        rows.append({
            "Rank": s.rs_rank,
            "Ticker": s.ticker,
            "Sector": s.name,
            "ETF Name": _ETF_FULL_NAMES.get(s.ticker, s.name),
            "RS 20d": f"{s.rs_20d:+.2%}",
            "Composite": f"{s.rs_composite:.1f}",
            "Pump": f"{pump.pump_score:.2f}" if pump else "N/A",
            "State": state.state.value if state else "N/A",
            "Confidence": f"{state.confidence}%" if state else "N/A",
        })
    from dashboard.components.style_utils import color_row_by_state
    df = pd.DataFrame(rows)
    styled = df.style.apply(color_row_by_state, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    # Forward returns
    prices = result["prices"]
    if selected_date in prices.index.strftime("%Y-%m-%d").values:
        st.subheader(f"What happened AFTER {selected_date}?")
        date_loc = prices.index.get_indexer(pd.to_datetime([selected_date]), method="nearest")[0]

        for fwd_days, label in [(10, "10-day"), (20, "20-day")]:
            fwd_loc = min(date_loc + fwd_days, len(prices) - 1)
            if fwd_loc <= date_loc:
                continue

            spy_ret = (prices["SPY"].iloc[fwd_loc] / prices["SPY"].iloc[date_loc]) - 1 if "SPY" in prices.columns else 0
            st.markdown(f"**{label} forward** (SPY: {spy_ret:+.2%})")

            fwd_rows = []
            for s in sectors_sorted[:5]:  # Top 5 on that date
                if s.ticker in prices.columns:
                    sec_ret = (prices[s.ticker].iloc[fwd_loc] / prices[s.ticker].iloc[date_loc]) - 1
                    excess = sec_ret - spy_ret
                    fwd_rows.append({
                        "Rank (then)": s.rs_rank,
                        "Ticker": f"{s.ticker} ({s.name})",
                        f"{label} Return": f"{sec_ret:+.2%}",
                        f"vs SPY": f"{excess:+.2%}",
                    })
            if fwd_rows:
                st.dataframe(pd.DataFrame(fwd_rows), width="stretch", hide_index=True)
    else:
        st.info("Forward returns unavailable — selected date not in current price data range.")


def _render_regime_badge(state_value: str):
    colors = {"NORMAL": "#00d4aa", "FRAGILE": "#ffa500", "HOSTILE": "#ff4444"}
    color = colors.get(state_value, "#888")
    st.markdown(
        f"<span style='background: {color}; color: black; padding: 4px 12px; "
        f"border-radius: 4px; font-weight: bold; font-size: 1.2em;'>{state_value}</span>",
        unsafe_allow_html=True,
    )
