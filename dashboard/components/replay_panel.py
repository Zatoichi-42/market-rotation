"""
Panel 5: History Replay — date navigation with ◄/► buttons, regime comparison,
sector rankings snapshot, forward returns.
"""
import streamlit as st
import pandas as pd
from data.snapshots import list_snapshots, load_snapshot
from dashboard.components.style_utils import color_row_by_state

_ETF_FULL_NAMES = {
    "XLK": "Technology Select Sector SPDR", "XLV": "Health Care Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR", "XLE": "Energy Select Sector SPDR",
    "XLI": "Industrial Select Sector SPDR", "XLU": "Utilities Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR", "XLC": "Communication Services Select Sector SPDR",
    "XLY": "Consumer Discretionary Select Sector SPDR", "XLP": "Consumer Staples Select Sector SPDR",
    "XLB": "Materials Select Sector SPDR",
}


def _regime_badge(state_value: str):
    colors = {"NORMAL": "#22c55e", "FRAGILE": "#eab308", "HOSTILE": "#ef4444"}
    color = colors.get(state_value, "#888")
    st.markdown(
        f"<span style='background: {color}; color: black; padding: 4px 12px; "
        f"border-radius: 4px; font-weight: bold; font-size: 1.2em;'>{state_value}</span>",
        unsafe_allow_html=True,
    )


def render_replay_panel(result: dict):
    available = list_snapshots()
    if not available:
        st.warning("No historical snapshots. Run `python scripts/backfill.py` first.")
        return

    st.subheader("History Replay")

    # ── Date navigation with ◄ slider ► ──────────────
    if "replay_idx" not in st.session_state:
        st.session_state.replay_idx = len(available) - 1

    nav_left, nav_slider, nav_right = st.columns([1, 8, 1])
    with nav_left:
        if st.button("◄", key="replay_prev", help="Previous trading day"):
            st.session_state.replay_idx = max(0, st.session_state.replay_idx - 1)
    with nav_right:
        if st.button("►", key="replay_next", help="Next trading day"):
            st.session_state.replay_idx = min(len(available) - 1, st.session_state.replay_idx + 1)
    with nav_slider:
        selected_idx = st.slider(
            "Date", min_value=0, max_value=len(available) - 1,
            value=st.session_state.replay_idx, format="",
            key="replay_slider",
        )
        st.session_state.replay_idx = selected_idx

    selected_date = available[st.session_state.replay_idx]
    st.markdown(f"**Selected: {selected_date}** &nbsp;&nbsp; "
                f"({available[0]} to {available[-1]}, {len(available)} snapshots)")

    try:
        snap = load_snapshot(selected_date)
    except Exception as e:
        st.error(f"Failed to load snapshot: {e}")
        return

    # ── Regime Comparison: then vs now ────────────────
    st.subheader("Regime Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{selected_date}**")
        _regime_badge(snap.regime.state.value)
        # Show signal details
        for sig in snap.regime.signals:
            st.caption(f"{sig.name}: {sig.raw_value:.2f} [{sig.level.value}]")
    with col2:
        st.markdown("**Current**")
        _regime_badge(result["regime"].state.value)
        for sig in result["regime"].signals:
            st.caption(f"{sig.name}: {sig.raw_value:.2f} [{sig.level.value}]")

    # ── Sector Rankings on that date ──────────────────
    st.subheader(f"Sector Rankings: {selected_date}")
    sectors_sorted = sorted(snap.sectors, key=lambda s: s.rs_rank)
    state_map = {s.ticker: s for s in snap.states} if snap.states else {}
    pump_map = {p.ticker: p for p in snap.pump_scores} if snap.pump_scores else {}

    rows = []
    for s in sectors_sorted:
        state = state_map.get(s.ticker)
        pump = pump_map.get(s.ticker)
        rows.append({
            "Rank": s.rs_rank,
            "Ticker": s.ticker,
            "Sector": s.name,
            "RS 20d": f"{s.rs_20d:+.2%}",
            "Composite": f"{s.rs_composite:.1f}",
            "Pump": f"{pump.pump_score:.2f}" if pump else "—",
            "Delta": f"{pump.pump_delta:+.3f}" if pump else "—",
            "State": state.state.value if state else "—",
            "Confidence": f"{state.confidence}%" if state else "—",
        })
    df = pd.DataFrame(rows)
    styled = df.style.apply(color_row_by_state, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── What happened AFTER this date? ────────────────
    prices = result["prices"]
    if selected_date in prices.index.strftime("%Y-%m-%d").values:
        st.subheader(f"What happened AFTER {selected_date}?")
        date_loc = prices.index.get_indexer(pd.to_datetime([selected_date]), method="nearest")[0]

        for fwd_days, label in [(10, "10-day forward"), (20, "20-day forward")]:
            fwd_loc = min(date_loc + fwd_days, len(prices) - 1)
            if fwd_loc <= date_loc:
                continue

            spy_ret = (prices["SPY"].iloc[fwd_loc] / prices["SPY"].iloc[date_loc]) - 1 if "SPY" in prices.columns else 0

            fwd_rows = []
            for s in sectors_sorted:
                if s.ticker in prices.columns:
                    sec_ret = (prices[s.ticker].iloc[fwd_loc] / prices[s.ticker].iloc[date_loc]) - 1
                    excess = sec_ret - spy_ret
                    state = state_map.get(s.ticker)
                    fwd_rows.append({
                        "Rank (then)": s.rs_rank,
                        "Ticker": f"{s.ticker} ({s.name})",
                        "State (then)": state.state.value if state else "—",
                        f"{label}": f"{sec_ret:+.2%}",
                        "vs SPY": f"{excess:+.2%}",
                    })

            if fwd_rows:
                st.markdown(f"**{label}** (SPY: {spy_ret:+.2%})")
                st.dataframe(pd.DataFrame(fwd_rows), width="stretch", hide_index=True)
    else:
        st.info("Forward returns unavailable — selected date not in current price data range.")

    # ── 1d Rolling Metrics ────────────────────────────
    st.subheader("Rolling 24h Snapshot Diff")
    prev_idx = st.session_state.replay_idx - 1
    if prev_idx >= 0:
        prev_date = available[prev_idx]
        try:
            prev_snap = load_snapshot(prev_date)
            prev_state_map = {s.ticker: s for s in prev_snap.states} if prev_snap.states else {}
            prev_pump_map = {p.ticker: p for p in prev_snap.pump_scores} if prev_snap.pump_scores else {}
            prev_sectors = {s.ticker: s for s in prev_snap.sectors}

            diff_rows = []
            for s in sectors_sorted:
                prev_s = prev_sectors.get(s.ticker)
                prev_p = prev_pump_map.get(s.ticker)
                curr_p = pump_map.get(s.ticker)
                curr_state = state_map.get(s.ticker)
                prev_state = prev_state_map.get(s.ticker)

                rs_chg = (s.rs_20d - prev_s.rs_20d) * 100 if prev_s else 0
                rank_chg = (prev_s.rs_rank - s.rs_rank) if prev_s else 0
                pump_chg = (curr_p.pump_score - prev_p.pump_score) if curr_p and prev_p else 0
                state_changed = "→" if (curr_state and prev_state and curr_state.state != prev_state.state) else ""
                prev_st = prev_state.state.value if prev_state else "—"
                curr_st = curr_state.state.value if curr_state else "—"

                diff_rows.append({
                    "Ticker": s.ticker,
                    "RS 20d Δ": f"{rs_chg:+.2f}%",
                    "Rank Δ": f"{rank_chg:+d}" if rank_chg != 0 else "—",
                    "Pump Δ": f"{pump_chg:+.4f}" if pump_chg != 0 else "—",
                    "State": f"{prev_st} {state_changed} {curr_st}" if state_changed else curr_st,
                })
            st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)
        except Exception:
            st.caption("Previous day snapshot unavailable.")
    else:
        st.caption("No prior snapshot for comparison.")
