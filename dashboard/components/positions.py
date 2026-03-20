"""Open positions panel — shows tracked positions with entry vs current comparisons."""
import streamlit as st

def render_positions_panel(result: dict):
    """Render open positions status."""
    positions = result.get("open_positions", [])
    if not positions:
        st.info("No open positions tracked.")
        return

    st.subheader("Open Positions")
    for pos in positions:
        ticker = pos.get("ticker", "?")
        name = pos.get("name", ticker)
        sessions = pos.get("sessions_held", 0)
        expected = pos.get("expected_hold_sessions", 0)
        entry_price = pos.get("entry_price", 0)
        current_price = pos.get("current_price", entry_price)
        pnl = ((current_price / entry_price - 1) * 100) if entry_price > 0 else 0

        entry_state = pos.get("entry_analysis_state", "\u2014")
        current_state = result.get("states", {}).get(ticker)
        current_state_str = current_state.state.value if current_state else "\u2014"

        horizon = result.get("horizon_readings", {}).get(ticker)
        horizon_str = horizon.pattern.value if horizon else "\u2014"

        pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"

        st.markdown(
            f"**{ticker} {name}** \u2014 day {sessions} of ~{expected} expected  \n"
            f"Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | "
            f'<span style="color:{pnl_color}">PnL: {pnl:+.1f}%</span>  \n'
            f"Entry state: {entry_state} \u2192 Current: {current_state_str}  \n"
            f"Horizon: {horizon_str}",
            unsafe_allow_html=True,
        )

        # Exit assessment
        ea = result.get("exit_assessments", {}).get(ticker)
        if ea and ea.signals:
            st.warning(f"Exit signals: {ea.recommendation} ({len(ea.signals)} active)")

        st.divider()
