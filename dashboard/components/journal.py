"""
Trade Journal dashboard panel — equity curve, hit rates, open/closed calls.
"""
import streamlit as st
from engine.schemas import JournalSummary, TradeCall


def render_journal_panel(result: dict):
    """Render the trade journal panel."""
    journal_calls = result.get("journal_calls", [])
    journal_summary = result.get("journal_summary")

    if not journal_calls and journal_summary is None:
        st.info("No trade journal data yet. Calls are generated when the system produces actionable signals.")
        return

    st.subheader("Trade Journal — Running P&L")

    if journal_summary:
        _render_summary(journal_summary, journal_calls)
    _render_open_calls(journal_calls, result)
    _render_recent_closed(journal_calls)


def _render_summary(summary: JournalSummary, journal_calls: list = None):
    """Render aggregate stats."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", summary.total_calls)
    with col2:
        st.metric("Open", summary.open_calls)
    with col3:
        st.metric("Closed", summary.closed_calls)
    with col4:
        pnl_color = "#16a34a" if summary.total_pnl_10d >= 0 else "#dc2626"
        st.markdown(
            f"**P&L (10d)**<br>"
            f"<span style='color:{pnl_color};font-size:1.3em;'>"
            f"{summary.total_pnl_10d:+,.0f}</span>",
            unsafe_allow_html=True,
        )

    # Hit rates — check if any calls have settled 10d data
    has_settled_10d = any(
        (c.hit_10d if hasattr(c, 'hit_10d') else c.get("hit_10d")) is not None
        for c in journal_calls
    ) if journal_calls else False
    has_settled_20d = any(
        (c.hit_20d if hasattr(c, 'hit_20d') else c.get("hit_20d")) is not None
        for c in journal_calls
    ) if journal_calls else False

    col1, col2 = st.columns(2)
    with col1:
        if has_settled_10d:
            hr10 = summary.hit_rate_10d * 100
            st.markdown(f"**Hit Rate 10d:** {hr10:.1f}%"
                        f" {'✅' if hr10 > 55 else '⚠️' if hr10 > 45 else '❌'}")
        else:
            st.markdown("**Hit Rate 10d:** No settled calls yet")
    with col2:
        if has_settled_20d:
            hr20 = summary.hit_rate_20d * 100
            st.markdown(f"**Hit Rate 20d:** {hr20:.1f}%"
                        f" {'✅' if hr20 > 55 else '⚠️' if hr20 > 45 else '❌'}")
        else:
            st.markdown("**Hit Rate 20d:** No settled calls yet")

    # Breakdown by state
    if summary.pnl_by_state:
        st.markdown("**By Analysis State:**")
        rows = []
        for state, pnl in sorted(summary.pnl_by_state.items(), key=lambda x: x[1], reverse=True):
            hr = summary.hit_rate_by_state.get(state, 0) * 100
            rows.append(f"- {state}: {pnl:+,.0f} ({hr:.0f}% hit)")
        st.markdown("\n".join(rows))

    # Breakdown by regime character
    if summary.pnl_by_regime:
        st.markdown("**By Regime Character:**")
        rows = []
        for regime, pnl in sorted(summary.pnl_by_regime.items(), key=lambda x: x[1], reverse=True):
            hr = summary.hit_rate_by_regime.get(regime, 0) * 100
            rows.append(f"- {regime}: {pnl:+,.0f} ({hr:.0f}% hit)")
        st.markdown("\n".join(rows))

    # Breakdown by horizon pattern
    if summary.pnl_by_pattern:
        st.markdown("**By Horizon Pattern:**")
        rows = []
        for pat, pnl in sorted(summary.pnl_by_pattern.items(), key=lambda x: x[1], reverse=True):
            hr = summary.hit_rate_by_pattern.get(pat, 0) * 100
            rows.append(f"- {pat}: {pnl:+,.0f} ({hr:.0f}% hit)")
        st.markdown("\n".join(rows))

    # Cumulative P&L sparkline (text-based)
    if summary.cumulative_pnl and len(summary.cumulative_pnl) > 1:
        st.markdown("**Cumulative P&L Curve:**")
        import pandas as pd
        curve_data = pd.DataFrame(summary.cumulative_pnl, columns=["date", "pnl"])
        curve_data["date"] = pd.to_datetime(curve_data["date"])
        st.line_chart(curve_data.set_index("date")["pnl"], height=200)


def _render_open_calls(calls: list, result: dict):
    """Show open calls with current status."""
    open_calls = [c for c in calls if (c.status if hasattr(c, 'status') else c.get("status")) == "open"]
    if not open_calls:
        return

    st.markdown("---")
    st.markdown(f"**Open Calls ({len(open_calls)}):**")

    for c in open_calls:
        ticker = c.ticker if hasattr(c, 'ticker') else c.get("ticker", "?")
        name = c.name if hasattr(c, 'name') else c.get("name", "")
        label = f"{ticker} ({name})" if name else ticker
        target = c.target_pct if hasattr(c, 'target_pct') else c.get("target_pct", 0)
        conf = c.confidence if hasattr(c, 'confidence') else c.get("confidence", 0)
        date = c.date if hasattr(c, 'date') else c.get("date", "?")
        entry_price = c.entry_price if hasattr(c, 'entry_price') else c.get("entry_price", 0)
        state = c.analysis_state if hasattr(c, 'analysis_state') else c.get("analysis_state", "?")
        pattern = c.horizon_pattern if hasattr(c, 'horizon_pattern') else c.get("horizon_pattern", "?")

        direction = "Long" if target > 0 else "Short" if target < 0 else "Flat"

        st.markdown(
            f"**{label}** {direction} {target:+d}% | "
            f"Entry ${entry_price:.2f} on {date} | "
            f"State: {state} | Pattern: {pattern} | "
            f"Conf: {conf}",
        )


def _render_recent_closed(calls: list):
    """Show recently closed calls."""
    closed = [c for c in calls if (c.status if hasattr(c, 'status') else c.get("status")) != "open"]
    if not closed:
        return

    # Show last 10
    recent = closed[-10:]
    st.markdown("---")
    st.markdown(f"**Recent Closed Calls ({len(closed)} total):**")

    for c in recent:
        ticker = c.ticker if hasattr(c, 'ticker') else c.get("ticker", "?")
        name = c.name if hasattr(c, 'name') else c.get("name", "")
        label = f"{ticker} ({name})" if name else ticker
        reason = c.close_reason if hasattr(c, 'close_reason') else c.get("close_reason", "?")
        pnl = c.pnl_10d if hasattr(c, 'pnl_10d') else c.get("pnl_10d")
        target = c.target_pct if hasattr(c, 'target_pct') else c.get("target_pct", 0)

        pnl_str = f"{pnl:+,.0f}" if pnl is not None else "pending"
        direction = "Long" if target > 0 else "Short" if target < 0 else "Flat"

        st.caption(f"{label} {direction} {target:+d}% | P&L(10d): {pnl_str} | Reason: {reason}")
