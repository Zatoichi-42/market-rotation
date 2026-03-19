"""
Reversal Diagnostics — top 5 most fragile groups with sub-signal detail.

Shows: Rev Score, CLV, Gap Fade, Follow-Through, Stretch (distance from MA), RVOL, State
"""
import streamlit as st
import pandas as pd
import math


def render_reversal_diagnostics(result: dict):
    """Render the top-5 fragile groups table with sub-signal breakdowns."""
    reversal_scores = result.get("reversal_scores", [])
    if not reversal_scores:
        return

    states = result.get("states", {})
    state_map = states if isinstance(states, dict) else {}

    # Sort by reversal score descending, take top 5
    sorted_rev = sorted(reversal_scores, key=lambda r: r.reversal_score, reverse=True)[:5]

    if not sorted_rev or sorted_rev[0].reversal_score < 0.01:
        return

    st.subheader("Reversal Diagnostics (top 5 fragile groups)")

    rows = []
    for r in sorted_rev:
        subs = r.sub_signals
        state = state_map.get(r.ticker)
        state_val = state.state.value if state else "—"

        # Extract sub-signals with safe defaults
        clv = subs.get("clv_trend", 0)
        gap_fade = subs.get("gap_fade_rate", 0)
        follow = subs.get("follow_through", 0)
        dist = subs.get("distance_from_ma", 0)
        rvol = subs.get("rvol", 1.0)

        rows.append({
            "Group": f"{r.ticker} ({r.name})",
            "Rev Score": f"{r.reversal_score:.2f}",
            "Rev %ile": f"{r.reversal_percentile:.0f}%",
            "CLV": f"{clv:.2f}" if not _is_nan(clv) else "—",
            "Gap Fade": f"{gap_fade:.2f}" if not _is_nan(gap_fade) else "—",
            "Follow": f"{follow:.2f}" if not _is_nan(follow) else "—",
            "Stretch": f"{dist:+.1f}σ" if not _is_nan(dist) else "—",
            "RVOL": f"{rvol:.1f}x" if not _is_nan(rvol) else "—",
            "State": state_val,
        })

    from dashboard.components.style_utils import color_row_by_state
    df = pd.DataFrame(rows)
    styled = df.style.apply(color_row_by_state, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    # Pillar breakdown for the most fragile
    top = sorted_rev[0]
    st.caption(
        f"**{top.ticker}** — Breadth Det: {top.breadth_det_pillar:.0f}/100, "
        f"Price Break: {top.price_break_pillar:.0f}/100, "
        f"Crowding: {top.crowding_pillar:.0f}/100"
    )


def _is_nan(v):
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False
