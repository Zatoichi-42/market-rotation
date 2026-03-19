"""
Panel: Signal Reliability — evaluates past signals against actual outcomes.

For each lookback (1d, 5d, 20d, 60d):
- What did the system call? (state, rank, pump delta)
- What actually happened? (forward RS return, rank change)
- Did the signal work? (scored and color-coded)
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data.snapshots import list_snapshots, load_snapshot

_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}

# States where we expect positive forward RS
_BULLISH_STATES = {"Overt Pump", "Accumulation"}
# States where we expect negative forward RS or exit
_BEARISH_STATES = {"Exhaustion", "Overt Dump"}


def _label(ticker):
    return f"{ticker} ({_SECTOR_NAMES.get(ticker, ticker)})"


def render_signal_reliability(result: dict):
    prices = result["prices"]
    available = list_snapshots()

    if len(available) < 5:
        st.warning("Need at least 5 snapshots for reliability analysis.")
        return

    st.subheader("Signal Reliability Scorecard")
    st.caption(
        "Evaluates what the system said N days ago vs what actually happened. "
        "A 'hit' means the signal direction matched the forward return. "
        "Green = signal worked. Red = signal failed."
    )

    # ── Lookback periods ──────────────────────────────
    lookbacks = []
    for days, label in [(1, "Yesterday"), (5, "5 days ago"), (20, "20 days ago"), (60, "60 days ago")]:
        if len(available) > days + 1:
            lookbacks.append((days, label, available[-(days + 1)]))

    if not lookbacks:
        st.warning("Not enough snapshots.")
        return

    last_date = prices.index[-1]
    st.caption(
        f"Forward returns measured to **{last_date.strftime('%Y-%m-%d')}** "
        f"(today's latest available price — intraday during market hours)"
    )

    # ── Summary scorecard across all lookbacks ────────
    all_results = []
    for days, label, snap_date in lookbacks:
        try:
            snap = load_snapshot(snap_date)
        except Exception:
            continue

        state_map = {s.ticker: s for s in snap.states} if snap.states else {}
        pump_map = {p.ticker: p for p in snap.pump_scores} if snap.pump_scores else {}
        sector_map = {s.ticker: s for s in snap.sectors}

        # Compute actual forward RS from snap_date to today
        if snap_date not in prices.index.strftime("%Y-%m-%d").values:
            continue
        dloc = prices.index.get_indexer(pd.to_datetime([snap_date]), method="nearest")[0]
        now_loc = len(prices) - 1

        if now_loc <= dloc:
            continue

        spy_fwd = (prices["SPY"].iloc[now_loc] / prices["SPY"].iloc[dloc]) - 1 if "SPY" in prices.columns else 0

        for ticker in _SECTOR_NAMES:
            if ticker not in prices.columns or ticker not in sector_map:
                continue

            sec_fwd = (prices[ticker].iloc[now_loc] / prices[ticker].iloc[dloc]) - 1
            fwd_rs = sec_fwd - spy_fwd

            snap_sector = sector_map[ticker]
            snap_state = state_map.get(ticker)
            snap_pump = pump_map.get(ticker)

            state_val = snap_state.state.value if snap_state else "—"
            pump_delta = snap_pump.pump_delta if snap_pump else 0
            rank_then = snap_sector.rs_rank
            composite_then = snap_sector.rs_composite

            # Score: did the signal direction match?
            if state_val in _BULLISH_STATES:
                expected_dir = "positive"
                hit = fwd_rs > 0
            elif state_val in _BEARISH_STATES:
                expected_dir = "negative"
                hit = fwd_rs < 0
            else:
                expected_dir = "neutral"
                hit = abs(fwd_rs) < 0.02  # Ambiguous = small move is "correct"

            all_results.append({
                "lookback": label,
                "days": days,
                "ticker": ticker,
                "state_then": state_val,
                "rank_then": rank_then,
                "composite_then": composite_then,
                "pump_delta_then": pump_delta,
                "fwd_rs": fwd_rs,
                "fwd_return": sec_fwd,
                "expected_dir": expected_dir,
                "hit": hit,
            })

    if not all_results:
        st.info("No data to evaluate.")
        return

    df_all = pd.DataFrame(all_results)

    # ── Overall hit rate by lookback ──────────────────
    st.subheader("Hit Rate by Lookback")
    summary_rows = []
    for days, label, _ in lookbacks:
        subset = df_all[df_all["days"] == days]
        if len(subset) == 0:
            continue
        total = len(subset)
        hits = subset["hit"].sum()
        rate = hits / total
        bullish = subset[subset["expected_dir"] == "positive"]
        bearish = subset[subset["expected_dir"] == "negative"]
        bull_rate = bullish["hit"].mean() if len(bullish) > 0 else float("nan")
        bear_rate = bearish["hit"].mean() if len(bearish) > 0 else float("nan")

        summary_rows.append({
            "Lookback": label,
            "Signals": total,
            "Hits": int(hits),
            "Hit Rate": f"{rate:.0%}",
            "Bullish Hit Rate": f"{bull_rate:.0%}" if not np.isnan(bull_rate) else "—",
            "Bearish Hit Rate": f"{bear_rate:.0%}" if not np.isnan(bear_rate) else "—",
        })

    sdf = pd.DataFrame(summary_rows)

    def _color_rate(val):
        try:
            n = float(str(val).replace("%", "")) / 100
            if n >= 0.6:
                return "color: #22c55e"
            elif n < 0.4:
                return "color: #ef4444"
        except (ValueError, TypeError):
            pass
        return ""

    styled = sdf.style.map(_color_rate, subset=["Hit Rate", "Bullish Hit Rate", "Bearish Hit Rate"])
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── Per-lookback detail tabs ──────────────────────
    for days, label, snap_date in lookbacks:
        subset = df_all[df_all["days"] == days].copy()
        if len(subset) == 0:
            continue

        with st.expander(f"{label} ({snap_date}) — Signal Detail", expanded=(days == 1)):
            subset_sorted = subset.sort_values("rank_then")

            rows = []
            for _, r in subset_sorted.iterrows():
                verdict = "HIT" if r["hit"] else "MISS"
                rows.append({
                    "Sector": _label(r["ticker"]),
                    "State (then)": r["state_then"],
                    "Rank (then)": f"#{int(r['rank_then'])}",
                    "Comp (then)": f"{r['composite_then']:.0f}",
                    "Pump Δ (then)": f"{r['pump_delta_then']:+.3f}",
                    "Expected": r["expected_dir"],
                    f"Fwd RS ({label})": f"{r['fwd_rs']:+.2%}",
                    f"Fwd Return": f"{r['fwd_return']:+.2%}",
                    "Verdict": verdict,
                })

            detail_df = pd.DataFrame(rows)

            def _color_verdict(val):
                if val == "HIT":
                    return "background-color: #064e3b; color: #34d399"
                elif val == "MISS":
                    return "background-color: #7f1d1d; color: #f87171"
                return ""

            def _color_fwd(val):
                try:
                    n = float(str(val).replace("+", "").replace("%", ""))
                    if n > 0.5:
                        return "color: #22c55e"
                    elif n < -0.5:
                        return "color: #ef4444"
                except (ValueError, TypeError):
                    pass
                return ""

            fwd_col = f"Fwd RS ({label})"
            styled_detail = detail_df.style
            styled_detail = styled_detail.map(_color_verdict, subset=["Verdict"])
            if fwd_col in detail_df.columns:
                styled_detail = styled_detail.map(_color_fwd, subset=[fwd_col])

            st.dataframe(styled_detail, width="stretch", hide_index=True)

    # ── State Reliability Heat Map ────────────────────
    st.subheader("State Reliability by Lookback")
    st.caption(
        "Each cell shows the average forward RS for sectors in that state at that lookback. "
        "Green = state predicted correctly (bullish states had positive RS, bearish had negative). "
        "Red = state prediction was wrong."
    )

    states_list = ["Overt Pump", "Accumulation", "Ambiguous", "Exhaustion", "Overt Dump"]
    lookback_labels = [label for _, label, _ in lookbacks]

    hm_z = []
    hm_text = []
    for state in states_list:
        row_z = []
        row_text = []
        for days, label, _ in lookbacks:
            subset = df_all[(df_all["days"] == days) & (df_all["state_then"] == state)]
            if len(subset) > 0:
                avg_rs = subset["fwd_rs"].mean()
                count = len(subset)
                row_z.append(avg_rs * 100)
                row_text.append(f"{avg_rs*100:+.1f}%\n(n={count})")
            else:
                row_z.append(0)
                row_text.append("—")
        hm_z.append(row_z)
        hm_text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=hm_z, x=lookback_labels, y=states_list,
        text=hm_text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#22c55e"]],
        zmid=0,
        colorbar=dict(title="Avg Fwd RS (%)"),
    ))
    fig.update_layout(
        height=max(250, len(states_list) * 40 + 60),
        margin=dict(t=20, b=20, l=10, r=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Key Insights ──────────────────────────────────
    st.subheader("Key Insights")

    # Best/worst state reliability
    for days, label, _ in lookbacks:
        subset = df_all[df_all["days"] == days]
        if len(subset) == 0:
            continue
        hit_rate = subset["hit"].mean()
        best_state = None
        best_rate = 0
        worst_state = None
        worst_rate = 1
        for state in states_list:
            s = subset[subset["state_then"] == state]
            if len(s) >= 2:
                rate = s["hit"].mean()
                if rate > best_rate:
                    best_rate, best_state = rate, state
                if rate < worst_rate:
                    worst_rate, worst_state = rate, state

        if best_state:
            st.markdown(
                f"**{label}**: Overall {hit_rate:.0%} hit rate. "
                f"Best: **{best_state}** ({best_rate:.0%}). "
                f"{'Worst: **' + worst_state + '** (' + f'{worst_rate:.0%}' + ').' if worst_state else ''}"
            )

    # Biggest wins and misses from yesterday
    yesterday = df_all[df_all["days"] == 1].copy() if 1 in df_all["days"].values else pd.DataFrame()
    if not yesterday.empty:
        st.markdown("---")
        st.markdown("**Yesterday's Biggest Signals:**")
        yesterday_sorted = yesterday.sort_values("fwd_rs", ascending=False)
        top = yesterday_sorted.iloc[0]
        bottom = yesterday_sorted.iloc[-1]
        st.markdown(
            f"🟢 Best: **{_label(top['ticker'])}** — called {top['state_then']}, "
            f"delivered {top['fwd_rs']:+.2%} RS ({'HIT' if top['hit'] else 'MISS'})"
        )
        st.markdown(
            f"🔴 Worst: **{_label(bottom['ticker'])}** — called {bottom['state_then']}, "
            f"delivered {bottom['fwd_rs']:+.2%} RS ({'HIT' if bottom['hit'] else 'MISS'})"
        )
