"""
Panel 5: History Replay — ◄/► nav, regime comparison, rankings, forward returns,
rolling 1d/5d/20d/60d metrics, color coded throughout.
"""
import streamlit as st
import pandas as pd
from data.snapshots import list_snapshots, load_snapshot
from dashboard.components.style_utils import color_row_by_state

_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


def _name(ticker):
    return _SECTOR_NAMES.get(ticker, ticker)


def _regime_badge(state_value):
    colors = {"NORMAL": "#22c55e", "FRAGILE": "#eab308", "HOSTILE": "#ef4444"}
    c = colors.get(state_value, "#888")
    st.markdown(
        f"<span style='background:{c};color:black;padding:4px 12px;"
        f"border-radius:4px;font-weight:bold;font-size:1.2em;'>{state_value}</span>",
        unsafe_allow_html=True,
    )


def render_replay_panel(result: dict):
    available = list_snapshots()
    if not available:
        st.warning("No snapshots. Run `python scripts/backfill.py`.")
        return

    st.subheader("History Replay")

    # ── Navigation ────────────────────────────────────
    max_idx = len(available) - 1

    # Initialize
    if "rp_pos" not in st.session_state:
        st.session_state.rp_pos = max_idx

    step_map = {"1 day": 1, "5 days": 5, "20 days": 20, "60 days": 60}

    # Buttons use on_click to modify rp_pos BEFORE widgets render
    def _back():
        s = step_map.get(st.session_state.get("rp_step_sel2", "1 day"), 1)
        st.session_state.rp_pos = max(0, st.session_state.rp_pos - s)

    def _fwd():
        s = step_map.get(st.session_state.get("rp_step_sel2", "1 day"), 1)
        st.session_state.rp_pos = min(max_idx, st.session_state.rp_pos + s)

    step_col, nav_l, nav_r = st.columns([2, 1, 1])
    with step_col:
        st.selectbox("Step size", list(step_map.keys()), index=0, key="rp_step_sel2")
    with nav_l:
        st.button("◄ Back", on_click=_back, key="rp_back5")
    with nav_r:
        st.button("Forward ►", on_click=_fwd, key="rp_fwd5")

    # Display current position (read-only — buttons control it)
    current_idx = st.session_state.rp_pos
    sel_date = available[current_idx]
    st.markdown(f"**{sel_date}** &nbsp; (#{current_idx} of {len(available)} | "
                f"{available[0]} → {available[-1]})")

    try:
        snap = load_snapshot(sel_date)
    except Exception as e:
        st.error(f"Load failed: {e}")
        return

    # ── Regime Comparison ─────────────────────────────
    st.subheader("Regime Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{sel_date}**")
        _regime_badge(snap.regime.state.value)
        for s in snap.regime.signals:
            st.caption(f"{s.name}: {s.raw_value:.2f} [{s.level.value}]")
    with c2:
        st.markdown("**Current**")
        _regime_badge(result["regime"].state.value)
        for s in result["regime"].signals:
            st.caption(f"{s.name}: {s.raw_value:.2f} [{s.level.value}]")

    # ── Sector Rankings with color coding ─────────────
    st.subheader(f"Sector Rankings: {sel_date}")
    sectors_sorted = sorted(snap.sectors, key=lambda s: s.rs_rank)
    state_map = {s.ticker: s for s in snap.states} if snap.states else {}
    pump_map = {p.ticker: p for p in snap.pump_scores} if snap.pump_scores else {}

    rows = []
    for s in sectors_sorted:
        st_obj = state_map.get(s.ticker)
        pm = pump_map.get(s.ticker)
        rows.append({
            "Rank": s.rs_rank,
            "Sector": f"{s.ticker} ({_name(s.ticker)})",
            "RS 20d": f"{s.rs_20d:+.2%}",
            "Composite": f"{s.rs_composite:.1f}",
            "Pump": f"{pm.pump_score:.2f}" if pm else "—",
            "Delta": f"{pm.pump_delta:+.3f}" if pm else "—",
            "State": st_obj.state.value if st_obj else "—",
            "Conf": f"{st_obj.confidence}%" if st_obj else "—",
        })
    df = pd.DataFrame(rows)
    styled = df.style.apply(color_row_by_state, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── Forward Returns ───────────────────────────────
    prices = result["prices"]
    date_strs = prices.index.strftime("%Y-%m-%d").values
    if sel_date in date_strs:
        st.subheader(f"What happened AFTER {sel_date}?")
        st.caption(f"Forward returns measured to **{prices.index[-1].strftime('%Y-%m-%d')}** (latest available)")
        dloc = prices.index.get_indexer(pd.to_datetime([sel_date]), method="nearest")[0]

        for fwd, label in [(10, "10-day"), (20, "20-day")]:
            floc = min(dloc + fwd, len(prices) - 1)
            if floc <= dloc:
                continue
            spy_r = (prices["SPY"].iloc[floc] / prices["SPY"].iloc[dloc]) - 1 if "SPY" in prices.columns else 0
            fwd_rows = []
            for s in sectors_sorted:
                if s.ticker in prices.columns:
                    sec_r = (prices[s.ticker].iloc[floc] / prices[s.ticker].iloc[dloc]) - 1
                    excess = sec_r - spy_r
                    st_obj = state_map.get(s.ticker)
                    fwd_rows.append({
                        "Rank": s.rs_rank,
                        "Sector": f"{s.ticker} ({_name(s.ticker)})",
                        "State (then)": st_obj.state.value if st_obj else "—",
                        f"{label}": f"{sec_r:+.2%}",
                        "vs SPY": f"{excess:+.2%}",
                    })
            if fwd_rows:
                st.markdown(f"**{label} forward** (SPY: {spy_r:+.2%})")

                def _color_vs_spy(val):
                    try:
                        n = float(str(val).replace("+", "").replace("%", ""))
                        if n > 0.5:
                            return "color: #22c55e"
                        elif n < -0.5:
                            return "color: #ef4444"
                    except (ValueError, TypeError):
                        pass
                    return ""

                fdf = pd.DataFrame(fwd_rows)
                styled_fwd = fdf.style.map(_color_vs_spy, subset=["vs SPY"])
                st.dataframe(styled_fwd, width="stretch", hide_index=True)
    else:
        st.info("Forward returns unavailable — date not in current price range.")

    # ── Rolling 1d / 5d / 20d / 60d Metrics ──────────
    st.subheader("Rolling Period Metrics")
    prev_idx = current_idx - 1
    if prev_idx >= 0 and sel_date in date_strs:
        dloc = prices.index.get_indexer(pd.to_datetime([sel_date]), method="nearest")[0]

        roll_rows = []
        for s in sectors_sorted:
            if s.ticker not in prices.columns:
                continue
            sec_prices = prices[s.ticker]
            spy_prices = prices["SPY"] if "SPY" in prices.columns else sec_prices

            def _rs(w):
                if dloc >= w:
                    sec_w = sec_prices.iloc[dloc] / sec_prices.iloc[dloc - w] - 1
                    spy_w = spy_prices.iloc[dloc] / spy_prices.iloc[dloc - w] - 1
                    return sec_w - spy_w
                return 0

            st_obj = state_map.get(s.ticker)
            roll_rows.append({
                "Sector": f"{s.ticker} ({_name(s.ticker)})",
                "RS 1d": f"{_rs(1):+.2%}",
                "RS 5d": f"{_rs(5):+.2%}",
                "RS 20d": f"{_rs(20):+.2%}",
                "RS 60d": f"{_rs(60):+.2%}" if dloc >= 60 else "—",
                "State": st_obj.state.value if st_obj else "—",
            })

        rdf = pd.DataFrame(roll_rows)
        styled_roll = rdf.style.apply(color_row_by_state, axis=1)
        st.dataframe(styled_roll, width="stretch", hide_index=True)

    # ── 24h Snapshot Diff ─────────────────────────────
    st.subheader("24h Snapshot Diff")
    if prev_idx >= 0:
        try:
            prev_snap = load_snapshot(available[prev_idx])
            prev_state_map = {s.ticker: s for s in prev_snap.states} if prev_snap.states else {}
            prev_pump_map = {p.ticker: p for p in prev_snap.pump_scores} if prev_snap.pump_scores else {}
            prev_sectors = {s.ticker: s for s in prev_snap.sectors}

            diff_rows = []
            for s in sectors_sorted:
                ps = prev_sectors.get(s.ticker)
                pp = prev_pump_map.get(s.ticker)
                cp = pump_map.get(s.ticker)
                cs = state_map.get(s.ticker)
                pst = prev_state_map.get(s.ticker)

                rs_chg = (s.rs_20d - ps.rs_20d) * 100 if ps else 0
                rank_chg = (ps.rs_rank - s.rs_rank) if ps else 0
                pump_chg = (cp.pump_score - pp.pump_score) if cp and pp else 0
                changed = cs and pst and cs.state != pst.state
                prev_st = pst.state.value if pst else "—"
                curr_st = cs.state.value if cs else "—"

                diff_rows.append({
                    "Sector": f"{s.ticker} ({_name(s.ticker)})",
                    "RS 20d Δ": f"{rs_chg:+.2f}%",
                    "Rank Δ": f"{rank_chg:+d}" if rank_chg else "—",
                    "Pump Δ": f"{pump_chg:+.4f}" if abs(pump_chg) > 0.0001 else "—",
                    "State": f"{prev_st} → {curr_st}" if changed else curr_st,
                })
            st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)
        except Exception:
            st.caption("Previous snapshot unavailable.")
    else:
        st.caption("No prior snapshot for comparison.")
