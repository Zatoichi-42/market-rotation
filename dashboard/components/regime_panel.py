"""Panel 1: Regime Gate traffic light + VIX, Credit, and Breadth charts."""
import math
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from engine.schemas import RegimeState, SignalLevel, BreadthSignal


_COLORS = {
    RegimeState.NORMAL: "#00d4aa",
    RegimeState.FRAGILE: "#ffa500",
    RegimeState.HOSTILE: "#ff4444",
}

_SIGNAL_COLORS = {
    SignalLevel.NORMAL: "#00d4aa",
    SignalLevel.FRAGILE: "#ffa500",
    SignalLevel.HOSTILE: "#ff4444",
}

_BREADTH_COLORS = {
    BreadthSignal.HEALTHY: "#00d4aa",
    BreadthSignal.NARROWING: "#ffa500",
    BreadthSignal.DIVERGING: "#ff4444",
}

# ═══════════════════════════════════════════════════════
# GLOSSARY — popups via st.popover
# ═══════════════════════════════════════════════════════

_GLOSSARY = {
    "vix": {
        "title": "VIX — CBOE Volatility Index",
        "body": (
            "The VIX measures the market's expectation of 30-day forward volatility, "
            "derived from S&P 500 index option prices. Often called the 'fear gauge'.\n\n"
            "**How we use it:** VIX < 20 = NORMAL (low fear), 20–30 = FRAGILE (elevated concern), "
            "> 30 = HOSTILE (high fear, likely sell-off in progress).\n\n"
            "**Boundary rule:** exact threshold (e.g. VIX = 20.0) goes to the worse bucket."
        ),
    },
    "term_structure": {
        "title": "VIX Term Structure (VIX / VIX3M)",
        "body": (
            "The ratio of 30-day VIX to 3-month VIX3M reveals the shape of the volatility curve.\n\n"
            "**Contango** (ratio < 0.95): VIX < VIX3M — markets expect short-term calm. NORMAL.\n\n"
            "**Flat** (0.95–1.05): Uncertain — near-term and medium-term fear are similar. FRAGILE.\n\n"
            "**Backwardation** (ratio > 1.05): VIX > VIX3M — panic is greatest right now. HOSTILE.\n\n"
            "Backwardation is rare and typically accompanies sharp sell-offs."
        ),
    },
    "breadth": {
        "title": "Market Breadth (RSP/SPY Ratio)",
        "body": (
            "Compares the equal-weight S&P 500 (RSP) to the cap-weight S&P 500 (SPY). "
            "When RSP outperforms, most stocks are participating in the rally. "
            "When SPY outperforms, gains are concentrated in mega-caps.\n\n"
            "**How we use it:** We compute a z-score of the RSP/SPY ratio against its "
            "504-day history. Z > 0 = HEALTHY, 0 to −1 = NARROWING, < −1 = DIVERGING.\n\n"
            "**Why it matters:** Narrow markets (few leaders) are fragile — "
            "when leadership falters, there's nothing underneath to hold the index up."
        ),
    },
    "credit": {
        "title": "Credit Spread (HYG/LQD Ratio)",
        "body": (
            "HYG (iShares iBoxx High Yield Bond ETF) tracks junk bonds. "
            "LQD (iShares iBoxx Investment Grade Bond ETF) tracks quality bonds. "
            "Their ratio is a real-time proxy for credit risk appetite.\n\n"
            "**Rising ratio:** investors prefer risk (junk outperforming quality). NORMAL.\n\n"
            "**Falling ratio:** flight to quality — investors dumping junk for safety. HOSTILE.\n\n"
            "We z-score the ratio against history. Z > −0.5 = NORMAL, −0.5 to −1.5 = FRAGILE, "
            "< −1.5 = HOSTILE."
        ),
    },
    "fred_hy_oas": {
        "title": "FRED HY OAS (BAMLH0A0HYM2)",
        "body": (
            "The ICE BofA US High Yield Option-Adjusted Spread measures the spread between "
            "high-yield corporate bonds and a risk-free benchmark (US Treasuries).\n\n"
            "**Low spread** (< 3.5%): Tight credit conditions, risk appetite strong.\n\n"
            "**High spread** (> 5%): Credit stress — investors demanding large premiums "
            "for default risk.\n\n"
            "**Spike:** A sudden widening often precedes or accompanies equity sell-offs.\n\n"
            "⚠️ **Lag disclaimer:** FRED data is published with a 1–2 business day delay. "
            "The most recent data point may not reflect today's conditions. "
            "Use HYG/LQD ratio (real-time) as the primary signal; FRED OAS as confirmation."
        ),
    },
}


def _render_glossary_popover(key: str):
    """Render a (?) popover with glossary explanation."""
    entry = _GLOSSARY.get(key, {})
    with st.popover("ℹ️"):
        st.markdown(f"**{entry.get('title', key)}**")
        st.markdown(entry.get("body", "No description available."))


def render_regime_panel(result: dict):
    regime = result["regime"]
    color = _COLORS[regime.state]

    # ── Traffic light header ──────────────────────────
    col1, col2 = st.columns([1, 3])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="number",
            value=result["vix_val"],
            title={"text": "VIX"},
            number={"font": {"size": 48}},
        ))
        fig.update_layout(height=150, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown(
            f"<h1 style='color: {color}; margin-bottom: 0;'>{regime.state.value}</h1>",
            unsafe_allow_html=True,
        )
        # Catalyst gate badge
        catalyst = result.get("catalyst")
        if catalyst:
            cat_badges = {
                "Clear": ("✓ CLEAR", "#22c55e"),
                "Caution": ("⚡ CAUTION", "#eab308"),
                "Embargo": ("🚫 EMBARGO", "#ef4444"),
                "Shock Pause": ("⏸ SHOCK PAUSE", "#a78bfa"),
            }
            badge_text, badge_color = cat_badges.get(catalyst.action.value, ("—", "#888"))
            st.markdown(
                f"<span style='background:{badge_color};color:black;padding:3px 10px;"
                f"border-radius:4px;font-weight:bold;font-size:0.9em;'>{badge_text}</span>"
                f"&nbsp; {catalyst.explanation[:120]}",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='font-size: 1.15em; line-height: 1.6;'>{regime.explanation}</div>",
            unsafe_allow_html=True,
        )

    # ── Signal breakdown with glossary popovers ───────
    st.subheader("Signal Breakdown")
    cols = st.columns(len(regime.signals))
    for i, sig in enumerate(regime.signals):
        with cols[i]:
            sig_color = _SIGNAL_COLORS[sig.level]
            # Title row with popover
            title_col, info_col = st.columns([4, 1])
            with title_col:
                st.markdown(
                    f"**{sig.name.replace('_', ' ').title()}**",
                )
            with info_col:
                _render_glossary_popover(sig.name)
            st.markdown(
                f"<span style='color: {sig_color}; font-size: 1.5em;'>{sig.raw_value:.2f}</span><br>"
                f"<span style='color: {sig_color};'>{sig.level.value}</span>",
                unsafe_allow_html=True,
            )
            st.caption(sig.description)

    # ══════════════════════════════════════════════════
    # CHARTS — VIX, Credit, Breadth all on page 1
    # ══════════════════════════════════════════════════

    # ── VIX Term Structure ────────────────────────────
    vix = result["vix"]
    vix3m = result["vix3m"]
    if len(vix) > 0 and len(vix3m) > 0:
        hdr_col, info_col = st.columns([6, 1])
        with hdr_col:
            st.subheader("VIX Term Structure (60d)")
        with info_col:
            _render_glossary_popover("term_structure")
        st.caption("VIX = CBOE Volatility Index | VIX3M = CBOE 3-Month Volatility Index")
        combined = pd.DataFrame({"VIX": vix.tail(60), "VIX3M": vix3m.tail(60)}).dropna()
        if not combined.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=combined.index, y=combined["VIX"],
                name="^VIX (30-day implied vol)", line=dict(color="#ff4444"),
            ))
            fig.add_trace(go.Scatter(
                x=combined.index, y=combined["VIX3M"],
                name="^VIX3M (3-month implied vol)", line=dict(color="#4488ff"),
            ))
            fig.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Fragile (20)")
            fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Hostile (30)")
            fig.update_layout(
                height=300, margin=dict(t=20, b=20), legend=dict(orientation="h"),
                xaxis=dict(range=[combined.index[0], combined.index[-1]]),
            )
            st.plotly_chart(fig, width="stretch")

    # ── Credit chart (HYG/LQD + FRED OAS) ────────────
    credit = result.get("credit")
    fred_hy_oas = result.get("fred_hy_oas")

    has_credit_ratio = credit is not None and "hyg_close" in credit.columns and "lqd_close" in credit.columns
    has_fred = fred_hy_oas is not None and not fred_hy_oas.empty

    if has_credit_ratio or has_fred:
        hdr_col, info_col = st.columns([6, 1])
        with hdr_col:
            st.subheader("Credit Conditions (60d)")
        with info_col:
            _render_glossary_popover("credit")

        n_rows = 1 + (1 if has_fred else 0)
        row_heights = [0.5, 0.5] if n_rows == 2 else [1.0]
        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            vertical_spacing=0.08, row_heights=row_heights,
            subplot_titles=(
                ["HYG/LQD Ratio (real-time credit proxy)"] +
                (["FRED HY OAS — BAMLH0A0HYM2 (1–2 day lag)"] if has_fred else [])
            ),
        )

        if has_credit_ratio:
            hyg = credit["hyg_close"].tail(120)
            lqd = credit["lqd_close"].tail(120)
            ratio = (hyg / lqd).dropna()
            ratio_60 = ratio.tail(60)
            ma20 = ratio.rolling(20).mean().tail(60)

            fig.add_trace(go.Scatter(
                x=ratio_60.index, y=ratio_60.values,
                name="HYG/LQD", line=dict(color="#ff8800", width=2),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=ma20.index, y=ma20.values,
                name="20d MA", line=dict(color="#888", width=1, dash="dot"),
            ), row=1, col=1)

            st.caption(
                "HYG = iShares iBoxx $ High Yield Corp Bond ETF | "
                "LQD = iShares iBoxx $ Investment Grade Corp Bond ETF | "
                "**Falling ratio = flight to quality (risk-off)**"
            )

        # Determine shared x-axis range from HYG/LQD data (or FRED if no HYG/LQD)
        if has_credit_ratio:
            x_min, x_max = ratio_60.index[0], ratio_60.index[-1]
        else:
            x_min, x_max = None, None

        if has_fred:
            fred_row = 2 if has_credit_ratio else 1
            oas_raw = fred_hy_oas["hy_oas"].dropna()
            # FRED reports in percentage points (e.g. 3.22 = 322 bps). Convert to bps.
            oas_bps = oas_raw * 100

            # Align FRED to same date window as HYG/LQD, or last 60 points
            if x_min is not None:
                oas_window = oas_bps[(oas_bps.index >= x_min) & (oas_bps.index <= x_max)]
            else:
                oas_window = oas_bps.tail(60)
                x_min, x_max = oas_window.index[0], oas_window.index[-1]

            if not oas_window.empty:
                fig.add_trace(go.Scatter(
                    x=oas_window.index, y=oas_window.values,
                    name="HY OAS (bps)", line=dict(color="#cc44ff", width=2),
                    fill="tozeroy", fillcolor="rgba(204,68,255,0.08)",
                ), row=fred_row, col=1)

                # Only show threshold lines if they're within reasonable range of the data
                oas_max = oas_window.max()
                y_top = max(oas_max * 1.15, oas_max + 30)  # 15% headroom or 30 bps
                if 350 <= y_top:
                    fig.add_hline(y=350, line_dash="dash", line_color="orange",
                                  annotation_text="Elevated (350 bps)", row=fred_row, col=1)
                if 500 <= y_top:
                    fig.add_hline(y=500, line_dash="dash", line_color="red",
                                  annotation_text="Stress (500 bps)", row=fred_row, col=1)

                # Pin y-axis to data range
                oas_min = oas_window.min()
                fig.update_yaxes(range=[max(0, oas_min - 20), y_top], row=fred_row, col=1)

            fred_info_col, fred_link_col = st.columns([5, 1])
            with fred_link_col:
                _render_glossary_popover("fred_hy_oas")
            with fred_info_col:
                st.caption(
                    "⚠️ **FRED data is delayed 1–2 business days.** "
                    "Source: ICE BofA US High Yield Index OAS (BAMLH0A0HYM2). "
                    "Use HYG/LQD above for real-time signal; FRED OAS for confirmation."
                )

        # Pin x-axis to actual data range on all subplots
        x_range_update = {}
        if x_min is not None and x_max is not None:
            x_range_update = dict(range=[x_min, x_max])

        chart_height = 450 if n_rows == 2 else 300
        fig.update_layout(height=chart_height, margin=dict(t=30, b=20), legend=dict(orientation="h"))
        if x_range_update:
            fig.update_xaxes(**x_range_update)
        st.plotly_chart(fig, width="stretch")

    # ── Breadth chart ─────────────────────────────────
    breadth = result["breadth"]
    prices = result["prices"]
    breadth_color = _BREADTH_COLORS[breadth.signal]

    hdr_col, info_col = st.columns([6, 1])
    with hdr_col:
        st.subheader("Market Breadth (60d)")
    with info_col:
        _render_glossary_popover("breadth")

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("RSP/SPY Ratio", f"{breadth.rsp_spy_ratio:.4f}",
                   delta=f"{breadth.rsp_spy_ratio_20d_change:+.4f} (20d)")
    with m2:
        z_display = f"{breadth.rsp_spy_ratio_zscore:.2f}" if not math.isnan(breadth.rsp_spy_ratio_zscore) else "N/A"
        st.metric("Z-Score", z_display)
    with m3:
        st.markdown(
            f"<span style='background: {breadth_color}; color: black; padding: 4px 12px; "
            f"border-radius: 4px; font-weight: bold; font-size: 1.1em;'>"
            f"{breadth.signal.value}</span>",
            unsafe_allow_html=True,
        )

    st.caption("RSP = Invesco S&P 500 Equal Weight ETF | SPY = SPDR S&P 500 ETF Trust")

    if "SPY" in prices.columns and "RSP" in prices.columns:
        ratio = (prices["RSP"] / prices["SPY"]).dropna()
        ratio_60 = ratio.tail(60)
        ma20 = ratio.rolling(20).mean().tail(60)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ratio_60.index, y=ratio_60.values,
            name="RSP/SPY", line=dict(color="#4488ff", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ma20.index, y=ma20.values,
            name="20d MA", line=dict(color="#888", width=1, dash="dot"),
        ))
        fig.update_layout(
            height=250, margin=dict(t=20, b=20),
            legend=dict(orientation="h"),
            yaxis_title="RSP / SPY",
            xaxis=dict(range=[ratio_60.index[0], ratio_60.index[-1]]),
        )
        st.plotly_chart(fig, width="stretch")

    # ── 1d Rolling Metrics ────────────────────────────
    st.subheader("1d Rolling Metrics (24h change)")
    _render_1d_metrics(result)


def _render_1d_metrics(result: dict):
    """Show rolling metrics using TODAY's latest price as endpoint.
    1d = today vs yesterday. 5d/20d/60d = today vs N days ago."""
    import pandas as pd
    prices = result["prices"]
    vix = result["vix"]

    if len(prices) < 2 or len(vix) < 2:
        st.caption("Insufficient data for 1d metrics.")
        return

    # Confirm we're using today's data
    last_date = prices.index[-1]
    st.caption(f"Data as of: **{last_date.strftime('%Y-%m-%d')}** (latest available — intraday during market hours)")

    rows = []

    # VIX 1d change
    vix_now, vix_prev = vix.iloc[-1], vix.iloc[-2]
    vix_chg = vix_now - vix_prev
    rows.append({"Metric": "VIX", "Current": f"{vix_now:.1f}", "1d Change": f"{vix_chg:+.1f}",
                 "Signal": "rising" if vix_chg > 0.5 else ("falling" if vix_chg < -0.5 else "stable")})

    # SPY 1d return
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().iloc[-1]
        rows.append({"Metric": "SPY", "Current": f"{prices['SPY'].iloc[-1]:.2f}",
                     "1d Change": f"{spy_ret:+.2%}", "Signal": ""})

    # RSP/SPY 1d change
    if "RSP" in prices.columns and "SPY" in prices.columns:
        ratio = prices["RSP"] / prices["SPY"]
        r_chg = ratio.iloc[-1] - ratio.iloc[-2]
        rows.append({"Metric": "RSP/SPY", "Current": f"{ratio.iloc[-1]:.4f}",
                     "1d Change": f"{r_chg:+.4f}",
                     "Signal": "broadening" if r_chg > 0 else "narrowing"})

    # HYG/LQD 1d change
    credit = result.get("credit")
    if credit is not None and "hyg_close" in credit.columns and "lqd_close" in credit.columns:
        cr = (credit["hyg_close"] / credit["lqd_close"]).dropna()
        if len(cr) >= 2:
            cr_chg = cr.iloc[-1] - cr.iloc[-2]
            rows.append({"Metric": "HYG/LQD", "Current": f"{cr.iloc[-1]:.4f}",
                         "1d Change": f"{cr_chg:+.4f}",
                         "Signal": "risk-on" if cr_chg > 0 else "risk-off"})

    # Top/bottom 1d RS movers (with names)
    rs_readings = result.get("rs_readings", [])
    name_map = {r.ticker: r.name for r in rs_readings}
    if rs_readings and "SPY" in prices.columns and len(prices) >= 2:
        spy_1d = prices["SPY"].pct_change().iloc[-1]
        movers = []
        for r in rs_readings:
            if r.ticker in prices.columns:
                sec_1d = prices[r.ticker].pct_change().iloc[-1]
                movers.append((r.ticker, r.name, sec_1d - spy_1d))
        if movers:
            movers.sort(key=lambda x: x[2], reverse=True)
            t, n, v = movers[0]
            rows.append({"Metric": f"Top 1d RS: {t} ({n})", "Current": "", "1d Change": f"{v:+.2%}", "Signal": "leading today"})
            t, n, v = movers[-1]
            rows.append({"Metric": f"Bottom 1d RS: {t} ({n})", "Current": "", "1d Change": f"{v:+.2%}", "Signal": "lagging today"})

    # 5d / 20d / 60d rolling RS leaders
    if rs_readings and "SPY" in prices.columns:
        for window, label in [(5, "5d"), (20, "20d"), (60, "60d")]:
            if len(prices) > window:
                spy_w = prices["SPY"].pct_change(window).iloc[-1]
                best_t, best_n, best_v = None, None, -999
                for r in rs_readings:
                    if r.ticker in prices.columns:
                        sec_w = prices[r.ticker].pct_change(window).iloc[-1]
                        rs_w = sec_w - spy_w
                        if rs_w > best_v:
                            best_t, best_n, best_v = r.ticker, r.name, rs_w
                if best_t:
                    rows.append({"Metric": f"Top {label} RS: {best_t} ({best_n})",
                                 "Current": "", "1d Change": f"{best_v:+.2%}",
                                 "Signal": f"leading {label}"})

    # Color code the Signal column
    df_metrics = pd.DataFrame(rows)

    def _color_signal(val):
        v = str(val).lower()
        if any(w in v for w in ["leading", "risk-on", "broadening", "falling", "stable"]):
            return "color: #22c55e"
        if any(w in v for w in ["lagging", "risk-off", "narrowing", "rising"]):
            return "color: #ef4444"
        return ""

    if not df_metrics.empty and "Signal" in df_metrics.columns:
        styled = df_metrics.style.map(_color_signal, subset=["Signal"])
        st.dataframe(styled, width="stretch", hide_index=True)
    else:
        st.dataframe(df_metrics, width="stretch", hide_index=True)
