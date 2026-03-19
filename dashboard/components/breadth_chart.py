"""Panel 3: Breadth divergence chart."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from engine.schemas import BreadthSignal


_SIGNAL_COLORS = {
    BreadthSignal.HEALTHY: "#00d4aa",
    BreadthSignal.NARROWING: "#ffa500",
    BreadthSignal.DIVERGING: "#ff4444",
}


def render_breadth_chart(result: dict):
    breadth = result["breadth"]
    prices = result["prices"]
    color = _SIGNAL_COLORS[breadth.signal]

    # Header
    st.caption("RSP = Invesco S&P 500 Equal Weight ETF | SPY = SPDR S&P 500 ETF Trust")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RSP/SPY Ratio", f"{breadth.rsp_spy_ratio:.4f}",
                   delta=f"{breadth.rsp_spy_ratio_20d_change:+.4f} (20d)")
    with col2:
        import math
        z_display = f"{breadth.rsp_spy_ratio_zscore:.2f}" if not math.isnan(breadth.rsp_spy_ratio_zscore) else "N/A"
        st.metric("Z-Score", z_display)
    with col3:
        st.markdown(
            f"<h3 style='color: {color};'>{breadth.signal.value}</h3>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='font-size: 1.15em; line-height: 1.6;'>{breadth.explanation}</div>",
        unsafe_allow_html=True,
    )

    # RSP/SPY ratio chart
    if "SPY" in prices.columns and "RSP" in prices.columns:
        ratio = (prices["RSP"] / prices["SPY"]).dropna()
        ratio_60d = ratio.tail(120)

        fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                            shared_xaxes=True, vertical_spacing=0.08)

        # Ratio line
        fig.add_trace(go.Scatter(
            x=ratio_60d.index, y=ratio_60d.values,
            name="RSP/SPY Ratio", line=dict(color="#4488ff", width=2),
        ), row=1, col=1)
        # 20d MA
        ma20 = ratio_60d.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=ma20.index, y=ma20.values,
            name="20d MA", line=dict(color="#888", width=1, dash="dot"),
        ), row=1, col=1)

        # SPY vs RSP normalized
        spy_norm = (prices["SPY"] / prices["SPY"].iloc[-120]).tail(120) * 100 if len(prices) >= 120 else (prices["SPY"] / prices["SPY"].iloc[0]) * 100
        rsp_norm = (prices["RSP"] / prices["RSP"].iloc[-120]).tail(120) * 100 if len(prices) >= 120 else (prices["RSP"] / prices["RSP"].iloc[0]) * 100

        fig.add_trace(go.Scatter(
            x=spy_norm.index, y=spy_norm.values,
            name="SPY (indexed)", line=dict(color="#ff8800", width=1.5),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=rsp_norm.index, y=rsp_norm.values,
            name="RSP (indexed)", line=dict(color="#4488ff", width=1.5),
        ), row=2, col=1)

        fig.update_layout(
            height=500, margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=1.02),
        )
        fig.update_xaxes(range=[ratio_60d.index[0], ratio_60d.index[-1]])
        fig.update_yaxes(title_text="RSP/SPY", row=1, col=1)
        fig.update_yaxes(title_text="Indexed (100)", row=2, col=1)

        st.plotly_chart(fig, width="stretch")
