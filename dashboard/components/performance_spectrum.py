"""
Performance Spectrum — finviz-style heatmap showing individual stocks within sectors/industries.

Each row = sector or industry. Each cell = a stock, sized by market cap weight,
colored by 1d performance (deep red → red → neutral → green → deep green).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf

_SECTOR_LEADERS = {
    "XLK": {"name": "Technology", "holdings": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "ACN", "ADBE", "AMD", "CSCO"]},
    "XLV": {"name": "Health Care", "holdings": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN"]},
    "XLF": {"name": "Financials", "holdings": ["BRK-B", "JPM", "V", "MA", "BAC", "GS", "MS", "AXP", "C", "SPGI"]},
    "XLE": {"name": "Energy", "holdings": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HES"]},
    "XLI": {"name": "Industrials", "holdings": ["GE", "RTX", "HON", "CAT", "UNP", "BA", "DE", "LMT", "ETN", "ITW"]},
    "XLU": {"name": "Utilities", "holdings": ["NEE", "SO", "DUK", "CEG", "SRE", "D", "AEP", "EXC", "XEL", "PEG"]},
    "XLRE": {"name": "Real Estate", "holdings": ["PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "DLR", "O", "CCI", "VICI"]},
    "XLC": {"name": "Comm Services", "holdings": ["META", "GOOGL", "NFLX", "T", "DIS", "CMCSA", "VZ", "TMUS", "CHTR", "EA"]},
    "XLY": {"name": "Cons Discretionary", "holdings": ["AMZN", "TSLA", "HD", "MCD", "LOW", "NKE", "SBUX", "TJX", "BKNG", "CMG"]},
    "XLP": {"name": "Cons Staples", "holdings": ["PG", "COST", "WMT", "KO", "PEP", "PM", "MO", "CL", "MDLZ", "STZ"]},
    "XLB": {"name": "Materials", "holdings": ["LIN", "SHW", "FCX", "APD", "ECL", "NEM", "NUE", "VMC", "MLM", "DOW"]},
}

_INDUSTRY_LEADERS = {
    "SMH": {"name": "Semiconductors", "holdings": ["NVDA", "TSM", "AVGO", "AMD", "QCOM", "TXN", "MU", "INTC"]},
    "XBI": {"name": "Biotech", "holdings": ["VRTX", "REGN", "GILEAD", "MRNA", "BIIB", "ALNY", "SGEN", "BMRN"]},
    "KRE": {"name": "Regional Banks", "holdings": ["CFG", "RF", "HBAN", "KEY", "ZION", "FHN", "CMA", "SIVB"]},
    "XOP": {"name": "Oil & Gas E&P", "holdings": ["COP", "EOG", "PXD", "DVN", "FANG", "MRO", "OVV", "APA"]},
    "XHB": {"name": "Homebuilders", "holdings": ["DHI", "LEN", "NVR", "PHM", "TOL", "KBH", "MDC", "MTH"]},
    "GDX": {"name": "Gold Miners", "holdings": ["NEM", "GOLD", "AEM", "FNV", "WPM", "RGLD", "AGI", "KGC"]},
}


def _perf_color(pct: float) -> str:
    """Map performance % to color. Deep red → neutral → deep green."""
    if pct <= -3:
        return "#7f1d1d"
    elif pct <= -1.5:
        return "#b91c1c"
    elif pct <= -0.5:
        return "#dc2626"
    elif pct <= -0.1:
        return "#991b1b"
    elif pct <= 0.1:
        return "#374151"
    elif pct <= 0.5:
        return "#166534"
    elif pct <= 1.5:
        return "#16a34a"
    elif pct <= 3:
        return "#15803d"
    else:
        return "#052e16"


def _text_color(pct: float) -> str:
    return "#ffffff" if abs(pct) > 0.3 else "#9ca3af"


@st.cache_data(ttl=300)
def _fetch_stock_returns(tickers: tuple) -> dict:
    """Fetch 1d returns for a batch of stock tickers. Cached 5 min."""
    returns = {}
    try:
        data = yf.download(list(tickers), period="2d", progress=False)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
            else:
                closes = data[["Close"]]
            for t in tickers:
                if t in closes.columns and len(closes[t].dropna()) >= 2:
                    vals = closes[t].dropna()
                    ret = (vals.iloc[-1] / vals.iloc[-2] - 1) * 100
                    returns[t] = ret
    except Exception:
        pass
    return returns


def render_sector_spectrum(result: dict):
    """Render the finviz-style 1-day performance spectrum for sectors."""
    st.subheader("1-Day Sector Performance Spectrum")
    st.caption("Each cell = stock, color = 1d performance. Deep red → neutral gray → deep green.")

    # Collect all tickers
    all_tickers = set()
    for info in _SECTOR_LEADERS.values():
        all_tickers.update(info["holdings"])
    returns = _fetch_stock_returns(tuple(sorted(all_tickers)))

    if not returns:
        st.warning("Could not fetch stock returns for spectrum.")
        return

    _render_spectrum(_SECTOR_LEADERS, returns, "Sector")


def render_industry_spectrum(result: dict):
    """Render the finviz-style 1-day performance spectrum for industries."""
    st.subheader("1-Day Industry Performance Spectrum")

    all_tickers = set()
    for info in _INDUSTRY_LEADERS.values():
        all_tickers.update(info["holdings"])
    returns = _fetch_stock_returns(tuple(sorted(all_tickers)))

    if not returns:
        st.warning("Could not fetch stock returns for spectrum.")
        return

    _render_spectrum(_INDUSTRY_LEADERS, returns, "Industry")


def _render_spectrum(leaders: dict, returns: dict, label: str):
    """Render the spectrum heatmap using Plotly."""
    rows = []
    for etf, info in leaders.items():
        name = info["name"]
        holdings = info["holdings"]
        row_data = []
        for ticker in holdings:
            pct = returns.get(ticker, 0)
            row_data.append({"ticker": ticker, "pct": pct})
        # Sort by performance descending (best on left, worst on right? or by weight?)
        # Keep original order (roughly by weight)
        rows.append({"etf": etf, "name": name, "stocks": row_data})

    # Build the figure as a series of horizontal bar-like rectangles
    fig = go.Figure()

    y_labels = []
    n_rows = len(rows)

    for i, row in enumerate(rows):
        y = n_rows - i - 1
        y_labels.append(f"{row['name']}")
        stocks = row["stocks"]
        n = len(stocks)
        if n == 0:
            continue
        cell_width = 1.0 / n

        for j, stock in enumerate(stocks):
            pct = stock["pct"]
            color = _perf_color(pct)
            x0 = j * cell_width
            x1 = (j + 1) * cell_width
            ticker = stock["ticker"]

            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=y - 0.4, y1=y + 0.4,
                fillcolor=color, line=dict(color="#1e293b", width=1),
                layer="below",
            )
            # Stock ticker label
            fig.add_annotation(
                x=(x0 + x1) / 2, y=y,
                text=f"<b>{ticker}</b><br>{pct:+.1f}%",
                showarrow=False,
                font=dict(size=9, color=_text_color(pct)),
                xanchor="center", yanchor="middle",
            )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(
        tickvals=list(range(n_rows)),
        ticktext=list(reversed(y_labels)),
        showgrid=False,
    )
    fig.update_layout(
        height=max(300, n_rows * 50 + 40),
        margin=dict(t=10, b=10, l=140, r=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
