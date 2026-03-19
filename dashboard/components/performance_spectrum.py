"""
Performance Spectrum — shows industry ETFs within each sector row,
and individual holdings within each industry row.
Sorted by 1d return (best left, worst right). No cutoff.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Sector → its industry ETFs
_SECTOR_INDUSTRIES = {
    "XLK": {"name": "Technology", "etfs": ["SMH", "IGV", "HACK", "SOXX"]},
    "XLV": {"name": "Health Care", "etfs": ["XBI", "IHI"]},
    "XLF": {"name": "Financials", "etfs": ["KRE", "IAI", "KIE"]},
    "XLE": {"name": "Energy", "etfs": ["XOP", "OIH", "URA"]},
    "XLI": {"name": "Industrials", "etfs": ["ITA", "XAR"]},
    "XLU": {"name": "Utilities", "etfs": ["TAN", "NLR"]},
    "XLRE": {"name": "Real Estate", "etfs": ["VNQ"]},
    "XLC": {"name": "Comm Services", "etfs": []},
    "XLY": {"name": "Cons Discretionary", "etfs": ["XHB", "ITB", "XRT", "IBUY"]},
    "XLP": {"name": "Cons Staples", "etfs": []},
    "XLB": {"name": "Materials", "etfs": ["XME", "GDX"]},
}

# Industry → its top holdings
_INDUSTRY_HOLDINGS = {
    "SMH": ["NVDA", "TSM", "AVGO", "AMD", "QCOM", "TXN", "MU", "INTC"],
    "IGV": ["CRM", "ADBE", "NOW", "INTU", "PANW", "SNPS", "CDNS", "WDAY"],
    "HACK": ["PANW", "CRWD", "FTNT", "ZS", "OKTA"],
    "SOXX": ["NVDA", "AVGO", "AMD", "QCOM", "TXN", "MU", "INTC", "MRVL"],
    "XBI": ["VRTX", "REGN", "MRNA", "BIIB", "ALNY", "BMRN", "EXAS", "IONS"],
    "IHI": ["ABT", "SYK", "BSX", "MDT", "ISRG", "EW", "ZBH", "HOLX"],
    "KRE": ["CFG", "RF", "HBAN", "KEY", "ZION", "FHN", "CMA", "WAL"],
    "IAI": ["GS", "MS", "SCHW", "RJF", "LPLA", "EVR", "SF", "HLI"],
    "KIE": ["ALL", "TRV", "PGR", "CB", "AFL", "MET", "AIG", "HIG"],
    "XOP": ["COP", "EOG", "DVN", "FANG", "MRO", "OVV", "APA", "CTRA"],
    "OIH": ["SLB", "HAL", "BKR", "FTI", "NOV", "WHD", "HP", "WFRD"],
    "URA": ["CCJ", "NXE", "UEC", "DNN", "LEU", "UUUU", "SMR", "OKLO"],
    "ITA": ["GE", "RTX", "LMT", "NOC", "BA", "GD", "TDG", "HWM"],
    "XAR": ["LHX", "AXON", "HEI", "TDG", "KTOS", "RKLB", "PLTR", "SPR"],
    "TAN": ["ENPH", "FSLR", "SEDG", "RUN", "NOVA", "ARRY", "CSIQ", "JKS"],
    "NLR": ["CCJ", "CEG", "VST", "NXE", "SMR", "LEU", "OKLO"],
    "VNQ": ["PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "DLR", "O"],
    "XHB": ["DHI", "LEN", "NVR", "PHM", "TOL", "KBH", "MDC", "MTH"],
    "ITB": ["DHI", "LEN", "NVR", "PHM", "TOL", "SHW", "LOW", "HD"],
    "XRT": ["GME", "CVNA", "W", "BBY", "EBAY", "KSS", "M", "GPS"],
    "IBUY": ["AMZN", "SHOP", "MELI", "SE", "ETSY", "W", "CHWY", "EBAY"],
    "XME": ["FCX", "NUE", "STLD", "CLF", "RS", "ATI", "CMC", "AA"],
    "GDX": ["NEM", "GOLD", "AEM", "FNV", "WPM", "RGLD", "AGI", "KGC"],
}


def _perf_color(pct: float) -> str:
    if pct <= -3:    return "#7f1d1d"
    elif pct <= -1.5: return "#b91c1c"
    elif pct <= -0.5: return "#dc2626"
    elif pct <= -0.1: return "#991b1b"
    elif pct <= 0.1:  return "#374151"
    elif pct <= 0.5:  return "#166534"
    elif pct <= 1.5:  return "#16a34a"
    elif pct <= 3:    return "#15803d"
    else:             return "#052e16"


def _text_color(pct: float) -> str:
    return "#ffffff" if abs(pct) > 0.3 else "#d1d5db"


@st.cache_data(ttl=300)
def _fetch_returns(tickers: tuple) -> dict:
    """Fetch 1d returns. Cached 5 min."""
    returns = {}
    if not tickers:
        return returns
    try:
        data = yf.download(list(tickers), period="2d", progress=False)
        if data is not None and not data.empty:
            closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
            for t in tickers:
                if t in closes.columns:
                    vals = closes[t].dropna()
                    if len(vals) >= 2:
                        returns[t] = (vals.iloc[-1] / vals.iloc[-2] - 1) * 100
    except Exception:
        pass
    return returns


def _render_spectrum_html(rows: list[dict]) -> str:
    """
    Render spectrum as raw HTML table — no plotly, no frame issues.
    Each row = sector/industry. Cells = ETFs/stocks sorted by return.
    """
    html = ['<div style="width:100%;overflow-x:auto;">']
    html.append('<table style="width:100%;border-collapse:collapse;border:0;">')

    for row in rows:
        name = row["name"]
        items = sorted(row["items"], key=lambda x: x["pct"], reverse=True)  # Best left

        html.append('<tr>')
        html.append(f'<td style="padding:4px 8px;font-weight:bold;font-size:13px;'
                    f'white-space:nowrap;vertical-align:middle;border:0;width:130px;">{name}</td>')
        html.append('<td style="padding:0;border:0;"><div style="display:flex;gap:0px;">')

        for item in items:
            pct = item["pct"]
            bg = _perf_color(pct)
            fg = _text_color(pct)
            ticker = item["ticker"]
            html.append(
                f'<div style="flex:1;min-width:55px;background:{bg};color:{fg};'
                f'padding:6px 3px;text-align:center;font-size:11px;line-height:1.3;">'
                f'<b>{ticker}</b><br>{pct:+.1f}%</div>'
            )

        html.append('</div></td></tr>')

    html.append('</table></div>')
    return "".join(html)


def render_sector_spectrum(result: dict):
    """Sector spectrum: each row = sector, cells = its industry ETFs sorted by return."""
    st.subheader("1-Day Sector Performance Spectrum")
    st.caption("Each row = sector. Cells = industry ETFs within that sector, sorted by return (best → worst).")

    # Collect all industry ETF tickers + sector ETFs themselves
    all_tickers = set()
    for etf, info in _SECTOR_INDUSTRIES.items():
        all_tickers.add(etf)
        all_tickers.update(info["etfs"])
    returns = _fetch_returns(tuple(sorted(all_tickers)))

    rows = []
    for etf in ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]:
        info = _SECTOR_INDUSTRIES[etf]
        items = []
        # Add the sector ETF itself
        if etf in returns:
            items.append({"ticker": etf, "pct": returns[etf]})
        # Add its industry ETFs
        for ind in info["etfs"]:
            if ind in returns:
                items.append({"ticker": ind, "pct": returns[ind]})
        if items:
            rows.append({"name": info["name"], "items": items})

    if rows:
        st.markdown(_render_spectrum_html(rows), unsafe_allow_html=True)
    else:
        st.warning("Could not fetch ETF returns for spectrum.")


def render_industry_spectrum(result: dict):
    """Industry spectrum: each row = industry ETF, cells = its top holdings sorted by return."""
    st.subheader("1-Day Industry Performance Spectrum")
    st.caption("Each row = industry ETF. Cells = top holdings, sorted by return (best → worst).")

    # Collect all stock tickers
    all_tickers = set()
    for holdings in _INDUSTRY_HOLDINGS.values():
        all_tickers.update(holdings)
    returns = _fetch_returns(tuple(sorted(all_tickers)))

    rows = []
    for ind_etf, holdings in _INDUSTRY_HOLDINGS.items():
        items = []
        for ticker in holdings:
            if ticker in returns:
                items.append({"ticker": ticker, "pct": returns[ticker]})
        if items:
            rows.append({"name": ind_etf, "items": items})

    # Sort rows by average return (best industry at top)
    rows.sort(key=lambda r: np.mean([i["pct"] for i in r["items"]]), reverse=True)

    if rows:
        st.markdown(_render_spectrum_html(rows), unsafe_allow_html=True)
    else:
        st.warning("Could not fetch stock returns for spectrum.")
