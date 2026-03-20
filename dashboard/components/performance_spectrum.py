"""
Performance Spectrum — shows industry ETFs within each sector row (1d/5d/20d/60d),
and individual holdings within each industry row.
Sorted by return (best left, worst right). No cutoff.
Tickers always shown with names.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

_SECTOR_INDUSTRIES = {
    "XLK": {"name": "Technology", "etfs": ["SMH", "IGV", "HACK", "SOXX"]},
    "XLV": {"name": "Health Care", "etfs": ["XBI", "IHI"]},
    "XLF": {"name": "Financials", "etfs": ["KRE", "IAI", "KIE"]},
    "XLE": {"name": "Energy", "etfs": ["XOP", "OIH", "URA"]},
    "XLI": {"name": "Industrials", "etfs": ["ITA", "XAR"]},
    "XLU": {"name": "Utilities", "etfs": ["TAN", "NLR"]},
    "XLRE": {"name": "Real Estate", "etfs": ["VNQ"]},
    "XLC": {"name": "Comm Services", "etfs": []},
    "XLY": {"name": "Cons Disc", "etfs": ["XHB", "ITB", "XRT", "IBUY"]},
    "XLP": {"name": "Cons Staples", "etfs": []},
    "XLB": {"name": "Materials", "etfs": ["XME", "GDX"]},
}

_ETF_NAMES = {
    "SMH": "Semis", "IGV": "Software", "HACK": "Cyber", "SOXX": "Semis(iSh)",
    "XBI": "Biotech", "IHI": "MedDev", "KRE": "RegBanks", "IAI": "Brokers", "KIE": "Insurance",
    "XOP": "Oil E&P", "OIH": "OilSvc", "URA": "Uranium", "ITA": "A&D", "XAR": "A&D(SPDR)",
    "TAN": "Solar", "NLR": "Nuclear", "VNQ": "REITs",
    "XHB": "Homebuild", "ITB": "HomeCon", "XRT": "Retail", "IBUY": "eRetail",
    "XME": "Metals", "GDX": "Gold",
    "XLK": "Tech", "XLV": "Health", "XLF": "Fins", "XLE": "Energy",
    "XLI": "Indust", "XLU": "Utils", "XLRE": "RealEst", "XLC": "CommSvc",
    "XLY": "ConDisc", "XLP": "ConStap", "XLB": "Materials",
}

_INDUSTRY_HOLDINGS = {
    "SMH": {"name": "Semiconductors", "stocks": {"NVDA": "Nvidia", "TSM": "TSMC", "AVGO": "Broadcom", "AMD": "AMD", "QCOM": "Qualcomm", "TXN": "TI", "MU": "Micron", "INTC": "Intel"}},
    "IGV": {"name": "Software", "stocks": {"CRM": "Salesforce", "ADBE": "Adobe", "NOW": "ServiceNow", "INTU": "Intuit", "PANW": "PaloAlto", "SNPS": "Synopsys", "CDNS": "Cadence", "WDAY": "Workday"}},
    "XBI": {"name": "Biotech", "stocks": {"VRTX": "Vertex", "REGN": "Regeneron", "MRNA": "Moderna", "BIIB": "Biogen", "ALNY": "Alnylam", "BMRN": "BioMarin"}},
    "KRE": {"name": "Regional Banks", "stocks": {"CFG": "Citizens", "RF": "Regions", "HBAN": "Huntington", "KEY": "KeyCorp", "ZION": "Zions", "CMA": "Comerica"}},
    "XOP": {"name": "Oil & Gas E&P", "stocks": {"COP": "Conoco", "EOG": "EOG", "DVN": "Devon", "FANG": "Diamondback", "MRO": "Marathon", "OVV": "Ovintiv"}},
    "GDX": {"name": "Gold Miners", "stocks": {"NEM": "Newmont", "GOLD": "Barrick", "AEM": "AgnicoEagle", "FNV": "FrancoNev", "WPM": "Wheaton"}},
    "XHB": {"name": "Homebuilders", "stocks": {"DHI": "DRHorton", "LEN": "Lennar", "NVR": "NVR", "PHM": "PulteGrp", "TOL": "TollBros"}},
    "OIH": {"name": "Oil Services", "stocks": {"SLB": "Schlumbrg", "HAL": "Halliburtn", "BKR": "BakerHugh"}},
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
def _fetch_returns_multi(tickers: tuple) -> dict:
    """Fetch 1d/5d/20d/60d returns. Returns {ticker: {period: pct}}."""
    results = {}
    if not tickers:
        return results
    try:
        data = yf.download(list(tickers), period="3mo", progress=False)
        if data is None or data.empty:
            return results
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        for t in tickers:
            if t not in closes.columns:
                continue
            vals = closes[t].dropna()
            r = {}
            if len(vals) >= 2:
                r["1d"] = (vals.iloc[-1] / vals.iloc[-2] - 1) * 100
            if len(vals) >= 6:
                r["5d"] = (vals.iloc[-1] / vals.iloc[-5] - 1) * 100
            if len(vals) >= 21:
                r["20d"] = (vals.iloc[-1] / vals.iloc[-20] - 1) * 100
            if len(vals) >= 61:
                r["60d"] = (vals.iloc[-1] / vals.iloc[-60] - 1) * 100
            if r:
                results[t] = r
    except Exception:
        pass
    return results


def _render_spectrum_html(rows: list[dict], period: str) -> str:
    """Render spectrum as HTML. Cells sorted by return for the selected period."""
    html = ['<div style="width:100%;overflow-x:auto;">']
    html.append('<table style="width:100%;border-collapse:collapse;border:0;">')

    for row in rows:
        name = row["name"]
        items = row["items"]
        # Sort by selected period return (best left)
        items_sorted = sorted(items, key=lambda x: x.get(period, 0), reverse=True)

        html.append('<tr>')
        html.append(f'<td style="padding:4px 8px;font-weight:bold;font-size:13px;'
                    f'white-space:nowrap;vertical-align:middle;border:0;width:120px;">{name}</td>')
        html.append('<td style="padding:0;border:0;"><div style="display:flex;gap:0px;">')

        for item in items_sorted:
            pct = item.get(period, 0)
            bg = _perf_color(pct)
            fg = _text_color(pct)
            ticker = item["ticker"]
            label = item.get("label", ticker)
            html.append(
                f'<div style="flex:1;min-width:65px;background:{bg};color:{fg};'
                f'padding:5px 2px;text-align:center;font-size:10px;line-height:1.3;">'
                f'<b>{label}</b><br>{pct:+.1f}%</div>'
            )

        html.append('</div></td></tr>')

    html.append('</table></div>')
    return "".join(html)


def render_sector_spectrum(result: dict):
    """Sector spectrum: rows = sectors, cells = industry ETFs, multi-period."""
    st.subheader("Sector Performance Spectrum")

    period = st.selectbox("Period", ["1d", "5d", "20d", "60d"], index=0, key="sec_spec_period")

    all_tickers = set()
    for etf, info in _SECTOR_INDUSTRIES.items():
        all_tickers.add(etf)
        all_tickers.update(info["etfs"])
    returns = _fetch_returns_multi(tuple(sorted(all_tickers)))

    if not returns:
        st.warning("Could not fetch ETF returns.")
        return

    rows = []
    for etf in ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]:
        info = _SECTOR_INDUSTRIES[etf]
        items = []
        if etf in returns:
            items.append({"ticker": etf, "label": f"{etf} ({_ETF_NAMES.get(etf, '')})", **returns[etf]})
        for ind in info["etfs"]:
            if ind in returns:
                items.append({"ticker": ind, "label": f"{ind} ({_ETF_NAMES.get(ind, '')})", **returns[ind]})
        if items:
            rows.append({"name": info["name"], "items": items})

    if rows:
        st.caption(f"Sorted by {period} return (best → worst). Each cell = ETF with name.")
        st.markdown(_render_spectrum_html(rows, period), unsafe_allow_html=True)


def render_industry_spectrum(result: dict):
    """Industry spectrum: rows = industry ETFs, cells = top holdings, multi-period."""
    st.subheader("Industry Performance Spectrum")

    period = st.selectbox("Period", ["1d", "5d", "20d", "60d"], index=0, key="ind_spec_period")

    all_tickers = set()
    for info in _INDUSTRY_HOLDINGS.values():
        all_tickers.update(info["stocks"].keys())
    returns = _fetch_returns_multi(tuple(sorted(all_tickers)))

    if not returns:
        st.warning("Could not fetch stock returns.")
        return

    rows = []
    for ind_etf, info in _INDUSTRY_HOLDINGS.items():
        items = []
        for ticker, name in info["stocks"].items():
            if ticker in returns:
                items.append({"ticker": ticker, "label": f"{ticker} ({name})", **returns[ticker]})
        if items:
            rows.append({"name": f"{ind_etf} ({info['name']})", "items": items})

    # Sort rows by average return for selected period
    rows.sort(key=lambda r: np.mean([i.get(period, 0) for i in r["items"]]), reverse=True)

    if rows:
        st.caption(f"Sorted by {period} return (best → worst). Each cell = stock with name.")
        st.markdown(_render_spectrum_html(rows, period), unsafe_allow_html=True)
