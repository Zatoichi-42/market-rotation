"""
Fetch, cache, and flag ETF valuation metrics.

Valuations are DISPLAY ONLY — NOT signal inputs.
They provide context for the operator but do not affect state classification.
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


# Approximate 5yr sector P/E averages (FactSet/Yardeni/SPDR, as of 2024-2025)
SECTOR_PE_AVERAGES = {
    "XLK":  {"pe_avg": 28.0, "pe_std": 5.0, "fwd_pe_avg": 24.0, "fwd_pe_std": 4.0},
    "XLV":  {"pe_avg": 18.0, "pe_std": 3.0, "fwd_pe_avg": 16.0, "fwd_pe_std": 2.5},
    "XLF":  {"pe_avg": 14.0, "pe_std": 2.5, "fwd_pe_avg": 12.5, "fwd_pe_std": 2.0},
    "XLE":  {"pe_avg": 15.0, "pe_std": 5.0, "fwd_pe_avg": 12.0, "fwd_pe_std": 4.0},
    "XLI":  {"pe_avg": 20.0, "pe_std": 3.0, "fwd_pe_avg": 18.0, "fwd_pe_std": 2.5},
    "XLU":  {"pe_avg": 19.0, "pe_std": 2.5, "fwd_pe_avg": 17.0, "fwd_pe_std": 2.0},
    "XLRE": {"pe_avg": 38.0, "pe_std": 8.0, "fwd_pe_avg": 32.0, "fwd_pe_std": 6.0},
    "XLC":  {"pe_avg": 22.0, "pe_std": 5.0, "fwd_pe_avg": 18.0, "fwd_pe_std": 4.0},
    "XLY":  {"pe_avg": 25.0, "pe_std": 5.0, "fwd_pe_avg": 22.0, "fwd_pe_std": 4.0},
    "XLP":  {"pe_avg": 22.0, "pe_std": 2.5, "fwd_pe_avg": 20.0, "fwd_pe_std": 2.0},
    "XLB":  {"pe_avg": 18.0, "pe_std": 4.0, "fwd_pe_avg": 15.0, "fwd_pe_std": 3.0},
}


@st.cache_data(ttl=86400)
def fetch_valuations_raw(tickers: tuple) -> pd.DataFrame:
    """Fetch raw valuation metrics (before formatting). Cached 24h."""
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker": t,
                "_pe_raw": info.get("trailingPE"),
                "_fwd_pe_raw": info.get("forwardPE"),
                "_pb_raw": info.get("priceToBook"),
                "_div_yield_raw": info.get("dividendYield"),
                "_expense_raw": info.get("annualReportExpenseRatio"),
                "_aum_raw": info.get("totalAssets"),
                "_52w_high_raw": info.get("fiftyTwoWeekHigh"),
                "_price_raw": info.get("previousClose") or info.get("regularMarketPrice"),
            })
        except Exception:
            rows.append({"Ticker": t})
    return pd.DataFrame(rows)


def fetch_valuations(tickers: list[str]) -> pd.DataFrame:
    """Fetch and format valuation metrics."""
    raw = fetch_valuations_raw(tuple(tickers))
    if raw.empty:
        return raw

    df = raw.copy()
    df["P/E"] = df["_pe_raw"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    df["Fwd P/E"] = df["_fwd_pe_raw"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    df["P/B"] = df["_pb_raw"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    df["Div Yield"] = df["_div_yield_raw"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    df["AUM ($B)"] = df["_aum_raw"].apply(
        lambda x: f"{x/1e9:.1f}" if pd.notna(x) and x > 0 else "—")
    df["Expense Ratio"] = df["_expense_raw"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")

    if "_price_raw" in df.columns and "_52w_high_raw" in df.columns:
        df["% from 52w High"] = df.apply(
            lambda r: f"{(r['_price_raw']/r['_52w_high_raw']-1):.1%}"
            if pd.notna(r.get("_price_raw")) and pd.notna(r.get("_52w_high_raw")) and r["_52w_high_raw"] > 0
            else "—", axis=1)

    return df


def compute_valuation_flags(
    tickers: list[str],
    sigma_threshold: float = 1.0,
) -> list[dict]:
    """
    Flag valuations >1σ from approximate 5-year average.
    Returns list of flag dicts with ticker, field, current, avg, sigma, direction, message.
    """
    raw = fetch_valuations_raw(tuple(tickers))
    if raw.empty:
        return []

    flags = []
    for _, row in raw.iterrows():
        ticker = row.get("Ticker")
        if not ticker:
            continue
        avgs = SECTOR_PE_AVERAGES.get(ticker)
        if not avgs:
            continue

        # Check P/E
        pe = row.get("_pe_raw")
        if pd.notna(pe) and avgs.get("pe_avg") and avgs.get("pe_std") and avgs["pe_std"] > 0:
            sigma = (pe - avgs["pe_avg"]) / avgs["pe_std"]
            if abs(sigma) >= sigma_threshold:
                direction = "above" if sigma > 0 else "below"
                lang = "historically expensive" if sigma > 0 else "historically cheap"
                flags.append({
                    "ticker": ticker, "field": "P/E",
                    "current": pe, "avg_5yr": avgs["pe_avg"], "std_5yr": avgs["pe_std"],
                    "sigma": round(sigma, 1), "direction": direction,
                    "message": f"{ticker} P/E ({pe:.1f}) is {abs(sigma):.1f}σ {direction.upper()} 5yr avg ({avgs['pe_avg']:.0f}) — {lang}",
                })

        # Check Fwd P/E
        fwd_pe = row.get("_fwd_pe_raw")
        if pd.notna(fwd_pe) and avgs.get("fwd_pe_avg") and avgs.get("fwd_pe_std") and avgs["fwd_pe_std"] > 0:
            sigma = (fwd_pe - avgs["fwd_pe_avg"]) / avgs["fwd_pe_std"]
            if abs(sigma) >= sigma_threshold:
                direction = "above" if sigma > 0 else "below"
                lang = "historically expensive" if sigma > 0 else "historically cheap"
                flags.append({
                    "ticker": ticker, "field": "Fwd P/E",
                    "current": fwd_pe, "avg_5yr": avgs["fwd_pe_avg"], "std_5yr": avgs["fwd_pe_std"],
                    "sigma": round(sigma, 1), "direction": direction,
                    "message": f"{ticker} Fwd P/E ({fwd_pe:.1f}) is {abs(sigma):.1f}σ {direction.upper()} 5yr avg ({avgs['fwd_pe_avg']:.0f}) — {lang}",
                })

    return flags


def render_valuations_panel(tickers: list[str], tab_label: str = "Sector"):
    """Render the full valuations panel with table + sigma flags."""
    with st.expander(f"{tab_label} Valuations (Context Only — Not a Signal)", expanded=False):
        val_df = fetch_valuations(tickers)
        if val_df.empty:
            st.info("Valuation data unavailable.")
            return

        display_cols = [c for c in [
            "Ticker", "P/E", "Fwd P/E", "P/B", "Div Yield",
            "AUM ($B)", "% from 52w High", "Expense Ratio",
        ] if c in val_df.columns]

        # Add ⚠ markers to flagged cells
        flags = compute_valuation_flags(tickers)
        flagged_cells = {}  # (ticker, field) → sigma
        for f in flags:
            flagged_cells[(f["ticker"], f["field"])] = f["sigma"]

        # Mark flagged values in the display DataFrame
        df_display = val_df[display_cols].copy()
        for (ticker, field), sigma in flagged_cells.items():
            mask = df_display["Ticker"] == ticker
            if field in df_display.columns and mask.any():
                idx = df_display.index[mask][0]
                current_val = df_display.at[idx, field]
                if current_val != "—":
                    df_display.at[idx, field] = f"{current_val} ⚠"

        st.dataframe(df_display, width="stretch", hide_index=True)

        # Extreme flags
        if flags:
            st.markdown("**Valuation Flags** (>1σ from approximate 5-year average):")
            for f in sorted(flags, key=lambda x: abs(x["sigma"]), reverse=True):
                icon = "🔴" if f["direction"] == "above" else "🟢"
                st.markdown(f"&nbsp;&nbsp;{icon} {f['message']}")

        st.caption(
            "⚠ Valuations are operator context — NOT a trading signal in this system. "
            "5yr averages are approximate (FactSet/Yardeni). "
            "Data from yfinance, cached 24h."
        )
