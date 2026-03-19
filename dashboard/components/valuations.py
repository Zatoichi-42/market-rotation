"""Fetch and cache ETF valuation metrics from yfinance."""
import streamlit as st
import pandas as pd
import yfinance as yf


@st.cache_data(ttl=86400)  # Cache for 24h — valuations don't change fast
def fetch_valuations(tickers: list[str]) -> pd.DataFrame:
    """Fetch basic valuation metrics for ETFs. Returns DataFrame with ticker as index."""
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker": t,
                "P/E": info.get("trailingPE"),
                "Fwd P/E": info.get("forwardPE"),
                "P/B": info.get("priceToBook"),
                "Div Yield": info.get("dividendYield"),
                "Expense Ratio": info.get("annualReportExpenseRatio"),
                "AUM ($B)": round(info.get("totalAssets", 0) / 1e9, 2) if info.get("totalAssets") else None,
                "52w High": info.get("fiftyTwoWeekHigh"),
                "52w Low": info.get("fiftyTwoWeekLow"),
                "Price": info.get("previousClose") or info.get("regularMarketPrice"),
            })
        except Exception:
            rows.append({"Ticker": t})
    df = pd.DataFrame(rows)
    if not df.empty:
        # Compute % from 52w high/low
        if "Price" in df.columns and "52w High" in df.columns:
            df["% from 52w High"] = ((df["Price"] / df["52w High"]) - 1).apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
        if "Div Yield" in df.columns:
            df["Div Yield"] = df["Div Yield"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )
        if "P/E" in df.columns:
            df["P/E"] = df["P/E"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        if "Fwd P/E" in df.columns:
            df["Fwd P/E"] = df["Fwd P/E"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        if "P/B" in df.columns:
            df["P/B"] = df["P/B"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        if "Expense Ratio" in df.columns:
            df["Expense Ratio"] = df["Expense Ratio"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )
        if "AUM ($B)" in df.columns:
            df["AUM ($B)"] = df["AUM ($B)"].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "—")
    return df
