"""
Export — generates a full system report in multiple formats.

Formats:
- Claude XML: structured for LLM ingestion with all signals, states, and context
- Markdown: human-readable summary
- CSV: raw data tables for spreadsheet analysis
- JSON: machine-readable complete dump
"""
import io
import json
import csv
from datetime import datetime, timezone

import streamlit as st
import pandas as pd

from engine.schemas import AnalysisState


_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


def render_export_button(result: dict):
    """Render the export sidebar with format selection and download button."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Report")
    fmt = st.sidebar.selectbox("Format", ["Claude XML", "Markdown", "CSV", "JSON"], key="export_fmt")

    if st.sidebar.button("Generate Export", key="export_btn"):
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        if fmt == "Claude XML":
            content = _build_claude_xml(result)
            st.sidebar.download_button("Download XML", content, f"pump_report_{now}.xml",
                                        mime="application/xml", key="dl_xml")
        elif fmt == "Markdown":
            content = _build_markdown(result)
            st.sidebar.download_button("Download MD", content, f"pump_report_{now}.md",
                                        mime="text/markdown", key="dl_md")
        elif fmt == "CSV":
            content = _build_csv(result)
            st.sidebar.download_button("Download CSV", content, f"pump_report_{now}.csv",
                                        mime="text/csv", key="dl_csv")
        elif fmt == "JSON":
            content = _build_json(result)
            st.sidebar.download_button("Download JSON", content, f"pump_report_{now}.json",
                                        mime="application/json", key="dl_json")


# ═══════════════════════════════════════════════════════
# CLAUDE XML — optimized for LLM ingestion
# ═══════════════════════════════════════════════════════

def _build_claude_xml(result: dict) -> str:
    regime = result["regime"]
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    breadth = result.get("breadth")
    industry_rs = result.get("industry_rs", [])
    reversal_scores = result.get("reversal_scores", [])
    prices = result["prices"]
    last_date = prices.index[-1].strftime("%Y-%m-%d")

    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append(f'<pump_rotation_report date="{last_date}" '
                 f'generated="{datetime.now(timezone.utc).isoformat()}">')

    # Regime
    lines.append(f'  <regime state="{regime.state.value}" '
                 f'hostile_count="{regime.hostile_count}" '
                 f'fragile_count="{regime.fragile_count}">')
    for sig in regime.signals:
        lines.append(f'    <signal name="{sig.name}" value="{sig.raw_value:.4f}" '
                     f'level="{sig.level.value}"/>')
    lines.append(f'    <explanation>{_xml_escape(regime.explanation)}</explanation>')
    lines.append('  </regime>')

    # Breadth
    if breadth:
        lines.append(f'  <breadth signal="{breadth.signal.value}" '
                     f'ratio="{breadth.rsp_spy_ratio:.6f}" '
                     f'zscore="{breadth.rsp_spy_ratio_zscore:.4f}" '
                     f'change_20d="{breadth.rsp_spy_ratio_20d_change:.6f}"/>')

    # Sectors
    lines.append('  <sectors>')
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        rev_map = {rv.ticker: rv for rv in reversal_scores}
        rev = rev_map.get(r.ticker)
        lines.append(f'    <sector ticker="{r.ticker}" name="{r.name}" rank="{r.rs_rank}">')
        lines.append(f'      <rs rs_5d="{r.rs_5d:.6f}" rs_20d="{r.rs_20d:.6f}" '
                     f'rs_60d="{r.rs_60d:.6f}" slope="{r.rs_slope:.6f}" '
                     f'composite="{r.rs_composite:.2f}"/>')
        if pump:
            lines.append(f'      <pump score="{pump.pump_score:.4f}" '
                         f'delta="{pump.pump_delta:.4f}" '
                         f'delta_5d_avg="{pump.pump_delta_5d_avg:.4f}"/>')
        if rev:
            lines.append(f'      <reversal score="{rev.reversal_score:.4f}" '
                         f'percentile="{rev.reversal_percentile:.1f}" '
                         f'above_75th="{rev.above_75th}" '
                         f'breadth_det="{rev.breadth_det_pillar:.1f}" '
                         f'price_break="{rev.price_break_pillar:.1f}" '
                         f'crowding="{rev.crowding_pillar:.1f}"/>')
        if state:
            lines.append(f'      <state value="{state.state.value}" '
                         f'confidence="{state.confidence}" '
                         f'sessions="{state.sessions_in_state}" '
                         f'pressure="{state.transition_pressure.value}"/>')
            lines.append(f'      <explanation>{_xml_escape(state.explanation)}</explanation>')
        lines.append('    </sector>')
    lines.append('  </sectors>')

    # Industries
    if industry_rs:
        lines.append('  <industries>')
        for ir in sorted(industry_rs, key=lambda x: x.rs_rank):
            state = states.get(ir.ticker)
            parent_state = states.get(ir.parent_sector)
            lines.append(f'    <industry ticker="{ir.ticker}" name="{ir.name}" '
                         f'parent="{ir.parent_sector}" rank="{ir.rs_rank}">')
            lines.append(f'      <rs rs_5d="{ir.rs_5d:.6f}" rs_20d="{ir.rs_20d:.6f}" '
                         f'rs_60d="{ir.rs_60d:.6f}" slope="{ir.rs_slope:.6f}" '
                         f'composite="{ir.industry_composite:.2f}"/>')
            lines.append(f'      <rs_vs_parent rs_5d="{ir.rs_5d_vs_parent:.6f}" '
                         f'rs_20d="{ir.rs_20d_vs_parent:.6f}" '
                         f'rs_60d="{ir.rs_60d_vs_parent:.6f}"/>')
            if state:
                lines.append(f'      <state value="{state.state.value}" '
                             f'confidence="{state.confidence}"/>')
            if parent_state:
                lines.append(f'      <parent_state value="{parent_state.state.value}"/>')
            lines.append('    </industry>')
        lines.append('  </industries>')

    # 1d moves
    lines.append('  <today_moves>')
    spy_1d = prices["SPY"].pct_change().iloc[-1] if "SPY" in prices.columns else 0
    for r in rs_readings:
        if r.ticker in prices.columns:
            sec_1d = prices[r.ticker].pct_change().iloc[-1]
            rs_1d = sec_1d - spy_1d
            lines.append(f'    <move ticker="{r.ticker}" return_1d="{sec_1d:.6f}" '
                         f'rs_1d="{rs_1d:.6f}"/>')
    lines.append('  </today_moves>')

    lines.append('</pump_rotation_report>')
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# MARKDOWN
# ═══════════════════════════════════════════════════════

def _build_markdown(result: dict) -> str:
    regime = result["regime"]
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    breadth = result.get("breadth")
    industry_rs = result.get("industry_rs", [])
    prices = result["prices"]
    last_date = prices.index[-1].strftime("%Y-%m-%d")

    lines = [f"# Pump Rotation System Report — {last_date}", ""]

    # Regime
    lines.append(f"## Regime: **{regime.state.value}**")
    lines.append(f"> {regime.explanation}")
    lines.append("")
    lines.append("| Signal | Value | Level |")
    lines.append("|--------|-------|-------|")
    for sig in regime.signals:
        lines.append(f"| {sig.name} | {sig.raw_value:.2f} | {sig.level.value} |")
    lines.append("")

    # Breadth
    if breadth:
        import math
        z = f"{breadth.rsp_spy_ratio_zscore:.2f}" if not math.isnan(breadth.rsp_spy_ratio_zscore) else "N/A"
        lines.append(f"## Breadth: **{breadth.signal.value}** (z: {z})")
        lines.append("")

    # Sectors
    lines.append("## Sector Rankings")
    lines.append("")
    lines.append("| Rank | Sector | RS 5d | RS 20d | RS 60d | Comp | Pump | Delta | State | Conf |")
    lines.append("|------|--------|-------|--------|--------|------|------|-------|-------|------|")
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        sv = state.state.value if state else "—"
        sc = f"{state.confidence}%" if state else "—"
        ps = f"{pump.pump_score:.2f}" if pump else "—"
        pd_val = f"{pump.pump_delta:+.3f}" if pump else "—"
        lines.append(f"| #{r.rs_rank} | {r.ticker} ({r.name}) | {r.rs_5d:+.2%} | "
                     f"{r.rs_20d:+.2%} | {r.rs_60d:+.2%} | {r.rs_composite:.0f} | "
                     f"{ps} | {pd_val} | {sv} | {sc} |")
    lines.append("")

    # Industries
    if industry_rs:
        lines.append("## Industry Rankings")
        lines.append("")
        lines.append("| Rank | Industry | Parent | RS 20d | vs Parent | Comp | State |")
        lines.append("|------|----------|--------|--------|-----------|------|-------|")
        for ir in sorted(industry_rs, key=lambda x: x.rs_rank):
            state = states.get(ir.ticker)
            sv = state.state.value if state else "—"
            lines.append(f"| #{ir.rs_rank} | {ir.ticker} ({ir.name}) | "
                         f"{ir.parent_sector} | {ir.rs_20d:+.2%} | "
                         f"{ir.rs_20d_vs_parent:+.2%} | {ir.industry_composite:.0f} | {sv} |")
        lines.append("")

    # 1d moves
    spy_1d = prices["SPY"].pct_change().iloc[-1] if "SPY" in prices.columns else 0
    lines.append("## Today's Moves")
    lines.append(f"SPY: {spy_1d:+.2%}")
    lines.append("")
    for r in sorted(rs_readings, key=lambda x: x.rs_rank):
        if r.ticker in prices.columns:
            sec_1d = prices[r.ticker].pct_change().iloc[-1]
            rs_1d = sec_1d - spy_1d
            lines.append(f"- {r.ticker} ({r.name}): {sec_1d:+.2%} (RS: {rs_1d:+.2%})")
    lines.append("")

    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════

def _build_csv(result: dict) -> str:
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    prices = result["prices"]
    spy_1d = prices["SPY"].pct_change().iloc[-1] if "SPY" in prices.columns else 0

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Rank", "Ticker", "Sector", "RS_5d", "RS_20d", "RS_60d", "Slope",
        "Composite", "Pump_Score", "Pump_Delta", "State", "Confidence",
        "Return_1d", "RS_1d",
    ])
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        sec_1d = prices[r.ticker].pct_change().iloc[-1] if r.ticker in prices.columns else 0
        writer.writerow([
            r.rs_rank, r.ticker, r.name,
            f"{r.rs_5d:.6f}", f"{r.rs_20d:.6f}", f"{r.rs_60d:.6f}", f"{r.rs_slope:.6f}",
            f"{r.rs_composite:.2f}",
            f"{pump.pump_score:.4f}" if pump else "",
            f"{pump.pump_delta:.4f}" if pump else "",
            state.state.value if state else "",
            state.confidence if state else "",
            f"{sec_1d:.6f}", f"{sec_1d - spy_1d:.6f}",
        ])
    return output.getvalue()


# ═══════════════════════════════════════════════════════
# JSON
# ═══════════════════════════════════════════════════════

def _build_json(result: dict) -> str:
    regime = result["regime"]
    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    industry_rs = result.get("industry_rs", [])
    reversal_scores = result.get("reversal_scores", [])
    prices = result["prices"]
    spy_1d = prices["SPY"].pct_change().iloc[-1] if "SPY" in prices.columns else 0

    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "price_date": prices.index[-1].strftime("%Y-%m-%d"),
        "regime": {
            "state": regime.state.value,
            "hostile_count": regime.hostile_count,
            "fragile_count": regime.fragile_count,
            "signals": [{"name": s.name, "value": s.raw_value, "level": s.level.value}
                        for s in regime.signals],
            "explanation": regime.explanation,
        },
        "sectors": [],
        "industries": [],
    }

    rev_map = {rv.ticker: rv for rv in reversal_scores}
    for r in rs_readings:
        state = states.get(r.ticker)
        pump = pumps.get(r.ticker)
        rev = rev_map.get(r.ticker)
        sec_1d = prices[r.ticker].pct_change().iloc[-1] if r.ticker in prices.columns else 0
        entry = {
            "rank": r.rs_rank, "ticker": r.ticker, "name": r.name,
            "rs": {"5d": r.rs_5d, "20d": r.rs_20d, "60d": r.rs_60d,
                   "slope": r.rs_slope, "composite": r.rs_composite},
            "pump": {"score": pump.pump_score, "delta": pump.pump_delta,
                     "delta_5d_avg": pump.pump_delta_5d_avg} if pump else None,
            "reversal": {"score": rev.reversal_score, "percentile": rev.reversal_percentile,
                         "above_75th": bool(rev.above_75th)} if rev else None,
            "state": {"value": state.state.value, "confidence": state.confidence,
                      "sessions": state.sessions_in_state,
                      "pressure": state.transition_pressure.value} if state else None,
            "today": {"return_1d": sec_1d, "rs_1d": sec_1d - spy_1d},
        }
        data["sectors"].append(entry)

    for ir in sorted(industry_rs, key=lambda x: x.rs_rank):
        state = states.get(ir.ticker)
        data["industries"].append({
            "rank": ir.rs_rank, "ticker": ir.ticker, "name": ir.name,
            "parent": ir.parent_sector,
            "rs": {"5d": ir.rs_5d, "20d": ir.rs_20d, "60d": ir.rs_60d,
                   "vs_parent_20d": ir.rs_20d_vs_parent,
                   "composite": ir.industry_composite},
            "state": state.state.value if state else None,
        })

    return json.dumps(data, indent=2, default=str)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
