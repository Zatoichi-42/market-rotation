"""
LLM Export — structured briefing packet for non-deterministic intelligence.
Gives an LLM everything needed to add event context, sentiment, and judgment.
"""
from datetime import date
from engine.contradiction_detector import detect_contradictions
from engine.market_calendar import get_market_status


_SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


def _safe(val, fmt=".4f", default="—"):
    """Safely format a numeric value."""
    if val is None:
        return default
    try:
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return str(val)


def _get_price_date(result: dict) -> date:
    """Extract the price date from result."""
    prices = result.get("prices")
    if prices is not None and len(prices) > 0:
        return prices.index[-1].date()
    return date.today()


def generate_llm_briefing(result: dict) -> str:
    """Generate the full LLM briefing packet as structured text."""
    sections = []

    # ── SECTION 1: SYSTEM STATE SUMMARY ──────────────────
    s1 = []
    s1.append("## SECTION 1: SYSTEM STATE SUMMARY")
    s1.append("")

    regime = result.get("regime")
    regime_char = result.get("regime_character")
    treasury = result.get("treasury_context")

    regime_state = regime.state.value if regime else "UNKNOWN"
    s1.append(f"Regime Gate: {regime_state}")

    if regime:
        s1.append(f"Hostile signals: {regime.hostile_count}, Fragile signals: {regime.fragile_count}, Normal signals: {regime.normal_count}")

    if regime_char:
        char_val = regime_char.character.value if hasattr(regime_char.character, "value") else str(regime_char.character)
        s1.append(f"Regime Character: {char_val} (confidence: {regime_char.confidence}%, sessions: {regime_char.sessions_in_character})")

    if treasury:
        fit_val = treasury.treasury_fit.value if hasattr(treasury.treasury_fit, "value") else str(treasury.treasury_fit)
        s1.append(f"Treasury Fit: {fit_val}")
        s1.append(f"Cash Hurdle: {treasury.cash_hurdle:.2f}% annualized")
        s1.append(f"SB-Correlation (TLT/SPY 21d): {treasury.sb_correlation:.3f}")
        s1.append(f"MOVE Index: {treasury.move_level:.1f}")

    # Market snapshot values
    vix = result.get("vix")
    vix3m = result.get("vix3m")
    vix_val = result.get("vix_val", None)
    breadth = result.get("breadth")
    credit = result.get("credit")
    correlation_reading = result.get("correlation_reading")

    if regime_char:
        s1.append(f"SPY 20d Return: {regime_char.spy_20d_return:.4f}")

    if vix_val is not None:
        s1.append(f"VIX: {vix_val:.1f}")
    elif vix is not None and len(vix) > 0:
        s1.append(f"VIX: {vix.iloc[-1]:.1f}")

    if vix is not None and vix3m is not None and len(vix) > 0 and len(vix3m) > 0:
        ratio = vix.iloc[-1] / vix3m.iloc[-1] if vix3m.iloc[-1] > 0 else 0
        struct = "backwardation" if ratio > 1.0 else "contango"
        s1.append(f"VIX Term Structure: {ratio:.3f} ({struct})")

    if breadth:
        s1.append(f"Breadth: {breadth.signal.value} (RSP/SPY z-score: {_safe(breadth.rsp_spy_ratio_zscore, '.2f')})")

    if credit is not None:
        s1.append(f"Credit (HYG-LQD z-score): {_safe(credit, '.3f')}")

    if correlation_reading:
        s1.append(f"Cross-Sector Correlation: {correlation_reading.avg_correlation:.3f} (z: {correlation_reading.avg_corr_zscore:.2f}, level: {correlation_reading.level.value})")

    # Oil signal from regime signals
    if regime:
        for sig in regime.signals:
            if "oil" in sig.name.lower():
                s1.append(f"Oil Signal: {sig.level.value} (raw: {sig.raw_value:.4f})")

    s1.append("")
    sections.append("\n".join(s1))

    # ── SECTION 2: SECTOR ROTATION TABLE ─────────────────
    s2 = []
    s2.append("## SECTION 2: SECTOR ROTATION TABLE")
    s2.append("")

    rs_readings = sorted(result.get("rs_readings", []), key=lambda r: r.rs_rank)
    states = result.get("states", {})
    pumps = result.get("pumps", {})
    reversal_map = result.get("reversal_map", {})
    trade_states = result.get("trade_states", {})
    horizon_readings = result.get("horizon_readings", {})

    s2.append("| Rank | Ticker | Name | 2d | 5d | 10d | 20d | 60d | 120d | Slope | Horizon | State | Conf | Pump | Delta | Delta_5d | Rev Score | Rev Pctl | Trade State |")
    s2.append("|------|--------|------|-----|-----|------|------|------|-------|-------|---------|-------|------|------|-------|----------|-----------|----------|-------------|")

    for r in rs_readings:
        ticker = r.ticker
        sc = states.get(ticker)
        pump = pumps.get(ticker)
        rev = reversal_map.get(ticker)
        ts = trade_states.get(ticker)
        hr = horizon_readings.get(ticker)

        state_val = sc.state.value if sc else "—"
        conf = sc.confidence if sc else "—"

        pump_score = f"{pump.pump_score:.3f}" if pump else "—"
        pump_delta = f"{pump.pump_delta:+.4f}" if pump else "—"
        pump_delta_5d = f"{pump.pump_delta_5d_avg:+.4f}" if pump else "—"

        rev_score = f"{rev.reversal_score:.3f}" if rev else "—"
        rev_pctl = f"{rev.reversal_percentile:.0f}" if rev else "—"

        trade_val = ts.trade_state.value if ts else "—"

        horizon_val = hr.pattern.value if hr else "—"

        rs_2d = f"{getattr(r, 'rs_2d', 0.0)*100:+.2f}%"
        rs_5d = f"{r.rs_5d*100:+.2f}%"
        rs_10d = f"{getattr(r, 'rs_10d', 0.0)*100:+.2f}%"
        rs_20d = f"{r.rs_20d*100:+.2f}%"
        rs_60d = f"{r.rs_60d*100:+.2f}%"
        rs_120d = f"{getattr(r, 'rs_120d', 0.0)*100:+.2f}%"
        slope = f"{r.rs_slope:+.4f}"

        s2.append(f"| {r.rs_rank} | {ticker} | {r.name} | {rs_2d} | {rs_5d} | {rs_10d} | {rs_20d} | {rs_60d} | {rs_120d} | {slope} | {horizon_val} | {state_val} | {conf} | {pump_score} | {pump_delta} | {pump_delta_5d} | {rev_score} | {rev_pctl} | {trade_val} |")

    s2.append("")
    sections.append("\n".join(s2))

    # ── SECTION 3: INDUSTRY ROTATION TABLE ───────────────
    s3 = []
    s3.append("## SECTION 3: INDUSTRY ROTATION TABLE")
    s3.append("")

    industry_rs = sorted(result.get("industry_rs", []), key=lambda x: x.rs_rank)
    if industry_rs:
        s3.append("| Rank | Ticker | Name | Parent | 2d | 5d | 10d | 20d | 60d | 120d | Slope | vs_parent_20d | Horizon | State | Conf | Pump | Delta | Rev Score | Rev Pctl | Trade State |")
        s3.append("|------|--------|------|--------|-----|-----|------|------|------|-------|-------|---------------|---------|-------|------|------|-------|-----------|----------|-------------|")

        for ir in industry_rs:
            ticker = ir.ticker
            sc = states.get(ticker)
            pump = pumps.get(ticker)
            rev = reversal_map.get(ticker)
            ts = trade_states.get(ticker)
            hr = horizon_readings.get(ticker)

            state_val = sc.state.value if sc else "—"
            conf = sc.confidence if sc else "—"

            pump_score = f"{pump.pump_score:.3f}" if pump else "—"
            pump_delta = f"{pump.pump_delta:+.4f}" if pump else "—"

            rev_score = f"{rev.reversal_score:.3f}" if rev else "—"
            rev_pctl = f"{rev.reversal_percentile:.0f}" if rev else "—"

            trade_val = ts.trade_state.value if ts else "—"
            horizon_val = hr.pattern.value if hr else "—"

            rs_2d = f"{getattr(ir, 'rs_2d', 0.0)*100:+.2f}%"
            rs_5d = f"{ir.rs_5d*100:+.2f}%"
            rs_10d = f"{getattr(ir, 'rs_10d', 0.0)*100:+.2f}%"
            rs_20d = f"{ir.rs_20d*100:+.2f}%"
            rs_60d = f"{ir.rs_60d*100:+.2f}%"
            rs_120d = f"{getattr(ir, 'rs_120d', 0.0)*100:+.2f}%"
            slope = f"{ir.rs_slope:+.4f}"
            vs_parent = f"{ir.rs_20d_vs_parent*100:+.2f}%"

            s3.append(f"| {ir.rs_rank} | {ticker} | {ir.name} | {ir.parent_sector} | {rs_2d} | {rs_5d} | {rs_10d} | {rs_20d} | {rs_60d} | {rs_120d} | {slope} | {vs_parent} | {horizon_val} | {state_val} | {conf} | {pump_score} | {pump_delta} | {rev_score} | {rev_pctl} | {trade_val} |")
    else:
        s3.append("No industry data available.")

    s3.append("")
    sections.append("\n".join(s3))

    # ── SECTION 4: REVERSAL DIAGNOSTICS ──────────────────
    s4 = []
    s4.append("## SECTION 4: REVERSAL DIAGNOSTICS")
    s4.append("")

    reversal_scores = result.get("reversal_scores", [])
    top_reversals = sorted(reversal_scores, key=lambda x: x.reversal_percentile, reverse=True)[:5]

    if top_reversals:
        for rev in top_reversals:
            s4.append(f"### {rev.ticker} ({rev.name}) — Percentile: {rev.reversal_percentile:.0f}")
            s4.append(f"  Reversal Score: {rev.reversal_score:.4f}")
            s4.append(f"  Breadth Deterioration: {rev.breadth_det_pillar:.2f}")
            s4.append(f"  Price Break: {rev.price_break_pillar:.2f}")
            s4.append(f"  Crowding: {rev.crowding_pillar:.2f}")
            s4.append(f"  Above 75th: {rev.above_75th}")
            if rev.sub_signals:
                s4.append("  Sub-signals:")
                for k, v in rev.sub_signals.items():
                    s4.append(f"    {k}: {v:.4f}")
            s4.append("")
    else:
        s4.append("No reversal data available.")
        s4.append("")

    sections.append("\n".join(s4))

    # ── SECTION 5: CONTRADICTIONS AND OPEN QUESTIONS ─────
    s5 = []
    s5.append("## SECTION 5: CONTRADICTIONS AND OPEN QUESTIONS")
    s5.append("")

    try:
        contradictions = detect_contradictions(result)
    except Exception:
        contradictions = []

    if contradictions:
        for c in contradictions:
            s5.append(f"### [{c['severity']}] {c['type']} — {c['ticker']}")
            s5.append(f"  Detail: {c['detail']}")
            s5.append(f"  Question: {c['question']}")
            s5.append("")
    else:
        s5.append("No contradictions detected.")
        s5.append("")

    sections.append("\n".join(s5))

    # ── SECTION 6: WHAT THE LLM SHOULD EVALUATE ─────────
    s6 = []
    s6.append("## SECTION 6: WHAT THE LLM SHOULD EVALUATE")
    s6.append("")
    s6.append("The system above is purely quantitative. You should add judgment on:")
    s6.append("")
    s6.append("1. **Event Context**: Are there earnings, Fed meetings, CPI releases, or geopolitical events")
    s6.append("   in the next 1-5 days that could invalidate or amplify any signal above?")
    s6.append("")
    s6.append("2. **Crisis Alignment**: If a crisis type is active, does the crisis narrative match the")
    s6.append("   quantitative signal? E.g., if OIL_SHOCK is active, is there a real supply disruption")
    s6.append("   or is it just price action noise?")
    s6.append("")
    s6.append("3. **Catalyst Calendar**: Which sectors have binary events (earnings, regulatory decisions)")
    s6.append("   that make entry risky regardless of quantitative signal strength?")
    s6.append("")
    s6.append("4. **Sentiment Read**: Is positioning crowded in any direction? Are retail/institutional")
    s6.append("   flows confirming or contradicting the rotation signals?")
    s6.append("")
    s6.append("5. **Weekend/Gap Risk**: If this is a Friday or pre-holiday reading, which open positions")
    s6.append("   carry unacceptable overnight gap risk? Should any be closed or hedged?")
    s6.append("")

    sections.append("\n".join(s6))

    # ── SECTION 7: SIGNAL RELIABILITY ────────────────────
    s7 = []
    s7.append("## SECTION 7: SIGNAL RELIABILITY")
    s7.append("")

    journal_summary = result.get("journal_summary")
    if journal_summary:
        s7.append("### Hit Rates by Analysis State")
        hr_by_state = getattr(journal_summary, "hit_rate_by_state", {})
        pnl_by_state = getattr(journal_summary, "pnl_by_state", {})
        if hr_by_state:
            s7.append("| State | Hit Rate | Avg P&L |")
            s7.append("|-------|----------|---------|")
            for state, rate in sorted(hr_by_state.items()):
                pnl = pnl_by_state.get(state, 0)
                s7.append(f"| {state} | {rate:.0%} | {pnl:+.2f}% |")
            s7.append("")

        s7.append("### Hit Rates by Regime")
        hr_by_regime = getattr(journal_summary, "hit_rate_by_regime", {})
        pnl_by_regime = getattr(journal_summary, "pnl_by_regime", {})
        if hr_by_regime:
            s7.append("| Regime | Hit Rate | Avg P&L |")
            s7.append("|--------|----------|---------|")
            for regime_key, rate in sorted(hr_by_regime.items()):
                pnl = pnl_by_regime.get(regime_key, 0)
                s7.append(f"| {regime_key} | {rate:.0%} | {pnl:+.2f}% |")
            s7.append("")

        s7.append("### Hit Rates by Horizon Pattern")
        hr_by_pattern = getattr(journal_summary, "hit_rate_by_pattern", {})
        pnl_by_pattern = getattr(journal_summary, "pnl_by_pattern", {})
        if hr_by_pattern:
            s7.append("| Pattern | Hit Rate | Avg P&L |")
            s7.append("|---------|----------|---------|")
            for pat, rate in sorted(hr_by_pattern.items()):
                pnl = pnl_by_pattern.get(pat, 0)
                s7.append(f"| {pat} | {rate:.0%} | {pnl:+.2f}% |")
            s7.append("")

        s7.append("### Hit Rates by Confidence Bucket")
        hr_by_conf = getattr(journal_summary, "hit_rate_by_confidence", {})
        pnl_by_conf = getattr(journal_summary, "pnl_by_confidence", {})
        if hr_by_conf:
            s7.append("| Confidence | Hit Rate | Avg P&L |")
            s7.append("|------------|----------|---------|")
            for conf_key, rate in sorted(hr_by_conf.items()):
                pnl = pnl_by_conf.get(conf_key, 0)
                s7.append(f"| {conf_key} | {rate:.0%} | {pnl:+.2f}% |")
            s7.append("")
    else:
        s7.append("No journal data available for signal reliability analysis.")
        s7.append("")

    sections.append("\n".join(s7))

    # ── SECTION 8: TRADE JOURNAL CONTEXT ─────────────────
    s8 = []
    s8.append("## SECTION 8: TRADE JOURNAL CONTEXT")
    s8.append("")

    journal_calls = result.get("journal_calls", [])
    open_calls = [c for c in journal_calls
                  if (c.status if hasattr(c, "status") else c.get("status", "")) == "open"]

    if journal_summary:
        s8.append(f"Total calls: {journal_summary.total_calls}")
        s8.append(f"Open calls: {journal_summary.open_calls}")
        s8.append(f"Closed calls: {journal_summary.closed_calls}")
        s8.append(f"Cumulative P&L (10d): {journal_summary.total_pnl_10d:+.2f}%")
        s8.append(f"Cumulative P&L (20d): {journal_summary.total_pnl_20d:+.2f}%")
        s8.append(f"Hit Rate (10d): {journal_summary.hit_rate_10d:.0%}")
        s8.append(f"Hit Rate (20d): {journal_summary.hit_rate_20d:.0%}")
        s8.append("")

    if open_calls:
        s8.append("### Open Calls")
        s8.append("| Ticker | Name | State | Trade | Target | Entry Price | P&L 5d | P&L 10d |")
        s8.append("|--------|------|-------|-------|--------|-------------|--------|---------|")
        for c in open_calls:
            ticker = c.ticker if hasattr(c, "ticker") else c.get("ticker", "?")
            name = c.name if hasattr(c, "name") else c.get("name", "?")
            a_state = c.analysis_state if hasattr(c, "analysis_state") else c.get("analysis_state", "?")
            t_state = c.trade_state if hasattr(c, "trade_state") else c.get("trade_state", "?")
            target = c.target_pct if hasattr(c, "target_pct") else c.get("target_pct", 0)
            entry = c.entry_price if hasattr(c, "entry_price") else c.get("entry_price", 0)
            pnl_5 = c.pnl_5d if hasattr(c, "pnl_5d") else c.get("pnl_5d", None)
            pnl_10 = c.pnl_10d if hasattr(c, "pnl_10d") else c.get("pnl_10d", None)
            pnl_5_str = f"{pnl_5:+.2f}%" if pnl_5 is not None else "—"
            pnl_10_str = f"{pnl_10:+.2f}%" if pnl_10 is not None else "—"
            s8.append(f"| {ticker} | {name} | {a_state} | {t_state} | {target:+d}% | {entry:.2f} | {pnl_5_str} | {pnl_10_str} |")
        s8.append("")
    elif not journal_summary:
        s8.append("No trade journal data available.")
        s8.append("")

    sections.append("\n".join(s8))

    # ── SECTION 9: RAW PARAMETERS ────────────────────────
    s9 = []
    s9.append("## SECTION 9: RAW PARAMETERS")
    s9.append("")

    for r in rs_readings:
        ticker = r.ticker
        s9.append(f"### {ticker} ({r.name})")

        # RS readings
        s9.append(f"  rs_2d={getattr(r, 'rs_2d', 0.0):.6f}")
        s9.append(f"  rs_5d={r.rs_5d:.6f}")
        s9.append(f"  rs_10d={getattr(r, 'rs_10d', 0.0):.6f}")
        s9.append(f"  rs_20d={r.rs_20d:.6f}")
        s9.append(f"  rs_60d={r.rs_60d:.6f}")
        s9.append(f"  rs_120d={getattr(r, 'rs_120d', 0.0):.6f}")
        s9.append(f"  rs_slope={r.rs_slope:.6f}")
        s9.append(f"  rs_composite={r.rs_composite:.4f}")
        s9.append(f"  rs_rank={r.rs_rank}")
        s9.append(f"  rs_rank_change={r.rs_rank_change}")

        # Pump
        pump = pumps.get(ticker)
        if pump:
            s9.append(f"  pump_score={pump.pump_score:.4f}")
            s9.append(f"  pump_delta={pump.pump_delta:+.4f}")
            s9.append(f"  pump_delta_5d_avg={pump.pump_delta_5d_avg:+.4f}")
            s9.append(f"  rs_pillar={pump.rs_pillar:.4f}")
            s9.append(f"  participation_pillar={pump.participation_pillar:.4f}")
            s9.append(f"  flow_pillar={pump.flow_pillar:.4f}")

        # State
        sc = states.get(ticker)
        if sc:
            s9.append(f"  analysis_state={sc.state.value}")
            s9.append(f"  confidence={sc.confidence}")
            s9.append(f"  sessions_in_state={sc.sessions_in_state}")
            s9.append(f"  transition_pressure={sc.transition_pressure.value}")
            s9.append(f"  state_changed={sc.state_changed}")

        # Reversal
        rev = reversal_map.get(ticker)
        if rev:
            s9.append(f"  reversal_score={rev.reversal_score:.4f}")
            s9.append(f"  reversal_percentile={rev.reversal_percentile:.1f}")
            s9.append(f"  above_75th={rev.above_75th}")

        # Horizon
        hr = horizon_readings.get(ticker)
        if hr:
            s9.append(f"  horizon_pattern={hr.pattern.value}")
            s9.append(f"  horizon_conviction={hr.conviction}")
            s9.append(f"  horizon_is_rotation={hr.is_rotation_signal}")
            s9.append(f"  horizon_is_trap={hr.is_trap}")
            s9.append(f"  horizon_is_entry_zone={hr.is_entry_zone}")

        s9.append("")

    sections.append("\n".join(s9))

    # ── SECTION 10: MARKET STATUS ────────────────────────
    s10 = []
    s10.append("## SECTION 10: MARKET STATUS")
    s10.append("")

    price_date = _get_price_date(result)
    try:
        status = get_market_status(price_date)
        s10.append(f"Price Date: {price_date.isoformat()}")
        s10.append(f"Is Trading Day: {status['is_trading_day']}")
        s10.append(f"Last Close: {status['last_close']}")
        s10.append(f"Next Open: {status['next_open']}")
        s10.append(f"Staleness (calendar days): {status['staleness_calendar_days']}")
        s10.append(f"Staleness (trading days): {status['staleness_trading_days']}")
        s10.append(f"Reason: {status['reason']}")
        s10.append(f"Note: {status['note']}")
        s10.append("")

        # Weekend risk factors
        if status["reason"] == "weekend":
            s10.append("WEEKEND RISK FACTORS:")
            s10.append("  - All positions carry 2+ days of overnight gap risk")
            s10.append("  - Geopolitical events over the weekend cannot be hedged")
            s10.append("  - Consider reducing directional exposure Friday afternoon")
            s10.append("  - VIX term structure may shift significantly at Monday open")
            s10.append("")
        elif status["reason"] == "holiday":
            s10.append("HOLIDAY RISK FACTORS:")
            s10.append("  - Extended closure increases gap risk")
            s10.append("  - Liquidity may be thin at re-open")
            s10.append("  - International markets may trade during US closure, creating dislocations")
            s10.append("")
    except Exception:
        s10.append(f"Price Date: {price_date.isoformat()}")
        s10.append("Market status computation failed.")
        s10.append("")

    sections.append("\n".join(s10))

    return "\n".join(sections)
