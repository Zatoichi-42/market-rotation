"""Executive Briefing — Page 1 of the dashboard. Plain English."""
import streamlit as st
from engine.language import generate_executive_briefing
from engine.crisis_alignment import detect_crisis_type


SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


def render_briefing(result: dict):
    """Render the executive briefing panel."""
    from datetime import datetime, timezone
    from engine.trade_journal import compute_target_pct

    regime = result.get("regime")
    regime_char = result.get("regime_character")
    crisis_types = result.get("crisis_types", [])
    trade_states = result.get("trade_states", {})
    horizon_readings = result.get("horizon_readings", {})
    journal_summary = result.get("journal_summary")
    gold_div = result.get("gold_divergence_reading")
    vix_val = result.get("vix_val", 20.0)
    journal_calls = result.get("journal_calls", [])
    treasury_ctx = result.get("treasury_context")

    # ── Compute fresh targets for all sectors ──
    fresh_targets = {}
    for ticker in SECTOR_NAMES:
        ts = trade_states.get(ticker)
        if not ts:
            continue
        hr = horizon_readings.get(ticker)
        pattern = hr.pattern if hr else "No Pattern"
        target, *_ = compute_target_pct(
            ts.analysis_state, ts.confidence,
            regime.state.value if regime else "NORMAL",
            regime_char.character.value if regime_char else "Choppy",
            pattern.value if hasattr(pattern, 'value') else str(pattern),
            vix_level=vix_val,
            ticker=ticker,
            crisis_types=crisis_types,
        )
        fresh_targets[ticker] = target

    # ── Build open call lookup ──
    open_calls = []
    open_call_map = {}  # ticker → call
    for c in journal_calls:
        status = c.status if hasattr(c, 'status') else c.get("status", "")
        if status == "open":
            open_calls.append(c)
            t = c.ticker if hasattr(c, 'ticker') else c.get("ticker", "")
            open_call_map[t] = c

    # Sector targets: use journal targets for open calls, fresh for others
    sector_targets = {}
    for ticker in SECTOR_NAMES:
        if ticker in open_call_map:
            sector_targets[ticker] = (open_call_map[ticker].target_pct
                                      if hasattr(open_call_map[ticker], 'target_pct')
                                      else open_call_map[ticker].get("target_pct", 0))
        else:
            sector_targets[ticker] = fresh_targets.get(ticker, 0)

    # Determine oil signal level
    oil_level = "NORMAL"
    if regime:
        for sig in regime.signals:
            if "oil" in sig.name.lower():
                oil_level = sig.level.value

    # ── Header ──
    st.markdown(f"**DAILY BRIEFING** — {datetime.now(timezone.utc).strftime('%B %d, %Y')}")
    st.caption(f"Generated {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    # ── CURRENT PORTFOLIO (open calls) ──
    if open_calls:
        st.markdown("**━━━ CURRENT PORTFOLIO ━━━**")
        for c in open_calls:
            ticker = c.ticker if hasattr(c, 'ticker') else c.get("ticker", "?")
            name = c.name if hasattr(c, 'name') else c.get("name", "")
            label = f"{ticker} ({name})" if name else ticker
            entry_target = c.target_pct if hasattr(c, 'target_pct') else c.get("target_pct", 0)
            cur_target = c.current_target_pct if hasattr(c, 'current_target_pct') else c.get("current_target_pct")
            if cur_target is None:
                cur_target = fresh_targets.get(ticker, entry_target)
            is_prov = c.is_provisional if hasattr(c, 'is_provisional') else c.get("is_provisional", True)
            ts = c.timestamp if hasattr(c, 'timestamp') else c.get("timestamp", "")
            conf = c.confidence if hasattr(c, 'confidence') else c.get("confidence", 0)
            entry_price = c.entry_price if hasattr(c, 'entry_price') else c.get("entry_price", 0)

            direction = "Long" if entry_target > 0 else "Short" if entry_target < 0 else "Flat"

            # Detect edge decay
            decayed = False
            if entry_target > 0 and cur_target is not None and cur_target < 15:
                decayed = True
            elif entry_target < 0 and cur_target is not None and abs(cur_target) < 15:
                decayed = True

            if decayed:
                status_badge = " `EDGE DECAYED`"
            elif is_prov:
                status_badge = " `PROVISIONAL`"
            else:
                status_badge = " `OPEN`"
            ts_str = f" @ {ts}" if ts and is_prov else ""

            color = "#dc2626" if decayed else ("#f59e0b" if is_prov else "#16a34a")
            st.markdown(
                f"<div style='padding:6px 10px;background:rgba(0,0,0,0.2);border-left:3px solid {color};"
                f"border-radius:4px;margin-bottom:6px;'>"
                f"<b>{label}</b> — {direction} {entry_target:+d}%"
                f"{status_badge}{ts_str}<br>"
                f"<span style='font-size:0.85em;color:#9ca3af;'>"
                f"Entry: ${entry_price:.2f} | Conf: {conf} | "
                f"Current target: {cur_target:+d}%"
                f"{'  ⚠ below action threshold' if decayed else ''}"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        # Recently closed (edge decayed)
        recently_decayed = [c for c in journal_calls
                           if (c.status if hasattr(c, 'status') else c.get("status", "")) == "closed"
                           and (c.close_reason if hasattr(c, 'close_reason') else c.get("close_reason", "")).startswith("Edge decayed")]
        if recently_decayed:
            st.caption("Recently closed (edge decayed):")
            for c in recently_decayed[-3:]:
                ticker = c.ticker if hasattr(c, 'ticker') else c.get("ticker", "?")
                name = c.name if hasattr(c, 'name') else c.get("name", "")
                reason = c.close_reason if hasattr(c, 'close_reason') else c.get("close_reason", "")
                st.caption(f"  {ticker} ({name}): {reason}")
        st.markdown("")
    else:
        st.markdown("**━━━ CURRENT PORTFOLIO ━━━**")
        st.markdown("No open positions. Capital in bills at cash hurdle rate.")
        st.markdown("")

    # ── Generate and render the executive briefing ──
    briefing_text = generate_executive_briefing(
        regime=regime, regime_character=regime_char,
        crisis_types=crisis_types, trade_states=trade_states,
        horizon_readings=horizon_readings, sector_targets=sector_targets,
        journal_summary=journal_summary, gold_divergence=gold_div,
        oil_signal_level=oil_level, vix_level=vix_val,
        treasury_context=treasury_ctx,
    )

    for line in briefing_text.split("\n"):
        if line.startswith("━━━"):
            st.markdown(f"**{line.strip('━ ')}**")
        else:
            st.markdown(line)
