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
    # Extract all needed data from result
    regime = result.get("regime")
    regime_char = result.get("regime_character")
    crisis_types = result.get("crisis_types", [])
    trade_states = result.get("trade_states", {})
    horizon_readings = result.get("horizon_readings", {})
    journal_summary = result.get("journal_summary")
    gold_div = result.get("gold_divergence_reading")
    vix_val = result.get("vix_val", 20.0)

    # Compute sector targets for briefing
    from engine.trade_journal import compute_target_pct
    sector_targets = {}
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
        sector_targets[ticker] = target

    # Determine oil signal level from regime signals
    oil_level = "NORMAL"
    if regime:
        for sig in regime.signals:
            if "oil" in sig.name.lower():
                oil_level = sig.level.value

    # Generate briefing text
    treasury_ctx = result.get("treasury_context")
    briefing_text = generate_executive_briefing(
        regime=regime, regime_character=regime_char,
        crisis_types=crisis_types, trade_states=trade_states,
        horizon_readings=horizon_readings, sector_targets=sector_targets,
        journal_summary=journal_summary, gold_divergence=gold_div,
        oil_signal_level=oil_level, vix_level=vix_val,
        treasury_context=treasury_ctx,
    )

    # Display
    from datetime import datetime, timezone
    st.markdown(f"**DAILY BRIEFING** — {datetime.now(timezone.utc).strftime('%B %d, %Y')}")
    st.caption(f"Generated {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    # Render each section with styling
    for line in briefing_text.split("\n"):
        if line.startswith("━━━"):
            st.markdown(f"**{line.strip('━ ')}**")
        else:
            st.markdown(line)
