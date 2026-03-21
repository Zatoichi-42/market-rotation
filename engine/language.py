"""
Language Module — plain-English templates for executive briefings.
Translates system state into human-readable narratives.
"""
from engine.schemas import (
    CrisisType, RegimeState, RegimeCharacter, HorizonPattern,
    AnalysisState, TradeState,
)

# Default ticker → name mapping for all sectors and key industries
_TICKER_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
    "SMH": "Semiconductors", "IGV": "Software", "HACK": "Cybersecurity", "SOXX": "Semis (iShares)",
    "XBI": "Biotech", "IHI": "Medical Devices",
    "KRE": "Regional Banks", "IAI": "Broker-Dealers", "KIE": "Insurance",
    "XOP": "Oil & Gas E&P", "OIH": "Oil Services", "URA": "Uranium",
    "ITA": "Aerospace & Defense", "XAR": "A&D (SPDR)",
    "TAN": "Solar", "NLR": "Nuclear", "VNQ": "REITs",
    "XHB": "Homebuilders", "ITB": "Home Construction", "XRT": "Retail", "IBUY": "eCommerce Retail",
    "XME": "Metals & Mining", "GDX": "Gold Miners", "SIL": "Silver Miners",
}


def _tn(ticker: str, names: dict | None = None) -> str:
    """Return 'TICKER (Name)' for any ticker."""
    n = (names or _TICKER_NAMES).get(ticker, "")
    return f"{ticker} ({n})" if n else ticker


# ── Template Dictionaries ────────────────────────────────

REGIME_LANGUAGE = {
    "NORMAL": {
        "headline": "Markets are functioning normally. Rotation signals are reliable.",
        "action": "Follow system calls with standard sizing.",
        "risk": "No systemic risk flags. Watch for complacency.",
    },
    "FRAGILE": {
        "headline": "Markets are under stress. Protect capital first, rotate second.",
        "action": "Reduce position sizes. Tighten stops. Favor defensive sectors.",
        "risk": "Elevated volatility. Correlations may spike. Watch for regime deterioration.",
    },
    "HOSTILE": {
        "headline": "Crisis conditions active. Capital protection is the only priority.",
        "action": "Minimize exposure. Only defined-risk shorts or cash. No new longs.",
        "risk": "Systemic stress detected. Correlations near 1.0. Liquidity may vanish.",
    },
}

CRISIS_LANGUAGE = {
    CrisisType.OIL_SHOCK: {
        "headline": "Oil shock detected. Energy sector dislocated from broad market.",
        "detail": "Energy names may benefit while consumer and transport sectors suffer. Watch oil inventory data and OPEC signals.",
        "risk": "Oil shocks can cascade into inflation fears and rate volatility.",
    },
    CrisisType.RATE_SHOCK: {
        "headline": "Rate shock detected. Interest-rate-sensitive sectors under pressure.",
        "detail": "Real estate, utilities, and growth names face headwinds. Financials may benefit from steeper curve.",
        "risk": "Duration risk is elevated. Avoid rate-sensitive longs.",
    },
    CrisisType.CREDIT_CRISIS: {
        "headline": "Credit crisis detected. Financial sector stress is spreading.",
        "detail": "Banks and financials face forced selling. Defensive staples and healthcare offer relative safety.",
        "risk": "Credit contagion risk is high. Counterparty risk may surface.",
    },
    CrisisType.MARGIN_CALL: {
        "headline": "Margin call regime detected. Forced selling across all asset classes.",
        "detail": "Gold/VIX divergence signals institutional liquidation. Cash is king. Even safe havens may sell off.",
        "risk": "Forced selling ignores fundamentals. Correlations spike to 1.0. Raise cash immediately.",
    },
    CrisisType.GEOPOLITICAL: {
        "headline": "Geopolitical shock detected. Defense and energy may outperform.",
        "detail": "Uncertainty premium rising. Defense contractors and energy names historically benefit. Consumer discretionary suffers.",
        "risk": "Geopolitical risks are inherently unpredictable. Size positions conservatively.",
    },
    CrisisType.MULTI_CRISIS: {
        "headline": "Multiple crisis types active simultaneously. Maximum caution.",
        "detail": "Overlapping crises amplify risk. Standard sector rotations may not work.",
        "risk": "Multi-crisis environments are the most dangerous. Protect capital above all.",
    },
    CrisisType.NONE: {
        "headline": "",
        "detail": "",
        "risk": "",
    },
}

STATE_LANGUAGE = {
    "Overt Pump": {
        "description": "Maximum capital inflow. This sector is in full acceleration mode.",
        "action": "Full long position. Let it run with a trailing relative stop.",
    },
    "Broadening": {
        "description": "Participation expanding. Smart money is being joined by broad flows.",
        "action": "Add to longs on dips. This is the sweet spot for sector rotation.",
    },
    "Accumulation": {
        "description": "Early smart money positioning detected. Quiet accumulation phase.",
        "action": "Initiate starter position. Build on confirmation.",
    },
    "Ambiguous": {
        "description": "Conflicting signals. No clear edge in either direction.",
        "action": "Stay flat. Wait for clarity before committing capital.",
    },
    "Distribution": {
        "description": "Smart money exiting quietly. Surface looks calm but internals weakening.",
        "action": "Consider defined-risk put spreads or reduce long exposure.",
    },
    "Exhaustion": {
        "description": "Participation contracting. The rally is running out of buyers.",
        "action": "Defined-risk short via put spreads. Don't fight the exhaustion.",
    },
    "Overt Dump": {
        "description": "Full capital flight. Institutional selling across the board.",
        "action": "Maximum short conviction via defined-risk put spreads. Don't buy the dip.",
    },
}

HORIZON_LANGUAGE = {
    "Full Confirm": {
        "meaning": "All timeframes agree: short, medium, and long-term relative performance confirms the trend.",
        "implication": "Highest conviction signal. Ride the trend.",
    },
    "Rotation In": {
        "meaning": "Short and medium-term relative performance turning positive while long-term is still negative.",
        "implication": "New rotation beginning. Early entry opportunity.",
    },
    "Rotation Out": {
        "meaning": "Short and medium-term weakening while long-term still positive.",
        "implication": "Rotation ending. Take profits or tighten stops.",
    },
    "Full Reject": {
        "meaning": "All timeframes confirm weakness. This sector is being abandoned.",
        "implication": "Avoid longs entirely. Short candidates only.",
    },
    "Dead Cat": {
        "meaning": "TRAP WARNING: Short-term bounce in a sector with weak medium and long-term performance.",
        "implication": "Do NOT buy this bounce. It's a dead cat. The sector is still in structural decline.",
    },
    "Healthy Dip": {
        "meaning": "Short-term weakness in a sector with strong medium and long-term performance.",
        "implication": "Buy the dip. Long-term trend is intact.",
    },
    "No Pattern": {
        "meaning": "Mixed signals across timeframes. No clear pattern.",
        "implication": "No horizon edge. Rely on other signals.",
    },
}

CHARACTER_LANGUAGE = {
    "Trending Bull": {
        "description": "Broad uptrend with healthy participation. Standard long-biased rotation works.",
    },
    "Trending Bear": {
        "description": "Broad downtrend. Defensive positioning and short-biased rotation.",
    },
    "Choppy": {
        "description": "Range-bound market. Quick rotations, small sizes, tight stops.",
    },
    "Crisis": {
        "description": "Systemic stress. Crisis alignment determines sector-specific positioning.",
    },
    "Recovery": {
        "description": "Post-crisis recovery. Early longs in recovery leaders with standard sizing.",
    },
    "Rotation": {
        "description": "Active sector rotation. Follow the relative performance signals.",
    },
}


# ── Executive Briefing Generator ─────────────────────────

def generate_executive_briefing(
    regime,                    # RegimeAssessment
    regime_character,          # RegimeCharacterReading
    crisis_types,              # list[CrisisType]
    trade_states,              # dict[str, TradeStateAssignment]
    horizon_readings,          # dict[str, HorizonReading]
    sector_targets,            # dict[str, int] — ticker -> target %
    journal_summary=None,      # JournalSummary | None
    gold_divergence=None,      # GoldDivergenceReading | None
    oil_signal_level="NORMAL", # str
    vix_level=20.0,            # float
    ticker_names: dict | None = None,  # ticker → name override
) -> str:
    """Generate a plain-English executive briefing from system state."""
    # Build name lookup: merge defaults with trade_state names and any overrides
    names = dict(_TICKER_NAMES)
    for t, ts in trade_states.items():
        if hasattr(ts, "name") and ts.name:
            names[t] = ts.name
    if ticker_names:
        names.update(ticker_names)

    sections = []

    # ── THE SITUATION ────────────────────────────────
    situation_lines = []
    regime_state_val = regime.state.value if regime else "NORMAL"
    regime_lang = REGIME_LANGUAGE.get(regime_state_val, REGIME_LANGUAGE["NORMAL"])
    situation_lines.append(regime_lang["headline"])

    # Crisis headlines
    active_crises = [ct for ct in crisis_types if ct not in (CrisisType.NONE,)]
    for ct in active_crises:
        crisis_lang = CRISIS_LANGUAGE.get(ct, {})
        headline = crisis_lang.get("headline", "")
        if headline:
            situation_lines.append(headline)

    # Margin call detail
    if gold_divergence and getattr(gold_divergence, "is_margin_call_regime", False):
        situation_lines.append(
            "Gold/VIX divergence confirms forced selling pressure. "
            "Raise cash immediately. Even safe havens may correlate to 1.0."
        )

    # Character context
    if regime_character:
        char_val = regime_character.character.value if hasattr(regime_character.character, "value") else str(regime_character.character)
        char_lang = CHARACTER_LANGUAGE.get(char_val, {})
        desc = char_lang.get("description", "")
        if desc:
            situation_lines.append(desc)

    sections.append("THE SITUATION")
    sections.append("\n".join(situation_lines))

    # ── WHAT TO DO ────────────────────────────────────
    action_lines = []
    buys = []
    shorts = []
    avoids = []

    for ticker, target in sorted(sector_targets.items(), key=lambda x: -abs(x[1])):
        if target >= 15:
            buys.append((ticker, target))
        elif target <= -15:
            shorts.append((ticker, target))
        else:
            avoids.append((ticker, target))

    if buys:
        action_lines.append("BUY:")
        for ticker, target in buys:
            ts = trade_states.get(ticker)
            state_val = ts.analysis_state.value if ts and hasattr(ts.analysis_state, "value") else str(ts.analysis_state) if ts else "Unknown"
            state_lang = STATE_LANGUAGE.get(state_val, {})
            desc = state_lang.get("description", "")
            action_lines.append(f"  {_tn(ticker, names)}: target {target:+d}% — {desc}")

    if shorts:
        action_lines.append("SHORT (defined-risk put spreads only):")
        for ticker, target in shorts:
            ts = trade_states.get(ticker)
            state_val = ts.analysis_state.value if ts and hasattr(ts.analysis_state, "value") else str(ts.analysis_state) if ts else "Unknown"
            state_lang = STATE_LANGUAGE.get(state_val, {})
            desc = state_lang.get("description", "")
            action_lines.append(f"  {_tn(ticker, names)}: target {target:+d}% — {desc}")

    if avoids:
        action_lines.append(f"AVOID ({len(avoids)} sectors with no actionable edge):")
        for ticker, target in avoids:
            action_lines.append(f"  {_tn(ticker, names)}: target {target:+d}%")

    sections.append("WHAT TO DO")
    sections.append("\n".join(action_lines) if action_lines else "No actionable calls. Stay in cash and avoid forcing trades.")

    # ── KEY RISKS ──────────────────────────────────────
    risk_lines = []

    # VIX proximity
    if vix_level > 25:
        risk_lines.append(f"VIX at {vix_level:.1f} — approaching HOSTILE threshold. Position sizes already reduced.")
    elif vix_level > 20:
        risk_lines.append(f"VIX at {vix_level:.1f} — elevated but manageable. Stay alert for spikes.")

    # Crisis-specific risks
    for ct in active_crises:
        crisis_lang = CRISIS_LANGUAGE.get(ct, {})
        risk = crisis_lang.get("risk", "")
        if risk:
            risk_lines.append(risk)

    # Counter-secular warnings
    for ticker, hr in horizon_readings.items():
        pattern = hr.pattern if hasattr(hr, "pattern") else hr
        pattern_val = pattern.value if hasattr(pattern, "value") else str(pattern)
        if pattern_val == "Dead Cat":
            risk_lines.append(f"WARNING: {_tn(ticker, names)} showing Dead Cat pattern. Do NOT buy the bounce — it's a TRAP.")
        elif pattern_val == "Rotation Out":
            risk_lines.append(f"CAUTION: {_tn(ticker, names)} rotation ending. Tighten stops or take profits.")

    sections.append("KEY RISKS")
    sections.append("\n".join(risk_lines) if risk_lines else "No elevated risk signals at this time.")

    # ── HORIZON CHECK ──────────────────────────────────
    horizon_lines = []
    # Top 3-5 tickers by abs(target)
    sorted_tickers = sorted(sector_targets.items(), key=lambda x: -abs(x[1]))[:5]
    for ticker, target in sorted_tickers:
        hr = horizon_readings.get(ticker)
        if hr:
            pattern = hr.pattern if hasattr(hr, "pattern") else hr
            pattern_val = pattern.value if hasattr(pattern, "value") else str(pattern)
            horizon_lang = HORIZON_LANGUAGE.get(pattern_val, HORIZON_LANGUAGE.get("No Pattern", {}))
            meaning = horizon_lang.get("meaning", "")
            implication = horizon_lang.get("implication", "")
            horizon_lines.append(f"{_tn(ticker, names)} [{pattern_val}]: {meaning}")
            if implication:
                horizon_lines.append(f"  -> {implication}")

    sections.append("HORIZON CHECK")
    sections.append("\n".join(horizon_lines) if horizon_lines else "No horizon data available.")

    # ── SYSTEM CONFIDENCE ──────────────────────────────
    confidence_lines = []
    total = len(sector_targets)
    actionable = sum(1 for t in sector_targets.values() if abs(t) >= 15)
    confidence_lines.append(
        f"{actionable} of {total} sectors have actionable calls."
    )

    if actionable == 0:
        confidence_lines.append(
            "Low system confidence. The system sees no clear edge. Avoid forcing trades."
        )
    elif actionable <= 3:
        confidence_lines.append(
            "Moderate confidence. A few clear opportunities but selectivity is key."
        )
    else:
        confidence_lines.append(
            "High system confidence. Multiple sectors showing clear signals."
        )

    if journal_summary:
        pnl_10d = getattr(journal_summary, "total_pnl_10d", None)
        hit_rate = getattr(journal_summary, "hit_rate_10d", None)
        if pnl_10d is not None:
            confidence_lines.append(
                f"Journal P&L (10-day): {pnl_10d:+.1f}% cumulative."
            )
        if hit_rate is not None:
            confidence_lines.append(
                f"Hit rate (10-day): {hit_rate:.0%}."
            )

    sections.append("SYSTEM CONFIDENCE")
    sections.append("\n".join(confidence_lines))

    # ── Assemble ──────────────────────────────────────
    output_parts = []
    for i in range(0, len(sections), 2):
        header = sections[i]
        body = sections[i + 1]
        output_parts.append(f"{'━' * 3} {header} {'━' * 3}")
        output_parts.append(body)
        output_parts.append("")  # blank line between sections

    return "\n".join(output_parts)
