"""
Shared styling — 7-state symmetric color model.

Bullish:  Accumulation (light green) → Broadening (green) → Overt Pump (deep green)
Bearish:  Distribution (light salmon) → Exhaustion (red) → Overt Dump (deep red)
Neutral:  Ambiguous (gray)
"""
from engine.schemas import AnalysisState

# Row background + text colors for dark theme
STATE_COLORS = {
    "Overt Dump":    ("#7f1d1d", "#f87171"),   # Deep red bg, red text
    "Exhaustion":    ("#7c2d12", "#fb923c"),   # Red bg, orange text
    "Distribution":  ("#78350f", "#fbbf24"),   # Salmon bg, yellow text
    "Ambiguous":     ("", ""),                  # No color — transparent
    "Accumulation":  ("#064e3b", "#34d399"),   # Light green bg, green text
    "Broadening":    ("#065f46", "#6ee7b7"),   # Green bg, light green text
    "Overt Pump":    ("#052e16", "#22c55e"),   # Deep green bg, bright green text
}

# Bar chart colors
STATE_BAR_COLORS = {
    "Overt Dump":    "#ef4444",   # Red
    "Exhaustion":    "#fb923c",   # Orange-red
    "Distribution":  "#fbbf24",   # Yellow/salmon
    "Ambiguous":     "#64748b",   # Gray
    "Accumulation":  "#a5d6a7",   # Light green
    "Broadening":    "#4ade80",   # Green
    "Overt Pump":    "#22c55e",   # Deep green
}

# Momentum spectrum (for interpretation panel)
MOMENTUM_COLORS = {
    "Overt Dump":    "#7f1d1d",
    "Exhaustion":    "#ef4444",
    "Distribution":  "#fbbf24",
    "Ambiguous":     "#64748b",
    "Accumulation":  "#a5d6a7",
    "Broadening":    "#4ade80",
    "Overt Pump":    "#064e3b",
}

TRADE_STATE_COLORS = {
    "Long Entry":       "#22c55e",
    "Selective Add":    "#4ade80",
    "Hold":             "#94a3b8",
    "Hold Smaller":     "#94a3b8",
    "Reduce":           "#f97316",
    "Short Candidate":  "#ef4444",
    "Pair Candidate":   "#a78bfa",
    "Pair Cand.":       "#a78bfa",
    "Hedge":            "#ef4444",
    "Watchlist":        "#64748b",
    "No Trade":         "#475569",
}


def color_row_by_state(row, state_col: str = "State"):
    """Apply bg + text color based on State column."""
    state_val = row.get(state_col, "")
    colors = STATE_COLORS.get(state_val)
    if colors and colors[0]:
        bg, fg = colors
        return [f"background-color: {bg}; color: {fg}"] * len(row)
    return [""] * len(row)


def color_delta(val: str) -> str:
    """Color a delta string green if positive, red if negative."""
    try:
        # Strip arrow symbols
        cleaned = val
        for sym in ("↑↑", "↑", "↗", "→", "↘", "↓", "↓↓"):
            cleaned = cleaned.replace(sym, "")
        num = float(cleaned.strip().replace("+", ""))
        if num > 0.005:
            return "color: #22c55e"
        elif num < -0.005:
            return "color: #ef4444"
    except (ValueError, AttributeError):
        pass
    return ""


def style_dataframe(df, state_col="State", delta_col="Δ"):
    """Apply row coloring + delta coloring."""
    styled = df.style.apply(color_row_by_state, axis=1, state_col=state_col)
    if delta_col in df.columns:
        styled = styled.map(color_delta, subset=[delta_col])
    # Also try legacy "Delta" column name
    if "Delta" in df.columns and "Delta" != delta_col:
        styled = styled.map(color_delta, subset=["Delta"])
    return styled
