"""
Shared styling — exact Tailwind dark-theme colors per Phase 3 spec.

Analysis state → (background hex, text hex)
"""
from engine.schemas import AnalysisState

# Spec §Feature 5: exact hex colors for dark theme
STATE_COLORS = {
    "Broadening":        ("#064e3b", "#34d399"),  # Dark green bg, green text
    "Overt Pump":        ("#064e3b", "#34d399"),  # Same green family
    "Exhaustion":        ("#7c2d12", "#fb923c"),  # Dark orange bg, orange text
    "Rotation/Reversal": ("#7f1d1d", "#f87171"),  # Dark red bg, red text
    "Ambiguous":         ("#713f12", "#fbbf24"),  # Dark yellow bg, yellow text
    "Accumulation":      ("#1e293b", "#94a3b8"),  # Dark slate bg, gray text
}

TRADE_STATE_COLORS = {
    "Long Entry":       "#22c55e",
    "Selective Add":    "#4ade80",
    "Hold":             "#94a3b8",
    "Hold Smaller":     "#94a3b8",
    "Reduce":           "#f97316",
    "Short Candidate":  "#ef4444",
    "Pair Candidate":   "#a78bfa",
    "Hedge":            "#ef4444",
    "Watchlist":        "#64748b",
    "No Trade":         "#475569",
}

# Bar chart colors keyed by state (for composite chart)
STATE_BAR_COLORS = {
    "Broadening":        "#22c55e",  # Green
    "Overt Pump":        "#22c55e",  # Green
    "Exhaustion":        "#f97316",  # Orange
    "Rotation/Reversal": "#ef4444",  # Red
    "Ambiguous":         "#eab308",  # Yellow
    "Accumulation":      "#94a3b8",  # Gray
}


def color_row_by_state(row, state_col: str = "State"):
    """Apply bg + text color from STATE_COLORS based on the State column."""
    state_val = row.get(state_col, "")
    colors = STATE_COLORS.get(state_val)
    if colors:
        bg, fg = colors
        return [f"background-color: {bg}; color: {fg}"] * len(row)
    return [""] * len(row)


def color_delta(val: str) -> str:
    """Color a delta string green if positive, red if negative."""
    try:
        num = float(val.replace("+", ""))
        if num > 0.005:
            return "color: #22c55e"
        elif num < -0.005:
            return "color: #ef4444"
    except (ValueError, AttributeError):
        pass
    return ""


def style_dataframe(df, state_col="State", delta_col="Delta"):
    """Apply row bg/text coloring + delta coloring."""
    styled = df.style.apply(color_row_by_state, axis=1, state_col=state_col)
    if delta_col in df.columns:
        styled = styled.map(color_delta, subset=[delta_col])
    return styled
