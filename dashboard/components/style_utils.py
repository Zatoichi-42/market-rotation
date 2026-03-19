"""Shared styling utilities for dashboard tables — Tailwind-based dark theme colors."""
from engine.schemas import AnalysisState

# Analysis state → row background color (dark theme, subtle tints)
STATE_ROW_COLORS = {
    AnalysisState.OVERT_PUMP.value:  "rgba(34, 197, 94, 0.18)",    # Green — strongest IN
    AnalysisState.BROADENING.value:  "rgba(34, 197, 94, 0.10)",    # Light green — building
    AnalysisState.ACCUMULATION.value: "rgba(148, 163, 184, 0.06)", # Slate — early/neutral
    AnalysisState.EXHAUSTION.value:  "rgba(249, 115, 22, 0.15)",   # Orange — fading
    AnalysisState.ROTATION.value:    "rgba(239, 68, 68, 0.18)",    # Red — rotating OUT
    AnalysisState.AMBIGUOUS.value:   "rgba(234, 179, 8, 0.10)",    # Yellow — unclear
}

# Trade state colors (for future Phase 3 trade panel)
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


def color_row_by_state(row, state_col: str = "State"):
    """Return CSS styles for each cell based on the State column value."""
    state_val = row.get(state_col, "")
    bg = STATE_ROW_COLORS.get(state_val, "")
    if bg:
        return [f"background-color: {bg}"] * len(row)
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
    """Apply full row coloring + delta coloring to a DataFrame."""
    styled = df.style.apply(color_row_by_state, axis=1, state_col=state_col)
    if delta_col in df.columns:
        styled = styled.map(color_delta, subset=[delta_col])
    return styled
