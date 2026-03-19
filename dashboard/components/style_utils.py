"""Shared styling utilities for dashboard tables."""
from engine.schemas import AnalysisState

# State → color mapping
# Green family: rotating IN (momentum building)
# Red family: rotating OUT (momentum fading)
# Yellow/gray: ambiguous/neutral
STATE_ROW_COLORS = {
    AnalysisState.OVERT_PUMP.value: "rgba(0, 212, 170, 0.18)",       # green — strongest IN
    AnalysisState.BROADENING.value: "rgba(0, 212, 170, 0.10)",       # light green — building
    AnalysisState.ACCUMULATION.value: "rgba(0, 212, 170, 0.05)",     # very light green — early
    AnalysisState.EXHAUSTION.value: "rgba(255, 165, 0, 0.15)",       # orange — fading
    AnalysisState.ROTATION.value: "rgba(255, 68, 68, 0.18)",         # red — rotating OUT
    AnalysisState.AMBIGUOUS.value: "rgba(255, 255, 0, 0.08)",        # yellow — unclear
}


def color_row_by_state(row, state_col: str = "State"):
    """Return a list of CSS styles for each cell in the row based on the State column."""
    state_val = row.get(state_col, "")
    bg = STATE_ROW_COLORS.get(state_val, "")
    if bg:
        return [f"background-color: {bg}"] * len(row)
    return [""] * len(row)
