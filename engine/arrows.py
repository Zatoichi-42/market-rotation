"""
Arrow Indicators — dashboard-wide delta direction system.

7-level vocabulary: ↑↑ ↑ ↗ → ↘ ↓ ↓↓
Each arrow has a color and optional counter-trend flag.
"""
from engine.schemas import ArrowDirection, ArrowIndicator


_ARROW_CONFIG = {
    ArrowDirection.STRONG_UP:   ("#1B5E20", "Strongly accelerating UP"),
    ArrowDirection.UP:          ("#4CAF50", "Improving"),
    ArrowDirection.SLIGHT_UP:   ("#81C784", "Slightly improving"),
    ArrowDirection.FLAT:        ("#9E9E9E", "Flat / stable"),
    ArrowDirection.SLIGHT_DOWN: ("#EF9A9A", "Slightly declining"),
    ArrowDirection.DOWN:        ("#E53935", "Declining"),
    ArrowDirection.STRONG_DOWN: ("#B71C1C", "Strongly accelerating DOWN"),
}


def compute_arrow(
    delta: float,
    delta_prior: float = 0.0,
    rank_change: int = 0,
    flat_threshold: float = 0.005,
    is_counter_trend: bool = False,
) -> ArrowIndicator:
    """
    Compute arrow indicator from delta values.

    Args:
        delta: Current session-over-session pump delta
        delta_prior: Prior session's delta (for delta-of-delta / acceleration)
        rank_change: Rank change (positive = improved)
        flat_threshold: |delta| below this = flat
        is_counter_trend: Whether today's move opposes the 5d trend

    Returns ArrowIndicator with direction, color, and label.
    """
    delta_of_delta = delta - delta_prior

    if abs(delta) < flat_threshold:
        direction = ArrowDirection.FLAT
    elif delta > 0:
        if delta_of_delta > 0 and rank_change > 0:
            direction = ArrowDirection.STRONG_UP
        elif delta_of_delta >= 0:
            direction = ArrowDirection.UP
        else:
            direction = ArrowDirection.SLIGHT_UP
    else:
        if delta_of_delta < 0 and rank_change < 0:
            direction = ArrowDirection.STRONG_DOWN
        elif delta_of_delta <= 0:
            direction = ArrowDirection.DOWN
        else:
            direction = ArrowDirection.SLIGHT_DOWN

    color, label = _ARROW_CONFIG[direction]

    return ArrowIndicator(
        direction=direction,
        color_hex=color,
        label=label,
        is_counter_trend=is_counter_trend,
    )


def arrow_symbol(indicator: ArrowIndicator) -> str:
    """Get the unicode arrow symbol."""
    return indicator.direction.value


def arrow_html(indicator: ArrowIndicator) -> str:
    """Render arrow as colored HTML span."""
    sym = indicator.direction.value
    return f"<span style='color:{indicator.color_hex};font-weight:bold;'>{sym}</span>"
