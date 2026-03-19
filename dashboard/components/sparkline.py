"""
Sparkline generators — Unicode block characters for inline table display.
"""
import pandas as pd
import numpy as np

# Unicode block elements: 8 levels from lowest to highest
_BLOCKS = "▁▂▃▄▅▆▇█"


def make_sparkline_unicode(series: pd.Series, width: int = 12) -> str:
    """
    Create a Unicode sparkline from a numeric series.

    Bins the series into 8 levels and returns a string of block characters.
    Green/red coloring applied by the caller via markdown.

    Args:
        series: Numeric values (e.g., 20d RS history, last N days)
        width: Number of characters in the sparkline
    """
    if series is None or len(series) == 0:
        return "—"

    values = series.dropna().values
    if len(values) == 0:
        return "—"

    # Resample to `width` points if longer
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width, dtype=int)
        values = values[indices]

    # Normalize to 0-7 range for block indexing
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return _BLOCKS[4] * len(values)  # All middle if flat

    normalized = ((values - vmin) / (vmax - vmin) * 7).astype(int)
    normalized = np.clip(normalized, 0, 7)

    return "".join(_BLOCKS[i] for i in normalized)


def sparkline_with_color(series: pd.Series, width: int = 12) -> str:
    """
    Return a sparkline string with direction indicator.
    Caller wraps in colored markdown.
    """
    spark = make_sparkline_unicode(series, width)
    if series is not None and len(series.dropna()) >= 2:
        first, last = series.dropna().iloc[0], series.dropna().iloc[-1]
        if last > first:
            return f"↑ {spark}"
        elif last < first:
            return f"↓ {spark}"
    return f"  {spark}"
