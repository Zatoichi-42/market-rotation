"""
Cross-Horizon Divergence Patterns — classifies 5d/20d/60d RS into 7 named patterns.

Each sector/industry gets exactly one pattern based on the sign of its 3 RS horizons.
Patterns feed the state classifier as input (DEAD_CAT vetoes Accumulation, etc.).
"""
from engine.schemas import HorizonPattern, HorizonReading


def _sign(value: float, threshold: float) -> str:
    """Return '+', '-', or '~' based on threshold."""
    if value > threshold:
        return "+"
    elif value < -threshold:
        return "-"
    return "~"


_PATTERN_MAP = {
    ("+" , "+", "+"): HorizonPattern.FULL_CONFIRM,
    ("+" , "+", "-"): HorizonPattern.ROTATION_IN,
    ("-" , "-", "+"): HorizonPattern.ROTATION_OUT,
    ("-" , "-", "-"): HorizonPattern.FULL_REJECT,
    ("+" , "-", "-"): HorizonPattern.DEAD_CAT,
    ("-" , "+", "+"): HorizonPattern.HEALTHY_DIP,
}

# Flat (~) handling: map ambiguous combos to conservative pattern
_FLAT_MAP = {
    # ~↑↑ → HEALTHY_DIP (flat short-term in uptrend)
    ("~", "+", "+"): HorizonPattern.HEALTHY_DIP,
    # ~↓↓ → ROTATION_OUT (flat short-term in downtrend)
    ("~", "-", "-"): HorizonPattern.ROTATION_OUT,
    # ↑↑~ → lean FULL_CONFIRM (but lower conviction)
    ("+", "+", "~"): HorizonPattern.FULL_CONFIRM,
    # ↓↓~ → lean FULL_REJECT (but lower conviction)
    ("-", "-", "~"): HorizonPattern.FULL_REJECT,
    # ↑~↓ → lean ROTATION_IN
    ("+", "~", "-"): HorizonPattern.ROTATION_IN,
    # ↓~↑ → lean ROTATION_OUT
    ("-", "~", "+"): HorizonPattern.ROTATION_OUT,
    # ~↑↓ → lean ROTATION_IN
    ("~", "+", "-"): HorizonPattern.ROTATION_IN,
    # ~↓↑ → lean ROTATION_OUT
    ("~", "-", "+"): HorizonPattern.ROTATION_OUT,
    # ↑~+ → lean FULL_CONFIRM
    ("+", "~", "+"): HorizonPattern.FULL_CONFIRM,
    # -~- → lean FULL_REJECT
    ("-", "~", "-"): HorizonPattern.FULL_REJECT,
    # +~~ → NO_PATTERN (2+ flat)
    # ~~+ → NO_PATTERN
    # etc — handled by fallback
}


def classify_horizon_pattern(
    rs_5d: float,
    rs_20d: float,
    rs_60d: float,
    near_zero_threshold: float = 0.003,
) -> HorizonPattern:
    """Classify the cross-horizon pattern from 3 RS values."""
    s5 = _sign(rs_5d, near_zero_threshold)
    s20 = _sign(rs_20d, near_zero_threshold)
    s60 = _sign(rs_60d, near_zero_threshold)

    key = (s5, s20, s60)

    # Count flat horizons
    n_flat = sum(1 for s in key if s == "~")
    if n_flat >= 2:
        return HorizonPattern.NO_PATTERN

    # Exact match (no flats)
    if key in _PATTERN_MAP:
        return _PATTERN_MAP[key]

    # Flat handling (1 flat)
    if key in _FLAT_MAP:
        return _FLAT_MAP[key]

    return HorizonPattern.NO_PATTERN


def compute_horizon_conviction(
    rs_5d: float, rs_20d: float, rs_60d: float,
    pattern: HorizonPattern,
) -> int:
    """
    Conviction = average absolute RS across confirming horizons, normalized to 0-100.

    For FULL_CONFIRM/FULL_REJECT: all 3 horizons contribute
    For ROTATION_IN: 5d and 20d contribute (60d is counter — expected)
    For ROTATION_OUT: 20d and 60d contribute (5d is counter — expected)
    For DEAD_CAT: 20d and 60d contribute (bearish direction)
    For HEALTHY_DIP: 20d and 60d contribute (bullish direction)
    """
    if pattern == HorizonPattern.NO_PATTERN:
        return 15

    values = [abs(rs_5d), abs(rs_20d), abs(rs_60d)]

    if pattern in (HorizonPattern.FULL_CONFIRM, HorizonPattern.FULL_REJECT):
        confirming = values  # all 3
    elif pattern == HorizonPattern.ROTATION_IN:
        confirming = [abs(rs_5d), abs(rs_20d)]
    elif pattern == HorizonPattern.ROTATION_OUT:
        confirming = [abs(rs_20d), abs(rs_60d)]
    elif pattern == HorizonPattern.DEAD_CAT:
        confirming = [abs(rs_20d), abs(rs_60d)]
    elif pattern == HorizonPattern.HEALTHY_DIP:
        confirming = [abs(rs_20d), abs(rs_60d)]
    else:
        confirming = values

    total = sum(confirming)

    if total > 0.15:
        return 90
    elif total > 0.10:
        return 75
    elif total > 0.05:
        return 55
    elif total > 0.02:
        return 35
    else:
        return 20


def classify_horizon_reading(
    ticker: str,
    name: str,
    rs_5d: float,
    rs_20d: float,
    rs_60d: float,
    near_zero_threshold: float = 0.003,
) -> HorizonReading:
    """Build a complete HorizonReading for a sector/industry."""
    pattern = classify_horizon_pattern(rs_5d, rs_20d, rs_60d, near_zero_threshold)
    conviction = compute_horizon_conviction(rs_5d, rs_20d, rs_60d, pattern)

    s5 = _sign(rs_5d, near_zero_threshold)
    s20 = _sign(rs_20d, near_zero_threshold)
    s60 = _sign(rs_60d, near_zero_threshold)

    # Arrow symbols for description
    arrow_map = {"+": "↑", "-": "↓", "~": "~"}
    arrows = f"{arrow_map[s5]}{arrow_map[s20]}{arrow_map[s60]}"

    description = f"{ticker}: {pattern.value} ({arrows})"
    if pattern == HorizonPattern.DEAD_CAT:
        description += " — TRAP: short-term bounce in downtrend"
    elif pattern == HorizonPattern.HEALTHY_DIP:
        description += " — entry zone: short-term pullback in uptrend"
    elif pattern == HorizonPattern.ROTATION_IN:
        description += " — new leader emerging from weakness"
    elif pattern == HorizonPattern.ROTATION_OUT:
        description += " — former leader breaking down"
    elif pattern == HorizonPattern.FULL_CONFIRM:
        description += " — all horizons aligned bullish"
    elif pattern == HorizonPattern.FULL_REJECT:
        description += " — all horizons aligned bearish"

    return HorizonReading(
        ticker=ticker,
        name=name,
        pattern=pattern,
        rs_5d=rs_5d,
        rs_20d=rs_20d,
        rs_60d=rs_60d,
        rs_5d_sign=s5,
        rs_20d_sign=s20,
        rs_60d_sign=s60,
        conviction=conviction,
        description=description,
        is_rotation_signal=pattern in (HorizonPattern.ROTATION_IN, HorizonPattern.ROTATION_OUT),
        is_trap=pattern == HorizonPattern.DEAD_CAT,
        is_entry_zone=pattern == HorizonPattern.HEALTHY_DIP,
    )


def classify_all_horizon_patterns(
    rs_readings,
    industry_rs_readings=None,
    near_zero_threshold: float = 0.003,
) -> dict[str, HorizonReading]:
    """Classify horizon patterns for all sectors and industries.

    Args:
        rs_readings: list of RSReading (sectors)
        industry_rs_readings: list of IndustryRSReading (industries)
        near_zero_threshold: threshold for flat classification

    Returns: dict[ticker, HorizonReading]
    """
    results = {}

    for r in rs_readings:
        results[r.ticker] = classify_horizon_reading(
            r.ticker, r.name, r.rs_5d, r.rs_20d, r.rs_60d, near_zero_threshold,
        )

    if industry_rs_readings:
        for ir in industry_rs_readings:
            results[ir.ticker] = classify_horizon_reading(
                ir.ticker, ir.name, ir.rs_5d, ir.rs_20d, ir.rs_60d, near_zero_threshold,
            )

    return results
