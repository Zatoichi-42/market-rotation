"""
Crisis Alignment — sector-specific modifiers during different crisis types.
Replaces the flat Crisis character modifier with nuanced sector alignment.
"""
from engine.schemas import CrisisType

# Matrix: CRISIS_ALIGNMENT[crisis_type][sector_ticker] = (long_modifier, short_modifier)
CRISIS_ALIGNMENT = {
    CrisisType.OIL_SHOCK: {
        "XLE": (1.3, 0.5), "XOP": (1.3, 0.5), "OIH": (1.3, 0.5),
        "XLK": (0.5, 1.0), "XLY": (0.5, 1.0), "XLI": (0.7, 0.8),
        "XLP": (1.1, 0.7), "XLV": (1.0, 0.8), "XLU": (1.0, 0.8),
        "XLF": (0.8, 0.8), "XLC": (0.7, 0.8), "XLRE": (0.7, 0.9), "XLB": (0.9, 0.8),
    },
    CrisisType.RATE_SHOCK: {
        "XLRE": (0.3, 1.3), "XLU": (0.5, 1.1), "XLK": (0.5, 1.0),
        "XHB": (0.3, 1.3), "ITB": (0.3, 1.3),
        "XLF": (1.1, 0.7), "KRE": (1.1, 0.7),
        "XLP": (1.1, 0.7), "XLV": (1.0, 0.8),
        "XLE": (0.9, 0.8), "XLY": (0.5, 1.0), "XLC": (0.7, 0.8),
        "XLI": (0.8, 0.8), "XLB": (0.8, 0.8),
    },
    CrisisType.CREDIT_CRISIS: {
        "XLF": (0.3, 1.3), "KRE": (0.2, 1.4), "IAI": (0.3, 1.3),
        "XLP": (1.2, 0.5), "XLV": (1.1, 0.6), "XLU": (1.1, 0.6),
        "XLE": (0.7, 0.9), "XLK": (0.8, 0.8), "XLC": (0.7, 0.8),
        "XLI": (0.6, 0.9), "XLY": (0.5, 1.0), "XLRE": (0.5, 1.0), "XLB": (0.6, 0.9),
    },
    CrisisType.MARGIN_CALL: {
        "GDX": (0.3, 1.4), "SIL": (0.3, 1.4), "XLK": (0.5, 1.0),
        "XLY": (0.5, 1.0), "SMH": (0.4, 1.1),
        "XLP": (0.6, 0.8), "XLV": (0.6, 0.8), "XLU": (0.6, 0.8),
        "XLF": (0.5, 0.9), "XLE": (0.6, 0.8), "XLC": (0.5, 0.9),
        "XLI": (0.5, 0.9), "XLRE": (0.5, 0.9), "XLB": (0.5, 0.9),
    },
    CrisisType.GEOPOLITICAL: {
        "XLI": (0.9, 0.7), "XAR": (1.2, 0.5), "ITA": (1.2, 0.5),
        "XLE": (1.1, 0.7), "XLP": (1.1, 0.7), "XLU": (1.0, 0.8), "XLV": (1.0, 0.8),
        "XLK": (0.7, 0.8), "XLY": (0.6, 0.9), "XLF": (0.7, 0.8),
        "XLC": (0.7, 0.8), "XLRE": (0.7, 0.9), "XLB": (0.8, 0.8),
    },
}

DEFAULT_ALIGNMENT = (1.0, 1.0)


def detect_crisis_type(
    oil_level: str = "NORMAL",
    credit_level: str = "NORMAL",
    gold_divergence_active: bool = False,
    gold_silver_stress: bool = False,
    vix_level: float = 20.0,
    correlation_level: str = "NORMAL",
    term_structure_level: str = "NORMAL",
    breadth_trend: str = "stable",
) -> list[CrisisType]:
    """Detect active crisis types from regime signals."""
    types = []
    if oil_level == "HOSTILE":
        types.append(CrisisType.OIL_SHOCK)
    if credit_level == "HOSTILE":
        types.append(CrisisType.CREDIT_CRISIS)
    if gold_divergence_active or gold_silver_stress:
        types.append(CrisisType.MARGIN_CALL)
    if term_structure_level == "HOSTILE" and breadth_trend == "deteriorating":
        types.append(CrisisType.RATE_SHOCK)
    if not types and vix_level > 30 and correlation_level == "HOSTILE":
        types.append(CrisisType.GEOPOLITICAL)
    if len(types) >= 2:
        types.append(CrisisType.MULTI_CRISIS)
    if not types:
        types.append(CrisisType.NONE)
    return types


def get_crisis_modifier(ticker: str, crisis_types: list[CrisisType], is_long: bool) -> float:
    """
    Get the most extreme crisis alignment modifier for a ticker.
    For longs: use the highest long_mod across active crisis types.
    For shorts: use the highest short_mod across active crisis types.
    """
    best_mod = 0.25  # Default crisis floor (same as flat Crisis modifier)
    for ct in crisis_types:
        if ct in (CrisisType.NONE, CrisisType.MULTI_CRISIS):
            continue
        alignment = CRISIS_ALIGNMENT.get(ct, {})
        long_mod, short_mod = alignment.get(ticker, DEFAULT_ALIGNMENT)
        mod = long_mod if is_long else short_mod
        best_mod = max(best_mod, mod)
    return best_mod
