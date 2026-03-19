"""
Concentration Monitor — distinguishes healthy narrow leadership from unhealthy exhaustion.

Solves: The Mag 7 problem (2023-24). When breadth is DIVERGING because a few mega-caps
are driving returns, but those leaders are fundamentally strong, the system should NOT
classify the sector as Exhaustion.

Two metrics:
1. Concentration Ratio: EW/CW spread z-score (RSP/SPY ratio already computed by breadth module)
2. Leader Health: RS of top-5 holdings vs their parent sector

Decision matrix:
- BROAD_HEALTHY: Normal breadth, no concern
- CONCENTRATED_HEALTHY: Narrow but leaders strong → suppress false exhaustion (+15 modifier)
- CONCENTRATED_FRAGILE: Narrow and leaders mixed → monitor
- CONCENTRATED_UNHEALTHY: Narrow and leaders deteriorating → amplify exhaustion (-15 modifier)
"""
import math

import numpy as np
import pandas as pd

from engine.schemas import ConcentrationRegime, ConcentrationReading


def compute_concentration(
    prices: pd.DataFrame,
    sector_ticker: str,
    leader_tickers: list[str],
    ew_cw_zscore: float,
    settings: dict | None = None,
) -> ConcentrationReading:
    """
    Compute concentration reading for a single sector.

    Args:
        prices: Full price DataFrame (must contain sector + leader columns)
        sector_ticker: The sector ETF (e.g., "XLK")
        leader_tickers: Top-N constituent tickers (e.g., ["AAPL", "MSFT", ...])
        ew_cw_zscore: RSP/SPY z-score from breadth module (market-wide, not per-sector)
        settings: Concentration config from settings.yaml
    """
    if settings is None:
        settings = {}

    elevated_z = settings.get("elevated_zscore", -0.5)
    extreme_z = settings.get("extreme_zscore", -1.5)
    strong_min_rs = settings.get("leader_strong_min_avg_rs", 0.0)
    deteriorating_max_rs = settings.get("leader_deteriorating_max_rs", -0.02)
    dispersion_mixed = settings.get("leader_dispersion_mixed", 0.03)
    healthy_mod = settings.get("healthy_modifier", 15)
    unhealthy_mod = settings.get("unhealthy_modifier", -15)

    # Compute leader RS vs sector (20-day relative return)
    available_leaders = [t for t in leader_tickers if t in prices.columns]

    if not available_leaders or sector_ticker not in prices.columns or len(prices) < 20:
        return _default_reading(sector_ticker, leader_tickers, ew_cw_zscore)

    # 20-day returns
    sector_ret = prices[sector_ticker].pct_change(20).iloc[-1]
    leader_rets = {}
    for t in available_leaders:
        r = prices[t].pct_change(20).iloc[-1]
        if not math.isnan(r):
            leader_rets[t] = r

    if not leader_rets:
        return _default_reading(sector_ticker, leader_tickers, ew_cw_zscore)

    # Leader RS vs sector = leader_return - sector_return
    leader_rs = {t: r - sector_ret for t, r in leader_rets.items()}
    avg_rs = np.mean(list(leader_rs.values()))
    dispersion = np.std(list(leader_rs.values())) if len(leader_rs) > 1 else 0.0

    # Classify leader health
    leader_health = _classify_leader_health(
        avg_rs, dispersion, strong_min_rs, deteriorating_max_rs, dispersion_mixed,
    )

    # Classify concentration regime
    regime, modifier = _classify_concentration_regime(
        ew_cw_zscore, leader_health, elevated_z, extreme_z,
        healthy_mod, unhealthy_mod,
    )

    explanation = _build_explanation(
        sector_ticker, regime, leader_health, avg_rs, dispersion,
        ew_cw_zscore, available_leaders,
    )

    return ConcentrationReading(
        sector_ticker=sector_ticker,
        ew_cw_zscore=ew_cw_zscore,
        leader_health=leader_health,
        leader_tickers=available_leaders,
        leader_avg_rs=avg_rs,
        leader_dispersion=dispersion,
        regime=regime,
        participation_modifier=modifier,
        explanation=explanation,
    )


def compute_concentration_all(
    prices: pd.DataFrame,
    sector_leaders: dict[str, list[str]],
    ew_cw_zscore: float,
    settings: dict | None = None,
) -> list[ConcentrationReading]:
    """
    Compute concentration readings for all sectors.

    Args:
        prices: Full price DataFrame
        sector_leaders: {sector_ticker: [leader1, leader2, ...]} from universe.yaml
        ew_cw_zscore: Market-wide RSP/SPY z-score
        settings: Concentration config
    """
    readings = []
    for sector, leaders in sector_leaders.items():
        readings.append(compute_concentration(
            prices, sector, leaders, ew_cw_zscore, settings,
        ))
    return readings


# ═══════════════════════════════════════════════════════
# INTERNAL
# ═══════════════════════════════════════════════════════

def _classify_leader_health(
    avg_rs: float,
    dispersion: float,
    strong_min: float,
    deteriorating_max: float,
    dispersion_mixed: float,
) -> str:
    """Classify leader health as 'strong', 'mixed', or 'deteriorating'."""
    if avg_rs <= deteriorating_max:
        return "deteriorating"
    elif avg_rs >= strong_min and dispersion < dispersion_mixed:
        return "strong"
    else:
        return "mixed"


def _classify_concentration_regime(
    ew_cw_zscore: float,
    leader_health: str,
    elevated_z: float,
    extreme_z: float,
    healthy_mod: int,
    unhealthy_mod: int,
) -> tuple[ConcentrationRegime, int]:
    """
    Classify concentration regime and return participation modifier.

    Returns: (regime, modifier)
    """
    # Determine concentration level from EW/CW z-score
    if math.isnan(ew_cw_zscore) or ew_cw_zscore > elevated_z:
        # Normal breadth — no concentration concern
        return ConcentrationRegime.BROAD_HEALTHY, 0

    is_extreme = ew_cw_zscore <= extreme_z

    if leader_health == "strong":
        if is_extreme:
            # Extreme concentration — even strong leaders can't fully protect
            return ConcentrationRegime.CONCENTRATED_FRAGILE, healthy_mod // 2
        else:
            # Elevated but leaders healthy — suppress false exhaustion
            return ConcentrationRegime.CONCENTRATED_HEALTHY, healthy_mod

    elif leader_health == "deteriorating":
        # Leaders failing — amplify exhaustion signal
        return ConcentrationRegime.CONCENTRATED_UNHEALTHY, unhealthy_mod

    else:
        # Mixed — no strong signal either way
        return ConcentrationRegime.CONCENTRATED_FRAGILE, 0


def _default_reading(
    sector_ticker: str,
    leader_tickers: list[str],
    ew_cw_zscore: float,
) -> ConcentrationReading:
    """Return a neutral default when data is insufficient."""
    return ConcentrationReading(
        sector_ticker=sector_ticker,
        ew_cw_zscore=ew_cw_zscore,
        leader_health="mixed",
        leader_tickers=leader_tickers,
        leader_avg_rs=0.0,
        leader_dispersion=0.0,
        regime=ConcentrationRegime.BROAD_HEALTHY,
        participation_modifier=0,
        explanation=f"Concentration data unavailable for {sector_ticker}. Default: BROAD_HEALTHY.",
    )


def _build_explanation(
    sector_ticker: str,
    regime: ConcentrationRegime,
    leader_health: str,
    avg_rs: float,
    dispersion: float,
    ew_cw_zscore: float,
    leaders: list[str],
) -> str:
    """Build human-readable explanation."""
    z_str = f"{ew_cw_zscore:.2f}" if not math.isnan(ew_cw_zscore) else "N/A"
    leader_str = ", ".join(leaders[:3])
    if len(leaders) > 3:
        leader_str += f" +{len(leaders)-3}"

    if regime == ConcentrationRegime.BROAD_HEALTHY:
        return (
            f"{sector_ticker}: BROAD HEALTHY. EW/CW z: {z_str}. "
            f"Normal breadth — no concentration concern."
        )
    elif regime == ConcentrationRegime.CONCENTRATED_HEALTHY:
        return (
            f"{sector_ticker}: CONCENTRATED HEALTHY. EW/CW z: {z_str} (narrow). "
            f"But leaders ({leader_str}) are strong (avg RS vs sector: {avg_rs:+.3f}). "
            f"Suppressing false exhaustion signal. +15 participation bonus."
        )
    elif regime == ConcentrationRegime.CONCENTRATED_FRAGILE:
        return (
            f"{sector_ticker}: CONCENTRATED FRAGILE. EW/CW z: {z_str} (narrow). "
            f"Leaders ({leader_str}) are {leader_health} (avg RS: {avg_rs:+.3f}, "
            f"dispersion: {dispersion:.3f}). Monitor — could go either way."
        )
    else:  # CONCENTRATED_UNHEALTHY
        return (
            f"{sector_ticker}: CONCENTRATED UNHEALTHY. EW/CW z: {z_str} (narrow). "
            f"Leaders ({leader_str}) are deteriorating (avg RS: {avg_rs:+.3f}). "
            f"Amplifying exhaustion signal. -15 participation penalty."
        )
