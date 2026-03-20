"""
Catalyst Gate — sits between Layer 1 (Regime Gate) and Layer 2 (State Classifier).

Two sub-components:
A. Scheduled catalyst lookup — FOMC, CPI, NFP, OPEC dates from catalysts.yaml
B. Unscheduled shock detector — abnormal multi-sector returns + VIX spikes

Actions:
- CLEAR: No catalyst concern. Proceed normally.
- CAUTION: Catalyst nearby or earnings season. Reduce confidence.
- EMBARGO: Catalyst imminent. No new entries.
- SHOCK_PAUSE: Unscheduled shock detected. Pause classification 1 session.

Never overrides the regime gate (respects signal hierarchy).
"""
import math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import yaml

from engine.schemas import (
    CatalystAction, CatalystCategory, CatalystImpact, ShockType,
    ScheduledCatalyst, CatalystShock, CatalystAssessment,
)


# ═══════════════════════════════════════════════════════
# A. SCHEDULED CATALYST LOOKUP
# ═══════════════════════════════════════════════════════

def load_catalyst_calendar(path: str = "config/catalysts.yaml") -> list[ScheduledCatalyst]:
    """Load all scheduled catalysts from YAML. Returns flat list of dated events."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    catalysts = []
    for event in raw.get("recurring", []):
        name = event["name"]
        category = CatalystCategory(event["category"].capitalize() if event["category"].upper() != "MACRO" else "Macro")
        if event["category"].upper() == "MACRO":
            category = CatalystCategory.MACRO
        elif event["category"].upper() == "SECTOR":
            category = CatalystCategory.SECTOR
        else:
            category = CatalystCategory.EARNINGS

        impact = CatalystImpact.HIGH if event["impact"].upper() == "HIGH" else CatalystImpact.MEDIUM
        affected = event.get("affected", ["ALL"])
        embargo = event.get("embargo_sessions", event.get("embargo", 0))

        for d in event.get("dates", event.get("dates_2026", [])):
            catalysts.append(ScheduledCatalyst(
                date=str(d),
                name=name,
                category=category,
                impact=impact,
                affected_sectors=affected,
                embargo_sessions=embargo,
            ))

    return catalysts


def check_scheduled_catalyst(
    today: str,
    catalysts: list[ScheduledCatalyst],
    sector_ticker: str = "ALL",
) -> tuple[CatalystAction, str | None, int]:
    """
    Check if today is within the embargo/caution window of any scheduled catalyst.

    Returns: (action, catalyst_name, confidence_modifier)
    """
    today_date = _parse_date(today)

    best_action = CatalystAction.CLEAR
    best_name = None
    best_modifier = 0

    for cat in catalysts:
        cat_date = _parse_date(cat.date)
        if cat_date is None:
            continue

        # Check if this catalyst affects the given sector
        if "ALL" not in cat.affected_sectors and sector_ticker not in cat.affected_sectors:
            continue

        days_until = (cat_date - today_date).days  # positive=future, 0=today, negative=past

        # EMBARGO: day of event and days leading up (within embargo window)
        if 0 <= days_until <= cat.embargo_sessions:
            if cat.impact == CatalystImpact.HIGH:
                if _action_priority(CatalystAction.EMBARGO) > _action_priority(best_action):
                    best_action = CatalystAction.EMBARGO
                    best_name = cat.name
                    best_modifier = -25
            elif cat.impact == CatalystImpact.MEDIUM:
                if _action_priority(CatalystAction.CAUTION) > _action_priority(best_action):
                    best_action = CatalystAction.CAUTION
                    best_name = cat.name
                    best_modifier = -10

        # CAUTION: one day AFTER a HIGH event (settling period)
        elif days_until == -1 and cat.impact == CatalystImpact.HIGH:
            if _action_priority(CatalystAction.CAUTION) > _action_priority(best_action):
                best_action = CatalystAction.CAUTION
                best_name = f"{cat.name} (post-event)"
                best_modifier = -10

    return best_action, best_name, best_modifier


def next_scheduled_catalyst(
    today: str,
    catalysts: list[ScheduledCatalyst],
) -> tuple[str | None, str | None, int]:
    """Find the next upcoming catalyst. Returns (name, date, sessions_away)."""
    today_date = _parse_date(today)
    best_name = None
    best_date = None
    best_dist = 999

    for cat in catalysts:
        cat_date = _parse_date(cat.date)
        if cat_date is None:
            continue
        dist = (cat_date - today_date).days
        if 0 < dist < best_dist:
            best_dist = dist
            best_name = cat.name
            best_date = cat.date

    if best_name:
        return best_name, best_date, best_dist
    return None, None, 0


# ═══════════════════════════════════════════════════════
# B. UNSCHEDULED SHOCK DETECTOR
# ═══════════════════════════════════════════════════════

_SECTOR_TICKERS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]


def detect_shock(
    prices: pd.DataFrame,
    vix_series: pd.Series | None = None,
    shock_settings: dict | None = None,
) -> CatalystShock:
    """
    Detect unscheduled market shocks from abnormal price action.

    Checks:
    1. Multi-sector selloff/rally (>70% of sectors moving >1σ in same direction)
    2. Single-sector dislocation (one sector >2.5σ while others normal)
    3. VIX spike (>3 points or >15% in one session)

    Returns CatalystShock (shock_type=NONE if no shock detected).
    """
    if shock_settings is None:
        shock_settings = {}

    today_str = str(prices.index[-1].date()) if not prices.empty else ""

    # Compute 1-day returns for sectors
    available = [t for t in _SECTOR_TICKERS if t in prices.columns]
    if len(available) < 5 or len(prices) < 60:
        return _no_shock(today_str)

    returns_1d = prices[available].pct_change().iloc[-1]
    # Compute 60-day rolling σ for each sector
    returns_60d = prices[available].pct_change().iloc[-60:]
    sigma_60d = returns_60d.std()

    # Z-score today's return vs 60d σ
    zscore_today = {}
    for t in available:
        if sigma_60d[t] > 0:
            zscore_today[t] = returns_1d[t] / sigma_60d[t]
        else:
            zscore_today[t] = 0.0

    # ── Check 1: Multi-sector selloff/rally ──
    threshold = shock_settings.get("broad_selloff_sector_fraction", 0.70)
    sigma_bar = shock_settings.get("broad_selloff_sigma", 1.0)

    n_down = sum(1 for z in zscore_today.values() if z < -sigma_bar)
    n_up = sum(1 for z in zscore_today.values() if z > sigma_bar)
    n_total = len(available)

    if n_down / n_total >= threshold:
        avg_z = np.mean([z for z in zscore_today.values() if z < -sigma_bar])
        return CatalystShock(
            date=today_str,
            shock_type=ShockType.BROAD_SELLOFF,
            magnitude=abs(avg_z),
            affected_tickers=[t for t, z in zscore_today.items() if z < -sigma_bar],
            confidence=80,
            explanation=f"Broad selloff: {n_down}/{n_total} sectors below -{sigma_bar}σ. "
                        f"Avg z-score: {avg_z:.2f}.",
        )

    if n_up / n_total >= threshold:
        avg_z = np.mean([z for z in zscore_today.values() if z > sigma_bar])
        return CatalystShock(
            date=today_str,
            shock_type=ShockType.BROAD_RALLY,
            magnitude=abs(avg_z),
            affected_tickers=[t for t, z in zscore_today.items() if z > sigma_bar],
            confidence=70,
            explanation=f"Broad rally: {n_up}/{n_total} sectors above +{sigma_bar}σ. "
                        f"Avg z-score: {avg_z:.2f}.",
        )

    # ── Check 2: Single-sector dislocation ──
    disloc_sigma = shock_settings.get("dislocation_sigma", 2.5)
    disloc_gap = shock_settings.get("dislocation_gap_sigma", 2.0)

    zvals = pd.Series(zscore_today)
    median_z = zvals.median()
    outliers = zvals[zvals.abs() > disloc_sigma]

    if not outliers.empty:
        worst = outliers.iloc[0] if outliers.abs().max() == outliers.abs().iloc[0] else outliers[outliers.abs() == outliers.abs().max()].iloc[0]
        worst_ticker = outliers.abs().idxmax()
        gap = abs(worst) - abs(median_z)
        if gap >= disloc_gap:
            return CatalystShock(
                date=today_str,
                shock_type=ShockType.SECTOR_DISLOCATION,
                magnitude=abs(worst),
                affected_tickers=[worst_ticker],
                confidence=75,
                explanation=f"Sector dislocation: {worst_ticker} at {worst:.2f}σ, "
                            f"median sector at {median_z:.2f}σ (gap: {gap:.2f}σ).",
            )

    # ── Check 3: VIX spike ──
    if vix_series is not None and len(vix_series) >= 2:
        vix_today = vix_series.iloc[-1]
        vix_yesterday = vix_series.iloc[-2]
        if not (math.isnan(vix_today) or math.isnan(vix_yesterday)):
            vix_change = vix_today - vix_yesterday
            vix_pct = vix_change / vix_yesterday if vix_yesterday > 0 else 0.0
            jump_pts = shock_settings.get("vix_jump_points", 3.0)
            jump_pct = shock_settings.get("vix_jump_pct", 0.15)

            if vix_change >= jump_pts or vix_pct >= jump_pct:
                return CatalystShock(
                    date=today_str,
                    shock_type=ShockType.BROAD_SELLOFF,
                    magnitude=vix_pct * 10,  # Normalized
                    affected_tickers=["^VIX"],
                    confidence=70,
                    explanation=f"VIX spike: {vix_yesterday:.1f} → {vix_today:.1f} "
                                f"(+{vix_change:.1f} pts, +{vix_pct:.1%}).",
                )

    return _no_shock(today_str)


def count_multi_sector_direction(prices: pd.DataFrame, window: int = 20) -> int:
    """Count how many sectors are trending in the same direction over window days."""
    available = [t for t in _SECTOR_TICKERS if t in prices.columns]
    if len(available) < 5 or len(prices) < window:
        return 0

    returns = prices[available].pct_change(window).iloc[-1]
    n_positive = sum(1 for r in returns if r > 0)
    n_negative = sum(1 for r in returns if r < 0)

    return max(n_positive, n_negative)


# ═══════════════════════════════════════════════════════
# C. COMBINED CATALYST ASSESSMENT
# ═══════════════════════════════════════════════════════

def assess_catalyst(
    today: str,
    prices: pd.DataFrame,
    catalysts: list[ScheduledCatalyst] | None = None,
    vix_series: pd.Series | None = None,
    catalyst_settings: dict | None = None,
    shock_settings: dict | None = None,
) -> CatalystAssessment:
    """
    Full catalyst gate assessment. Combines scheduled + unscheduled detection.

    Priority: SHOCK_PAUSE > EMBARGO > CAUTION > CLEAR
    """
    if catalyst_settings is None:
        catalyst_settings = {}

    # Check unscheduled shocks first (highest priority)
    shock = detect_shock(prices, vix_series, shock_settings)

    pause_threshold = catalyst_settings.get("shock_pause_threshold", 2.0)
    if shock.shock_type != ShockType.NONE and shock.magnitude >= pause_threshold:
        multi = count_multi_sector_direction(prices)
        return CatalystAssessment(
            action=CatalystAction.SHOCK_PAUSE,
            scheduled_catalyst=None,
            shock_detected=shock.shock_type,
            shock_magnitude=shock.magnitude,
            affected_sectors=shock.affected_tickers,
            confidence_modifier=-30,
            explanation=f"SHOCK PAUSE: {shock.explanation} "
                        f"Classification paused for 1 session. Re-run pipeline tomorrow.",
            multi_sector_count=multi,
        )

    # Check scheduled catalysts
    if catalysts:
        action, name, modifier = check_scheduled_catalyst(today, catalysts)
    else:
        action, name, modifier = CatalystAction.CLEAR, None, 0

    # If shock detected but below pause threshold, upgrade to at least CAUTION
    if shock.shock_type != ShockType.NONE and action == CatalystAction.CLEAR:
        action = CatalystAction.CAUTION
        modifier = min(modifier, -15)

    multi = count_multi_sector_direction(prices)

    explanation = _build_assessment_explanation(action, name, shock, multi)

    return CatalystAssessment(
        action=action,
        scheduled_catalyst=name,
        shock_detected=shock.shock_type,
        shock_magnitude=shock.magnitude if shock.shock_type != ShockType.NONE else 0.0,
        affected_sectors=shock.affected_tickers if shock.shock_type != ShockType.NONE else [],
        confidence_modifier=modifier,
        explanation=explanation,
        multi_sector_count=multi,
    )


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def _parse_date(d: str) -> date | None:
    """Parse ISO date string to date object."""
    try:
        return datetime.strptime(str(d), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _action_priority(action: CatalystAction) -> int:
    """Higher number = more restrictive."""
    return {
        CatalystAction.CLEAR: 0,
        CatalystAction.CAUTION: 1,
        CatalystAction.EMBARGO: 2,
        CatalystAction.SHOCK_PAUSE: 3,
    }[action]


def _no_shock(today_str: str) -> CatalystShock:
    return CatalystShock(
        date=today_str,
        shock_type=ShockType.NONE,
        magnitude=0.0,
        affected_tickers=[],
        confidence=0,
        explanation="No unscheduled shock detected.",
    )


def _build_assessment_explanation(
    action: CatalystAction,
    scheduled_name: str | None,
    shock: CatalystShock,
    multi_sector_count: int,
) -> str:
    parts = []

    if action == CatalystAction.CLEAR:
        parts.append("CLEAR: No catalyst concerns.")
    elif action == CatalystAction.CAUTION:
        if scheduled_name:
            parts.append(f"CAUTION: {scheduled_name} nearby. Reduce confidence, prefer smaller entries.")
        else:
            parts.append("CAUTION: Mild shock detected. Monitor closely.")
    elif action == CatalystAction.EMBARGO:
        parts.append(f"EMBARGO: {scheduled_name} imminent. No new entries until event passes.")

    if shock.shock_type != ShockType.NONE:
        parts.append(f"Shock: {shock.explanation}")

    parts.append(f"Multi-sector alignment: {multi_sector_count}/11 sectors trending same direction.")

    return " ".join(parts)
