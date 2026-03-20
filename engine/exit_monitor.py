"""
Exit Monitor — Phase 4, Step 5.

Checks open positions against seven exit signal types, then aggregates
signals into per-ticker ExitAssessments with urgency and recommendation.
"""
from __future__ import annotations

from engine.schemas import (
    ExitAssessment,
    ExitSignal,
    ExitSignalType,
    ExitUrgency,
    HorizonPattern,
    RegimeCharacter,
)


# ── Urgency ordering helper ──────────────────────────────────

_URGENCY_ORDER = {
    ExitUrgency.WATCH: 0,
    ExitUrgency.WARNING: 1,
    ExitUrgency.ALERT: 2,
    ExitUrgency.IMMEDIATE: 3,
}


def _max_urgency(urgencies: list[ExitUrgency]) -> ExitUrgency:
    """Return the highest urgency from a list."""
    if not urgencies:
        return ExitUrgency.WATCH
    return max(urgencies, key=lambda u: _URGENCY_ORDER[u])


# ── Individual signal checks ─────────────────────────────────


def check_delta_deceleration(
    ticker: str,
    delta_history: list[float],
    sessions_threshold: int = 3,
) -> ExitSignal | None:
    """
    Fires when pump_delta is positive but has been shrinking for N+
    consecutive sessions.

    delta_today < delta_yesterday < delta_2d_ago, ALL must be positive.
    WATCH at threshold sessions, WARNING at threshold+2.
    """
    if len(delta_history) < 2:
        return None

    # Count consecutive shrinking positive deltas from the end
    decel_count = 0
    for i in range(len(delta_history) - 1, 0, -1):
        if (
            delta_history[i] > 0
            and delta_history[i - 1] > 0
            and delta_history[i] < delta_history[i - 1]
        ):
            decel_count += 1
        else:
            break

    if decel_count < sessions_threshold:
        return None

    urgency = (
        ExitUrgency.WARNING
        if decel_count >= sessions_threshold + 2
        else ExitUrgency.WATCH
    )

    return ExitSignal(
        signal_type=ExitSignalType.DELTA_DECEL,
        ticker=ticker,
        urgency=urgency,
        sessions_active=decel_count,
        value=delta_history[-1],
        threshold=float(sessions_threshold),
        description=(
            f"Pump delta positive but shrinking for {decel_count} consecutive "
            f"sessions (current: {delta_history[-1]:.3f})"
        ),
    )


def check_reversal_erosion(
    ticker: str,
    entry_reversal_score: float,
    current_reversal_score: float,
    threshold: float = 0.15,
) -> ExitSignal | None:
    """
    Fires when current reversal_score > entry + threshold.
    Always WARNING.
    """
    diff = current_reversal_score - entry_reversal_score
    if diff <= threshold:
        return None

    return ExitSignal(
        signal_type=ExitSignalType.REVERSAL_EROSION,
        ticker=ticker,
        urgency=ExitUrgency.WARNING,
        sessions_active=0,
        value=round(diff, 4),
        threshold=threshold,
        description=(
            f"Reversal score increased {diff:.2f} above entry "
            f"(entry: {entry_reversal_score:.2f}, current: {current_reversal_score:.2f})"
        ),
    )


def check_breadth_narrowing(
    ticker: str,
    leader_health: str,
    participation_modifier: int,
    sessions_active: int = 0,
) -> ExitSignal | None:
    """
    Fires when leader_health == "deteriorating" AND participation_modifier < 0.
    WATCH initially, WARNING if sessions >= 5.
    """
    if leader_health != "deteriorating" or participation_modifier >= 0:
        return None

    urgency = ExitUrgency.WARNING if sessions_active >= 5 else ExitUrgency.WATCH

    return ExitSignal(
        signal_type=ExitSignalType.BREADTH_NARROWING,
        ticker=ticker,
        urgency=urgency,
        sessions_active=sessions_active,
        value=float(participation_modifier),
        threshold=0.0,
        description=(
            f"Leader health deteriorating with negative participation modifier "
            f"({participation_modifier}) for {sessions_active} sessions"
        ),
    )


def check_volume_climax(
    ticker: str,
    rvol_history: list[float],
    return_history: list[float],
    rvol_threshold: float = 2.0,
    follow_through_pct: float = 0.003,
) -> ExitSignal | None:
    """
    Fires when: RVOL > threshold on some day, then 2+ subsequent days
    with no new high AND return < follow_through_pct.
    Need at least 3 data points. Look at last 5 days of rvol_history.
    WARNING.
    """
    if len(rvol_history) < 3 or len(return_history) < 3:
        return None

    # Look at last 5 days
    window_rvol = rvol_history[-5:]
    window_ret = return_history[-5:]

    # Find a climax day (RVOL > threshold), then check subsequent days
    for i in range(len(window_rvol) - 2):  # Need at least 2 days after
        if window_rvol[i] >= rvol_threshold:
            # Check that all subsequent days have return < follow_through_pct
            subsequent_returns = window_ret[i + 1:]
            if len(subsequent_returns) >= 2 and all(
                r < follow_through_pct for r in subsequent_returns
            ):
                return ExitSignal(
                    signal_type=ExitSignalType.VOLUME_CLIMAX,
                    ticker=ticker,
                    urgency=ExitUrgency.WARNING,
                    sessions_active=len(subsequent_returns),
                    value=window_rvol[i],
                    threshold=rvol_threshold,
                    description=(
                        f"Volume climax (RVOL {window_rvol[i]:.1f}x) with no "
                        f"follow-through for {len(subsequent_returns)} sessions"
                    ),
                )
    return None


def check_failed_breakouts(
    ticker: str,
    failed_breakout_rate: float,
    entry_rate: float = 0.0,
    warning_threshold: float = 0.60,
    alert_threshold: float = 0.70,
) -> ExitSignal | None:
    """
    Fires when rate > warning_threshold AND rate > entry_rate.
    WATCH at warning_threshold, WARNING at alert_threshold.
    """
    if failed_breakout_rate <= warning_threshold or failed_breakout_rate <= entry_rate:
        return None

    urgency = (
        ExitUrgency.WARNING
        if failed_breakout_rate >= alert_threshold
        else ExitUrgency.WATCH
    )

    return ExitSignal(
        signal_type=ExitSignalType.FAILED_BREAKOUTS,
        ticker=ticker,
        urgency=urgency,
        sessions_active=0,
        value=round(failed_breakout_rate, 4),
        threshold=warning_threshold,
        description=(
            f"Failed breakout rate {failed_breakout_rate:.0%} "
            f"(entry: {entry_rate:.0%}, warning: {warning_threshold:.0%})"
        ),
    )


def check_horizon_flip(
    ticker: str,
    entry_horizon: HorizonPattern,
    current_horizon: HorizonPattern,
) -> ExitSignal | None:
    """
    Fires when horizon flips from bullish to less bullish/bearish:
      FULL_CONFIRM -> anything else -> WARNING
      ROTATION_IN -> DEAD_CAT -> ALERT
      HEALTHY_DIP -> ROTATION_OUT -> ALERT
    Other flips -> None.
    """
    if entry_horizon == current_horizon:
        return None

    urgency: ExitUrgency | None = None

    if entry_horizon == HorizonPattern.FULL_CONFIRM and current_horizon != HorizonPattern.FULL_CONFIRM:
        urgency = ExitUrgency.WARNING
    elif entry_horizon == HorizonPattern.ROTATION_IN and current_horizon == HorizonPattern.DEAD_CAT:
        urgency = ExitUrgency.ALERT
    elif entry_horizon == HorizonPattern.HEALTHY_DIP and current_horizon == HorizonPattern.ROTATION_OUT:
        urgency = ExitUrgency.ALERT
    else:
        return None

    return ExitSignal(
        signal_type=ExitSignalType.HORIZON_FLIP,
        ticker=ticker,
        urgency=urgency,
        sessions_active=0,
        value=0.0,
        threshold=0.0,
        description=(
            f"Horizon flipped from {entry_horizon.value} to {current_horizon.value}"
        ),
    )


def check_relative_stop(
    ticker: str,
    peak_rs_20d: float,
    current_rs_20d: float,
    warning_pct: float = 0.05,
    alert_pct: float = 0.08,
    immediate_pct: float = 0.12,
) -> ExitSignal | None:
    """
    Fires when peak_rs_20d - current_rs_20d > threshold.
    WARNING at 5%, ALERT at 8%, IMMEDIATE at 12%.
    """
    decline = peak_rs_20d - current_rs_20d
    if decline <= warning_pct:
        return None

    if decline > immediate_pct:
        urgency = ExitUrgency.IMMEDIATE
    elif decline > alert_pct:
        urgency = ExitUrgency.ALERT
    else:
        urgency = ExitUrgency.WARNING

    return ExitSignal(
        signal_type=ExitSignalType.RELATIVE_STOP,
        ticker=ticker,
        urgency=urgency,
        sessions_active=0,
        value=round(decline, 6),
        threshold=warning_pct,
        description=(
            f"RS declined {decline:.1%} from peak "
            f"(peak: {peak_rs_20d:.4f}, current: {current_rs_20d:.4f})"
        ),
    )


# ── Assessment aggregation ────────────────────────────────────


_DEFAULT_SETTINGS = {
    "signals_to_tighten": 2,
    "signals_to_reduce": 3,
    "signals_to_exit": 4,
}


def assess_exit(
    ticker: str,
    signals: list[ExitSignal],
    regime_character: RegimeCharacter,
    settings: dict | None = None,
) -> ExitAssessment:
    """
    Aggregate signals into recommendation.

    Default thresholds (from settings or defaults):
      signals_to_tighten: 2
      signals_to_reduce: 3
      signals_to_exit: 4

    Rules:
      0 signals -> "Hold"
      1 signal  -> "Hold (monitor: {signal_name})"
      2 signals -> "Tighten stop"
      3 signals -> "Reduce position by 50%"
      4+ signals -> "Exit"
      Any IMMEDIATE signal -> "Exit" regardless of count

    Regime character modifier:
      CHOPPY: reduce thresholds by 1 (2 -> "Reduce", not "Tighten")
      TRENDING_BULL: increase thresholds by 1 (3 -> "Tighten", not "Reduce")
      CRISIS: any signal -> "Exit"
    """
    cfg = {**_DEFAULT_SETTINGS, **(settings or {})}
    tighten = cfg["signals_to_tighten"]
    reduce_ = cfg["signals_to_reduce"]
    exit_ = cfg["signals_to_exit"]

    # Regime modifier
    if regime_character == RegimeCharacter.CHOPPY:
        tighten -= 1
        reduce_ -= 1
        exit_ -= 1
    elif regime_character == RegimeCharacter.TRENDING_BULL:
        tighten += 1
        reduce_ += 1
        exit_ += 1

    n = len(signals)
    urgency = _max_urgency([s.urgency for s in signals]) if signals else ExitUrgency.WATCH

    # CRISIS override: any signal -> Exit
    if regime_character == RegimeCharacter.CRISIS and n > 0:
        return ExitAssessment(
            ticker=ticker,
            signals=signals,
            urgency=urgency,
            recommendation="Exit",
            description=f"CRISIS regime with {n} active signal(s) — immediate exit",
        )

    # IMMEDIATE override: any IMMEDIATE signal -> Exit
    if any(s.urgency == ExitUrgency.IMMEDIATE for s in signals):
        return ExitAssessment(
            ticker=ticker,
            signals=signals,
            urgency=urgency,
            recommendation="Exit",
            description=f"IMMEDIATE urgency signal detected — exit",
        )

    # Threshold-based recommendation
    if n >= exit_:
        recommendation = "Exit"
        description = f"{n} exit signals active — exit position"
    elif n >= reduce_:
        recommendation = "Reduce position by 50%"
        description = f"{n} exit signals active — reduce position"
    elif n >= tighten:
        recommendation = "Tighten stop"
        description = f"{n} exit signals active — tighten stop"
    elif n == 1:
        sig_name = signals[0].signal_type.value
        recommendation = f"Hold (monitor: {sig_name})"
        description = f"1 signal active: {sig_name}"
    else:
        recommendation = "Hold"
        description = "No exit signals active"

    return ExitAssessment(
        ticker=ticker,
        signals=signals,
        urgency=urgency,
        recommendation=recommendation,
        description=description,
    )


# ── Batch function ────────────────────────────────────────────


def assess_all_exits(
    positions: list[dict],
    market_data: dict,
    regime_character: RegimeCharacter,
    settings: dict | None = None,
) -> dict[str, ExitAssessment]:
    """
    Run all exit checks for all open positions, return dict[ticker, ExitAssessment].

    positions: list of dicts from PositionTracker.get_open_positions()
    market_data: dict keyed by ticker with keys:
        price, rs_20d, pump_score, reversal_score, confidence,
        horizon_pattern, delta_history, rvol_history, return_history,
        failed_breakout_rate, leader_health, participation_modifier,
        breadth_sessions_active
    """
    results: dict[str, ExitAssessment] = {}

    for pos in positions:
        ticker = pos["ticker"]
        md = market_data.get(ticker, {})
        if not md:
            # No market data — hold
            results[ticker] = ExitAssessment(
                ticker=ticker,
                signals=[],
                urgency=ExitUrgency.WATCH,
                recommendation="Hold",
                description="No market data available",
            )
            continue

        signals: list[ExitSignal] = []

        # 1. Delta deceleration
        delta_history = md.get("delta_history", [])
        sig = check_delta_deceleration(ticker, delta_history)
        if sig:
            signals.append(sig)

        # 2. Reversal erosion
        entry_reversal = pos.get("entry_reversal_score", 0.0)
        current_reversal = md.get("reversal_score", 0.0)
        sig = check_reversal_erosion(ticker, entry_reversal, current_reversal)
        if sig:
            signals.append(sig)

        # 3. Breadth narrowing
        leader_health = md.get("leader_health", "")
        participation_modifier = md.get("participation_modifier", 0)
        breadth_sessions = md.get("breadth_sessions_active", 0)
        sig = check_breadth_narrowing(
            ticker, leader_health, participation_modifier, breadth_sessions
        )
        if sig:
            signals.append(sig)

        # 4. Volume climax
        rvol_history = md.get("rvol_history", [])
        return_history = md.get("return_history", [])
        sig = check_volume_climax(ticker, rvol_history, return_history)
        if sig:
            signals.append(sig)

        # 5. Failed breakouts
        failed_rate = md.get("failed_breakout_rate", 0.0)
        entry_rate = pos.get("entry_failed_breakout_rate", 0.0)
        sig = check_failed_breakouts(ticker, failed_rate, entry_rate)
        if sig:
            signals.append(sig)

        # 6. Horizon flip
        entry_horizon_str = pos.get("entry_horizon_pattern", "No Pattern")
        current_horizon_str = md.get("horizon_pattern", "No Pattern")
        try:
            entry_horizon = (
                entry_horizon_str
                if isinstance(entry_horizon_str, HorizonPattern)
                else HorizonPattern(entry_horizon_str)
            )
            current_horizon = (
                current_horizon_str
                if isinstance(current_horizon_str, HorizonPattern)
                else HorizonPattern(current_horizon_str)
            )
        except ValueError:
            entry_horizon = HorizonPattern.NO_PATTERN
            current_horizon = HorizonPattern.NO_PATTERN
        sig = check_horizon_flip(ticker, entry_horizon, current_horizon)
        if sig:
            signals.append(sig)

        # 7. Relative stop
        peak_rs = pos.get("peak_rs_20d", 0.0)
        current_rs = md.get("rs_20d", 0.0)
        sig = check_relative_stop(ticker, peak_rs, current_rs)
        if sig:
            signals.append(sig)

        # Aggregate
        results[ticker] = assess_exit(ticker, signals, regime_character, settings)

    return results
