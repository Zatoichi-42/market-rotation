"""
Trade Journal — tracks every actionable call the system generates.

Computes target position percentages from analysis state, regime, character,
and horizon pattern inputs.  Generates TradeCall records when deltas exceed
thresholds, fills forward returns progressively, closes stale or invalidated
calls, and produces a JournalSummary for review.
"""
import json
import os
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import pandas as pd

from engine.schemas import (
    AnalysisState,
    HorizonPattern,
    JournalSummary,
    RegimeCharacter,
    RegimeState,
    TradeCall,
    TradeState,
    TradeStateAssignment,
)


# ── Lookup tables ────────────────────────────────────────

_STATE_DIRECTION = {
    AnalysisState.OVERT_PUMP: 1,
    AnalysisState.BROADENING: 1,
    AnalysisState.ACCUMULATION: 1,
    AnalysisState.AMBIGUOUS: 0,
    AnalysisState.DISTRIBUTION: -1,
    AnalysisState.EXHAUSTION: -1,
    AnalysisState.OVERT_DUMP: -1,
}

_BASE_SIZE = {
    AnalysisState.OVERT_PUMP: 100,
    AnalysisState.BROADENING: 75,
    AnalysisState.ACCUMULATION: 50,
    AnalysisState.AMBIGUOUS: 0,
    AnalysisState.DISTRIBUTION: 50,
    AnalysisState.EXHAUSTION: 75,
    AnalysisState.OVERT_DUMP: 100,
}

_REGIME_MULTIPLIER = {
    RegimeState.NORMAL: 1.0,
    RegimeState.FRAGILE: 0.5,
    RegimeState.HOSTILE: 0.25,
}

# Character modifier: keyed by (RegimeCharacter, is_long)
_CHARACTER_MODIFIER = {
    (RegimeCharacter.TRENDING_BULL, True): 1.2,
    (RegimeCharacter.TRENDING_BULL, False): 0.5,
    (RegimeCharacter.TRENDING_BEAR, True): 0.5,
    (RegimeCharacter.TRENDING_BEAR, False): 1.2,
    (RegimeCharacter.ROTATION, True): 0.5,
    (RegimeCharacter.ROTATION, False): 0.5,
    (RegimeCharacter.CHOPPY, True): 0.5,
    (RegimeCharacter.CHOPPY, False): 0.5,
    (RegimeCharacter.CRISIS, True): 0.25,
    (RegimeCharacter.CRISIS, False): 0.25,
    (RegimeCharacter.RECOVERY, True): 1.0,
    (RegimeCharacter.RECOVERY, False): 1.0,
}

# Horizon modifier: keyed by (HorizonPattern, is_long)
_HORIZON_MODIFIER = {
    (HorizonPattern.FULL_CONFIRM, True): 1.2,
    (HorizonPattern.FULL_CONFIRM, False): 0.5,
    (HorizonPattern.ROTATION_IN, True): 1.1,
    (HorizonPattern.ROTATION_IN, False): 0.8,
    (HorizonPattern.HEALTHY_DIP, True): 1.2,
    (HorizonPattern.HEALTHY_DIP, False): 0.5,
    (HorizonPattern.NO_PATTERN, True): 1.0,
    (HorizonPattern.NO_PATTERN, False): 1.0,
    (HorizonPattern.ROTATION_OUT, True): 0.5,
    (HorizonPattern.ROTATION_OUT, False): 1.1,
    (HorizonPattern.DEAD_CAT, True): 0.0,
    (HorizonPattern.DEAD_CAT, False): 1.2,
    (HorizonPattern.FULL_REJECT, True): 0.0,
    (HorizonPattern.FULL_REJECT, False): 1.2,
}

# Minimum delta thresholds
_MIN_NEW_POSITION = 15
_MIN_ADD = 10
_MIN_REDUCE = 15
_DUPLICATE_LOCKOUT_SESSIONS = 5

# Forward return horizons (days offset, field name pairs)
_FWD_HORIZONS = [
    (1, "fwd_1d"),
    (2, "fwd_2d"),
    (5, "fwd_5d"),
    (10, "fwd_10d"),
    (20, "fwd_20d"),
    (60, "fwd_60d"),
]

_FWD_RS_HORIZONS = [
    (1, "fwd_rs_1d"),
    (2, "fwd_rs_2d"),
    (5, "fwd_rs_5d"),
    (10, "fwd_rs_10d"),
    (20, "fwd_rs_20d"),
    (60, "fwd_rs_60d"),
]

_PNL_HORIZONS = [
    (1, "pnl_1d"),
    (5, "pnl_5d"),
    (10, "pnl_10d"),
    (20, "pnl_20d"),
    (60, "pnl_60d"),
]


# ── Helpers ──────────────────────────────────────────────

def _round5(value: float) -> int:
    """Round to nearest multiple of 5."""
    return int(round(value / 5) * 5)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _resolve_enum(value, enum_class):
    """Convert a string to an enum member if it isn't already one."""
    if isinstance(value, enum_class):
        return value
    # Try by value first, then by name
    for member in enum_class:
        if member.value == value:
            return member
    return enum_class[value]


def _enum_value(v) -> str:
    """Return .value if enum, else str."""
    return v.value if hasattr(v, "value") else str(v)


# ── Graduated FRAGILE scaling ───────────────────────────

def _graduated_regime_multiplier(regime_gate: RegimeState, vix_level: float = 25.0) -> float:
    """
    Graduated regime scaling.
    NORMAL: 1.0 (always)
    HOSTILE: 0.25 (always)
    FRAGILE: scales linearly from 0.7 (VIX=20) to 0.4 (VIX=30)
    """
    if regime_gate == RegimeState.NORMAL:
        return 1.0
    if regime_gate == RegimeState.HOSTILE:
        return 0.25
    # FRAGILE: linear interpolation
    if vix_level <= 20:
        return 0.7
    if vix_level >= 30:
        return 0.4
    return round(0.7 - (vix_level - 20) * 0.03, 3)


# ── Core computation ─────────────────────────────────────

def compute_target_pct(
    analysis_state,
    confidence: int,
    regime_gate,
    regime_character,
    horizon_pattern,
    vix_level: float = 25.0,
    ticker: str = "",
    crisis_types: list | None = None,
) -> tuple[int, int, int, float, float, float]:
    """
    Compute the target position percentage and its components.

    Returns
    -------
    (target_pct, direction, base_size, regime_mult, char_mod, horizon_mod)
    """
    analysis_state = _resolve_enum(analysis_state, AnalysisState)
    regime_gate = _resolve_enum(regime_gate, RegimeState)
    regime_character = _resolve_enum(regime_character, RegimeCharacter)
    horizon_pattern = _resolve_enum(horizon_pattern, HorizonPattern)

    direction = _STATE_DIRECTION[analysis_state]
    base_size = _BASE_SIZE[analysis_state]

    # Ambiguous → everything zero
    if direction == 0:
        return (0, 0, 0, 1.0, 1.0, 1.0)

    is_long = direction == 1
    confidence_scale = confidence / 100.0
    regime_mult = _graduated_regime_multiplier(regime_gate, vix_level)

    # Character modifier — use crisis alignment if in Crisis with known types
    from engine.schemas import RegimeCharacter as RC
    if (regime_character == RC.CRISIS
            and crisis_types
            and ticker):
        from engine.crisis_alignment import get_crisis_modifier, CrisisType
        # Filter out NONE
        active = [ct for ct in crisis_types
                  if ct != CrisisType.NONE and ct != CrisisType.MULTI_CRISIS]
        if active:
            char_mod = get_crisis_modifier(ticker, crisis_types, is_long)
        else:
            char_mod = _CHARACTER_MODIFIER.get((regime_character, is_long), 1.0)
    else:
        char_mod = _CHARACTER_MODIFIER.get((regime_character, is_long), 1.0)

    horizon_mod = _HORIZON_MODIFIER.get((horizon_pattern, is_long), 1.0)

    raw = direction * base_size * confidence_scale * regime_mult * char_mod * horizon_mod
    clamped = _clamp(raw, -100, 100)
    target_pct = _round5(clamped)

    return (target_pct, direction, base_size, regime_mult, char_mod, horizon_mod)


# ── Call type label ──────────────────────────────────────

def call_type_label(prior_target: int, new_target: int) -> str:
    """Human-readable label describing the position change."""
    delta = new_target - prior_target

    if prior_target == 0 and new_target > 0:
        return "New Long"
    if prior_target == 0 and new_target < 0:
        return "New Short"
    if new_target == 0:
        return "Close"
    if prior_target > 0 and new_target < 0:
        return "Reverse to Short"
    if prior_target < 0 and new_target > 0:
        return "Reverse to Long"
    if delta > 0 and new_target > 0:
        return "Add Long"
    if delta < 0 and new_target < 0:
        return "Add Short"
    if delta < 0 and new_target > 0:
        return "Reduce Long"
    if delta > 0 and new_target < 0:
        return "Reduce Short"
    return "Adjust"


# ── Call generation ──────────────────────────────────────

def _passes_threshold(prior_target: int, new_target: int) -> bool:
    """Check whether the delta exceeds the minimum actionable threshold."""
    delta = abs(new_target - prior_target)

    # Close to zero always generates
    if new_target == 0 and prior_target != 0:
        return True

    # Direction reversal always generates
    if prior_target > 0 and new_target < 0:
        return True
    if prior_target < 0 and new_target > 0:
        return True

    # New position from flat
    if prior_target == 0 and new_target != 0:
        return delta >= _MIN_NEW_POSITION

    # Adding to existing position
    if (prior_target > 0 and new_target > prior_target) or \
       (prior_target < 0 and new_target < prior_target):
        return delta >= _MIN_ADD

    # Reducing existing position
    return delta >= _MIN_REDUCE


def _has_recent_call(ticker: str, current_date: str, existing_open_calls: list) -> bool:
    """True if there is an open call for this ticker within the last 5 sessions."""
    try:
        cur = pd.Timestamp(current_date)
    except Exception:
        return False

    for call in existing_open_calls:
        if call.ticker != ticker:
            continue
        try:
            call_dt = pd.Timestamp(call.date)
        except Exception:
            continue
        bdays = len(pd.bdate_range(call_dt, cur)) - 1  # exclusive of start
        if bdays < _DUPLICATE_LOCKOUT_SESSIONS:
            return True
    return False


def _make_call_id(ticker: str, date: str, seq: int = 0) -> str:
    """Generate a deterministic call ID."""
    return f"CALL-{ticker}-{date}-{seq:03d}"


def generate_calls(
    current_states: dict[str, TradeStateAssignment],
    prior_targets: dict[str, int],
    market_data: dict,
    existing_open_calls: list[TradeCall],
) -> list[TradeCall]:
    """
    Generate TradeCall records for tickers whose delta_pct exceeds threshold.

    Parameters
    ----------
    current_states : dict mapping ticker -> TradeStateAssignment
    prior_targets  : dict mapping ticker -> previous target_pct (default 0)
    market_data    : dict with keys:
        prices           - DataFrame with ticker columns and date index
        regime_gate      - str or RegimeState
        regime_character - str or RegimeCharacter
        horizon_readings - dict[ticker, HorizonReading or similar with .pattern]
        pumps            - dict[ticker, PumpScoreReading]
        reversal_map     - dict[ticker, ReversalScoreReading]
        rs_readings      - list of RSReading
    existing_open_calls : list of currently-open TradeCall objects

    Returns
    -------
    list[TradeCall] — new calls generated this session.
    """
    prices = market_data.get("prices")
    regime_gate = market_data.get("regime_gate", "NORMAL")
    regime_character = market_data.get("regime_character", "Choppy")
    horizon_readings = market_data.get("horizon_readings", {})
    pumps = market_data.get("pumps", {})
    reversal_map = market_data.get("reversal_map", {})
    rs_readings = market_data.get("rs_readings", [])
    vix_level = market_data.get("vix_level", 25.0)
    crisis_types = market_data.get("crisis_types", None)

    # Build RS lookup by ticker
    rs_by_ticker: dict = {}
    for r in rs_readings:
        rs_by_ticker[r.ticker] = r

    # Determine current date from prices index
    current_date = None
    if prices is not None and len(prices) > 0:
        current_date = str(prices.index[-1].date()) if hasattr(prices.index[-1], "date") else str(prices.index[-1])

    # Track sequence per ticker-date to avoid ID collisions
    seq_counter: dict[str, int] = defaultdict(int)

    new_calls: list[TradeCall] = []

    for ticker, tsa in current_states.items():
        # Resolve horizon pattern for this ticker
        hr = horizon_readings.get(ticker)
        pattern = hr.pattern if hr is not None else HorizonPattern.NO_PATTERN

        # Compute target
        target_pct, direction, base_size_val, regime_mult, char_mod, horizon_mod = compute_target_pct(
            tsa.analysis_state,
            tsa.confidence,
            regime_gate,
            regime_character,
            pattern,
            vix_level=vix_level,
            ticker=ticker,
            crisis_types=crisis_types,
        )

        prior = prior_targets.get(ticker, 0)
        delta = target_pct - prior

        # Skip if delta doesn't meet threshold
        if delta == 0:
            continue
        if not _passes_threshold(prior, target_pct):
            continue

        # Skip duplicate calls within lockout window
        if current_date and _has_recent_call(ticker, current_date, existing_open_calls):
            continue

        # Gather context values
        pump = pumps.get(ticker)
        pump_score = pump.pump_score if pump else 0.0
        pump_delta = pump.pump_delta if pump else 0.0

        rev = reversal_map.get(ticker)
        reversal_score = rev.reversal_score if rev else 0.0
        reversal_percentile = rev.reversal_percentile if rev else 0.0

        rs = rs_by_ticker.get(ticker)
        rs_2d = rs.rs_2d if rs else 0.0
        rs_5d = rs.rs_5d if rs else 0.0
        rs_10d = rs.rs_10d if rs else 0.0
        rs_20d = rs.rs_20d if rs else 0.0
        rs_60d = rs.rs_60d if rs else 0.0
        rs_120d = rs.rs_120d if rs else 0.0
        rs_rank = rs.rs_rank if rs else 0

        # Entry price
        entry_price = 0.0
        if prices is not None and ticker in prices.columns:
            entry_price = float(prices[ticker].dropna().iloc[-1])

        # Notional = abs(target_pct) * confidence
        notional = abs(target_pct) * tsa.confidence

        date_str = current_date or datetime.now().strftime("%Y-%m-%d")

        seq_key = f"{ticker}-{date_str}"
        seq = seq_counter[seq_key]
        seq_counter[seq_key] += 1

        call = TradeCall(
            call_id=_make_call_id(ticker, date_str, seq),
            date=date_str,
            ticker=ticker,
            name=tsa.name,
            analysis_state=_enum_value(tsa.analysis_state),
            trade_state=_enum_value(tsa.trade_state),
            target_pct=target_pct,
            prior_target_pct=prior,
            delta_pct=delta,
            confidence=tsa.confidence,
            direction=direction,
            base_size=base_size_val,
            regime_multiplier=regime_mult,
            character_modifier=char_mod,
            horizon_modifier=horizon_mod,
            notional=notional,
            regime_gate=_enum_value(regime_gate),
            regime_character=_enum_value(regime_character),
            horizon_pattern=_enum_value(pattern),
            pump_score=pump_score,
            pump_delta=pump_delta,
            reversal_score=reversal_score,
            reversal_percentile=reversal_percentile,
            rs_2d=rs_2d,
            rs_5d=rs_5d,
            rs_10d=rs_10d,
            rs_20d=rs_20d,
            rs_60d=rs_60d,
            rs_120d=rs_120d,
            rs_rank=rs_rank,
            entry_price=entry_price,
        )
        new_calls.append(call)

    return new_calls


# ── Forward returns ──────────────────────────────────────

def update_forward_returns(
    calls: list[TradeCall],
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    current_date: str,
) -> list[TradeCall]:
    """
    Fill forward returns and P&L for open calls as time passes.

    Forward return = price return of the ticker from entry date to entry + N days.
    Forward RS = ticker forward return minus SPY forward return (excess return).
    P&L = direction * forward RS (positive means the call was correct).
    """
    if prices is None or prices.empty:
        return calls

    cur_dt = pd.Timestamp(current_date)

    for call in calls:
        if call.status != "open":
            continue

        entry_dt = pd.Timestamp(call.date)
        ticker = call.ticker

        if ticker not in prices.columns:
            continue

        ticker_series = prices[ticker].dropna()
        spy_series = spy_prices.dropna() if spy_prices is not None else None

        # Get entry price from the series (or fall back to stored)
        if entry_dt in ticker_series.index:
            entry_px = float(ticker_series.loc[entry_dt])
        else:
            entry_px = call.entry_price

        if entry_px == 0:
            continue

        # SPY entry price
        spy_entry_px = None
        if spy_series is not None and entry_dt in spy_series.index:
            spy_entry_px = float(spy_series.loc[entry_dt])

        # Build business day targets from entry date
        bdays = pd.bdate_range(entry_dt, periods=61)  # 0..60

        # Fill forward returns
        for offset, fld in _FWD_HORIZONS:
            if getattr(call, fld) is not None:
                continue  # already filled
            if offset >= len(bdays):
                continue
            target_dt = bdays[offset]
            if target_dt > cur_dt:
                continue  # future date, can't fill yet
            if target_dt in ticker_series.index:
                fwd_px = float(ticker_series.loc[target_dt])
                fwd_ret = (fwd_px / entry_px - 1) * 100
                setattr(call, fld, round(fwd_ret, 4))

        # Fill forward RS (excess over SPY)
        for offset, fld in _FWD_RS_HORIZONS:
            if getattr(call, fld) is not None:
                continue
            if offset >= len(bdays):
                continue
            target_dt = bdays[offset]
            if target_dt > cur_dt:
                continue
            if spy_series is None or spy_entry_px is None or spy_entry_px == 0:
                continue
            if target_dt in ticker_series.index and target_dt in spy_series.index:
                fwd_px = float(ticker_series.loc[target_dt])
                spy_fwd_px = float(spy_series.loc[target_dt])
                ticker_ret = (fwd_px / entry_px - 1) * 100
                spy_ret = (spy_fwd_px / spy_entry_px - 1) * 100
                excess = ticker_ret - spy_ret
                setattr(call, fld, round(excess, 4))

        # Fill P&L = direction * forward RS
        _pnl_offset_map = {1: "fwd_rs_1d", 5: "fwd_rs_5d", 10: "fwd_rs_10d",
                           20: "fwd_rs_20d", 60: "fwd_rs_60d"}
        for offset, pnl_fld in _PNL_HORIZONS:
            if getattr(call, pnl_fld) is not None:
                continue
            rs_fld = _pnl_offset_map.get(offset)
            if rs_fld is None:
                continue
            rs_val = getattr(call, rs_fld)
            if rs_val is not None:
                pnl = call.direction * rs_val
                setattr(call, pnl_fld, round(pnl, 4))

        # Compute hit flags (positive P&L at 10d/20d)
        if call.pnl_10d is not None and call.hit_10d is None:
            call.hit_10d = call.pnl_10d > 0
        if call.pnl_20d is not None and call.hit_20d is None:
            call.hit_20d = call.pnl_20d > 0

    return calls


# ── Close calls ──────────────────────────────────────────

def close_calls(
    calls: list[TradeCall],
    current_states: dict[str, TradeStateAssignment],
    regime_state,
    current_date: str,
) -> list[TradeCall]:
    """
    Close open calls when:
    1. Direction flips to opposite (analysis state reversal).
    2. Regime becomes HOSTILE.
    3. 60 sessions have elapsed since the call date.
    """
    regime_state = _resolve_enum(regime_state, RegimeState)
    cur_dt = pd.Timestamp(current_date)

    for call in calls:
        if call.status != "open":
            continue

        close_reason = None

        # 1. Opposite direction flip
        tsa = current_states.get(call.ticker)
        if tsa is not None:
            new_dir = _STATE_DIRECTION.get(
                _resolve_enum(tsa.analysis_state, AnalysisState), 0
            )
            # Flip: was long, now short (or vice versa)
            if call.direction != 0 and new_dir != 0 and new_dir == -call.direction:
                close_reason = "Direction flip"

        # 2. HOSTILE regime
        if regime_state == RegimeState.HOSTILE:
            close_reason = "Regime HOSTILE"

        # 3. 60 sessions elapsed
        entry_dt = pd.Timestamp(call.date)
        sessions = len(pd.bdate_range(entry_dt, cur_dt)) - 1
        if sessions >= 60:
            close_reason = "60 sessions elapsed"

        if close_reason:
            call.status = "closed"
            call.close_date = current_date
            call.close_reason = close_reason

    return calls


# ── Journal summary ──────────────────────────────────────

def _confidence_band(conf: int) -> str:
    """Bucket confidence into bands for aggregation."""
    if conf <= 30:
        return "low (<=30)"
    if conf <= 60:
        return "mid (31-60)"
    return "high (61+)"


def compute_journal_summary(calls: list[TradeCall]) -> JournalSummary:
    """Aggregate statistics across all calls."""
    total = len(calls)
    open_calls = [c for c in calls if c.status == "open"]
    closed_calls = [c for c in calls if c.status == "closed"]
    n_open = len(open_calls)
    n_closed = len(closed_calls)

    # Aggregate P&L
    total_pnl_10d = sum(c.pnl_10d for c in calls if c.pnl_10d is not None)
    total_pnl_20d = sum(c.pnl_20d for c in calls if c.pnl_20d is not None)

    calls_with_10d = [c for c in calls if c.pnl_10d is not None]
    calls_with_20d = [c for c in calls if c.pnl_20d is not None]

    avg_10d = total_pnl_10d / len(calls_with_10d) if calls_with_10d else 0.0
    avg_20d = total_pnl_20d / len(calls_with_20d) if calls_with_20d else 0.0

    # Hit rates
    hits_10d = [c for c in calls if c.hit_10d is True]
    hits_20d = [c for c in calls if c.hit_20d is True]
    eligible_10d = [c for c in calls if c.hit_10d is not None]
    eligible_20d = [c for c in calls if c.hit_20d is not None]

    hit_rate_10d = len(hits_10d) / len(eligible_10d) if eligible_10d else 0.0
    hit_rate_20d = len(hits_20d) / len(eligible_20d) if eligible_20d else 0.0

    # Breakdowns
    pnl_by_state = _breakdown_pnl(calls, lambda c: c.analysis_state)
    hit_rate_by_state = _breakdown_hit_rate(calls, lambda c: c.analysis_state)

    pnl_by_regime = _breakdown_pnl(calls, lambda c: c.regime_gate)
    hit_rate_by_regime = _breakdown_hit_rate(calls, lambda c: c.regime_gate)

    pnl_by_pattern = _breakdown_pnl(calls, lambda c: c.horizon_pattern)
    hit_rate_by_pattern = _breakdown_hit_rate(calls, lambda c: c.horizon_pattern)

    pnl_by_confidence = _breakdown_pnl(calls, lambda c: _confidence_band(c.confidence))
    hit_rate_by_confidence = _breakdown_hit_rate(calls, lambda c: _confidence_band(c.confidence))

    # Cumulative P&L curve (ordered by date, using pnl_10d)
    dated_calls = sorted(
        [c for c in calls if c.pnl_10d is not None],
        key=lambda c: c.date,
    )
    cum = 0.0
    cumulative_pnl = []
    for c in dated_calls:
        cum += c.pnl_10d
        cumulative_pnl.append({"date": c.date, "call_id": c.call_id, "cumulative_pnl": round(cum, 4)})

    return JournalSummary(
        total_calls=total,
        open_calls=n_open,
        closed_calls=n_closed,
        total_pnl_10d=round(total_pnl_10d, 4),
        total_pnl_20d=round(total_pnl_20d, 4),
        avg_pnl_per_call_10d=round(avg_10d, 4),
        avg_pnl_per_call_20d=round(avg_20d, 4),
        hit_rate_10d=round(hit_rate_10d, 4),
        hit_rate_20d=round(hit_rate_20d, 4),
        pnl_by_state=pnl_by_state,
        hit_rate_by_state=hit_rate_by_state,
        pnl_by_regime=pnl_by_regime,
        hit_rate_by_regime=hit_rate_by_regime,
        pnl_by_pattern=pnl_by_pattern,
        hit_rate_by_pattern=hit_rate_by_pattern,
        pnl_by_confidence=pnl_by_confidence,
        hit_rate_by_confidence=hit_rate_by_confidence,
        cumulative_pnl=cumulative_pnl,
    )


def _breakdown_pnl(calls: list[TradeCall], key_fn) -> dict:
    """Average pnl_10d grouped by key_fn."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for c in calls:
        if c.pnl_10d is not None:
            buckets[str(key_fn(c))].append(c.pnl_10d)
    return {k: round(sum(v) / len(v), 4) for k, v in buckets.items()}


def _breakdown_hit_rate(calls: list[TradeCall], key_fn) -> dict:
    """Hit rate (10d) grouped by key_fn."""
    hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for c in calls:
        if c.hit_10d is not None:
            k = str(key_fn(c))
            totals[k] += 1
            if c.hit_10d:
                hits[k] += 1
    return {k: round(hits[k] / totals[k], 4) if totals[k] else 0.0 for k in totals}


# ── Persistence ──────────────────────────────────────────

def save_journal(
    calls: list[TradeCall],
    summary: JournalSummary,
    path: str = "data/store/journal/trade_calls.json",
) -> None:
    """Serialize calls and summary to JSON."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    payload = {
        "calls": [asdict(c) for c in calls],
        "summary": asdict(summary) if summary is not None else None,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_journal(
    path: str = "data/store/journal/trade_calls.json",
) -> tuple[list[TradeCall], Optional[JournalSummary]]:
    """Load calls and summary from JSON. Return ([], None) if no file."""
    if not os.path.exists(path):
        return ([], None)

    with open(path, "r") as f:
        payload = json.load(f)

    calls = []
    for d in payload.get("calls", []):
        calls.append(TradeCall(**d))

    summary = None
    s = payload.get("summary")
    if s is not None:
        summary = JournalSummary(**s)

    return (calls, summary)
