"""
Exit Monitor tests — synthetic data, all 22 spec cases.
"""
import pytest

from engine.schemas import (
    ExitAssessment,
    ExitSignal,
    ExitSignalType,
    ExitUrgency,
    HorizonPattern,
    RegimeCharacter,
)
from engine.exit_monitor import (
    assess_exit,
    check_breadth_narrowing,
    check_delta_deceleration,
    check_failed_breakouts,
    check_horizon_flip,
    check_relative_stop,
    check_reversal_erosion,
    check_volume_climax,
)


# ── Helpers ───────────────────────────────────────────────────


def _make_signal(
    signal_type: ExitSignalType = ExitSignalType.DELTA_DECEL,
    ticker: str = "XLK",
    urgency: ExitUrgency = ExitUrgency.WATCH,
    sessions_active: int = 0,
    value: float = 0.0,
    threshold: float = 0.0,
    description: str = "test signal",
) -> ExitSignal:
    return ExitSignal(
        signal_type=signal_type,
        ticker=ticker,
        urgency=urgency,
        sessions_active=sessions_active,
        value=value,
        threshold=threshold,
        description=description,
    )


def _make_signals(n: int, **overrides) -> list[ExitSignal]:
    """Create n distinct signals using different ExitSignalType values."""
    types = list(ExitSignalType)
    return [
        _make_signal(signal_type=types[i % len(types)], **overrides)
        for i in range(n)
    ]


# ── TEST-EXIT-01 / 02: Delta Deceleration ────────────────────


class TestDeltaDeceleration:

    def test_exit01_decel_3_sessions_watch(self):
        """TEST-EXIT-01: Delta decel 3 sessions (all positive, shrinking) -> WATCH."""
        # 4 values: deltas going 4.0, 3.0, 2.0, 1.0 → 3 consecutive decels
        sig = check_delta_deceleration("XLK", [4.0, 3.0, 2.0, 1.0])
        assert sig is not None
        assert sig.signal_type == ExitSignalType.DELTA_DECEL
        assert sig.urgency == ExitUrgency.WATCH
        assert sig.sessions_active == 3

    def test_exit02_decel_5_sessions_warning(self):
        """TEST-EXIT-02: Delta decel 5+ sessions -> WARNING."""
        # 6 values: 6 consecutive positive shrinking → 5 decel sessions
        sig = check_delta_deceleration("XLK", [6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        assert sig is not None
        assert sig.urgency == ExitUrgency.WARNING
        assert sig.sessions_active == 5

    def test_no_signal_when_not_all_positive(self):
        """Negative value breaks the chain."""
        sig = check_delta_deceleration("XLK", [3.0, -1.0, 2.0, 1.0])
        assert sig is None or sig.sessions_active < 3

    def test_no_signal_when_not_shrinking(self):
        """Increasing deltas should not fire."""
        sig = check_delta_deceleration("XLK", [1.0, 2.0, 3.0, 4.0])
        assert sig is None

    def test_no_signal_short_history(self):
        """Need at least 2 data points."""
        sig = check_delta_deceleration("XLK", [3.0])
        assert sig is None


# ── TEST-EXIT-03 / 04: Reversal Erosion ──────────────────────


class TestReversalErosion:

    def test_exit03_erosion_fires(self):
        """TEST-EXIT-03: Reversal score +0.16 above entry -> WARNING."""
        sig = check_reversal_erosion("XLK", entry_reversal_score=0.30,
                                     current_reversal_score=0.46)
        assert sig is not None
        assert sig.urgency == ExitUrgency.WARNING
        assert sig.signal_type == ExitSignalType.REVERSAL_EROSION

    def test_exit04_no_signal_below_threshold(self):
        """TEST-EXIT-04: Reversal score +0.14 above entry -> no signal."""
        sig = check_reversal_erosion("XLK", entry_reversal_score=0.30,
                                     current_reversal_score=0.44)
        assert sig is None

    def test_exact_threshold_no_fire(self):
        """Exactly at threshold (<=) should not fire."""
        sig = check_reversal_erosion("XLK", entry_reversal_score=1.0,
                                     current_reversal_score=1.15)
        assert sig is None


# ── TEST-EXIT-05 / 06: Volume Climax ─────────────────────────


class TestVolumeClimax:

    def test_exit05_rvol_spike_then_flat(self):
        """TEST-EXIT-05: RVOL=2.5 then 2 days flat -> Volume Climax signal."""
        sig = check_volume_climax(
            "XLK",
            rvol_history=[2.5, 1.0, 0.9],
            return_history=[0.01, 0.001, 0.001],
        )
        assert sig is not None
        assert sig.signal_type == ExitSignalType.VOLUME_CLIMAX
        assert sig.urgency == ExitUrgency.WARNING

    def test_exit06_rvol_below_threshold(self):
        """TEST-EXIT-06: RVOL=1.5 then 2 days flat -> no signal."""
        sig = check_volume_climax(
            "XLK",
            rvol_history=[1.5, 1.0, 0.9],
            return_history=[0.01, 0.001, 0.001],
        )
        assert sig is None

    def test_no_signal_with_follow_through(self):
        """If subsequent returns are positive, no climax signal."""
        sig = check_volume_climax(
            "XLK",
            rvol_history=[2.5, 1.0, 0.9],
            return_history=[0.01, 0.005, 0.004],
        )
        assert sig is None

    def test_no_signal_short_history(self):
        """Need at least 3 data points."""
        sig = check_volume_climax("XLK", rvol_history=[2.5, 1.0],
                                  return_history=[0.01, 0.001])
        assert sig is None


# ── TEST-EXIT-07 / 08: Failed Breakouts ──────────────────────


class TestFailedBreakouts:

    def test_exit07_rate_above_warning(self):
        """TEST-EXIT-07: Failed breakout rate 0.65 (entry 0.5) -> WATCH."""
        sig = check_failed_breakouts("XLK", failed_breakout_rate=0.65,
                                     entry_rate=0.5)
        assert sig is not None
        assert sig.urgency == ExitUrgency.WATCH
        assert sig.signal_type == ExitSignalType.FAILED_BREAKOUTS

    def test_exit08_rate_above_alert(self):
        """TEST-EXIT-08: Failed breakout rate 0.72 (entry 0.5) -> WARNING."""
        sig = check_failed_breakouts("XLK", failed_breakout_rate=0.72,
                                     entry_rate=0.5)
        assert sig is not None
        assert sig.urgency == ExitUrgency.WARNING

    def test_no_signal_below_entry(self):
        """Rate below entry rate should not fire even if above warning."""
        sig = check_failed_breakouts("XLK", failed_breakout_rate=0.65,
                                     entry_rate=0.70)
        assert sig is None


# ── TEST-EXIT-09 / 10: Horizon Flip ──────────────────────────


class TestHorizonFlip:

    def test_exit09_full_confirm_to_rotation_out(self):
        """TEST-EXIT-09: FULL_CONFIRM -> ROTATION_OUT -> WARNING."""
        sig = check_horizon_flip("XLK", HorizonPattern.FULL_CONFIRM,
                                 HorizonPattern.ROTATION_OUT)
        assert sig is not None
        assert sig.urgency == ExitUrgency.WARNING
        assert sig.signal_type == ExitSignalType.HORIZON_FLIP

    def test_exit10_rotation_in_to_dead_cat(self):
        """TEST-EXIT-10: ROTATION_IN -> DEAD_CAT -> ALERT."""
        sig = check_horizon_flip("XLK", HorizonPattern.ROTATION_IN,
                                 HorizonPattern.DEAD_CAT)
        assert sig is not None
        assert sig.urgency == ExitUrgency.ALERT

    def test_healthy_dip_to_rotation_out(self):
        """HEALTHY_DIP -> ROTATION_OUT -> ALERT."""
        sig = check_horizon_flip("XLK", HorizonPattern.HEALTHY_DIP,
                                 HorizonPattern.ROTATION_OUT)
        assert sig is not None
        assert sig.urgency == ExitUrgency.ALERT

    def test_no_flip_same_pattern(self):
        """No signal when patterns are the same."""
        sig = check_horizon_flip("XLK", HorizonPattern.FULL_CONFIRM,
                                 HorizonPattern.FULL_CONFIRM)
        assert sig is None

    def test_no_flip_unrecognized_transition(self):
        """ROTATION_OUT -> FULL_REJECT is not a tracked transition."""
        sig = check_horizon_flip("XLK", HorizonPattern.ROTATION_OUT,
                                 HorizonPattern.FULL_REJECT)
        assert sig is None


# ── TEST-EXIT-11 / 12 / 13: Relative Stop ────────────────────


class TestRelativeStop:

    def test_exit11_warning(self):
        """TEST-EXIT-11: RS declined 5.5% from peak -> WARNING."""
        sig = check_relative_stop("XLK", peak_rs_20d=1.0, current_rs_20d=0.945)
        assert sig is not None
        assert sig.urgency == ExitUrgency.WARNING

    def test_exit12_alert(self):
        """TEST-EXIT-12: RS declined 8.5% from peak -> ALERT."""
        sig = check_relative_stop("XLK", peak_rs_20d=1.0, current_rs_20d=0.915)
        assert sig is not None
        assert sig.urgency == ExitUrgency.ALERT

    def test_exit13_immediate(self):
        """TEST-EXIT-13: RS declined 12.5% from peak -> IMMEDIATE."""
        sig = check_relative_stop("XLK", peak_rs_20d=1.0, current_rs_20d=0.875)
        assert sig is not None
        assert sig.urgency == ExitUrgency.IMMEDIATE

    def test_no_signal_within_threshold(self):
        """RS decline within 5% -> no signal."""
        sig = check_relative_stop("XLK", peak_rs_20d=1.0, current_rs_20d=0.96)
        assert sig is None


# ── TEST-EXIT-14..18: Assessment thresholds ───────────────────


class TestAssessExitThresholds:

    def test_exit14_zero_signals_hold(self):
        """TEST-EXIT-14: 0 signals -> 'Hold'."""
        a = assess_exit("XLK", [], RegimeCharacter.ROTATION)
        assert a.recommendation == "Hold"

    def test_exit15_one_signal_hold_monitor(self):
        """TEST-EXIT-15: 1 signal -> 'Hold (monitor: ...)'."""
        signals = [_make_signal(signal_type=ExitSignalType.DELTA_DECEL)]
        a = assess_exit("XLK", signals, RegimeCharacter.ROTATION)
        assert "Hold (monitor" in a.recommendation
        assert ExitSignalType.DELTA_DECEL.value in a.recommendation

    def test_exit16_two_signals_tighten(self):
        """TEST-EXIT-16: 2 signals -> 'Tighten stop'."""
        signals = _make_signals(2)
        a = assess_exit("XLK", signals, RegimeCharacter.ROTATION)
        assert a.recommendation == "Tighten stop"

    def test_exit17_three_signals_reduce(self):
        """TEST-EXIT-17: 3 signals -> 'Reduce position by 50%'."""
        signals = _make_signals(3)
        a = assess_exit("XLK", signals, RegimeCharacter.ROTATION)
        assert a.recommendation == "Reduce position by 50%"

    def test_exit18_four_signals_exit(self):
        """TEST-EXIT-18: 4 signals -> 'Exit'."""
        signals = _make_signals(4)
        a = assess_exit("XLK", signals, RegimeCharacter.ROTATION)
        assert a.recommendation == "Exit"


# ── TEST-EXIT-19: IMMEDIATE override ─────────────────────────


class TestImmediateOverride:

    def test_exit19_immediate_forces_exit(self):
        """TEST-EXIT-19: Any IMMEDIATE -> 'Exit' regardless of count."""
        signals = [_make_signal(urgency=ExitUrgency.IMMEDIATE)]
        a = assess_exit("XLK", signals, RegimeCharacter.ROTATION)
        assert a.recommendation == "Exit"
        assert a.urgency == ExitUrgency.IMMEDIATE


# ── TEST-EXIT-20..22: Regime modifiers ────────────────────────


class TestRegimeModifiers:

    def test_exit20_choppy_reduces_thresholds(self):
        """TEST-EXIT-20: CHOPPY -> thresholds reduced by 1 (2 signals -> 'Reduce')."""
        signals = _make_signals(2)
        a = assess_exit("XLK", signals, RegimeCharacter.CHOPPY)
        assert a.recommendation == "Reduce position by 50%"

    def test_exit21_trending_bull_increases_thresholds(self):
        """TEST-EXIT-21: TRENDING_BULL -> thresholds increased by 1 (3 signals -> 'Tighten')."""
        signals = _make_signals(3)
        a = assess_exit("XLK", signals, RegimeCharacter.TRENDING_BULL)
        assert a.recommendation == "Tighten stop"

    def test_exit22_crisis_any_signal_exit(self):
        """TEST-EXIT-22: CRISIS -> any signal -> 'Exit'."""
        signals = [_make_signal()]
        a = assess_exit("XLK", signals, RegimeCharacter.CRISIS)
        assert a.recommendation == "Exit"

    def test_crisis_zero_signals_hold(self):
        """CRISIS with no signals should still Hold."""
        a = assess_exit("XLK", [], RegimeCharacter.CRISIS)
        assert a.recommendation == "Hold"
