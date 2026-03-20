"""
Position Tracker tests — open/update/close lifecycle with synthetic data.
"""
import json
import os

import pytest

from engine.schemas import (
    AnalysisState,
    HorizonPattern,
    PositionEntrySnapshot,
    PositionExitRecord,
    PositionLiveState,
    RegimeCharacter,
    RegimeState,
    TradeState,
)
from engine.position_tracker import PositionTracker


# ── Helpers ───────────────────────────────────────────


def _make_snapshot(
    ticker: str = "XLK",
    name: str = "Technology",
    entry_date: str = "2026-03-15",
    entry_price: float = 100.0,
    position_id: str | None = None,
    entry_rs_20d: float = 0.05,
    entry_pump_score: float = 72.0,
    entry_reversal_score: float = 30.0,
    entry_confidence: int = 65,
    entry_horizon_pattern: HorizonPattern = HorizonPattern.FULL_CONFIRM,
) -> PositionEntrySnapshot:
    pid = position_id or f"POS-{ticker}-{entry_date.replace('-', '')}"
    return PositionEntrySnapshot(
        position_id=pid,
        ticker=ticker,
        name=name,
        entry_date=entry_date,
        entry_price=entry_price,
        entry_analysis_state=AnalysisState.BROADENING,
        entry_trade_state=TradeState.LONG_ENTRY,
        entry_regime_gate=RegimeState.NORMAL,
        entry_regime_character=RegimeCharacter.TRENDING_BULL,
        entry_horizon_pattern=entry_horizon_pattern,
        entry_pump_score=entry_pump_score,
        entry_pump_delta=1.5,
        entry_reversal_score=entry_reversal_score,
        entry_reversal_percentile=45.0,
        entry_confidence=entry_confidence,
        entry_rs_5d=0.03,
        entry_rs_20d=entry_rs_20d,
        entry_rs_60d=0.08,
        entry_rs_rank=3,
        expected_hold_sessions=10,
        invalidation_condition="RS rank drops below 8",
    )


def _make_market_data(
    ticker: str = "XLK",
    price: float = 105.0,
    rs_20d: float = 0.06,
    pump_score: float = 75.0,
    reversal_score: float = 32.0,
    reversal_percentile: float = 50.0,
    confidence: int = 68,
    horizon_pattern: HorizonPattern = HorizonPattern.FULL_CONFIRM,
    delta_history: list[float] | None = None,
) -> dict:
    return {
        ticker: {
            "price": price,
            "rs_20d": rs_20d,
            "pump_score": pump_score,
            "reversal_score": reversal_score,
            "reversal_percentile": reversal_percentile,
            "confidence": confidence,
            "horizon_pattern": horizon_pattern,
            "delta_history": delta_history or [1.0, 1.2, 1.5],
        }
    }


# ── TEST-POS-01: open_position creates entry with all fields ─


class TestOpenPosition:
    def test_open_position_creates_entry(self, tmp_path):
        """TEST-POS-01: open_position creates entry with all fields."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        snap = _make_snapshot()
        pid = tracker.open_position(snap)

        assert pid == "POS-XLK-20260315"

        pos = tracker.get_position_for_ticker("XLK")
        assert pos is not None
        assert pos["position_id"] == pid
        assert pos["ticker"] == "XLK"
        assert pos["name"] == "Technology"
        assert pos["entry_date"] == "2026-03-15"
        assert pos["entry_price"] == 100.0
        assert pos["entry_analysis_state"] == AnalysisState.BROADENING.value
        assert pos["entry_trade_state"] == TradeState.LONG_ENTRY.value
        assert pos["entry_regime_gate"] == RegimeState.NORMAL.value
        assert pos["entry_regime_character"] == RegimeCharacter.TRENDING_BULL.value
        assert pos["entry_horizon_pattern"] == HorizonPattern.FULL_CONFIRM.value
        assert pos["entry_pump_score"] == 72.0
        assert pos["entry_pump_delta"] == 1.5
        assert pos["entry_reversal_score"] == 30.0
        assert pos["entry_reversal_percentile"] == 45.0
        assert pos["entry_confidence"] == 65
        assert pos["entry_rs_5d"] == 0.03
        assert pos["entry_rs_20d"] == 0.05
        assert pos["entry_rs_60d"] == 0.08
        assert pos["entry_rs_rank"] == 3
        assert pos["expected_hold_sessions"] == 10
        assert pos["invalidation_condition"] == "RS rank drops below 8"
        # Live tracking fields initialised
        assert pos["sessions_held"] == 0
        assert pos["current_price"] == 100.0
        assert pos["peak_rs_20d"] == 0.05
        assert pos["peak_pump_score"] == 72.0


# ── TEST-POS-02: sessions_held tracks correctly ─────


class TestSessionsHeld:
    def test_sessions_held_increments(self, tmp_path):
        """TEST-POS-02: update_positions tracks sessions_held correctly."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot())

        md = _make_market_data()

        # First update
        states = tracker.update_positions(md)
        assert len(states) == 1
        assert states[0].sessions_held == 1

        # Second update
        states = tracker.update_positions(md)
        assert states[0].sessions_held == 2

        # Third update
        states = tracker.update_positions(md)
        assert states[0].sessions_held == 3


# ── TEST-POS-03: Peak RS tracks highest ─────────────


class TestPeakRS:
    def test_peak_rs_never_decreases(self, tmp_path):
        """TEST-POS-03: Peak RS tracks highest (never decreases)."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        # Entry rs_20d = 0.05
        tracker.open_position(_make_snapshot(entry_rs_20d=0.05))

        # Day 1: RS rises to 0.08
        states = tracker.update_positions(_make_market_data(rs_20d=0.08))
        assert states[0].peak_rs_20d_since_entry == 0.08

        # Day 2: RS drops to 0.03 — peak should stay at 0.08
        states = tracker.update_positions(_make_market_data(rs_20d=0.03))
        assert states[0].peak_rs_20d_since_entry == 0.08

        # Day 3: RS rises to 0.10 — new peak
        states = tracker.update_positions(_make_market_data(rs_20d=0.10))
        assert states[0].peak_rs_20d_since_entry == 0.10


# ── TEST-POS-04: Peak pump score tracks highest ─────


class TestPeakPumpScore:
    def test_peak_pump_score_never_decreases(self, tmp_path):
        """TEST-POS-04: Peak pump score tracks highest (never decreases)."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        # Entry pump_score = 72.0
        tracker.open_position(_make_snapshot(entry_pump_score=72.0))

        # Day 1: pump rises to 80
        states = tracker.update_positions(_make_market_data(pump_score=80.0))
        assert states[0].peak_pump_score_since_entry == 80.0

        # Day 2: pump drops to 65 — peak stays 80
        states = tracker.update_positions(_make_market_data(pump_score=65.0))
        assert states[0].peak_pump_score_since_entry == 80.0

        # Day 3: pump rises to 85 — new peak
        states = tracker.update_positions(_make_market_data(pump_score=85.0))
        assert states[0].peak_pump_score_since_entry == 85.0


# ── TEST-POS-05: rs_decline_from_peak = peak - current ─


class TestRSDecline:
    def test_rs_decline_from_peak(self, tmp_path):
        """TEST-POS-05: rs_decline_from_peak = peak - current."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot(entry_rs_20d=0.05))

        # Push peak to 0.10
        tracker.update_positions(_make_market_data(rs_20d=0.10))

        # Now drop to 0.04 — decline should be 0.10 - 0.04 = 0.06
        states = tracker.update_positions(_make_market_data(rs_20d=0.04))
        assert states[0].rs_decline_from_peak == pytest.approx(0.06, abs=1e-6)

        # At peak — decline should be 0
        states = tracker.update_positions(_make_market_data(rs_20d=0.10))
        assert states[0].rs_decline_from_peak == pytest.approx(0.0, abs=1e-6)


# ── TEST-POS-06: reversal_score_change = current - entry ─


class TestReversalScoreChange:
    def test_reversal_score_change(self, tmp_path):
        """TEST-POS-06: reversal_score_change = current - entry."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot(entry_reversal_score=30.0))

        # Current reversal = 42 → change = 42 - 30 = 12
        states = tracker.update_positions(
            _make_market_data(reversal_score=42.0)
        )
        assert states[0].reversal_score_change == pytest.approx(12.0, abs=1e-6)

        # Current reversal = 25 → change = 25 - 30 = -5
        states = tracker.update_positions(
            _make_market_data(reversal_score=25.0)
        )
        assert states[0].reversal_score_change == pytest.approx(-5.0, abs=1e-6)


# ── TEST-POS-07: close_position generates exit record with PnL ─


class TestClosePosition:
    def test_close_position_pnl(self, tmp_path):
        """TEST-POS-07: close_position generates exit record with PnL."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        snap = _make_snapshot(entry_price=100.0)
        pid = tracker.open_position(snap)

        # One update so sessions_held > 0
        tracker.update_positions(_make_market_data(price=110.0))

        record = tracker.close_position(
            position_id=pid,
            exit_reason="Target reached",
            exit_price=110.0,
            exit_pump_score=80.0,
            exit_reversal_score=35.0,
            exit_signals=["delta_decel"],
            rs_vs_spy=0.02,
        )

        assert isinstance(record, PositionExitRecord)
        assert record.position_id == pid
        assert record.ticker == "XLK"
        assert record.entry_price == 100.0
        assert record.exit_price == 110.0
        # PnL = (110-100)/100 * 100 = 10%
        assert record.pnl_pct == pytest.approx(10.0, abs=0.01)
        assert record.exit_reason == "Target reached"
        assert record.exit_signals_at_close == ["delta_decel"]
        assert record.sessions_held == 1

        # Position removed from open
        assert tracker.get_position_for_ticker("XLK") is None
        assert len(tracker.get_open_positions()) == 0


# ── TEST-POS-08: Exit record includes entry vs exit comparisons ─


class TestExitRecordComparisons:
    def test_entry_vs_exit_fields(self, tmp_path):
        """TEST-POS-08: Exit record includes entry vs exit comparisons."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        snap = _make_snapshot(
            entry_price=100.0,
            entry_pump_score=72.0,
            entry_reversal_score=30.0,
            entry_confidence=65,
        )
        pid = tracker.open_position(snap)
        tracker.update_positions(_make_market_data(price=95.0))

        record = tracker.close_position(
            position_id=pid,
            exit_reason="Stop hit",
            exit_price=95.0,
            exit_pump_score=55.0,
            exit_reversal_score=50.0,
        )

        assert record.entry_pump_score == 72.0
        assert record.exit_pump_score == 55.0
        assert record.entry_reversal_score == 30.0
        assert record.exit_reversal_score == 50.0
        assert record.entry_price == 100.0
        assert record.exit_price == 95.0
        # PnL negative → exit_quality "poor"
        assert record.pnl_pct == pytest.approx(-5.0, abs=0.01)
        assert record.exit_quality == "poor"
        # Entry confidence 65 → entry_quality "medium"
        assert record.entry_quality == "medium"
        # Trade state was LONG_ENTRY but PnL < 0 → state_was_correct False
        assert record.state_was_correct is False
        # Big loss → lesson tag
        assert "big_loss" in record.lesson_tags


# ── TEST-POS-09: Storage persists (write, re-read) ──


class TestPersistence:
    def test_persistence_across_instances(self, tmp_path):
        """TEST-POS-09: Storage persists (write, re-read)."""
        tracker1 = PositionTracker(storage_path=str(tmp_path))
        tracker1.open_position(_make_snapshot(ticker="XLK"))
        tracker1.open_position(
            _make_snapshot(
                ticker="XLF",
                position_id="POS-XLF-20260315",
                name="Financials",
            )
        )

        # Create a new tracker instance pointing to the same path
        tracker2 = PositionTracker(storage_path=str(tmp_path))
        positions = tracker2.get_open_positions()
        assert len(positions) == 2
        tickers = {p["ticker"] for p in positions}
        assert tickers == {"XLK", "XLF"}

        # Verify raw JSON file content
        filepath = os.path.join(str(tmp_path), "open_positions.json")
        with open(filepath) as f:
            raw = json.load(f)
        assert len(raw["positions"]) == 2
        assert len(raw["closed"]) == 0


# ── TEST-POS-10: get_open_positions returns only open ─


class TestGetOpenPositions:
    def test_only_open_returned(self, tmp_path):
        """TEST-POS-10: get_open_positions returns only open."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot(ticker="XLK"))
        tracker.open_position(
            _make_snapshot(
                ticker="XLE",
                position_id="POS-XLE-20260315",
                name="Energy",
            )
        )

        # Close one position
        tracker.update_positions(
            {
                "XLK": {
                    "price": 105.0, "rs_20d": 0.06, "pump_score": 75.0,
                    "reversal_score": 32.0, "reversal_percentile": 50.0,
                    "confidence": 68, "horizon_pattern": HorizonPattern.FULL_CONFIRM,
                    "delta_history": [1.0, 1.2],
                },
                "XLE": {
                    "price": 55.0, "rs_20d": 0.02, "pump_score": 40.0,
                    "reversal_score": 60.0, "reversal_percentile": 80.0,
                    "confidence": 30, "horizon_pattern": HorizonPattern.ROTATION_OUT,
                    "delta_history": [0.5],
                },
            }
        )
        tracker.close_position(
            position_id="POS-XLE-20260315",
            exit_reason="Rotation out",
            exit_price=55.0,
        )

        open_positions = tracker.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0]["ticker"] == "XLK"

        # Verify closed list has the closed one
        data = tracker._load()
        assert len(data["closed"]) == 1
        assert data["closed"][0]["ticker"] == "XLE"


# ── TEST-POS-11: Max 6 open positions enforced ──────


class TestMaxOpenPositions:
    def test_max_6_enforced(self, tmp_path):
        """TEST-POS-11: Max 6 open positions enforced."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC"]
        for t in tickers:
            tracker.open_position(
                _make_snapshot(
                    ticker=t,
                    position_id=f"POS-{t}-20260315",
                    name=t,
                )
            )

        assert len(tracker.get_open_positions()) == 6

        # 7th should raise
        with pytest.raises(ValueError, match="max"):
            tracker.open_position(
                _make_snapshot(
                    ticker="XLY",
                    position_id="POS-XLY-20260315",
                    name="Consumer Disc.",
                )
            )

        # Still 6
        assert len(tracker.get_open_positions()) == 6


# ── TEST-POS-12: Horizon change tracked ─────────────


class TestHorizonChange:
    def test_horizon_change_detected(self, tmp_path):
        """TEST-POS-12: Horizon change tracked."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(
            _make_snapshot(entry_horizon_pattern=HorizonPattern.FULL_CONFIRM)
        )

        # Same horizon — no change
        states = tracker.update_positions(
            _make_market_data(horizon_pattern=HorizonPattern.FULL_CONFIRM)
        )
        assert states[0].horizon_changed is False
        assert states[0].entry_horizon == HorizonPattern.FULL_CONFIRM
        assert states[0].current_horizon == HorizonPattern.FULL_CONFIRM

        # Horizon flips to ROTATION_OUT
        states = tracker.update_positions(
            _make_market_data(horizon_pattern=HorizonPattern.ROTATION_OUT)
        )
        assert states[0].horizon_changed is True
        assert states[0].entry_horizon == HorizonPattern.FULL_CONFIRM
        assert states[0].current_horizon == HorizonPattern.ROTATION_OUT

    def test_horizon_change_back_to_entry(self, tmp_path):
        """Horizon reverts to entry pattern — horizon_changed=False."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(
            _make_snapshot(entry_horizon_pattern=HorizonPattern.HEALTHY_DIP)
        )

        # Change away
        tracker.update_positions(
            _make_market_data(horizon_pattern=HorizonPattern.FULL_REJECT)
        )

        # Change back to entry
        states = tracker.update_positions(
            _make_market_data(horizon_pattern=HorizonPattern.HEALTHY_DIP)
        )
        assert states[0].horizon_changed is False


# ── Additional edge-case tests ───────────────────────


class TestDeltaDeceleration:
    def test_delta_decel_counting(self, tmp_path):
        """Delta deceleration counts consecutive declining positive deltas."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot())

        # 3 consecutive decelerations: 5.0 > 4.0 > 3.0 > 2.0
        states = tracker.update_positions(
            _make_market_data(delta_history=[5.0, 4.0, 3.0, 2.0])
        )
        assert states[0].delta_decel_sessions == 3

    def test_delta_decel_broken_chain(self, tmp_path):
        """Chain breaks when an increase occurs."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot())

        # 5.0 > 4.0 then 4.0 < 6.0 (increase) then 6.0 > 3.0
        # Only last pair counts from the right: 6.0 > 3.0 = 1
        states = tracker.update_positions(
            _make_market_data(delta_history=[5.0, 4.0, 6.0, 3.0])
        )
        assert states[0].delta_decel_sessions == 1

    def test_delta_decel_with_negatives(self, tmp_path):
        """Negative deltas break the chain."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot())

        # Negative value breaks chain
        states = tracker.update_positions(
            _make_market_data(delta_history=[3.0, -1.0, 2.0, 1.0])
        )
        # From right: 2.0 > 1.0 (both positive, decel) = 1,
        # then -1.0 breaks
        assert states[0].delta_decel_sessions == 1

    def test_delta_decel_empty_history(self, tmp_path):
        """Empty delta history returns 0."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot())

        states = tracker.update_positions(
            _make_market_data(delta_history=[])
        )
        assert states[0].delta_decel_sessions == 0


class TestClosePositionNotFound:
    def test_close_nonexistent_raises(self, tmp_path):
        """Closing a position that doesn't exist raises ValueError."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            tracker.close_position(
                position_id="POS-FAKE-20260101",
                exit_reason="test",
                exit_price=100.0,
            )


class TestGetPositionForTicker:
    def test_returns_none_for_missing(self, tmp_path):
        """get_position_for_ticker returns None when no position for ticker."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        assert tracker.get_position_for_ticker("XLK") is None

    def test_returns_correct_position(self, tmp_path):
        """get_position_for_ticker returns the right position."""
        tracker = PositionTracker(storage_path=str(tmp_path))
        tracker.open_position(_make_snapshot(ticker="XLK"))
        tracker.open_position(
            _make_snapshot(ticker="XLF", position_id="POS-XLF-20260315")
        )

        pos = tracker.get_position_for_ticker("XLF")
        assert pos is not None
        assert pos["ticker"] == "XLF"
