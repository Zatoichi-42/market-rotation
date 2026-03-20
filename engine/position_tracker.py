"""
Position Tracker — manages open/closed position lifecycle.

Stores positions in a JSON file. Tracks entry snapshots, daily live state
updates (price, PnL, peak metrics, delta deceleration, horizon changes),
and exit records with full entry-vs-exit comparisons.
"""
import json
import os
from datetime import datetime
from typing import Optional

from engine.schemas import (
    AnalysisState,
    ExitAssessment,
    HorizonPattern,
    PositionEntrySnapshot,
    PositionExitRecord,
    PositionLiveState,
    RegimeCharacter,
    RegimeState,
    TradeState,
)


class PositionTracker:
    """Manages position lifecycle: open, update, close."""

    MAX_OPEN = 6

    def __init__(self, storage_path: str = "data/store/positions"):
        self.storage_path = storage_path
        self.filepath = os.path.join(storage_path, "open_positions.json")
        os.makedirs(storage_path, exist_ok=True)
        if not os.path.exists(self.filepath):
            self._save({"positions": [], "closed": []})

    # ── Public API ────────────────────────────────────

    def open_position(self, snapshot: PositionEntrySnapshot) -> str:
        """Record new position. Returns position_id. Max 6 open."""
        data = self._load()
        if len(data["positions"]) >= self.MAX_OPEN:
            raise ValueError(
                f"Cannot open position: already at max ({self.MAX_OPEN} open)"
            )
        pos_dict = self._snapshot_to_dict(snapshot)
        # Initialise live-tracking fields stored alongside the entry snapshot
        pos_dict["sessions_held"] = 0
        pos_dict["current_price"] = snapshot.entry_price
        pos_dict["peak_rs_20d"] = snapshot.entry_rs_20d
        pos_dict["peak_pump_score"] = snapshot.entry_pump_score
        pos_dict["current_rs_20d"] = snapshot.entry_rs_20d
        pos_dict["current_pump_score"] = snapshot.entry_pump_score
        pos_dict["current_reversal_score"] = snapshot.entry_reversal_score
        pos_dict["current_confidence"] = snapshot.entry_confidence
        pos_dict["delta_decel_sessions"] = 0
        pos_dict["current_horizon"] = snapshot.entry_horizon_pattern.value
        data["positions"].append(pos_dict)
        self._save(data)
        return snapshot.position_id

    def update_positions(self, market_data: dict) -> list[PositionLiveState]:
        """
        Daily update of all open positions.

        market_data keyed by ticker::

            {ticker: {"price": float, "rs_20d": float, "pump_score": float,
                      "reversal_score": float, "reversal_percentile": float,
                      "confidence": int, "horizon_pattern": HorizonPattern,
                      "delta_history": list[float]}}

        Returns list of PositionLiveState (exit_assessment=None).
        """
        data = self._load()
        live_states: list[PositionLiveState] = []
        today = datetime.now().strftime("%Y-%m-%d")

        for pos in data["positions"]:
            ticker = pos["ticker"]
            if ticker not in market_data:
                continue
            md = market_data[ticker]

            # Increment session counter
            pos["sessions_held"] += 1

            # Price and PnL
            pos["current_price"] = md["price"]
            entry_price = pos["entry_price"]
            unrealized_pnl_pct = ((md["price"] - entry_price) / entry_price) * 100.0

            # Peak tracking — never decrease
            pos["current_rs_20d"] = md["rs_20d"]
            pos["peak_rs_20d"] = max(pos["peak_rs_20d"], md["rs_20d"])

            pos["current_pump_score"] = md["pump_score"]
            pos["peak_pump_score"] = max(pos["peak_pump_score"], md["pump_score"])

            # Reversal & confidence
            pos["current_reversal_score"] = md["reversal_score"]
            pos["current_confidence"] = md["confidence"]

            # Delta deceleration: consecutive sessions where delta[i] < delta[i-1]
            # and both are positive
            delta_hist = md.get("delta_history", [])
            pos["delta_decel_sessions"] = self._count_delta_decel(delta_hist)

            # Horizon change
            new_horizon = md["horizon_pattern"]
            if isinstance(new_horizon, HorizonPattern):
                new_horizon_str = new_horizon.value
            else:
                new_horizon_str = str(new_horizon)
            entry_horizon_str = pos["entry_horizon_pattern"]
            pos["current_horizon"] = new_horizon_str
            horizon_changed = new_horizon_str != entry_horizon_str

            # Build live state
            live_state = PositionLiveState(
                position_id=pos["position_id"],
                ticker=ticker,
                current_date=today,
                sessions_held=pos["sessions_held"],
                current_price=md["price"],
                unrealized_pnl_pct=round(unrealized_pnl_pct, 4),
                peak_rs_20d_since_entry=pos["peak_rs_20d"],
                peak_pump_score_since_entry=pos["peak_pump_score"],
                rs_decline_from_peak=round(
                    pos["peak_rs_20d"] - md["rs_20d"], 6
                ),
                pump_decline_from_peak=round(
                    pos["peak_pump_score"] - md["pump_score"], 6
                ),
                reversal_score_change=round(
                    md["reversal_score"] - pos["entry_reversal_score"], 6
                ),
                confidence_change=md["confidence"] - pos["entry_confidence"],
                delta_decel_sessions=pos["delta_decel_sessions"],
                exit_assessment=None,
                entry_horizon=HorizonPattern(entry_horizon_str),
                current_horizon=HorizonPattern(new_horizon_str),
                horizon_changed=horizon_changed,
            )
            live_states.append(live_state)

        self._save(data)
        return live_states

    def close_position(
        self,
        position_id: str,
        exit_reason: str,
        exit_price: float,
        exit_pump_score: float = 0.0,
        exit_reversal_score: float = 0.0,
        exit_signals: list[str] | None = None,
        rs_vs_spy: float = 0.0,
    ) -> PositionExitRecord:
        """Close position, move to closed list, return exit record."""
        data = self._load()
        pos = None
        pos_idx = None
        for i, p in enumerate(data["positions"]):
            if p["position_id"] == position_id:
                pos = p
                pos_idx = i
                break
        if pos is None:
            raise ValueError(f"Position {position_id} not found in open positions")

        entry_price = pos["entry_price"]
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0

        exit_date = datetime.now().strftime("%Y-%m-%d")

        # Entry quality heuristic based on confidence
        entry_confidence = pos["entry_confidence"]
        if entry_confidence >= 70:
            entry_quality = "high"
        elif entry_confidence >= 40:
            entry_quality = "medium"
        else:
            entry_quality = "low"

        # Exit quality heuristic based on PnL direction
        if pnl_pct > 2.0:
            exit_quality = "good"
        elif pnl_pct > -2.0:
            exit_quality = "neutral"
        else:
            exit_quality = "poor"

        # State correctness: entry state suggested long and PnL positive
        entry_trade_state = pos["entry_trade_state"]
        long_states = {TradeState.LONG_ENTRY.value, TradeState.HOLD.value,
                       TradeState.SELECTIVE_ADD.value}
        if entry_trade_state in long_states:
            state_was_correct = pnl_pct > 0
        else:
            state_was_correct = pnl_pct <= 0

        # Lesson tags
        lesson_tags: list[str] = []
        if pnl_pct > 5.0:
            lesson_tags.append("winner")
        if pnl_pct <= -5.0:
            lesson_tags.append("big_loss")
        if pos["sessions_held"] <= 2:
            lesson_tags.append("quick_exit")
        if pos.get("delta_decel_sessions", 0) >= 3:
            lesson_tags.append("delta_decel_exit")

        record = PositionExitRecord(
            position_id=position_id,
            ticker=pos["ticker"],
            entry_date=pos["entry_date"],
            exit_date=exit_date,
            sessions_held=pos["sessions_held"],
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=round(pnl_pct, 4),
            rs_vs_spy_during_hold=rs_vs_spy,
            exit_reason=exit_reason,
            exit_signals_at_close=exit_signals or [],
            entry_quality=entry_quality,
            exit_quality=exit_quality,
            state_was_correct=state_was_correct,
            entry_pump_score=pos["entry_pump_score"],
            exit_pump_score=exit_pump_score,
            entry_reversal_score=pos["entry_reversal_score"],
            exit_reversal_score=exit_reversal_score,
            lesson_tags=lesson_tags,
        )

        # Move from open to closed
        closed_dict = self._exit_record_to_dict(record)
        data["positions"].pop(pos_idx)
        data["closed"].append(closed_dict)
        self._save(data)
        return record

    def get_open_positions(self) -> list[dict]:
        """Return raw open position dicts."""
        data = self._load()
        return data["positions"]

    def get_position_for_ticker(self, ticker: str) -> dict | None:
        """Return open position dict for ticker, or None."""
        data = self._load()
        for pos in data["positions"]:
            if pos["ticker"] == ticker:
                return pos
        return None

    # ── Persistence ───────────────────────────────────

    def _load(self) -> dict:
        """Load JSON file."""
        with open(self.filepath, "r") as f:
            return json.load(f)

    def _save(self, data: dict):
        """Save JSON file."""
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    # ── Internal helpers ──────────────────────────────

    @staticmethod
    def _count_delta_decel(delta_history: list[float]) -> int:
        """Count consecutive sessions where delta[i] < delta[i-1] and both > 0."""
        if len(delta_history) < 2:
            return 0
        count = 0
        for i in range(len(delta_history) - 1, 0, -1):
            if delta_history[i] > 0 and delta_history[i - 1] > 0:
                if delta_history[i] < delta_history[i - 1]:
                    count += 1
                else:
                    break
            else:
                break
        return count

    @staticmethod
    def _snapshot_to_dict(snapshot: PositionEntrySnapshot) -> dict:
        """Convert a PositionEntrySnapshot to a JSON-safe dict."""
        return {
            "position_id": snapshot.position_id,
            "ticker": snapshot.ticker,
            "name": snapshot.name,
            "entry_date": snapshot.entry_date,
            "entry_price": snapshot.entry_price,
            "entry_analysis_state": snapshot.entry_analysis_state.value,
            "entry_trade_state": snapshot.entry_trade_state.value,
            "entry_regime_gate": snapshot.entry_regime_gate.value,
            "entry_regime_character": snapshot.entry_regime_character.value,
            "entry_horizon_pattern": snapshot.entry_horizon_pattern.value,
            "entry_pump_score": snapshot.entry_pump_score,
            "entry_pump_delta": snapshot.entry_pump_delta,
            "entry_reversal_score": snapshot.entry_reversal_score,
            "entry_reversal_percentile": snapshot.entry_reversal_percentile,
            "entry_confidence": snapshot.entry_confidence,
            "entry_rs_5d": snapshot.entry_rs_5d,
            "entry_rs_20d": snapshot.entry_rs_20d,
            "entry_rs_60d": snapshot.entry_rs_60d,
            "entry_rs_rank": snapshot.entry_rs_rank,
            "expected_hold_sessions": snapshot.expected_hold_sessions,
            "invalidation_condition": snapshot.invalidation_condition,
        }

    @staticmethod
    def _exit_record_to_dict(record: PositionExitRecord) -> dict:
        """Convert a PositionExitRecord to a JSON-safe dict."""
        return {
            "position_id": record.position_id,
            "ticker": record.ticker,
            "entry_date": record.entry_date,
            "exit_date": record.exit_date,
            "sessions_held": record.sessions_held,
            "entry_price": record.entry_price,
            "exit_price": record.exit_price,
            "pnl_pct": record.pnl_pct,
            "rs_vs_spy_during_hold": record.rs_vs_spy_during_hold,
            "exit_reason": record.exit_reason,
            "exit_signals_at_close": record.exit_signals_at_close,
            "entry_quality": record.entry_quality,
            "exit_quality": record.exit_quality,
            "state_was_correct": record.state_was_correct,
            "entry_pump_score": record.entry_pump_score,
            "exit_pump_score": record.exit_pump_score,
            "entry_reversal_score": record.entry_reversal_score,
            "exit_reversal_score": record.exit_reversal_score,
            "lesson_tags": record.lesson_tags,
        }
