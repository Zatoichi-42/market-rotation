"""
Snapshot persistence — save/load DailySnapshot as parquet files.

Each snapshot is flattened into tabular form for parquet storage,
then reconstructed on load.
"""
import os
import json
from dataclasses import asdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.schemas import (
    DailySnapshot, RegimeAssessment, RegimeSignal, RegimeState, SignalLevel,
    RSReading, BreadthReading, BreadthSignal,
    PumpScoreReading, StateClassification, AnalysisState, TransitionPressure,
)


_DEFAULT_BASE_PATH = "data/store/snapshots"


def save_snapshot(snapshot: DailySnapshot, base_path: str = _DEFAULT_BASE_PATH):
    """Save a DailySnapshot as a parquet file. Filename: {base_path}/{date}.parquet."""
    os.makedirs(base_path, exist_ok=True)
    filepath = os.path.join(base_path, f"{snapshot.date}.parquet")

    # Serialize the snapshot to a JSON string stored in a single-row parquet
    data = _snapshot_to_dict(snapshot)
    json_str = json.dumps(data, default=str)

    df = pd.DataFrame({"date": [snapshot.date], "snapshot_json": [json_str]})
    df.to_parquet(filepath, index=False)


def load_snapshot(date: str, base_path: str = _DEFAULT_BASE_PATH) -> DailySnapshot:
    """Load a snapshot for a specific date. Raises FileNotFoundError if not found."""
    filepath = os.path.join(base_path, f"{date}.parquet")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No snapshot for date {date} at {filepath}")

    df = pd.read_parquet(filepath)
    json_str = df["snapshot_json"].iloc[0]
    data = json.loads(json_str)
    return _dict_to_snapshot(data)


def list_snapshots(base_path: str = _DEFAULT_BASE_PATH) -> list[str]:
    """Return sorted list of available snapshot dates."""
    if not os.path.exists(base_path):
        return []
    files = [f.replace(".parquet", "") for f in os.listdir(base_path) if f.endswith(".parquet")]
    return sorted(files)


def load_snapshot_range(
    start_date: str, end_date: str, base_path: str = _DEFAULT_BASE_PATH,
) -> list[DailySnapshot]:
    """Load all snapshots in a date range for replay."""
    available = list_snapshots(base_path)
    in_range = [d for d in available if start_date <= d <= end_date]
    return [load_snapshot(d, base_path) for d in in_range]


# ═══════════════════════════════════════════════════════
# SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════

def _snapshot_to_dict(snap: DailySnapshot) -> dict:
    """Convert DailySnapshot to a JSON-serializable dict."""
    return {
        "date": snap.date,
        "timestamp": snap.timestamp,
        "regime": _regime_to_dict(snap.regime),
        "sectors": [_rs_to_dict(s) for s in snap.sectors],
        "breadth": _breadth_to_dict(snap.breadth),
        "pump_scores": [_pump_to_dict(p) for p in snap.pump_scores],
        "states": [_state_to_dict(s) for s in snap.states],
    }


def _regime_to_dict(r: RegimeAssessment) -> dict:
    return {
        "state": r.state.value,
        "signals": [
            {"name": s.name, "raw_value": s.raw_value, "level": s.level.value, "description": s.description}
            for s in r.signals
        ],
        "hostile_count": r.hostile_count,
        "fragile_count": r.fragile_count,
        "normal_count": r.normal_count,
        "timestamp": r.timestamp,
        "explanation": r.explanation,
    }


def _rs_to_dict(r: RSReading) -> dict:
    return {
        "ticker": r.ticker, "name": r.name,
        "rs_5d": r.rs_5d, "rs_20d": r.rs_20d, "rs_60d": r.rs_60d,
        "rs_slope": r.rs_slope, "rs_rank": r.rs_rank,
        "rs_rank_change": r.rs_rank_change, "rs_composite": r.rs_composite,
    }


def _breadth_to_dict(b: BreadthReading) -> dict:
    return {
        "rsp_spy_ratio": b.rsp_spy_ratio,
        "rsp_spy_ratio_20d_change": b.rsp_spy_ratio_20d_change,
        "rsp_spy_ratio_zscore": b.rsp_spy_ratio_zscore,
        "signal": b.signal.value,
        "explanation": b.explanation,
    }


def _pump_to_dict(p: PumpScoreReading) -> dict:
    return {
        "ticker": p.ticker, "name": p.name,
        "rs_pillar": p.rs_pillar, "participation_pillar": p.participation_pillar,
        "flow_pillar": p.flow_pillar, "pump_score": p.pump_score,
        "pump_delta": p.pump_delta, "pump_delta_5d_avg": p.pump_delta_5d_avg,
    }


def _state_to_dict(s: StateClassification) -> dict:
    return {
        "ticker": s.ticker, "name": s.name,
        "state": s.state.value, "confidence": s.confidence,
        "sessions_in_state": s.sessions_in_state,
        "transition_pressure": s.transition_pressure.value,
        "prior_state": s.prior_state.value if s.prior_state else None,
        "state_changed": s.state_changed,
        "explanation": s.explanation,
    }


def _dict_to_snapshot(d: dict) -> DailySnapshot:
    """Reconstruct DailySnapshot from dict."""
    regime = _dict_to_regime(d["regime"])
    sectors = [_dict_to_rs(s) for s in d["sectors"]]
    breadth = _dict_to_breadth(d["breadth"])
    pump_scores = [_dict_to_pump(p) for p in d["pump_scores"]]
    states = [_dict_to_state(s) for s in d["states"]]
    return DailySnapshot(
        date=d["date"], timestamp=d["timestamp"],
        regime=regime, sectors=sectors, breadth=breadth,
        pump_scores=pump_scores, states=states,
    )


def _dict_to_regime(d: dict) -> RegimeAssessment:
    signals = [
        RegimeSignal(name=s["name"], raw_value=s["raw_value"],
                     level=SignalLevel(s["level"]), description=s["description"])
        for s in d["signals"]
    ]
    return RegimeAssessment(
        state=RegimeState(d["state"]), signals=signals,
        hostile_count=d["hostile_count"], fragile_count=d["fragile_count"],
        normal_count=d["normal_count"], timestamp=d["timestamp"],
        explanation=d["explanation"],
    )


def _dict_to_rs(d: dict) -> RSReading:
    return RSReading(**d)


def _dict_to_breadth(d: dict) -> BreadthReading:
    return BreadthReading(
        rsp_spy_ratio=d["rsp_spy_ratio"],
        rsp_spy_ratio_20d_change=d["rsp_spy_ratio_20d_change"],
        rsp_spy_ratio_zscore=d["rsp_spy_ratio_zscore"],
        signal=BreadthSignal(d["signal"]),
        explanation=d["explanation"],
    )


def _dict_to_pump(d: dict) -> PumpScoreReading:
    return PumpScoreReading(**d)


def _dict_to_state(d: dict) -> StateClassification:
    return StateClassification(
        ticker=d["ticker"], name=d["name"],
        state=AnalysisState(d["state"]), confidence=d["confidence"],
        sessions_in_state=d["sessions_in_state"],
        transition_pressure=TransitionPressure(d["transition_pressure"]),
        prior_state=AnalysisState(d["prior_state"]) if d["prior_state"] else None,
        state_changed=d["state_changed"],
        explanation=d["explanation"],
    )
