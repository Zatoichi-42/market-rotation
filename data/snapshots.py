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
    IndustryRSReading, GroupType, ReversalScoreReading, PumpMapRow,
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
    d = {
        "date": snap.date,
        "timestamp": snap.timestamp,
        "regime": _regime_to_dict(snap.regime),
        "sectors": [_rs_to_dict(s) for s in snap.sectors],
        "breadth": _breadth_to_dict(snap.breadth),
        "pump_scores": [_pump_to_dict(p) for p in snap.pump_scores],
        "states": [_state_to_dict(s) for s in snap.states],
    }
    # Phase 2 fields
    if snap.industry_rs:
        d["industry_rs"] = [_industry_rs_to_dict(i) for i in snap.industry_rs]
    if snap.reversal_scores:
        d["reversal_scores"] = [_reversal_to_dict(r) for r in snap.reversal_scores]
    if snap.pump_map:
        d["pump_map"] = [_pump_map_row_to_dict(r) for r in snap.pump_map]
    return d


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


def _industry_rs_to_dict(i: IndustryRSReading) -> dict:
    return {
        "ticker": i.ticker, "name": i.name, "parent_sector": i.parent_sector,
        "group_type": i.group_type.value,
        "rs_5d": i.rs_5d, "rs_20d": i.rs_20d, "rs_60d": i.rs_60d,
        "rs_slope": i.rs_slope, "rs_composite": i.rs_composite,
        "rs_5d_vs_parent": i.rs_5d_vs_parent, "rs_20d_vs_parent": i.rs_20d_vs_parent,
        "rs_60d_vs_parent": i.rs_60d_vs_parent, "rs_slope_vs_parent": i.rs_slope_vs_parent,
        "rs_composite_vs_parent": i.rs_composite_vs_parent,
        "industry_composite": i.industry_composite,
        "rs_rank": i.rs_rank, "rs_rank_change": i.rs_rank_change,
        "rs_rank_within_sector": i.rs_rank_within_sector,
    }


def _reversal_to_dict(r: ReversalScoreReading) -> dict:
    return {
        "ticker": r.ticker, "name": r.name,
        "breadth_det_pillar": r.breadth_det_pillar,
        "price_break_pillar": r.price_break_pillar,
        "crowding_pillar": r.crowding_pillar,
        "reversal_score": r.reversal_score,
        "sub_signals": {k: float(v) for k, v in r.sub_signals.items()},
        "reversal_percentile": r.reversal_percentile,
        "above_75th": bool(r.above_75th),
    }


def _pump_map_row_to_dict(r: PumpMapRow) -> dict:
    return {
        "ticker": r.ticker, "name": r.name,
        "group_type": r.group_type.value,
        "parent_sector": r.parent_sector, "tier": r.tier,
        "regime_state": r.regime_state.value,
        "pump_score": r.pump_score, "pump_delta": r.pump_delta,
        "pump_delta_5d_avg": r.pump_delta_5d_avg,
        "reversal_score": r.reversal_score,
        "reversal_percentile": r.reversal_percentile,
        "analysis_state": r.analysis_state.value,
        "transition_pressure": r.transition_pressure.value,
        "confidence": r.confidence,
        "rs_composite": r.rs_composite, "rs_rank": r.rs_rank,
        "rs_rank_change": r.rs_rank_change, "rs_vs_parent": r.rs_vs_parent,
    }


def _dict_to_snapshot(d: dict) -> DailySnapshot:
    """Reconstruct DailySnapshot from dict. Backward compatible with Phase 1."""
    regime = _dict_to_regime(d["regime"])
    sectors = [_dict_to_rs(s) for s in d["sectors"]]
    breadth = _dict_to_breadth(d["breadth"])
    pump_scores = [_dict_to_pump(p) for p in d["pump_scores"]]
    states = [_dict_to_state(s) for s in d["states"]]

    # Phase 2 fields (optional — backward compatible)
    industry_rs = [_dict_to_industry_rs(i) for i in d.get("industry_rs", [])]
    reversal_scores = [_dict_to_reversal(r) for r in d.get("reversal_scores", [])]
    pump_map = [_dict_to_pump_map_row(r) for r in d.get("pump_map", [])]

    return DailySnapshot(
        date=d["date"], timestamp=d["timestamp"],
        regime=regime, sectors=sectors, breadth=breadth,
        pump_scores=pump_scores, states=states,
        industry_rs=industry_rs, reversal_scores=reversal_scores, pump_map=pump_map,
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


# Backward compat: map old state names to 7-state model
_STATE_MIGRATION = {
    "Rotation/Reversal": "Distribution",
    "Rotation": "Distribution",
    "Reversal": "Distribution",
}


def _migrate_state(val: str) -> str:
    return _STATE_MIGRATION.get(val, val)


def _dict_to_state(d: dict) -> StateClassification:
    state_str = _migrate_state(d["state"])
    prior_str = _migrate_state(d["prior_state"]) if d.get("prior_state") else None
    return StateClassification(
        ticker=d["ticker"], name=d["name"],
        state=AnalysisState(state_str), confidence=d["confidence"],
        sessions_in_state=d["sessions_in_state"],
        transition_pressure=TransitionPressure(d["transition_pressure"]),
        prior_state=AnalysisState(prior_str) if prior_str else None,
        state_changed=d["state_changed"],
        explanation=d["explanation"],
    )


def _dict_to_industry_rs(d: dict) -> IndustryRSReading:
    return IndustryRSReading(
        ticker=d["ticker"], name=d["name"], parent_sector=d["parent_sector"],
        group_type=GroupType(d["group_type"]),
        rs_5d=d["rs_5d"], rs_20d=d["rs_20d"], rs_60d=d["rs_60d"],
        rs_slope=d["rs_slope"], rs_composite=d["rs_composite"],
        rs_5d_vs_parent=d["rs_5d_vs_parent"], rs_20d_vs_parent=d["rs_20d_vs_parent"],
        rs_60d_vs_parent=d["rs_60d_vs_parent"], rs_slope_vs_parent=d["rs_slope_vs_parent"],
        rs_composite_vs_parent=d["rs_composite_vs_parent"],
        industry_composite=d["industry_composite"],
        rs_rank=d["rs_rank"], rs_rank_change=d["rs_rank_change"],
        rs_rank_within_sector=d["rs_rank_within_sector"],
    )


def _dict_to_reversal(d: dict) -> ReversalScoreReading:
    return ReversalScoreReading(
        ticker=d["ticker"], name=d["name"],
        breadth_det_pillar=d["breadth_det_pillar"],
        price_break_pillar=d["price_break_pillar"],
        crowding_pillar=d["crowding_pillar"],
        reversal_score=d["reversal_score"],
        sub_signals=d.get("sub_signals", {}),
        reversal_percentile=d["reversal_percentile"],
        above_75th=d["above_75th"],
    )


def _dict_to_pump_map_row(d: dict) -> PumpMapRow:
    return PumpMapRow(
        ticker=d["ticker"], name=d["name"],
        group_type=GroupType(d["group_type"]),
        parent_sector=d.get("parent_sector"),
        tier=d.get("tier", "T1"),
        regime_state=RegimeState(d["regime_state"]),
        pump_score=d["pump_score"], pump_delta=d["pump_delta"],
        pump_delta_5d_avg=d["pump_delta_5d_avg"],
        reversal_score=d["reversal_score"],
        reversal_percentile=d["reversal_percentile"],
        analysis_state=AnalysisState(_migrate_state(d["analysis_state"])),
        transition_pressure=TransitionPressure(d["transition_pressure"]),
        confidence=d["confidence"],
        rs_composite=d["rs_composite"], rs_rank=d["rs_rank"],
        rs_rank_change=d["rs_rank_change"], rs_vs_parent=d.get("rs_vs_parent"),
    )
