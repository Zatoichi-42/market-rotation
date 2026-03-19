"""
Centralized data schemas for the Pump Rotation System.
Every dataclass used by any engine module is defined here.
This is the single source of truth for data shapes.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ──────────────────────────────────────────────

class RegimeState(Enum):
    NORMAL = "NORMAL"
    FRAGILE = "FRAGILE"
    HOSTILE = "HOSTILE"


class SignalLevel(Enum):
    NORMAL = "NORMAL"
    FRAGILE = "FRAGILE"
    HOSTILE = "HOSTILE"


class AnalysisState(Enum):
    ACCUMULATION = "Accumulation"
    BROADENING = "Broadening"
    OVERT_PUMP = "Overt Pump"
    EXHAUSTION = "Exhaustion"
    ROTATION = "Rotation/Reversal"
    AMBIGUOUS = "Ambiguous"


class TransitionPressure(Enum):
    UP = "Up"
    STABLE = "Stable"
    DOWN = "Down"
    BREAK = "Break"


class BreadthSignal(Enum):
    HEALTHY = "HEALTHY"
    NARROWING = "NARROWING"
    DIVERGING = "DIVERGING"


# ── Regime Gate ────────────────────────────────────────

@dataclass
class RegimeSignal:
    name: str
    raw_value: float
    level: SignalLevel
    description: str


@dataclass
class RegimeAssessment:
    state: RegimeState
    signals: list[RegimeSignal]
    hostile_count: int
    fragile_count: int
    normal_count: int
    timestamp: str
    explanation: str


# ── RS Scanner ─────────────────────────────────────────

@dataclass
class RSReading:
    ticker: str
    name: str
    rs_5d: float
    rs_20d: float
    rs_60d: float
    rs_slope: float
    rs_rank: int
    rs_rank_change: int
    rs_composite: float


# ── Breadth ────────────────────────────────────────────

@dataclass
class BreadthReading:
    rsp_spy_ratio: float
    rsp_spy_ratio_20d_change: float
    rsp_spy_ratio_zscore: float
    signal: BreadthSignal
    explanation: str


# ── Pump Score ─────────────────────────────────────────

@dataclass
class PumpScoreReading:
    ticker: str
    name: str
    rs_pillar: float
    participation_pillar: float
    flow_pillar: float
    pump_score: float
    pump_delta: float
    pump_delta_5d_avg: float


# ── State Classifier ───────────────────────────────────

@dataclass
class StateClassification:
    ticker: str
    name: str
    state: AnalysisState
    confidence: int
    sessions_in_state: int
    transition_pressure: TransitionPressure
    prior_state: Optional[AnalysisState]
    state_changed: bool
    explanation: str


# ── Snapshot (for replay) ──────────────────────────────

@dataclass
class DailySnapshot:
    date: str
    timestamp: str
    regime: RegimeAssessment
    sectors: list[RSReading]
    breadth: BreadthReading
    pump_scores: list[PumpScoreReading]
    states: list[StateClassification]
