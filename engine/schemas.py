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
    OVERT_DUMP = "Overt Dump"        # Deep red — active rotation out
    DISTRIBUTION = "Distribution"     # Light red — momentum fading
    AMBIGUOUS = "Ambiguous"           # No color — conflicting signals
    ACCUMULATION = "Accumulation"     # Light green — momentum building
    OVERT_PUMP = "Overt Pump"        # Deep green — strongest inflow


class TransitionPressure(Enum):
    UP = "Up"
    STABLE = "Stable"
    DOWN = "Down"
    BREAK = "Break"


class BreadthSignal(Enum):
    HEALTHY = "HEALTHY"
    NARROWING = "NARROWING"
    DIVERGING = "DIVERGING"


class GroupType(Enum):
    """Whether a group is a sector or industry."""
    SECTOR = "sector"
    INDUSTRY = "industry"


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


# ── Industry RS (Phase 2) ──────────────────────────────

@dataclass
class IndustryRSReading:
    ticker: str
    name: str
    parent_sector: str
    group_type: GroupType

    # RS vs SPY
    rs_5d: float
    rs_20d: float
    rs_60d: float
    rs_slope: float
    rs_composite: float

    # RS vs parent sector
    rs_5d_vs_parent: float
    rs_20d_vs_parent: float
    rs_60d_vs_parent: float
    rs_slope_vs_parent: float
    rs_composite_vs_parent: float

    # Combined industry composite
    industry_composite: float

    # Ranking
    rs_rank: int
    rs_rank_change: int
    rs_rank_within_sector: int


# ── Reversal Score (Phase 2) ──────────────────────────

@dataclass
class ReversalScoreReading:
    ticker: str
    name: str

    breadth_det_pillar: float
    price_break_pillar: float
    crowding_pillar: float

    reversal_score: float
    sub_signals: dict
    reversal_percentile: float
    above_75th: bool


# ── Turnover Filter (Phase 2) ─────────────────────────

@dataclass
class TurnoverCheck:
    candidate_ticker: str
    current_ticker: str
    delta_advantage: float
    persistence_sessions: int
    current_state_exempt: bool
    passes_filter: bool
    reason: str


# ── Pump Map Row (Phase 2) ────────────────────────────

@dataclass
class PumpMapRow:
    ticker: str
    name: str
    group_type: GroupType
    parent_sector: Optional[str]
    tier: str

    regime_state: RegimeState

    pump_score: float
    pump_delta: float
    pump_delta_5d_avg: float

    reversal_score: float
    reversal_percentile: float

    analysis_state: AnalysisState
    transition_pressure: TransitionPressure
    confidence: int

    rs_composite: float
    rs_rank: int
    rs_rank_change: int
    rs_vs_parent: Optional[float]


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
    # Phase 2 additions (default to empty for backward compat)
    industry_rs: list = field(default_factory=list)
    reversal_scores: list = field(default_factory=list)
    pump_map: list = field(default_factory=list)
