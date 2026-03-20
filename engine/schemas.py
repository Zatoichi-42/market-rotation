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
    OVERT_DUMP = "Overt Dump"        # Deep red — full capital flight
    EXHAUSTION = "Exhaustion"         # Red — participation contracting
    DISTRIBUTION = "Distribution"     # Light salmon — smart money exiting quietly
    AMBIGUOUS = "Ambiguous"           # Gray — conflicting signals
    ACCUMULATION = "Accumulation"     # Light green — smart money entering quietly
    BROADENING = "Broadening"         # Green — participation expanding
    OVERT_PUMP = "Overt Pump"        # Deep green — maximum acceleration


class TradeState(Enum):
    """What the operator should DO, not what the sector IS."""
    LONG_ENTRY = "Long Entry"
    HOLD = "Hold"
    SELECTIVE_ADD = "Selective Add"
    REDUCE = "Reduce"
    PAIR_CANDIDATE = "Pair Cand."
    HEDGE = "Hedge"
    WATCHLIST = "Watchlist"
    NO_TRADE = "No Trade"


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


class CatalystCategory(Enum):
    MACRO = "Macro"
    SECTOR = "Sector"
    EARNINGS = "Earnings"


class CatalystImpact(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class CatalystAction(Enum):
    CLEAR = "Clear"
    CAUTION = "Caution"
    EMBARGO = "Embargo"
    SHOCK_PAUSE = "Shock Pause"


class ShockType(Enum):
    BROAD_SELLOFF = "Broad Selloff"
    BROAD_RALLY = "Broad Rally"
    SECTOR_DISLOCATION = "Sector Dislocation"
    CORRELATION_SPIKE = "Correlation Spike"
    NONE = "None"


class ConcentrationRegime(Enum):
    BROAD_HEALTHY = "Broad Healthy"
    CONCENTRATED_HEALTHY = "Concentrated Healthy"
    CONCENTRATED_FRAGILE = "Concentrated Fragile"
    CONCENTRATED_UNHEALTHY = "Concentrated Unhealthy"


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


# ── Catalyst Gate ─────────────────────────────────────
# (Enums defined above in Enums section)

@dataclass
class ScheduledCatalyst:
    """A known upcoming macro/sector/earnings event."""
    date: str                      # ISO date
    name: str
    category: CatalystCategory
    impact: CatalystImpact
    affected_sectors: list[str]    # ["XLE", "XOP"] or ["ALL"]
    embargo_sessions: int          # Sessions ±1 to suppress new entries


@dataclass
class CatalystShock:
    """Detected unscheduled market shock from abnormal price action."""
    date: str
    shock_type: ShockType
    magnitude: float               # Z-score of the shock vs history
    affected_tickers: list[str]
    confidence: int
    explanation: str


@dataclass
class CatalystAssessment:
    """Combined catalyst gate output."""
    action: CatalystAction
    scheduled_catalyst: Optional[str]
    shock_detected: ShockType
    shock_magnitude: float
    affected_sectors: list[str]
    confidence_modifier: int       # -30 to 0 (reduces downstream confidence)
    explanation: str
    multi_sector_count: int = 0    # How many sectors moving in same direction


# ── Concentration Monitor ────────────────────────────
# (Enum defined above in Enums section)

@dataclass
class ConcentrationReading:
    """Per-sector concentration and leader health assessment."""
    sector_ticker: str
    ew_cw_zscore: float
    leader_health: str             # "strong" | "mixed" | "deteriorating"
    leader_tickers: list[str]
    leader_avg_rs: float
    leader_dispersion: float
    regime: ConcentrationRegime
    participation_modifier: int    # -15 to +15
    explanation: str


# ── Trade State Assignment (Phase 3) ──────────────────

@dataclass
class TradeStateAssignment:
    """Dual-column output: Analysis State + Trade State."""
    ticker: str
    name: str
    analysis_state: AnalysisState
    trade_state: TradeState
    confidence: int
    entry_trigger: str
    invalidation: str
    size_class: str
    catalyst_note: str
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
    # Phase 2 additions (default to empty for backward compat)
    industry_rs: list = field(default_factory=list)
    reversal_scores: list = field(default_factory=list)
    pump_map: list = field(default_factory=list)
    trade_states: list = field(default_factory=list)
    # Gap fix additions
    catalyst: Optional[CatalystAssessment] = None
    concentration: list = field(default_factory=list)


# ── Gold/Silver Ratio Modifier ──────────────────────

@dataclass
class GoldSilverRatioReading:
    """Gold/silver ratio regime modifier output."""
    ratio: float                      # Current GLD/SLV ratio
    ratio_zscore: float               # Z-score vs 2-year history
    level: SignalLevel                # NORMAL / FRAGILE / HOSTILE
    gold_5d_return: float
    silver_5d_return: float
    silver_underperforming: bool      # True if silver falling harder than gold
    margin_call_amplifier: bool       # True if gold/VIX divergence active AND silver weaker
    description: str


# ── Arrow Indicators ────────────────────────────────

class ArrowDirection(Enum):
    STRONG_UP = "↑↑"
    UP = "↑"
    SLIGHT_UP = "↗"
    FLAT = "→"
    SLIGHT_DOWN = "↘"
    DOWN = "↓"
    STRONG_DOWN = "↓↓"


@dataclass
class ArrowIndicator:
    direction: ArrowDirection
    color_hex: str
    label: str
    is_counter_trend: bool = False


# ── Cross-Sector Correlation ────────────────────────

@dataclass
class CorrelationReading:
    avg_correlation: float
    avg_corr_zscore: float
    level: SignalLevel
    max_corr_pair: tuple
    min_corr_pair: tuple
    description: str


# ── Gold/VIX Divergence ────────────────────────────

@dataclass
class GoldDivergenceReading:
    gold_5d_return: float
    spy_5d_return: float
    vix_level: float
    is_margin_call_regime: bool
    level: SignalLevel
    description: str
