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
    # Extended horizons (Phase 4b — defaults for backward compat)
    rs_2d: float = 0.0
    rs_10d: float = 0.0
    rs_120d: float = 0.0


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

    # Extended horizons (Phase 4b — defaults for backward compat)
    rs_2d: float = 0.0
    rs_10d: float = 0.0
    rs_120d: float = 0.0
    rs_2d_vs_parent: float = 0.0
    rs_10d_vs_parent: float = 0.0
    rs_120d_vs_parent: float = 0.0


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


# ── Cross-Horizon Divergence Patterns (Phase 4) ────

class HorizonPattern(Enum):
    FULL_CONFIRM = "Full Confirm"       # ↑↑↑
    ROTATION_IN = "Rotation In"         # ↑↑↓
    ROTATION_OUT = "Rotation Out"       # ↓↓↑
    FULL_REJECT = "Full Reject"         # ↓↓↓
    DEAD_CAT = "Dead Cat"              # ↑↓↓
    HEALTHY_DIP = "Healthy Dip"        # ↓↑↑
    NO_PATTERN = "No Pattern"          # mixed / unclear


@dataclass
class HorizonReading:
    ticker: str
    name: str
    pattern: HorizonPattern
    rs_5d: float
    rs_20d: float
    rs_60d: float
    rs_5d_sign: str                    # "+" or "-" or "~" (near zero)
    rs_20d_sign: str
    rs_60d_sign: str
    conviction: int                    # 0-100
    description: str
    is_rotation_signal: bool           # True for ROTATION_IN or ROTATION_OUT
    is_trap: bool                      # True for DEAD_CAT
    is_entry_zone: bool                # True for HEALTHY_DIP


# ── Crisis Type (Phase 5) ─────────────────────────

class CrisisType(Enum):
    NONE = "None"
    OIL_SHOCK = "Oil Shock"
    RATE_SHOCK = "Rate Shock"
    CREDIT_CRISIS = "Credit Crisis"
    MARGIN_CALL = "Margin Call"
    GEOPOLITICAL = "Geopolitical"
    MULTI_CRISIS = "Multi-Crisis"


# ── Regime Character (Phase 4) ────────────────────

class RegimeCharacter(Enum):
    TRENDING_BULL = "Trending Bull"
    TRENDING_BEAR = "Trending Bear"
    CHOPPY = "Choppy"
    CRISIS = "Crisis"
    RECOVERY = "Recovery"
    ROTATION = "Rotation"


@dataclass
class RegimeCharacterReading:
    character: RegimeCharacter
    gate_level: RegimeState
    confidence: int
    spy_20d_return: float
    cross_sector_dispersion: float
    breadth_trend: str                 # "improving" / "stable" / "deteriorating"
    vix_trend: str                     # "declining" / "stable" / "rising"
    prior_character: Optional[RegimeCharacter]
    sessions_in_character: int
    description: str


# ── Exit Monitor (Phase 4) ────────────────────────

class ExitSignalType(Enum):
    DELTA_DECEL = "Delta Deceleration"
    REVERSAL_EROSION = "Reversal Score Erosion"
    BREADTH_NARROWING = "Breadth Narrowing"
    VOLUME_CLIMAX = "Volume Climax Without Follow-Through"
    FAILED_BREAKOUTS = "Failed Breakout Rate Rising"
    HORIZON_FLIP = "Cross-Horizon Flip"
    RELATIVE_STOP = "Relative Stop Hit"


class ExitUrgency(Enum):
    WATCH = "Watch"
    WARNING = "Warning"
    ALERT = "Alert"
    IMMEDIATE = "Immediate"


@dataclass
class ExitSignal:
    signal_type: ExitSignalType
    ticker: str
    urgency: ExitUrgency
    sessions_active: int
    value: float
    threshold: float
    description: str


@dataclass
class ExitAssessment:
    ticker: str
    signals: list[ExitSignal]
    urgency: ExitUrgency
    recommendation: str
    description: str


# ── Position Tracker (Phase 4) ────────────────────

@dataclass
class PositionEntrySnapshot:
    position_id: str
    ticker: str
    name: str
    entry_date: str
    entry_price: float
    entry_analysis_state: AnalysisState
    entry_trade_state: TradeState
    entry_regime_gate: RegimeState
    entry_regime_character: RegimeCharacter
    entry_horizon_pattern: HorizonPattern
    entry_pump_score: float
    entry_pump_delta: float
    entry_reversal_score: float
    entry_reversal_percentile: float
    entry_confidence: int
    entry_rs_5d: float
    entry_rs_20d: float
    entry_rs_60d: float
    entry_rs_rank: int
    expected_hold_sessions: int
    invalidation_condition: str


@dataclass
class PositionLiveState:
    position_id: str
    ticker: str
    current_date: str
    sessions_held: int
    current_price: float
    unrealized_pnl_pct: float
    peak_rs_20d_since_entry: float
    peak_pump_score_since_entry: float
    rs_decline_from_peak: float
    pump_decline_from_peak: float
    reversal_score_change: float
    confidence_change: int
    delta_decel_sessions: int
    exit_assessment: Optional[ExitAssessment]
    entry_horizon: HorizonPattern
    current_horizon: HorizonPattern
    horizon_changed: bool


@dataclass
class PositionExitRecord:
    position_id: str
    ticker: str
    entry_date: str
    exit_date: str
    sessions_held: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    rs_vs_spy_during_hold: float
    exit_reason: str
    exit_signals_at_close: list[str]
    entry_quality: str
    exit_quality: str
    state_was_correct: bool
    entry_pump_score: float
    exit_pump_score: float
    entry_reversal_score: float
    exit_reversal_score: float
    lesson_tags: list[str]


# ── Trade Journal (Phase 4b) ─────────────────────

@dataclass
class TradeCall:
    call_id: str                      # "CALL-{TICKER}-{DATE}-{SEQ}"
    date: str
    ticker: str
    name: str

    # What the system said
    analysis_state: str
    trade_state: str

    # Computed target (precise instruction)
    target_pct: int                   # -100 to +100, rounded to nearest 5
    prior_target_pct: int
    delta_pct: int                    # target_pct - prior_target_pct

    # Components of the computation
    confidence: int
    direction: int                    # +1, 0, -1
    base_size: int                    # 25, 50, 75, 100
    regime_multiplier: float
    character_modifier: float
    horizon_modifier: float
    notional: float                   # abs(target_pct) * confidence

    # Context
    regime_gate: str
    regime_character: str
    horizon_pattern: str
    pump_score: float
    pump_delta: float
    reversal_score: float
    reversal_percentile: float

    # RS at all horizons
    rs_2d: float
    rs_5d: float
    rs_10d: float
    rs_20d: float
    rs_60d: float
    rs_120d: float
    rs_rank: int

    entry_price: float

    # Forward returns (filled progressively)
    fwd_1d: Optional[float] = None
    fwd_2d: Optional[float] = None
    fwd_5d: Optional[float] = None
    fwd_10d: Optional[float] = None
    fwd_20d: Optional[float] = None
    fwd_60d: Optional[float] = None

    # Forward RS vs SPY
    fwd_rs_1d: Optional[float] = None
    fwd_rs_2d: Optional[float] = None
    fwd_rs_5d: Optional[float] = None
    fwd_rs_10d: Optional[float] = None
    fwd_rs_20d: Optional[float] = None
    fwd_rs_60d: Optional[float] = None

    # P&L
    pnl_1d: Optional[float] = None
    pnl_5d: Optional[float] = None
    pnl_10d: Optional[float] = None
    pnl_20d: Optional[float] = None
    pnl_60d: Optional[float] = None

    # Status
    status: str = "open"
    close_date: Optional[str] = None
    close_reason: Optional[str] = None
    hit_10d: Optional[bool] = None
    hit_20d: Optional[bool] = None

    # Pair trade
    pair_id: Optional[str] = None
    pair_leg: Optional[str] = None
    pair_counterpart: Optional[str] = None


@dataclass
class JournalSummary:
    total_calls: int
    open_calls: int
    closed_calls: int
    total_pnl_10d: float
    total_pnl_20d: float
    avg_pnl_per_call_10d: float
    avg_pnl_per_call_20d: float
    hit_rate_10d: float
    hit_rate_20d: float
    pnl_by_state: dict
    hit_rate_by_state: dict
    pnl_by_regime: dict
    hit_rate_by_regime: dict
    pnl_by_pattern: dict
    hit_rate_by_pattern: dict
    pnl_by_confidence: dict
    hit_rate_by_confidence: dict
    cumulative_pnl: list
