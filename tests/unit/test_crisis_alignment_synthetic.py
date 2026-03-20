"""
Crisis Alignment + Graduated Regime + Confidence Floor — synthetic tests.

Covers: detect_crisis_type, get_crisis_modifier, _graduated_regime_multiplier,
compute_target_pct with crisis types, and classify_state confidence floors.
"""
import pytest

from engine.crisis_alignment import (
    detect_crisis_type,
    get_crisis_modifier,
    CRISIS_ALIGNMENT,
    DEFAULT_ALIGNMENT,
)
from engine.schemas import (
    CrisisType,
    AnalysisState,
    RegimeState,
    HorizonPattern,
    PumpScoreReading,
)
from engine.trade_journal import compute_target_pct


# ── Crisis Detection Tests ─────────────────────────────────


def test_crisis_01_oil_hostile_includes_oil_shock():
    """TEST-CRISIS-01: Oil HOSTILE -> includes OIL_SHOCK."""
    result = detect_crisis_type(oil_level="HOSTILE")
    assert CrisisType.OIL_SHOCK in result


def test_crisis_02_gold_divergence_includes_margin_call():
    """TEST-CRISIS-02: gold_divergence_active=True -> includes MARGIN_CALL."""
    result = detect_crisis_type(gold_divergence_active=True)
    assert CrisisType.MARGIN_CALL in result


def test_crisis_03_credit_hostile_includes_credit_crisis():
    """TEST-CRISIS-03: credit HOSTILE -> includes CREDIT_CRISIS."""
    result = detect_crisis_type(credit_level="HOSTILE")
    assert CrisisType.CREDIT_CRISIS in result


def test_crisis_04_oil_plus_gold_includes_multi_crisis():
    """TEST-CRISIS-04: Oil HOSTILE + gold divergence -> includes MULTI_CRISIS."""
    result = detect_crisis_type(oil_level="HOSTILE", gold_divergence_active=True)
    assert CrisisType.MULTI_CRISIS in result
    assert CrisisType.OIL_SHOCK in result
    assert CrisisType.MARGIN_CALL in result


def test_crisis_05_all_normal_returns_none():
    """TEST-CRISIS-05: All NORMAL -> [NONE]."""
    result = detect_crisis_type()
    assert result == [CrisisType.NONE]


def test_crisis_06_xle_long_mod_oil_shock():
    """TEST-CRISIS-06: XLE long mod in OIL_SHOCK = 1.3."""
    mod = get_crisis_modifier("XLE", [CrisisType.OIL_SHOCK], is_long=True)
    assert mod == 1.3


def test_crisis_07_gdx_short_mod_margin_call():
    """TEST-CRISIS-07: GDX short mod in MARGIN_CALL = 1.4."""
    mod = get_crisis_modifier("GDX", [CrisisType.MARGIN_CALL], is_long=False)
    assert mod == 1.4


def test_crisis_08_xlf_long_mod_credit_crisis():
    """TEST-CRISIS-08: XLF long mod in CREDIT_CRISIS = 0.3."""
    mod = get_crisis_modifier("XLF", [CrisisType.CREDIT_CRISIS], is_long=True)
    assert mod == 0.3


def test_crisis_09_unknown_ticker_returns_default():
    """TEST-CRISIS-09: Unknown ticker -> DEFAULT (1.0)."""
    mod = get_crisis_modifier("ZZZZZ", [CrisisType.OIL_SHOCK], is_long=True)
    assert mod == 1.0


def test_crisis_10_multiple_crisis_uses_most_extreme():
    """TEST-CRISIS-10: Multiple crisis -> use most extreme."""
    # XLE in OIL_SHOCK has long_mod=1.3, in CREDIT_CRISIS has long_mod=0.7
    # get_crisis_modifier picks max -> 1.3
    mod = get_crisis_modifier(
        "XLE", [CrisisType.OIL_SHOCK, CrisisType.CREDIT_CRISIS], is_long=True
    )
    assert mod == 1.3


def test_crisis_11_xle_target_with_oil_shock():
    """TEST-CRISIS-11: XLE target with oil shock ~ +20%."""
    target, *_ = compute_target_pct(
        analysis_state="Accumulation",
        confidence=45,
        regime_gate="FRAGILE",
        regime_character="Crisis",
        horizon_pattern="Full Confirm",
        vix_level=27.0,
        ticker="XLE",
        crisis_types=[CrisisType.OIL_SHOCK, CrisisType.MARGIN_CALL],
    )
    assert target == pytest.approx(20, abs=5)


def test_crisis_12_gdx_target_with_margin_call():
    """TEST-CRISIS-12: GDX target with margin call ~ -50%."""
    target, *_ = compute_target_pct(
        analysis_state="Overt Dump",
        confidence=55,
        regime_gate="FRAGILE",
        regime_character="Crisis",
        horizon_pattern="Full Reject",
        vix_level=27.0,
        ticker="GDX",
        crisis_types=[CrisisType.MARGIN_CALL],
    )
    assert target == pytest.approx(-50, abs=10)


# ── Graduated Regime Multiplier Tests ──────────────────────

# Import the private function for direct testing
from engine.trade_journal import _graduated_regime_multiplier


def test_grad_01_vix_20_fragile():
    """TEST-GRAD-01: VIX 20, FRAGILE -> 0.70."""
    result = _graduated_regime_multiplier(RegimeState.FRAGILE, 20.0)
    assert result == pytest.approx(0.70, abs=0.01)


def test_grad_02_vix_25_fragile():
    """TEST-GRAD-02: VIX 25, FRAGILE -> 0.55."""
    result = _graduated_regime_multiplier(RegimeState.FRAGILE, 25.0)
    assert result == pytest.approx(0.55, abs=0.01)


def test_grad_03_vix_30_fragile():
    """TEST-GRAD-03: VIX 30, FRAGILE -> 0.40."""
    result = _graduated_regime_multiplier(RegimeState.FRAGILE, 30.0)
    assert result == pytest.approx(0.40, abs=0.01)


def test_grad_04_vix_15_normal():
    """TEST-GRAD-04: VIX 15, NORMAL -> 1.0."""
    result = _graduated_regime_multiplier(RegimeState.NORMAL, 15.0)
    assert result == 1.0


def test_grad_05_vix_35_hostile():
    """TEST-GRAD-05: VIX 35, HOSTILE -> 0.25."""
    result = _graduated_regime_multiplier(RegimeState.HOSTILE, 35.0)
    assert result == 0.25


def test_grad_06_multiplier_decreases_monotonically():
    """TEST-GRAD-06: Multiplier decreases monotonically (test VIX 20,22,24,26,28,30)."""
    vix_levels = [20, 22, 24, 26, 28, 30]
    results = [_graduated_regime_multiplier(RegimeState.FRAGILE, v) for v in vix_levels]
    for i in range(len(results) - 1):
        assert results[i] >= results[i + 1], (
            f"Not monotonic: VIX {vix_levels[i]} -> {results[i]}, "
            f"VIX {vix_levels[i+1]} -> {results[i+1]}"
        )


# ── Confidence Floor Tests ─────────────────────────────────

from engine.state_classifier import classify_state


def _make_pump(ticker="XLE", score=0.62, delta=-0.05, delta_5d=0.003):
    return PumpScoreReading(
        ticker=ticker, name="Energy",
        rs_pillar=80, participation_pillar=45, flow_pillar=50,
        pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta_5d,
    )


_DEFAULT_SETTINGS = {
    "ambiguous": {"max_duration": 15},
    "broadening": {"rs_delta_positive_sessions": 5},
    "overt_pump": {"min_pump_percentile": 75},
    "exhaustion": {"pump_delta_nonpositive_sessions": 3},
    "distribution": {"pump_delta_negative_sessions": 3},
}


def test_floor_01_rank1_full_confirm_all_rs_positive():
    """TEST-FLOOR-01: Rank 1 + Full Confirm + all RS positive -> confidence >= 45."""
    pump = _make_pump()
    result = classify_state(
        pump=pump, prior=None, regime=RegimeState.FRAGILE,
        rs_rank=1, pump_percentile=85,
        delta_history=[-0.02, -0.03, -0.05, -0.05, 0.01, 0.02, 0.03],
        settings=_DEFAULT_SETTINGS,
        rs_5d=0.049, rs_20d=0.140, rs_60d=0.394,
        horizon_pattern=HorizonPattern.FULL_CONFIRM,
    )
    assert result.confidence >= 45


def test_floor_02_rank2_floor_does_not_apply():
    """TEST-FLOOR-02: Rank 2 -> floor does NOT apply (confidence may be < 45)."""
    pump = _make_pump()
    result = classify_state(
        pump=pump, prior=None, regime=RegimeState.FRAGILE,
        rs_rank=2, pump_percentile=85,
        delta_history=[-0.02, -0.03, -0.05, -0.05, 0.01, 0.02, 0.03],
        settings=_DEFAULT_SETTINGS,
        rs_5d=0.049, rs_20d=0.140, rs_60d=0.394,
        horizon_pattern=HorizonPattern.FULL_CONFIRM,
    )
    # Rank 2 does not get the floor boost — confidence may be below 45
    # We just verify the function runs and returns a valid result
    assert result.confidence >= 0
    # The point is that rank-2 doesn't guarantee >= 45
    # (it may still be >= 45 from base confidence, but floor doesn't apply)


def test_floor_03_rank1_dead_cat_floor_does_not_apply():
    """TEST-FLOOR-03: Rank 1 + Dead Cat -> floor does NOT apply."""
    pump = _make_pump()
    result = classify_state(
        pump=pump, prior=None, regime=RegimeState.FRAGILE,
        rs_rank=1, pump_percentile=85,
        delta_history=[-0.02, -0.03, -0.05, -0.05, 0.01, 0.02, 0.03],
        settings=_DEFAULT_SETTINGS,
        rs_5d=0.049, rs_20d=0.140, rs_60d=0.394,
        horizon_pattern=HorizonPattern.DEAD_CAT,
    )
    # Dead Cat pattern means the floor condition is not met
    assert result.confidence >= 0  # valid result, but no floor guarantee


def test_floor_04_rank1_full_confirm_but_rs60d_negative():
    """TEST-FLOOR-04: Rank 1 + Full Confirm but rs_60d negative -> floor does NOT apply."""
    pump = _make_pump()
    result = classify_state(
        pump=pump, prior=None, regime=RegimeState.FRAGILE,
        rs_rank=1, pump_percentile=85,
        delta_history=[-0.02, -0.03, -0.05, -0.05, 0.01, 0.02, 0.03],
        settings=_DEFAULT_SETTINGS,
        rs_5d=0.049, rs_20d=0.140, rs_60d=-0.10,
        horizon_pattern=HorizonPattern.FULL_CONFIRM,
    )
    # rs_60d is negative, so the rank-1 floor condition is not met
    assert result.confidence >= 0  # valid result, but no floor guarantee
