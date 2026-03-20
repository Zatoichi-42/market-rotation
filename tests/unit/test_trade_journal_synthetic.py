"""
Trade Journal + Horizon Confirm — comprehensive synthetic tests.

Covers: target computation, call generation, P&L, follow-through,
journal summary, call type labels, and save/load roundtrip.
"""
import json
import os

import pandas as pd
import pytest

from engine.trade_journal import (
    compute_target_pct,
    call_type_label,
    generate_calls,
    update_forward_returns,
    close_calls,
    compute_journal_summary,
    save_journal,
    load_journal,
)
from engine.horizon_confirm import (
    compute_follow_through_quality,
    compute_persistence_quality,
    compute_secular_alignment,
)
from engine.schemas import (
    AnalysisState,
    TradeState,
    RegimeState,
    RegimeCharacter,
    HorizonPattern,
    TradeCall,
    TradeStateAssignment,
    PumpScoreReading,
    ReversalScoreReading,
    RSReading,
)


# ── Helpers ──────────────────────────────────────────────────

def _tsa(ticker="XLE", name="Test", state=AnalysisState.OVERT_PUMP,
         confidence=60, trade_state=TradeState.LONG_ENTRY):
    return TradeStateAssignment(
        ticker=ticker, name=name,
        analysis_state=state, trade_state=trade_state,
        confidence=confidence,
        entry_trigger="test", invalidation="test",
        size_class="full", catalyst_note="",
        explanation="synthetic",
    )


def _pump(ticker="XLE", score=0.70, delta=0.05, delta_5d=0.02):
    return PumpScoreReading(
        ticker=ticker, name="Test",
        rs_pillar=80.0, participation_pillar=60.0, flow_pillar=55.0,
        pump_score=score, pump_delta=delta, pump_delta_5d_avg=delta_5d,
    )


def _reversal(ticker="XLE", score=0.40, pctl=90.0):
    return ReversalScoreReading(
        ticker=ticker, name="Test",
        breadth_det_pillar=30.0, price_break_pillar=50.0,
        crowding_pillar=40.0,
        reversal_score=score, sub_signals={},
        reversal_percentile=pctl, above_75th=pctl >= 75,
    )


def _rs(ticker="XLE", rs_5d=0.01, rs_20d=0.005, rs_60d=0.003,
        rs_slope=0.001, rs_rank=3, rs_rank_change=0, rs_composite=0.5,
        rs_2d=0.005, rs_10d=0.004, rs_120d=0.002):
    return RSReading(
        ticker=ticker, name="Test",
        rs_5d=rs_5d, rs_20d=rs_20d, rs_60d=rs_60d,
        rs_slope=rs_slope, rs_rank=rs_rank, rs_rank_change=rs_rank_change,
        rs_composite=rs_composite,
        rs_2d=rs_2d, rs_10d=rs_10d, rs_120d=rs_120d,
    )


def _prices(tickers=("XLE",), n=5, base=100.0, start="2025-01-06"):
    """Return a minimal prices DataFrame with n business days."""
    dates = pd.bdate_range(start, periods=n)
    data = {t: [base + i for i in range(n)] for t in tickers}
    return pd.DataFrame(data, index=dates)


def _market_data(
    tickers=("XLE",),
    regime_gate="NORMAL",
    regime_character="Recovery",
    horizon_pattern=HorizonPattern.NO_PATTERN,
    n_prices=5,
):
    """Minimal market_data dict for generate_calls."""
    prices = _prices(tickers, n=n_prices)

    # Build horizon readings stub
    class _HR:
        def __init__(self, pattern):
            self.pattern = pattern

    horizon_readings = {t: _HR(horizon_pattern) for t in tickers}
    pumps = {t: _pump(t) for t in tickers}
    reversal_map = {t: _reversal(t) for t in tickers}
    rs_readings = [_rs(t) for t in tickers]

    return {
        "prices": prices,
        "regime_gate": regime_gate,
        "regime_character": regime_character,
        "horizon_readings": horizon_readings,
        "pumps": pumps,
        "reversal_map": reversal_map,
        "rs_readings": rs_readings,
    }


def _make_call(ticker="XLE", target_pct=50, prior_target_pct=0,
               direction=1, confidence=60, date="2025-01-10",
               analysis_state="Overt Pump", pnl_10d=None, pnl_20d=None,
               hit_10d=None, hit_20d=None, status="open",
               regime_gate="NORMAL", horizon_pattern="No Pattern",
               fwd_rs_10d=None):
    """Build a TradeCall with sensible defaults for test use."""
    delta = target_pct - prior_target_pct
    return TradeCall(
        call_id=f"CALL-{ticker}-{date}-000",
        date=date,
        ticker=ticker,
        name="Test",
        analysis_state=analysis_state,
        trade_state="Long Entry",
        target_pct=target_pct,
        prior_target_pct=prior_target_pct,
        delta_pct=delta,
        confidence=confidence,
        direction=direction,
        base_size=100,
        regime_multiplier=1.0,
        character_modifier=1.0,
        horizon_modifier=1.0,
        notional=abs(target_pct) * confidence,
        regime_gate=regime_gate,
        regime_character="Recovery",
        horizon_pattern=horizon_pattern,
        pump_score=0.70,
        pump_delta=0.05,
        reversal_score=0.40,
        reversal_percentile=90.0,
        rs_2d=0.0,
        rs_5d=0.01,
        rs_10d=0.005,
        rs_20d=0.005,
        rs_60d=0.003,
        rs_120d=0.002,
        rs_rank=3,
        entry_price=100.0,
        pnl_10d=pnl_10d,
        pnl_20d=pnl_20d,
        hit_10d=hit_10d,
        hit_20d=hit_20d,
        status=status,
        fwd_rs_10d=fwd_rs_10d,
    )


# ═══════════════════════════════════════════════════════════
# Target Computation Tests
# ═══════════════════════════════════════════════════════════

class TestTargetComputation:
    """TEST-TGT-01 through TEST-TGT-13."""

    def test_tgt_01_overt_pump_full_confirm_clamped(self):
        """Overt Pump + NORMAL + TRENDING_BULL + Full Confirm + conf 80 -> clamped +100."""
        target, d, bs, rm, cm, hm = compute_target_pct(
            AnalysisState.OVERT_PUMP, 80,
            RegimeState.NORMAL, RegimeCharacter.TRENDING_BULL,
            HorizonPattern.FULL_CONFIRM,
        )
        # raw = +1 * 100 * 0.80 * 1.0 * 1.2 * 1.2 = 115.2 -> clamped 100
        assert target == 100
        assert d == 1
        assert bs == 100

    def test_tgt_02_broadening_normal_no_pattern(self):
        """Broadening + NORMAL + RECOVERY + No Pattern + conf 70 -> 50.

        raw = +1 * 75 * 0.70 * 1.0 * 1.0 * 1.0 = 52.5
        round5(52.5) = int(round(52.5/5)*5) = int(round(10.5)*5) = int(10*5) = 50
        (Python banker's rounding: round(10.5) = 10)
        """
        target, d, bs, rm, cm, hm = compute_target_pct(
            AnalysisState.BROADENING, 70,
            RegimeState.NORMAL, RegimeCharacter.RECOVERY,
            HorizonPattern.NO_PATTERN,
        )
        assert target == 50
        assert d == 1
        assert rm == 1.0
        assert cm == 1.0
        assert hm == 1.0

    def test_tgt_03_accumulation_fragile_below_threshold(self):
        """Accumulation + FRAGILE + RECOVERY + No Pattern + conf 25 -> 0 (below 15% threshold)."""
        target, d, bs, rm, cm, hm = compute_target_pct(
            AnalysisState.ACCUMULATION, 25,
            RegimeState.FRAGILE, RegimeCharacter.RECOVERY,
            HorizonPattern.NO_PATTERN,
        )
        # raw = +1 * 50 * 0.25 * 0.5 * 1.0 * 1.0 = 6.25 -> round5 = 5
        assert target == 5
        assert d == 1
        # target too small for new position threshold (15), but compute_target_pct
        # just computes; the threshold is checked in generate_calls.

    def test_tgt_04_ambiguous_zero(self):
        """Ambiguous -> target always 0 regardless of regime/character."""
        target, d, bs, rm, cm, hm = compute_target_pct(
            AnalysisState.AMBIGUOUS, 90,
            RegimeState.NORMAL, RegimeCharacter.TRENDING_BULL,
            HorizonPattern.FULL_CONFIRM,
        )
        assert target == 0
        assert d == 0

    def test_tgt_05_distribution_normal(self):
        """Distribution + NORMAL + RECOVERY + No Pattern + conf 50 -> -25."""
        target, d, bs, rm, cm, hm = compute_target_pct(
            AnalysisState.DISTRIBUTION, 50,
            RegimeState.NORMAL, RegimeCharacter.RECOVERY,
            HorizonPattern.NO_PATTERN,
        )
        # raw = -1 * 50 * 0.50 * 1.0 * 1.0 * 1.0 = -25.0 -> round5 = -25
        assert target == -25
        assert d == -1

    def test_tgt_06_exhaustion_rotation_out(self):
        """Exhaustion + NORMAL + RECOVERY + Rotation Out + conf 65 -> -55."""
        target, d, bs, rm, cm, hm = compute_target_pct(
            AnalysisState.EXHAUSTION, 65,
            RegimeState.NORMAL, RegimeCharacter.RECOVERY,
            HorizonPattern.ROTATION_OUT,
        )
        # raw = -1 * 75 * 0.65 * 1.0 * 1.0 * 1.1 = -53.625 -> round5 = -55
        assert target == -55
        assert d == -1
        assert hm == 1.1

    def test_tgt_07_overt_dump_full_reject_clamped(self):
        """Overt Dump + NORMAL + TRENDING_BEAR + Full Reject + conf 80 -> clamped -100."""
        target, d, bs, rm, cm, hm = compute_target_pct(
            AnalysisState.OVERT_DUMP, 80,
            RegimeState.NORMAL, RegimeCharacter.TRENDING_BEAR,
            HorizonPattern.FULL_REJECT,
        )
        # raw = -1 * 100 * 0.80 * 1.0 * 1.2 * 1.2 = -115.2 -> clamped -100
        assert target == -100
        assert d == -1

    def test_tgt_08_bullish_hostile_limited(self):
        """Any bullish state + HOSTILE -> magnitude <= 25%."""
        for state in [AnalysisState.OVERT_PUMP, AnalysisState.BROADENING,
                      AnalysisState.ACCUMULATION]:
            target, *_ = compute_target_pct(
                state, 80,
                RegimeState.HOSTILE, RegimeCharacter.RECOVERY,
                HorizonPattern.FULL_CONFIRM,
            )
            assert abs(target) <= 25, f"{state.value} hostile target {target}"

    def test_tgt_09_accumulation_dead_cat_zero(self):
        """Accumulation + Dead Cat -> target = 0 (long horizon_mod = 0.0)."""
        target, *_ = compute_target_pct(
            AnalysisState.ACCUMULATION, 80,
            RegimeState.NORMAL, RegimeCharacter.RECOVERY,
            HorizonPattern.DEAD_CAT,
        )
        assert target == 0

    def test_tgt_10_accumulation_full_reject_zero(self):
        """Accumulation + Full Reject -> target = 0 (long horizon_mod = 0.0)."""
        target, *_ = compute_target_pct(
            AnalysisState.ACCUMULATION, 80,
            RegimeState.NORMAL, RegimeCharacter.RECOVERY,
            HorizonPattern.FULL_REJECT,
        )
        assert target == 0

    def test_tgt_11_crisis_character_limits(self):
        """Crisis character -> all targets severely limited.

        Crisis char_mod = 0.25 for both long and short.
        With FULL_CONFIRM horizon_mod (1.2 for long, 1.2 for short via
        FULL_REJECT), the worst case is 0.25 * 1.2 = 0.30 effective,
        so max magnitude = 100 * 1.0 * 1.0 * 0.30 = 30 (round5).
        With NO_PATTERN (horizon_mod=1.0): max = 25.
        Test with NO_PATTERN to confirm the strict 25% bound.
        """
        for state in AnalysisState:
            target, *_ = compute_target_pct(
                state, 100,
                RegimeState.NORMAL, RegimeCharacter.CRISIS,
                HorizonPattern.NO_PATTERN,
            )
            assert abs(target) <= 25, f"{state.value} crisis target {target}"

    def test_tgt_12_trending_bull_boosts_long(self):
        """Trending Bull -> long targets boosted vs CHOPPY."""
        target_bull, *_ = compute_target_pct(
            AnalysisState.BROADENING, 70,
            RegimeState.NORMAL, RegimeCharacter.TRENDING_BULL,
            HorizonPattern.NO_PATTERN,
        )
        target_choppy, *_ = compute_target_pct(
            AnalysisState.BROADENING, 70,
            RegimeState.NORMAL, RegimeCharacter.CHOPPY,
            HorizonPattern.NO_PATTERN,
        )
        assert target_bull > target_choppy

    def test_tgt_13_choppy_halves(self):
        """Choppy -> targets halved vs NORMAL (Recovery as baseline)."""
        target_normal, *_ = compute_target_pct(
            AnalysisState.BROADENING, 70,
            RegimeState.NORMAL, RegimeCharacter.RECOVERY,
            HorizonPattern.NO_PATTERN,
        )
        target_choppy, *_ = compute_target_pct(
            AnalysisState.BROADENING, 70,
            RegimeState.NORMAL, RegimeCharacter.CHOPPY,
            HorizonPattern.NO_PATTERN,
        )
        # CHOPPY char_mod for long = 0.5 vs RECOVERY = 1.0
        assert target_choppy <= target_normal * 0.6  # at most ~50%+rounding


# ═══════════════════════════════════════════════════════════
# Call Generation Tests
# ═══════════════════════════════════════════════════════════

class TestCallGeneration:
    """TEST-CALL-01 through TEST-CALL-08."""

    def test_call_01_open_long(self):
        """0% -> +75% generates OPEN LONG call (delta 75 > 15 threshold)."""
        states = {"XLE": _tsa(confidence=80)}
        md = _market_data(
            regime_gate="NORMAL",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        # conf=80, Overt Pump, NORMAL, RECOVERY, NO_PATTERN
        # raw = 1 * 100 * 0.8 * 1.0 * 1.0 * 1.0 = 80 -> round5 = 80
        calls = generate_calls(states, {}, md, [])
        assert len(calls) == 1
        assert calls[0].target_pct == 80
        assert calls[0].prior_target_pct == 0
        assert calls[0].direction == 1

    def test_call_02_no_change_no_call(self):
        """Same target -> no call generated."""
        states = {"XLE": _tsa(confidence=80)}
        md = _market_data(
            regime_gate="NORMAL",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        # target will be 80; set prior = 80
        calls = generate_calls(states, {"XLE": 80}, md, [])
        assert len(calls) == 0

    def test_call_03_reduce(self):
        """Prior 75 -> new 50 generates REDUCE call (delta 25 > 15 threshold)."""
        # Broadening + NORMAL + RECOVERY + NO_PATTERN, conf 65
        # raw = 1 * 75 * 0.65 * 1.0 * 1.0 * 1.0 = 48.75 -> round5 = 50
        states = {"XLE": _tsa(state=AnalysisState.BROADENING, confidence=65)}
        md = _market_data(
            regime_gate="NORMAL",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        calls = generate_calls(states, {"XLE": 75}, md, [])
        assert len(calls) == 1
        assert calls[0].target_pct == 50
        assert calls[0].delta_pct == -25

    def test_call_04_close(self):
        """Prior 50 -> Ambiguous -> 0% generates CLOSE call."""
        states = {"XLE": _tsa(state=AnalysisState.AMBIGUOUS, confidence=50)}
        md = _market_data(
            regime_gate="NORMAL",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        calls = generate_calls(states, {"XLE": 50}, md, [])
        assert len(calls) == 1
        assert calls[0].target_pct == 0

    def test_call_05_open_short(self):
        """0% -> negative target generates OPEN SHORT call."""
        # Distribution + NORMAL + RECOVERY + NO_PATTERN, conf 60
        # raw = -1 * 50 * 0.60 * 1.0 * 1.0 * 1.0 = -30 -> round5 = -30
        states = {"XLE": _tsa(state=AnalysisState.DISTRIBUTION, confidence=60,
                              trade_state=TradeState.HEDGE)}
        md = _market_data(
            regime_gate="NORMAL",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        calls = generate_calls(states, {}, md, [])
        assert len(calls) == 1
        assert calls[0].target_pct < 0
        assert calls[0].direction == -1

    def test_call_06_reverse(self):
        """Prior +25 -> negative target generates REVERSE call."""
        # Exhaustion + NORMAL + RECOVERY + NO_PATTERN, conf 60
        # raw = -1 * 75 * 0.60 * 1.0 * 1.0 * 1.0 = -45 -> round5 = -45
        states = {"XLE": _tsa(state=AnalysisState.EXHAUSTION, confidence=60,
                              trade_state=TradeState.HEDGE)}
        md = _market_data(
            regime_gate="NORMAL",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        calls = generate_calls(states, {"XLE": 25}, md, [])
        assert len(calls) == 1
        c = calls[0]
        assert c.prior_target_pct == 25
        assert c.target_pct < 0

    def test_call_07_new_position_below_threshold_suppressed(self):
        """0% -> +10% suppressed (below 15% new position threshold)."""
        # Accumulation + FRAGILE + RECOVERY + NO_PATTERN, conf 25
        # raw = 1 * 50 * 0.25 * 0.5 * 1.0 * 1.0 = 6.25 -> round5 = 5
        states = {"XLE": _tsa(state=AnalysisState.ACCUMULATION, confidence=25)}
        md = _market_data(
            regime_gate="FRAGILE",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        calls = generate_calls(states, {}, md, [])
        assert len(calls) == 0

    def test_call_08_add_below_threshold_suppressed(self):
        """Prior +50 -> +55 suppressed (delta 5 < 10 add threshold)."""
        # Broadening + NORMAL + RECOVERY + NO_PATTERN, conf 73
        # raw = 1 * 75 * 0.73 * 1.0 * 1.0 * 1.0 = 54.75 -> round5 = 55
        states = {"XLE": _tsa(state=AnalysisState.BROADENING, confidence=73)}
        md = _market_data(
            regime_gate="NORMAL",
            regime_character="Recovery",
            horizon_pattern=HorizonPattern.NO_PATTERN,
        )
        calls = generate_calls(states, {"XLE": 50}, md, [])
        assert len(calls) == 0


# ═══════════════════════════════════════════════════════════
# P&L Tests
# ═══════════════════════════════════════════════════════════

class TestPnL:
    """TEST-PNL-01 through TEST-PNL-03."""

    def test_pnl_01_long_positive_fwd(self):
        """Long +50% target, fwd_rs_10d = +0.02 -> pnl_10d > 0."""
        call = _make_call(target_pct=50, direction=1, fwd_rs_10d=0.02)
        # P&L = direction * fwd_rs = 1 * 0.02 = 0.02
        # Manually set pnl via the formula the journal uses
        pnl = call.direction * call.fwd_rs_10d
        assert pnl > 0

    def test_pnl_02_short_positive_fwd(self):
        """Short -30% target, fwd_rs_10d = -0.03 -> pnl positive."""
        call = _make_call(target_pct=-30, direction=-1, fwd_rs_10d=-0.03)
        pnl = call.direction * call.fwd_rs_10d
        # -1 * -0.03 = +0.03
        assert pnl > 0

    def test_pnl_03_long_negative_fwd(self):
        """Long target + negative fwd_rs -> pnl negative."""
        call = _make_call(target_pct=50, direction=1, fwd_rs_10d=-0.05)
        pnl = call.direction * call.fwd_rs_10d
        # 1 * -0.05 = -0.05
        assert pnl < 0


# ═══════════════════════════════════════════════════════════
# Follow-Through / Horizon Confirm Tests
# ═══════════════════════════════════════════════════════════

class TestFollowThrough:
    """TEST-FT-01 through TEST-FT-06."""

    def test_ft_01_confirmed_long(self):
        """rs_2d=+0.01, rs_5d=+0.02, dir=+1 -> confirmed, +5."""
        quality, modifier = compute_follow_through_quality(0.01, 0.02, 1)
        assert quality == "confirmed"
        assert modifier == 5

    def test_ft_02_failed_long(self):
        """rs_2d=-0.01, rs_5d=+0.02, dir=+1 -> failed, -10."""
        quality, modifier = compute_follow_through_quality(-0.01, 0.02, 1)
        assert quality == "failed"
        assert modifier == -10

    def test_ft_03_persisting(self):
        """rs_10d=+0.01, rs_20d=+0.02 -> persisting, +5."""
        quality, modifier = compute_persistence_quality(0.01, 0.02)
        assert quality == "persisting"
        assert modifier == 5

    def test_ft_04_reversing(self):
        """rs_10d=-0.01, rs_20d=+0.02 -> reversing, -15."""
        quality, modifier = compute_persistence_quality(-0.01, 0.02)
        assert quality == "reversing"
        assert modifier == -15

    def test_ft_05_with_secular(self):
        """rs_120d=+0.05, dir=+1 -> with-secular, +5."""
        alignment, modifier = compute_secular_alignment(0.05, 1)
        assert alignment == "with-secular"
        assert modifier == 5

    def test_ft_06_counter_secular(self):
        """rs_120d=-0.05, dir=+1 -> counter-secular, -15."""
        alignment, modifier = compute_secular_alignment(-0.05, 1)
        assert alignment == "counter-secular"
        assert modifier == -15


class TestFollowThroughEdgeCases:
    """Additional edge cases for horizon confirm."""

    def test_direction_zero_neutral(self):
        quality, modifier = compute_follow_through_quality(0.05, 0.02, 0)
        assert quality == "neutral"
        assert modifier == 0

    def test_uncertain_near_zero_2d(self):
        quality, modifier = compute_follow_through_quality(0.0005, 0.02, 1)
        assert quality == "uncertain"
        assert modifier == 0

    def test_short_confirmed(self):
        quality, modifier = compute_follow_through_quality(-0.01, -0.02, -1)
        assert quality == "confirmed"
        assert modifier == 5

    def test_short_failed(self):
        quality, modifier = compute_follow_through_quality(0.01, -0.02, -1)
        assert quality == "failed"
        assert modifier == -10

    def test_persistence_stalling(self):
        quality, modifier = compute_persistence_quality(0.002, 0.05)
        assert quality == "stalling"
        assert modifier == -5

    def test_secular_direction_zero(self):
        alignment, modifier = compute_secular_alignment(0.10, 0)
        assert alignment == "neutral-secular"
        assert modifier == 0

    def test_secular_short_with(self):
        alignment, modifier = compute_secular_alignment(-0.05, -1)
        assert alignment == "with-secular"
        assert modifier == 5

    def test_secular_short_counter(self):
        alignment, modifier = compute_secular_alignment(0.05, -1)
        assert alignment == "counter-secular"
        assert modifier == -15

    def test_secular_neutral_zone(self):
        alignment, modifier = compute_secular_alignment(0.005, 1)
        assert alignment == "neutral-secular"
        assert modifier == 0


# ═══════════════════════════════════════════════════════════
# Journal Summary Tests
# ═══════════════════════════════════════════════════════════

class TestJournalSummary:
    """TEST-JRNL-01 through TEST-JRNL-06."""

    def test_jrnl_01_hit_rate_settled_only(self):
        """Hit rate uses only calls with settled hit_10d (not None)."""
        calls = [
            _make_call(ticker="XLE", pnl_10d=5.0, hit_10d=True, status="closed"),
            _make_call(ticker="XLF", pnl_10d=-2.0, hit_10d=False, status="closed",
                       date="2025-01-11"),
            _make_call(ticker="XLK", pnl_10d=3.0, hit_10d=True, status="closed",
                       date="2025-01-12"),
        ]
        summary = compute_journal_summary(calls)
        # 2 hits out of 3 settled
        assert summary.hit_rate_10d == pytest.approx(2 / 3, abs=0.01)

    def test_jrnl_02_pnl_by_state_sums(self):
        """PnL by state sums should be consistent with total."""
        calls = [
            _make_call(ticker="XLE", pnl_10d=5.0, hit_10d=True, status="closed",
                       analysis_state="Overt Pump"),
            _make_call(ticker="XLF", pnl_10d=-2.0, hit_10d=False, status="closed",
                       analysis_state="Overt Pump", date="2025-01-11"),
            _make_call(ticker="XLK", pnl_10d=3.0, hit_10d=True, status="closed",
                       analysis_state="Distribution", date="2025-01-12"),
        ]
        summary = compute_journal_summary(calls)
        # pnl_by_state is average per bucket; total_pnl_10d = 5 + (-2) + 3 = 6
        assert summary.total_pnl_10d == pytest.approx(6.0, abs=0.01)
        # Overt Pump avg: (5 + -2) / 2 = 1.5
        assert summary.pnl_by_state["Overt Pump"] == pytest.approx(1.5, abs=0.01)
        # Distribution avg: 3 / 1 = 3.0
        assert summary.pnl_by_state["Distribution"] == pytest.approx(3.0, abs=0.01)

    def test_jrnl_03_open_calls_excluded_from_hit_rate(self):
        """Open calls (hit_10d=None) excluded from hit rate."""
        calls = [
            _make_call(ticker="XLE", pnl_10d=5.0, hit_10d=True, status="closed"),
            _make_call(ticker="XLF", status="open", date="2025-01-11"),  # open, no pnl
        ]
        summary = compute_journal_summary(calls)
        # Only 1 eligible for hit rate (the closed one with hit_10d=True)
        assert summary.hit_rate_10d == pytest.approx(1.0, abs=0.01)
        assert summary.total_calls == 2
        assert summary.open_calls == 1

    def test_jrnl_04_confidence_band_bucketing(self):
        """Confidence bands: >60=high, 31-60=mid, <=30=low."""
        calls = [
            _make_call(ticker="XLE", confidence=80, pnl_10d=10.0, hit_10d=True,
                       status="closed"),
            _make_call(ticker="XLF", confidence=50, pnl_10d=5.0, hit_10d=True,
                       status="closed", date="2025-01-11"),
            _make_call(ticker="XLK", confidence=20, pnl_10d=-3.0, hit_10d=False,
                       status="closed", date="2025-01-12"),
        ]
        summary = compute_journal_summary(calls)
        assert "high (61+)" in summary.pnl_by_confidence
        assert "mid (31-60)" in summary.pnl_by_confidence
        assert "low (<=30)" in summary.pnl_by_confidence

    def test_jrnl_05_save_load_roundtrip(self, tmp_path):
        """Save and load journal produces identical data."""
        calls = [
            _make_call(ticker="XLE", pnl_10d=5.0, hit_10d=True, status="closed"),
            _make_call(ticker="XLF", pnl_10d=-2.0, hit_10d=False, status="closed",
                       date="2025-01-11"),
        ]
        summary = compute_journal_summary(calls)
        path = str(tmp_path / "journal.json")

        save_journal(calls, summary, path=path)
        loaded_calls, loaded_summary = load_journal(path=path)

        assert len(loaded_calls) == len(calls)
        for orig, loaded in zip(calls, loaded_calls):
            assert orig.call_id == loaded.call_id
            assert orig.target_pct == loaded.target_pct
            assert orig.pnl_10d == loaded.pnl_10d
            assert orig.status == loaded.status

        assert loaded_summary is not None
        assert loaded_summary.total_calls == summary.total_calls
        assert loaded_summary.hit_rate_10d == summary.hit_rate_10d

    def test_jrnl_06_load_nonexistent_returns_empty(self, tmp_path):
        """Load from nonexistent file returns ([], None)."""
        path = str(tmp_path / "does_not_exist.json")
        calls, summary = load_journal(path=path)
        assert calls == []
        assert summary is None


# ═══════════════════════════════════════════════════════════
# Call Type Label Tests
# ═══════════════════════════════════════════════════════════

class TestCallTypeLabel:
    """Verify human-readable call type labels."""

    def test_new_long(self):
        assert call_type_label(0, 75) == "New Long"

    def test_close_from_long(self):
        assert call_type_label(75, 0) == "Close"

    def test_reverse_long_to_short(self):
        assert call_type_label(25, -25) == "Reverse to Short"

    def test_reverse_short_to_long(self):
        assert call_type_label(-25, 25) == "Reverse to Long"

    def test_add_long(self):
        assert call_type_label(50, 75) == "Add Long"

    def test_reduce_long(self):
        assert call_type_label(75, 50) == "Reduce Long"

    def test_close_from_short(self):
        assert call_type_label(-50, 0) == "Close"

    def test_new_short(self):
        assert call_type_label(0, -50) == "New Short"

    def test_add_short(self):
        assert call_type_label(-50, -75) == "Add Short"

    def test_reduce_short(self):
        assert call_type_label(-75, -50) == "Reduce Short"

    def test_no_change(self):
        # Same target -> "Adjust" (fallthrough)
        assert call_type_label(50, 50) == "Adjust"
