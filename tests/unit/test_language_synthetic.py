"""
Language Module — synthetic tests for executive briefing generation.

Covers: template content, crisis descriptions, horizon warnings,
actionable call formatting, and journal P&L inclusion.
"""
import re
from types import SimpleNamespace

import pytest

from engine.language import generate_executive_briefing
from engine.schemas import (
    CrisisType,
    RegimeState,
    RegimeCharacter,
    HorizonPattern,
    AnalysisState,
    SignalLevel,
)


# ── Mock Helpers ──────────────────────────────────────────

def _mock_regime(state=RegimeState.NORMAL):
    return SimpleNamespace(
        state=state,
        signals=[],
        hostile_count=0,
        fragile_count=0,
        normal_count=5,
        timestamp="2025-01-01",
        explanation="test",
    )


def _mock_regime_char(character=RegimeCharacter.TRENDING_BULL):
    return SimpleNamespace(
        character=character,
        gate_level=RegimeState.NORMAL,
        confidence=70,
        spy_20d_return=0.02,
        cross_sector_dispersion=0.01,
        breadth_trend="stable",
        vix_trend="stable",
        prior_character=None,
        sessions_in_character=5,
        description="test",
    )


def _mock_trade_state(ticker, state=AnalysisState.ACCUMULATION, confidence=50):
    return SimpleNamespace(
        ticker=ticker,
        name=ticker,
        analysis_state=state,
        trade_state="Long Entry",
        confidence=confidence,
        entry_trigger="test",
        invalidation="test",
        size_class="full",
        catalyst_note="",
        explanation="test",
    )


def _mock_horizon(ticker, pattern=HorizonPattern.NO_PATTERN):
    return SimpleNamespace(
        ticker=ticker,
        name=ticker,
        pattern=pattern,
        rs_5d=0.01,
        rs_20d=0.005,
        rs_60d=0.003,
        rs_5d_sign="+",
        rs_20d_sign="+",
        rs_60d_sign="+",
        conviction=50,
        description="test",
        is_rotation_signal=False,
        is_trap=(pattern == HorizonPattern.DEAD_CAT),
        is_entry_zone=False,
    )


def _mock_journal_summary(pnl_10d=5.2, hit_rate_10d=0.65):
    return SimpleNamespace(
        total_calls=20,
        open_calls=5,
        closed_calls=15,
        total_pnl_10d=pnl_10d,
        total_pnl_20d=8.0,
        avg_pnl_per_call_10d=0.35,
        avg_pnl_per_call_20d=0.53,
        hit_rate_10d=hit_rate_10d,
        hit_rate_20d=0.60,
        pnl_by_state={},
        hit_rate_by_state={},
        pnl_by_regime={},
        hit_rate_by_regime={},
        pnl_by_pattern={},
        hit_rate_by_pattern={},
        pnl_by_confidence={},
        hit_rate_by_confidence={},
        cumulative_pnl=[],
    )


def _build_briefing(
    regime_state=RegimeState.NORMAL,
    character=RegimeCharacter.TRENDING_BULL,
    crisis_types=None,
    tickers_and_targets=None,
    horizon_patterns=None,
    journal_summary=None,
    gold_divergence=None,
    vix_level=20.0,
):
    """Helper to build a briefing with minimal setup."""
    if crisis_types is None:
        crisis_types = [CrisisType.NONE]
    if tickers_and_targets is None:
        tickers_and_targets = {"XLE": 20, "XLK": -20, "XLF": 0}
    if horizon_patterns is None:
        horizon_patterns = {}

    regime = _mock_regime(regime_state)
    regime_char = _mock_regime_char(character)

    trade_states = {}
    horizon_readings = {}
    for ticker, target in tickers_and_targets.items():
        if target > 0:
            state = AnalysisState.ACCUMULATION
        elif target < 0:
            state = AnalysisState.DISTRIBUTION
        else:
            state = AnalysisState.AMBIGUOUS
        trade_states[ticker] = _mock_trade_state(ticker, state=state)
        hp = horizon_patterns.get(ticker, HorizonPattern.NO_PATTERN)
        horizon_readings[ticker] = _mock_horizon(ticker, pattern=hp)

    return generate_executive_briefing(
        regime=regime,
        regime_character=regime_char,
        crisis_types=crisis_types,
        trade_states=trade_states,
        horizon_readings=horizon_readings,
        sector_targets=tickers_and_targets,
        journal_summary=journal_summary,
        gold_divergence=gold_divergence,
        oil_signal_level="NORMAL",
        vix_level=vix_level,
    )


# ── Tests ─────────────────────────────────────────────────


def test_lang_01_hostile_regime_mentions_crisis_or_protect():
    """TEST-LANG-01: HOSTILE regime -> briefing contains 'crisis' or 'protect'."""
    text = _build_briefing(regime_state=RegimeState.HOSTILE)
    lower = text.lower()
    assert "crisis" in lower or "protect" in lower


def test_lang_02_oil_shock_mentions_oil_and_energy():
    """TEST-LANG-02: Oil shock crisis -> briefing mentions 'oil' and 'energy'."""
    text = _build_briefing(crisis_types=[CrisisType.OIL_SHOCK])
    lower = text.lower()
    assert "oil" in lower
    assert "energy" in lower


def test_lang_03_margin_call_mentions_forced_selling_or_cash():
    """TEST-LANG-03: Margin call -> briefing mentions 'forced selling' or 'cash'."""
    text = _build_briefing(crisis_types=[CrisisType.MARGIN_CALL])
    lower = text.lower()
    assert "forced selling" in lower or "cash" in lower


def test_lang_04_dead_cat_mentions_trap_or_dont_buy():
    """TEST-LANG-04: Dead Cat pattern -> briefing mentions 'TRAP' or 'don't buy'."""
    text = _build_briefing(
        tickers_and_targets={"XLE": 20, "XLK": -20},
        horizon_patterns={"XLE": HorizonPattern.DEAD_CAT},
    )
    lower = text.lower()
    assert "trap" in lower or "don't buy" in lower or "dont buy" in lower


def test_lang_05_no_actionable_calls_mentions_confidence_or_avoid():
    """TEST-LANG-05: No actionable calls (all targets 0) -> briefing says 'confidence' or 'avoid'."""
    text = _build_briefing(
        tickers_and_targets={"XLE": 0, "XLK": 0, "XLF": 0}
    )
    lower = text.lower()
    assert "confidence" in lower or "avoid" in lower


def test_lang_06_no_bare_rs_without_context():
    """TEST-LANG-06: Briefing never contains bare 'RS' without 'relative' or 'performance' nearby."""
    text = _build_briefing()
    # Find all occurrences of standalone "RS" (uppercase, word boundary)
    matches = list(re.finditer(r'\bRS\b', text))
    for m in matches:
        # Check 50 chars before and after for context words
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 50)
        context = text[start:end].lower()
        assert "relative" in context or "performance" in context, (
            f"Found bare 'RS' without context: ...{text[start:end]}..."
        )


def test_lang_07_briefing_contains_required_sections():
    """TEST-LANG-07: Briefing contains 'THE SITUATION' and 'WHAT TO DO' and 'KEY RISKS'."""
    text = _build_briefing()
    assert "THE SITUATION" in text
    assert "WHAT TO DO" in text
    assert "KEY RISKS" in text


def test_lang_08_buy_calls_include_percent():
    """TEST-LANG-08: Buy calls include '%' in target."""
    text = _build_briefing(
        tickers_and_targets={"XLE": 25, "XLK": 0}
    )
    # Find the BUY section and check for %
    lines = text.split("\n")
    in_buy_section = False
    found_percent = False
    for line in lines:
        if "BUY:" in line:
            in_buy_section = True
            continue
        if in_buy_section and line.strip() and not line.startswith("  "):
            break
        if in_buy_section and "XLE" in line:
            assert "%" in line
            found_percent = True
    assert found_percent, "Buy call for XLE should include '%'"


def test_lang_09_short_calls_mention_put_spread_or_defined_risk():
    """TEST-LANG-09: Short calls mention 'put spread' or 'defined-risk'."""
    text = _build_briefing(
        tickers_and_targets={"XLE": 0, "XLK": -25}
    )
    lower = text.lower()
    assert "put spread" in lower or "defined-risk" in lower or "defined risk" in lower


def test_lang_10_journal_pnl_included():
    """TEST-LANG-10: Journal P&L included when summary provided."""
    journal = _mock_journal_summary(pnl_10d=5.2, hit_rate_10d=0.65)
    text = _build_briefing(journal_summary=journal)
    assert "5.2" in text or "+5.2" in text
    assert "65%" in text or "65" in text
