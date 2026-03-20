"""
Causal Chain unit tests — synthetic data only.
"""
import pytest
from engine.schemas import (
    RegimeState, SignalLevel, RegimeSignal, RegimeAssessment,
)
from dashboard.components.chain import generate_causal_chain


def _make_assessment(signal_levels: dict) -> RegimeAssessment:
    """Build a RegimeAssessment with given signal levels."""
    signals = []
    for name, level in signal_levels.items():
        signals.append(RegimeSignal(name=name, raw_value=0.0, level=level, description="test"))
    hostile = sum(1 for s in signals if s.level == SignalLevel.HOSTILE)
    fragile = sum(1 for s in signals if s.level == SignalLevel.FRAGILE)
    normal = sum(1 for s in signals if s.level == SignalLevel.NORMAL)
    state = RegimeState.HOSTILE if hostile >= 2 else (
        RegimeState.FRAGILE if hostile >= 1 or fragile >= 2 else RegimeState.NORMAL
    )
    return RegimeAssessment(
        state=state, signals=signals,
        hostile_count=hostile, fragile_count=fragile, normal_count=normal,
        timestamp="", explanation="",
    )


class TestCausalChainGeneration:
    """TEST-CHAIN-01 through TEST-CHAIN-07."""

    def test_chain01_oil_vix(self):
        """TEST-CHAIN-01: Oil HOSTILE + VIX FRAGILE → chain includes oil shock"""
        regime = _make_assessment({
            "oil": SignalLevel.HOSTILE,
            "vix": SignalLevel.FRAGILE,
            "breadth": SignalLevel.NORMAL,
            "credit": SignalLevel.NORMAL,
        })
        chains = generate_causal_chain(regime)
        assert len(chains) > 0
        joined = " ".join(chains).lower()
        assert "oil" in joined
        assert "energy" in joined

    def test_chain02_correlation_breadth(self):
        """TEST-CHAIN-02: Correlation HOSTILE + Breadth FRAGILE → includes correlation"""
        regime = _make_assessment({
            "correlation": SignalLevel.HOSTILE,
            "breadth": SignalLevel.FRAGILE,
            "vix": SignalLevel.NORMAL,
            "credit": SignalLevel.NORMAL,
        })
        chains = generate_causal_chain(regime)
        joined = " ".join(chains).lower()
        assert "correlation" in joined

    def test_chain03_gold_divergence(self):
        """TEST-CHAIN-03: Gold divergence triggered → includes margin call"""
        regime = _make_assessment({
            "vix": SignalLevel.FRAGILE,
            "breadth": SignalLevel.FRAGILE,
        })
        chains = generate_causal_chain(regime, gold_divergence_active=True)
        joined = " ".join(chains).lower()
        assert "margin call" in joined

    def test_chain04_all_normal_empty(self):
        """TEST-CHAIN-04: All signals NORMAL → empty chain"""
        regime = _make_assessment({
            "vix": SignalLevel.NORMAL,
            "breadth": SignalLevel.NORMAL,
            "credit": SignalLevel.NORMAL,
            "oil": SignalLevel.NORMAL,
        })
        chains = generate_causal_chain(regime)
        assert chains == []

    def test_chain05_one_fragile_empty(self):
        """TEST-CHAIN-05: Only 1 signal FRAGILE → empty chain (need 2+)"""
        regime = _make_assessment({
            "vix": SignalLevel.FRAGILE,
            "breadth": SignalLevel.NORMAL,
            "credit": SignalLevel.NORMAL,
        })
        chains = generate_causal_chain(regime)
        assert chains == []

    def test_chain06_max_length(self):
        """TEST-CHAIN-06: Chain strings never exceed 200 characters each"""
        regime = _make_assessment({
            "oil": SignalLevel.HOSTILE,
            "vix": SignalLevel.HOSTILE,
            "breadth": SignalLevel.FRAGILE,
            "credit": SignalLevel.FRAGILE,
            "correlation": SignalLevel.HOSTILE,
        })
        chains = generate_causal_chain(regime, gold_divergence_active=True)
        for chain in chains:
            assert len(chain) <= 200, f"Chain too long ({len(chain)} chars): {chain}"

    def test_chain07_uses_arrows(self):
        """TEST-CHAIN-07: Chain strings use arrow symbols"""
        regime = _make_assessment({
            "oil": SignalLevel.HOSTILE,
            "vix": SignalLevel.FRAGILE,
        })
        chains = generate_causal_chain(regime)
        joined = " ".join(chains)
        assert "↑" in joined or "↓" in joined or "→" in joined

    def test_credit_vix_chain(self):
        """Credit + VIX chain generates."""
        regime = _make_assessment({
            "credit": SignalLevel.FRAGILE,
            "vix": SignalLevel.FRAGILE,
        })
        chains = generate_causal_chain(regime)
        joined = " ".join(chains).lower()
        assert "credit" in joined or "flight" in joined

    def test_breadth_hostile_chain(self):
        """Breadth HOSTILE alone (with another elevated) generates chain."""
        regime = _make_assessment({
            "breadth": SignalLevel.HOSTILE,
            "vix": SignalLevel.FRAGILE,
        })
        chains = generate_causal_chain(regime)
        joined = " ".join(chains).lower()
        assert "breadth" in joined or "narrowing" in joined
