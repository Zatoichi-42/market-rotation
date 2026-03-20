"""
Arrow Indicator unit tests — synthetic data only.
"""
import pytest
from engine.schemas import ArrowDirection, ArrowIndicator
from engine.arrows import compute_arrow, arrow_symbol, arrow_html


class TestArrowClassification:
    """TEST-ARROW-01 through TEST-ARROW-08."""

    def test_arrow01_strong_up(self):
        """TEST-ARROW-01: delta=+0.03, delta_prior=+0.01, rank_change=+2 → ↑↑"""
        result = compute_arrow(delta=0.03, delta_prior=0.01, rank_change=2)
        assert result.direction == ArrowDirection.STRONG_UP

    def test_arrow02_up(self):
        """TEST-ARROW-02: delta=+0.02, delta_prior=+0.02, rank_change=0 → ↑"""
        result = compute_arrow(delta=0.02, delta_prior=0.02, rank_change=0)
        assert result.direction == ArrowDirection.UP

    def test_arrow03_slight_up(self):
        """TEST-ARROW-03: delta=+0.008, delta_prior=+0.02, rank_change=0 → ↗"""
        result = compute_arrow(delta=0.008, delta_prior=0.02, rank_change=0)
        assert result.direction == ArrowDirection.SLIGHT_UP

    def test_arrow04_flat_positive(self):
        """TEST-ARROW-04: delta=+0.003 → → (flat, below threshold)"""
        result = compute_arrow(delta=0.003, delta_prior=0.003, rank_change=0)
        assert result.direction == ArrowDirection.FLAT

    def test_arrow05_flat_negative(self):
        """TEST-ARROW-05: delta=-0.003 → → (flat, below threshold)"""
        result = compute_arrow(delta=-0.003, delta_prior=-0.003, rank_change=0)
        assert result.direction == ArrowDirection.FLAT

    def test_arrow06_slight_down(self):
        """TEST-ARROW-06: delta=-0.008, delta_prior=-0.02, rank_change=0 → ↘"""
        result = compute_arrow(delta=-0.008, delta_prior=-0.02, rank_change=0)
        assert result.direction == ArrowDirection.SLIGHT_DOWN

    def test_arrow07_down(self):
        """TEST-ARROW-07: delta=-0.02, delta_prior=-0.02, rank_change=0 → ↓"""
        result = compute_arrow(delta=-0.02, delta_prior=-0.02, rank_change=0)
        assert result.direction == ArrowDirection.DOWN

    def test_arrow08_strong_down(self):
        """TEST-ARROW-08: delta=-0.03, delta_prior=-0.01, rank_change=-2 → ↓↓"""
        result = compute_arrow(delta=-0.03, delta_prior=-0.01, rank_change=-2)
        assert result.direction == ArrowDirection.STRONG_DOWN


class TestArrowProperties:
    """TEST-ARROW-09 and TEST-ARROW-10."""

    def test_arrow09_distinct_colors(self):
        """TEST-ARROW-09: All 7 arrow types have distinct color hex values"""
        arrows = [
            compute_arrow(0.03, 0.01, 2),   # ↑↑
            compute_arrow(0.02, 0.02, 0),   # ↑
            compute_arrow(0.008, 0.02, 0),  # ↗
            compute_arrow(0.001, 0.001, 0), # →
            compute_arrow(-0.008, -0.02, 0),# ↘
            compute_arrow(-0.02, -0.02, 0), # ↓
            compute_arrow(-0.03, -0.01, -2),# ↓↓
        ]
        colors = [a.color_hex for a in arrows]
        assert len(set(colors)) == 7

    def test_arrow10_returns_arrow_indicator(self):
        """TEST-ARROW-10: Arrow function returns valid ArrowIndicator dataclass"""
        result = compute_arrow(0.02, 0.01, 1)
        assert isinstance(result, ArrowIndicator)
        assert isinstance(result.direction, ArrowDirection)
        assert isinstance(result.color_hex, str)
        assert result.color_hex.startswith("#")
        assert isinstance(result.label, str)
        assert len(result.label) > 0


class TestArrowCounterTrend:
    """TEST-ARROW-11 and TEST-ARROW-12."""

    def test_arrow11_counter_trend(self):
        """TEST-ARROW-11: is_counter_trend=True is preserved"""
        result = compute_arrow(0.02, 0.01, 0, is_counter_trend=True)
        assert result.is_counter_trend == True

    def test_arrow12_with_trend(self):
        """TEST-ARROW-12: is_counter_trend=False by default"""
        result = compute_arrow(0.02, 0.01, 0)
        assert result.is_counter_trend == False


class TestArrowHelpers:
    def test_arrow_symbol(self):
        result = compute_arrow(0.02, 0.01, 0)
        sym = arrow_symbol(result)
        assert sym in ("↑↑", "↑", "↗", "→", "↘", "↓", "↓↓")

    def test_arrow_html(self):
        result = compute_arrow(0.02, 0.01, 0)
        html = arrow_html(result)
        assert "<span" in html
        assert result.color_hex in html

    def test_all_directions_have_symbols(self):
        for d in ArrowDirection:
            assert len(d.value) >= 1
