"""
Gold/Silver Ratio unit tests — synthetic data only.
Tests ALL classification paths, margin-call amplification, and regime modifier logic.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import SignalLevel, RegimeState, GoldSilverRatioReading
from engine.gold_silver_ratio import compute_gold_silver_ratio, apply_gold_silver_modifier


def _make_prices(n=600, gld_base=180.0, slv_base=22.0, gld_trend=0.0, slv_trend=0.0):
    """Generate synthetic GLD/SLV price series."""
    dates = pd.bdate_range(end="2026-03-19", periods=n)
    gld = pd.Series(
        gld_base + np.cumsum(np.random.randn(n) * 0.5 + gld_trend),
        index=dates, name="GLD"
    )
    slv = pd.Series(
        slv_base + np.cumsum(np.random.randn(n) * 0.3 + slv_trend),
        index=dates, name="SLV"
    )
    return gld, slv


def _make_ratio_at_zscore(target_z, n=600):
    """Create prices that produce approximately the target z-score for the ratio."""
    np.random.seed(123)
    dates = pd.bdate_range(end="2026-03-19", periods=n)
    # Base ratio with known mean and std
    ratio_mean = 8.0
    ratio_std = 0.5
    # Create stable history with realistic variance
    ratios = ratio_mean + np.random.randn(n) * ratio_std
    # Set last value to produce the target z-score relative to this history
    hist_mean = ratios[:-1].mean()
    hist_std = ratios[:-1].std()
    ratios[-1] = hist_mean + target_z * hist_std
    # Derive GLD = ratio * SLV
    slv = pd.Series(np.full(n, 22.0), index=dates, name="SLV")
    gld = pd.Series(ratios * slv.values, index=dates, name="GLD")
    return gld, slv


class TestGoldSilverRatioClassification:
    """TEST-GSR-01 through TEST-GSR-05: Z-score classification."""

    def test_gsr01_zscore_05_normal(self):
        """TEST-GSR-01: Ratio z-score = 0.5 → NORMAL"""
        gld, slv = _make_ratio_at_zscore(0.5)
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert reading.level == SignalLevel.NORMAL

    def test_gsr02_zscore_10_fragile(self):
        """TEST-GSR-02: Ratio z-score = 1.0 → FRAGILE (exact boundary, worse bucket)"""
        gld, slv = _make_ratio_at_zscore(1.0)
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert reading.level == SignalLevel.FRAGILE

    def test_gsr03_zscore_15_fragile(self):
        """TEST-GSR-03: Ratio z-score = 1.5 → FRAGILE"""
        gld, slv = _make_ratio_at_zscore(1.5)
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert reading.level == SignalLevel.FRAGILE

    def test_gsr04_zscore_20_hostile(self):
        """TEST-GSR-04: Ratio z-score = 2.0 → HOSTILE (exact boundary, worse bucket)"""
        gld, slv = _make_ratio_at_zscore(2.0)
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert reading.level == SignalLevel.HOSTILE

    def test_gsr05_zscore_25_hostile(self):
        """TEST-GSR-05: Ratio z-score = 2.5 → HOSTILE"""
        gld, slv = _make_ratio_at_zscore(2.5)
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert reading.level == SignalLevel.HOSTILE


class TestSilverUnderperformance:
    """TEST-GSR-06 and TEST-GSR-07: Silver underperformance detection."""

    def test_gsr06_silver_underperforming(self):
        """TEST-GSR-06: Silver 5d return < gold 5d return → silver_underperforming = True"""
        dates = pd.bdate_range(end="2026-03-19", periods=100)
        # Gold flat, silver declining over last 5 days
        gld = pd.Series(np.full(100, 180.0), index=dates)
        gld.iloc[-5:] = [180, 179, 178, 179, 179]  # flat-ish
        slv = pd.Series(np.full(100, 22.0), index=dates)
        slv.iloc[-5:] = [22, 21.5, 21, 20.5, 20]   # declining
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert reading.silver_underperforming == True

    def test_gsr07_silver_not_underperforming(self):
        """TEST-GSR-07: Silver 5d return > gold 5d return → silver_underperforming = False"""
        dates = pd.bdate_range(end="2026-03-19", periods=100)
        gld = pd.Series(np.full(100, 180.0), index=dates)
        gld.iloc[-5:] = [180, 179, 178, 177, 176]   # declining
        slv = pd.Series(np.full(100, 22.0), index=dates)
        slv.iloc[-5:] = [22, 22, 22, 22.5, 23]      # rising
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert reading.silver_underperforming == False


class TestMarginCallAmplifier:
    """TEST-GSR-08 and TEST-GSR-09: Margin-call amplification logic."""

    def test_gsr08_margin_call_amplifier_active(self):
        """TEST-GSR-08: Gold/VIX triggered AND silver underperforming → margin_call_amplifier = True"""
        dates = pd.bdate_range(end="2026-03-19", periods=100)
        gld = pd.Series(np.full(100, 180.0), index=dates)
        gld.iloc[-5:] = [180, 179, 178, 179, 179]
        slv = pd.Series(np.full(100, 22.0), index=dates)
        slv.iloc[-5:] = [22, 21.5, 21, 20.5, 20]
        reading = compute_gold_silver_ratio(gld, slv, gold_vix_divergence_active=True)
        assert reading is not None
        assert reading.silver_underperforming == True
        assert reading.margin_call_amplifier == True

    def test_gsr09_no_gold_vix_no_amplifier(self):
        """TEST-GSR-09: Gold/VIX NOT triggered AND silver underperforming → margin_call_amplifier = False"""
        dates = pd.bdate_range(end="2026-03-19", periods=100)
        gld = pd.Series(np.full(100, 180.0), index=dates)
        gld.iloc[-5:] = [180, 179, 178, 179, 179]
        slv = pd.Series(np.full(100, 22.0), index=dates)
        slv.iloc[-5:] = [22, 21.5, 21, 20.5, 20]
        reading = compute_gold_silver_ratio(gld, slv, gold_vix_divergence_active=False)
        assert reading is not None
        assert reading.silver_underperforming == True
        assert reading.margin_call_amplifier == False


class TestGracefulDegradation:
    """TEST-GSR-10: Graceful degradation when data unavailable."""

    def test_gsr10_slv_none(self):
        """TEST-GSR-10: SLV data unavailable → modifier inactive, no crash"""
        gld = pd.Series([180.0] * 100, index=pd.bdate_range(end="2026-03-19", periods=100))
        reading = compute_gold_silver_ratio(gld, None)
        assert reading is None

    def test_gsr10_gld_none(self):
        """SLV data available but GLD None → modifier inactive"""
        slv = pd.Series([22.0] * 100, index=pd.bdate_range(end="2026-03-19", periods=100))
        reading = compute_gold_silver_ratio(None, slv)
        assert reading is None

    def test_gsr10_empty_series(self):
        """Empty series → modifier inactive"""
        reading = compute_gold_silver_ratio(pd.Series(dtype=float), pd.Series(dtype=float))
        assert reading is None

    def test_gsr10_single_point(self):
        """Only 1 data point → modifier inactive"""
        gld = pd.Series([180.0])
        slv = pd.Series([22.0])
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is None


class TestRegimeModifier:
    """TEST-GSR-11 through TEST-GSR-14: Modifier integration with regime gate."""

    def _make_reading(self, level, margin_call_amplifier=False, zscore=0.0):
        return GoldSilverRatioReading(
            ratio=8.0, ratio_zscore=zscore, level=level,
            gold_5d_return=-0.02, silver_5d_return=-0.05,
            silver_underperforming=True, margin_call_amplifier=margin_call_amplifier,
            description="test",
        )

    def test_gsr11_normal_to_fragile(self):
        """TEST-GSR-11: Gate NORMAL + gold/silver FRAGILE → gate becomes FRAGILE"""
        reading = self._make_reading(SignalLevel.FRAGILE, zscore=1.3)
        new_state, explanation = apply_gold_silver_modifier(
            RegimeState.NORMAL, reading
        )
        assert new_state == RegimeState.FRAGILE

    def test_gsr12_hostile_stays_hostile(self):
        """TEST-GSR-12: Gate HOSTILE + gold/silver HOSTILE → gate stays HOSTILE"""
        reading = self._make_reading(SignalLevel.HOSTILE, zscore=2.5)
        new_state, explanation = apply_gold_silver_modifier(
            RegimeState.HOSTILE, reading
        )
        assert new_state == RegimeState.HOSTILE

    def test_gsr13_dual_modifier_explanation(self):
        """TEST-GSR-13: Both modifiers triggered → explanation contains 'DUAL PRECIOUS METALS STRESS'"""
        reading = self._make_reading(SignalLevel.FRAGILE, margin_call_amplifier=True, zscore=1.5)
        new_state, explanation = apply_gold_silver_modifier(
            RegimeState.NORMAL, reading, gold_vix_divergence_active=True
        )
        assert "DUAL PRECIOUS METALS STRESS" in explanation
        assert new_state == RegimeState.FRAGILE

    def test_gsr14_industrial_slowdown_not_margin_call(self):
        """TEST-GSR-14: Only gold/silver triggered (not gold/VIX) → 'industrial slowdown signal'"""
        reading = self._make_reading(SignalLevel.FRAGILE, margin_call_amplifier=False, zscore=1.3)
        new_state, explanation = apply_gold_silver_modifier(
            RegimeState.NORMAL, reading, gold_vix_divergence_active=False
        )
        assert "industrial slowdown" in explanation.lower()
        assert "margin call" not in explanation.lower()

    def test_modifier_cannot_loosen(self):
        """Modifier with NORMAL reading does not change gate."""
        reading = self._make_reading(SignalLevel.NORMAL, zscore=0.3)
        new_state, explanation = apply_gold_silver_modifier(
            RegimeState.FRAGILE, reading
        )
        assert new_state == RegimeState.FRAGILE
        assert explanation == ""

    def test_modifier_none_reading(self):
        """None reading → no change."""
        new_state, explanation = apply_gold_silver_modifier(
            RegimeState.NORMAL, None
        )
        assert new_state == RegimeState.NORMAL
        assert explanation == ""


class TestGoldSilverReadingFields:
    """Verify the GoldSilverRatioReading dataclass fields."""

    def test_reading_has_all_fields(self):
        gld, slv = _make_prices()
        reading = compute_gold_silver_ratio(gld, slv)
        assert reading is not None
        assert isinstance(reading.ratio, float)
        assert isinstance(reading.ratio_zscore, float)
        assert isinstance(reading.level, SignalLevel)
        assert isinstance(reading.gold_5d_return, float)
        assert isinstance(reading.silver_5d_return, float)
        assert reading.silver_underperforming in (True, False)
        assert reading.margin_call_amplifier in (True, False)
        assert isinstance(reading.description, str)
        assert len(reading.description) > 0
