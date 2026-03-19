"""
Normalizer unit tests — synthetic data only.
Tests z-score computation, percentile ranking, blended normalization, and decay.
"""
import pytest
import numpy as np
import pandas as pd
from engine.normalizer import (
    compute_zscore,
    compute_zscore_series,
    percentile_rank,
    blend_normalize,
    apply_decay,
)


class TestZScore:
    def test_zscore_of_mean_is_zero(self):
        """A value equal to the series mean should have z-score = 0."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        z = compute_zscore(30.0, series)
        assert abs(z) < 1e-10

    def test_zscore_of_one_std_above_is_one(self):
        """A value one std above the mean should have z-score ≈ 1."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        mean = series.mean()  # 30
        std = series.std()    # ~15.81
        z = compute_zscore(mean + std, series)
        assert abs(z - 1.0) < 1e-10

    def test_zscore_with_zero_std_returns_zero(self):
        """If all values are identical (std=0), z-score should be 0.0."""
        series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        z = compute_zscore(5.0, series)
        assert z == 0.0

    def test_zscore_with_zero_std_nonequal_value_returns_zero(self):
        """If std=0 and value != mean, z-score should still be 0.0 (no division by zero)."""
        series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        z = compute_zscore(10.0, series)
        assert z == 0.0

    def test_zscore_insufficient_data_returns_nan(self):
        """With fewer than 2 data points, z-score should be NaN."""
        series = pd.Series([5.0])
        z = compute_zscore(5.0, series)
        assert np.isnan(z)

    def test_zscore_empty_series_returns_nan(self):
        """Empty series → NaN."""
        series = pd.Series([], dtype=float)
        z = compute_zscore(5.0, series)
        assert np.isnan(z)

    def test_zscore_negative_value(self):
        """Z-score of a value below the mean should be negative."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        z = compute_zscore(10.0, series)
        assert z < 0

    def test_zscore_series_rolling(self):
        """compute_zscore_series returns a Series of rolling z-scores with correct window."""
        np.random.seed(100)
        values = pd.Series(np.random.normal(50, 10, 100))
        zs = compute_zscore_series(values, window=50)
        # First 49 values should be NaN (not enough history)
        assert zs.iloc[:49].isna().all()
        # Rest should be finite
        assert zs.iloc[49:].notna().all()
        # Z-scores should have mean ~0 and std ~1 over a long window
        valid = zs.dropna()
        assert abs(valid.mean()) < 0.5  # Approximate
        assert 0.5 < valid.std() < 1.5


class TestPercentileRank:
    def test_best_sector_is_100(self):
        """The highest value in a cross-section should rank at ~100."""
        values = pd.Series({"XLK": 10.0, "XLV": 5.0, "XLF": 1.0})
        pcts = percentile_rank(values)
        assert pcts["XLK"] == 100.0

    def test_worst_sector_is_near_zero(self):
        """The lowest value should rank near 0."""
        values = pd.Series({"XLK": 10.0, "XLV": 5.0, "XLF": 1.0})
        pcts = percentile_rank(values)
        # With 3 items: worst = 0 or close to it
        assert pcts["XLF"] < 50.0

    def test_all_identical_values(self):
        """If all values are the same, all percentiles should be equal."""
        values = pd.Series({"XLK": 5.0, "XLV": 5.0, "XLF": 5.0})
        pcts = percentile_rank(values)
        # All should be the same
        assert pcts["XLK"] == pcts["XLV"] == pcts["XLF"]

    def test_two_values(self):
        """With only 2 values, one should be 0 or near 0, other 100."""
        values = pd.Series({"A": 1.0, "B": 10.0})
        pcts = percentile_rank(values)
        assert pcts["B"] > pcts["A"]

    def test_output_range_0_to_100(self):
        """All percentiles should be in [0, 100]."""
        values = pd.Series({f"S{i}": float(i) for i in range(11)})
        pcts = percentile_rank(values)
        assert all(0 <= v <= 100 for v in pcts.values)

    def test_eleven_sectors_correct_ranking(self):
        """11 sectors with known ordering → rank ordering preserved."""
        values = pd.Series({
            "XLK": 11.0, "XLV": 10.0, "XLF": 9.0, "XLE": 8.0, "XLI": 7.0,
            "XLU": 6.0, "XLRE": 5.0, "XLC": 4.0, "XLY": 3.0, "XLP": 2.0, "XLB": 1.0,
        })
        pcts = percentile_rank(values)
        assert pcts["XLK"] > pcts["XLV"] > pcts["XLF"]
        assert pcts["XLP"] > pcts["XLB"]


class TestBlendNormalize:
    def test_average_of_zscore_pct_and_xsection_pct(self):
        """Blend should be (zscore_percentile + cross_section_percentile) / 2."""
        # zscore_percentile = 80, cross_section_percentile = 60 → blend = 70
        result = blend_normalize(zscore_pct=80.0, xsection_pct=60.0)
        assert abs(result - 70.0) < 1e-10

    def test_output_range_0_to_100(self):
        """Output must be clamped to [0, 100]."""
        # Even with extreme inputs
        result_high = blend_normalize(zscore_pct=120.0, xsection_pct=90.0)
        assert result_high <= 100.0
        result_low = blend_normalize(zscore_pct=-10.0, xsection_pct=5.0)
        assert result_low >= 0.0

    def test_both_zero(self):
        """Both inputs 0 → output 0."""
        assert blend_normalize(0.0, 0.0) == 0.0

    def test_both_hundred(self):
        """Both inputs 100 → output 100."""
        assert blend_normalize(100.0, 100.0) == 100.0

    def test_symmetric(self):
        """blend(a, b) == blend(b, a) — order doesn't matter."""
        assert blend_normalize(30.0, 70.0) == blend_normalize(70.0, 30.0)


class TestDecay:
    def test_no_decay_when_signal_changing(self):
        """If signal changed recently, no decay should be applied."""
        signal = 80.0
        unchanged_sessions = 0
        result = apply_decay(signal, unchanged_sessions, halflife=10, start_after=10)
        assert result == signal

    def test_no_decay_before_threshold(self):
        """If unchanged sessions < start_after, no decay."""
        signal = 80.0
        result = apply_decay(signal, unchanged_sessions=5, halflife=10, start_after=10)
        assert result == signal

    def test_decay_starts_after_threshold(self):
        """Once unchanged_sessions >= start_after, decay begins."""
        signal = 80.0
        result = apply_decay(signal, unchanged_sessions=15, halflife=10, start_after=10)
        assert result < signal

    def test_halflife_correct(self):
        """After exactly halflife sessions past start, signal should be ~50% of original."""
        signal = 100.0
        # unchanged = start_after + halflife = 10 + 10 = 20
        result = apply_decay(signal, unchanged_sessions=20, halflife=10, start_after=10)
        assert abs(result - 50.0) < 1.0  # Allow small floating point tolerance

    def test_decay_never_below_zero(self):
        """Even with very long unchanged period, signal >= 0."""
        result = apply_decay(80.0, unchanged_sessions=1000, halflife=10, start_after=10)
        assert result >= 0.0

    def test_decay_monotonically_decreasing(self):
        """More unchanged sessions → lower signal."""
        results = [apply_decay(80.0, s, halflife=10, start_after=10) for s in range(10, 50)]
        for i in range(1, len(results)):
            assert results[i] <= results[i - 1]

    def test_zero_signal_stays_zero(self):
        """Decaying a 0 signal should still be 0."""
        result = apply_decay(0.0, unchanged_sessions=20, halflife=10, start_after=10)
        assert result == 0.0
