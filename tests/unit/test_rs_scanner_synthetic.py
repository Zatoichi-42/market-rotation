"""
RS Scanner unit tests — synthetic data only.
Tests RS computation (relative return vs SPY), ranking, slope, composite.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import RSReading
from engine.rs_scanner import compute_rs, compute_rs_all, compute_rs_readings


SECTOR_TICKERS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB"]
SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


def _make_simple_prices(spy_returns: list[float], sector_returns: dict[str, list[float]],
                        start=100.0) -> pd.DataFrame:
    """Build a price DataFrame from explicit daily return lists."""
    n = len(spy_returns)
    dates = pd.bdate_range(end="2026-03-18", periods=n)
    data = {}
    # SPY
    prices = [start]
    for r in spy_returns:
        prices.append(prices[-1] * (1 + r))
    data["SPY"] = prices[1:]
    # Sectors
    for ticker, rets in sector_returns.items():
        prices = [start]
        for r in rets:
            prices.append(prices[-1] * (1 + r))
        data[ticker] = prices[1:]
    return pd.DataFrame(data, index=dates)


class TestRSComputation:
    """Test that RS = sector return - SPY return over a window."""

    def test_outperforming_sector_positive_rs(self):
        """Sector +5%, SPY +3% over 20d → RS ≈ +2%."""
        n = 25
        # SPY: ~+0.15%/day for 20 days = ~3% total
        spy_rets = [0.0015] * n
        # XLK: ~+0.25%/day for 20 days = ~5% total
        xlk_rets = [0.0025] * n
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        rs = compute_rs(prices, "XLK", window=20)
        # Last RS value should be positive (sector outperforming)
        assert rs.iloc[-1] > 0

    def test_underperforming_sector_negative_rs(self):
        """Sector +1%, SPY +3% over 20d → RS ≈ -2%."""
        n = 25
        spy_rets = [0.0015] * n
        xlf_rets = [0.0005] * n
        prices = _make_simple_prices(spy_rets, {"XLF": xlf_rets})
        rs = compute_rs(prices, "XLF", window=20)
        assert rs.iloc[-1] < 0

    def test_flat_sector_flat_spy_zero_rs(self):
        """Both flat → RS ≈ 0."""
        n = 25
        spy_rets = [0.0] * n
        xlk_rets = [0.0] * n
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        rs = compute_rs(prices, "XLK", window=20)
        assert abs(rs.iloc[-1]) < 1e-10

    def test_all_windows_computed(self):
        """Verify 5d, 20d, 60d RS all present and reasonable."""
        n = 70
        spy_rets = [0.001] * n
        xlk_rets = [0.002] * n
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        for window in [5, 20, 60]:
            rs = compute_rs(prices, "XLK", window=window)
            assert rs.iloc[-1] > 0
            assert not np.isnan(rs.iloc[-1])

    def test_rs_with_insufficient_data(self):
        """Only 3 days of data → 5d RS should be NaN."""
        n = 3
        spy_rets = [0.001] * n
        xlk_rets = [0.002] * n
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        rs = compute_rs(prices, "XLK", window=5)
        # Not enough data for 5d window → all NaN
        assert rs.isna().all()

    def test_rs_is_relative_not_absolute(self):
        """RS must be sector return MINUS SPY return, not just sector return."""
        n = 25
        # SPY up 2%, sector up 2% → RS should be ~0, not ~2%
        spy_rets = [0.001] * n
        xlk_rets = [0.001] * n
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        rs = compute_rs(prices, "XLK", window=20)
        assert abs(rs.iloc[-1]) < 0.001  # Near zero, not near 2%

    def test_rs_returns_series_with_correct_length(self):
        """Output is a pd.Series with same index as input prices."""
        n = 30
        spy_rets = [0.001] * n
        xlk_rets = [0.002] * n
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        rs = compute_rs(prices, "XLK", window=20)
        assert isinstance(rs, pd.Series)
        assert len(rs) == n


class TestRSRanking:
    """Test ranking across all 11 sectors."""

    def _make_ranked_prices(self, n=30):
        """Create prices where sectors have clearly differentiated returns."""
        rates = {
            "XLK": 0.003, "XLC": 0.0025, "XLY": 0.002, "XLF": 0.0015,
            "XLI": 0.001, "XLV": 0.0005, "XLP": 0.0, "XLU": -0.0005,
            "XLRE": -0.001, "XLB": -0.0015, "XLE": -0.002,
        }
        spy_rets = [0.001] * n
        sector_rets = {t: [r] * n for t, r in rates.items()}
        return _make_simple_prices(spy_rets, sector_rets)

    def test_ranking_correct_order(self):
        """Feed known returns for 11 sectors → rank 1 has highest 20d RS."""
        prices = self._make_ranked_prices(n=25)
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        # XLK should be rank 1 (strongest)
        xlk = next(r for r in readings if r.ticker == "XLK")
        assert xlk.rs_rank == 1
        # XLE should be rank 11 (weakest)
        xle = next(r for r in readings if r.ticker == "XLE")
        assert xle.rs_rank == 11

    def test_rank_1_is_strongest(self):
        """Rank 1 = highest RS, not lowest."""
        prices = self._make_ranked_prices(n=25)
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        rank1 = next(r for r in readings if r.rs_rank == 1)
        rank11 = next(r for r in readings if r.rs_rank == 11)
        assert rank1.rs_20d > rank11.rs_20d

    def test_all_ranks_present(self):
        """Ranks should be exactly {1, 2, ..., 11}."""
        prices = self._make_ranked_prices(n=25)
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        ranks = sorted(r.rs_rank for r in readings)
        assert ranks == list(range(1, 12))

    def test_rank_change_on_overtake(self):
        """Sector improves rank → positive rank_change."""
        # Build 2 snapshots: day 1 rankings, day 2 rankings where XLV overtakes
        n = 30
        half = n // 2
        spy_rets = [0.001] * n
        sector_rets = {}
        # XLV weak first half, strong second half → rank improves
        sector_rets["XLV"] = [0.0] * half + [0.005] * half
        # XLK strong first half, weak second half → rank declines
        sector_rets["XLK"] = [0.005] * half + [0.0] * half
        # Others constant
        for t in SECTOR_TICKERS:
            if t not in sector_rets:
                sector_rets[t] = [0.001] * n
        prices = _make_simple_prices(spy_rets, sector_rets)

        # Get readings with prior_ranks from midpoint
        readings_early = compute_rs_readings(
            prices.iloc[:half+5], SECTOR_NAMES,
            windows=[5, 20, 60], slope_window=5,
            composite_weights={5: 0.2, 20: 0.5, 60: 0.3},
        )
        prior_ranks = {r.ticker: r.rs_rank for r in readings_early}

        readings_late = compute_rs_readings(
            prices, SECTOR_NAMES,
            windows=[5, 20, 60], slope_window=5,
            composite_weights={5: 0.2, 20: 0.5, 60: 0.3},
            prior_ranks=prior_ranks,
        )
        xlv_late = next(r for r in readings_late if r.ticker == "XLV")
        # XLV should have improved (positive rank_change)
        assert xlv_late.rs_rank_change > 0

    def test_rank_change_on_decline(self):
        """Sector declines rank → negative rank_change."""
        n = 30
        half = n // 2
        spy_rets = [0.001] * n
        sector_rets = {}
        sector_rets["XLK"] = [0.005] * half + [-0.001] * half
        for t in SECTOR_TICKERS:
            if t not in sector_rets:
                sector_rets[t] = [0.001] * n
        prices = _make_simple_prices(spy_rets, sector_rets)

        readings_early = compute_rs_readings(
            prices.iloc[:half+5], SECTOR_NAMES,
            windows=[5, 20, 60], slope_window=5,
            composite_weights={5: 0.2, 20: 0.5, 60: 0.3},
        )
        prior_ranks = {r.ticker: r.rs_rank for r in readings_early}

        readings_late = compute_rs_readings(
            prices, SECTOR_NAMES,
            windows=[5, 20, 60], slope_window=5,
            composite_weights={5: 0.2, 20: 0.5, 60: 0.3},
            prior_ranks=prior_ranks,
        )
        xlk_late = next(r for r in readings_late if r.ticker == "XLK")
        assert xlk_late.rs_rank_change < 0

    def test_tied_returns_stable_ranking(self):
        """Two sectors with identical returns → ranks are assigned stably."""
        n = 25
        spy_rets = [0.001] * n
        sector_rets = {}
        for t in SECTOR_TICKERS:
            sector_rets[t] = [0.002] * n  # All identical
        prices = _make_simple_prices(spy_rets, sector_rets)
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        ranks = sorted(r.rs_rank for r in readings)
        # All 11 ranks should still be assigned (1-11), even with ties
        assert ranks == list(range(1, 12))


class TestRSSlope:
    """Test RS slope (rate of change of 20d RS over slope_window)."""

    def test_improving_rs_positive_slope(self):
        """RS increasing over 5 sessions → slope positive."""
        # Sector accelerating: returns increase over time
        n = 30
        spy_rets = [0.001] * n
        # XLK returns increase each day
        xlk_rets = [0.001 + 0.0002 * i for i in range(n)]
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        # Fill other sectors for compute_rs_readings
        for t in SECTOR_TICKERS:
            if t != "XLK" and t not in prices.columns:
                s_rets = [0.001] * n
                p = [100.0]
                for r in s_rets:
                    p.append(p[-1] * (1 + r))
                prices[t] = p[1:]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        xlk = next(r for r in readings if r.ticker == "XLK")
        assert xlk.rs_slope > 0

    def test_declining_rs_negative_slope(self):
        """RS decreasing over 5 sessions → slope negative."""
        n = 30
        spy_rets = [0.001] * n
        # XLK returns decrease over time
        xlk_rets = [0.005 - 0.0003 * i for i in range(n)]
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        for t in SECTOR_TICKERS:
            if t != "XLK" and t not in prices.columns:
                s_rets = [0.001] * n
                p = [100.0]
                for r in s_rets:
                    p.append(p[-1] * (1 + r))
                prices[t] = p[1:]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        xlk = next(r for r in readings if r.ticker == "XLK")
        assert xlk.rs_slope < 0

    def test_flat_rs_zero_slope(self):
        """RS unchanged → slope ≈ 0."""
        n = 30
        spy_rets = [0.001] * n
        xlk_rets = [0.002] * n  # Constant excess return
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        for t in SECTOR_TICKERS:
            if t != "XLK" and t not in prices.columns:
                s_rets = [0.001] * n
                p = [100.0]
                for r in s_rets:
                    p.append(p[-1] * (1 + r))
                prices[t] = p[1:]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        xlk = next(r for r in readings if r.ticker == "XLK")
        assert abs(xlk.rs_slope) < 0.01


class TestRSComposite:
    """Test composite score weighting."""

    def test_composite_weights(self):
        """Verify 20/50/30 weighting applied correctly."""
        n = 70
        spy_rets = [0.001] * n
        # XLK: strong at all windows
        xlk_rets = [0.003] * n
        prices = _make_simple_prices(spy_rets, {"XLK": xlk_rets})
        for t in SECTOR_TICKERS:
            if t != "XLK" and t not in prices.columns:
                s_rets = [0.001] * n
                p = [100.0]
                for r in s_rets:
                    p.append(p[-1] * (1 + r))
                prices[t] = p[1:]

        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        xlk = next(r for r in readings if r.ticker == "XLK")
        # XLK is the strongest → composite should be high (near 100)
        assert xlk.rs_composite > 80

    def test_composite_range(self):
        """Composite always in 0-100 range."""
        n = 70
        spy_rets = [0.001] * n
        sector_rets = {}
        rates = [0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0, -0.0005, -0.001, -0.0015, -0.002]
        for i, t in enumerate(SECTOR_TICKERS):
            sector_rets[t] = [rates[i]] * n
        prices = _make_simple_prices(spy_rets, sector_rets)

        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        for r in readings:
            assert 0 <= r.rs_composite <= 100, f"{r.ticker} composite {r.rs_composite} out of range"

    def test_weakest_sector_low_composite(self):
        """The weakest sector should have a low composite score."""
        n = 70
        spy_rets = [0.001] * n
        sector_rets = {}
        rates = [0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0, -0.0005, -0.001, -0.0015, -0.002]
        for i, t in enumerate(SECTOR_TICKERS):
            sector_rets[t] = [rates[i]] * n
        prices = _make_simple_prices(spy_rets, sector_rets)

        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        # XLB is at index 10 → rate -0.002, the weakest
        xlb = next(r for r in readings if r.ticker == "XLB")
        assert xlb.rs_composite < 20


class TestRSAllHistory:
    """Test that compute_rs_all returns a full history DataFrame."""

    def test_returns_dataframe(self):
        """compute_rs_all returns a DataFrame with a column per sector."""
        n = 30
        spy_rets = [0.001] * n
        sector_rets = {t: [0.001 + 0.0001 * i for i in range(n)] for i_t, t in enumerate(SECTOR_TICKERS)}
        # Fix: each sector gets slightly different returns
        sector_rets = {}
        for idx, t in enumerate(SECTOR_TICKERS):
            sector_rets[t] = [0.001 + 0.0001 * idx] * n
        prices = _make_simple_prices(spy_rets, sector_rets)

        rs_df = compute_rs_all(prices, SECTOR_TICKERS, window=20)
        assert isinstance(rs_df, pd.DataFrame)
        assert set(SECTOR_TICKERS).issubset(set(rs_df.columns))

    def test_history_length_matches_input(self):
        """RS history has same length as input prices."""
        n = 30
        spy_rets = [0.001] * n
        sector_rets = {t: [0.002] * n for t in SECTOR_TICKERS}
        prices = _make_simple_prices(spy_rets, sector_rets)
        rs_df = compute_rs_all(prices, SECTOR_TICKERS, window=20)
        assert len(rs_df) == n


class TestFactoryScenarios:
    """Test RS scanner with factory-generated market data."""

    def test_normal_market_xlk_top_rank(self, normal_market):
        """In normal market factory, XLK should be near top rank."""
        prices = normal_market["prices"]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        xlk = next(r for r in readings if r.ticker == "XLK")
        # XLK has highest return rate in factory → should be top 3
        assert xlk.rs_rank <= 3

    def test_normal_market_xle_bottom_rank(self, normal_market):
        """In normal market factory, XLE should be near bottom rank."""
        prices = normal_market["prices"]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        xle = next(r for r in readings if r.ticker == "XLE")
        assert xle.rs_rank >= 9

    def test_single_sector_pump_leader_is_rank_1(self, single_sector_pump):
        """In single sector pump (XLE), the pumping sector should be rank 1."""
        prices = single_sector_pump["prices"]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        xle = next(r for r in readings if r.ticker == "XLE")
        assert xle.rs_rank == 1

    def test_identical_sectors_all_near_zero_rs(self, identical_sectors):
        """When all sectors have identical returns, RS values near zero."""
        prices = identical_sectors["prices"]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        for r in readings:
            assert abs(r.rs_20d) < 0.02, f"{r.ticker} RS 20d = {r.rs_20d}, expected near 0"

    def test_all_readings_are_rs_reading_type(self, normal_market):
        """Every element returned should be an RSReading."""
        prices = normal_market["prices"]
        readings = compute_rs_readings(prices, SECTOR_NAMES, windows=[5, 20, 60],
                                        slope_window=5, composite_weights={5: 0.2, 20: 0.5, 60: 0.3})
        assert len(readings) == 11
        for r in readings:
            assert isinstance(r, RSReading)
