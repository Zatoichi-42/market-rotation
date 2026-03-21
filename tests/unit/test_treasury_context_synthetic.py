"""
Treasury Context unit tests — synthetic data only.
Tests classification logic, defensive vehicle selection, gate watch, and integration.
"""
import pytest
import numpy as np
import pandas as pd

from engine.schemas import (
    RegimeState,
    SignalLevel,
    RegimeSignal,
    TreasuryFit,
    TreasuryShockType,
    DefensiveVehicle,
    TreasuryContextReading,
)
from engine.treasury_context import (
    compute_sb_correlation,
    classify_treasury_fit,
    classify_shock_type,
    select_defensive_vehicle,
    compute_gate_watch,
    compute_treasury_context,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_price_series(n: int = 60, start: float = 100.0, daily_return: float = 0.0005) -> pd.Series:
    """Generate a synthetic price series with constant daily return."""
    dates = pd.bdate_range(end="2026-03-18", periods=n)
    prices = start * (1 + daily_return) ** np.arange(n)
    return pd.Series(prices, index=dates)


def _make_prices_df(
    n: int = 60,
    tlt_daily: float = 0.0005,
    shy_daily: float = 0.0001,
    spy_daily: float = 0.001,
) -> pd.DataFrame:
    """Build a DataFrame with TLT, SHY, SPY columns."""
    dates = pd.bdate_range(end="2026-03-18", periods=n)
    return pd.DataFrame({
        "TLT": 100.0 * (1 + tlt_daily) ** np.arange(n),
        "SHY": 100.0 * (1 + shy_daily) ** np.arange(n),
        "SPY": 100.0 * (1 + spy_daily) ** np.arange(n),
    }, index=dates)


def _make_regime_signal(name: str, raw_value: float, level: SignalLevel) -> RegimeSignal:
    return RegimeSignal(name=name, raw_value=raw_value, level=level, description="test")


DEFAULT_SETTINGS = {
    "regime": {
        "vix": {"normal_max": 20, "fragile_max": 30},
        "breadth": {"normal_min_zscore": 0.0, "fragile_min_zscore": -1.0},
        "credit": {"normal_min_zscore": -0.5, "fragile_min_zscore": -1.5},
        "gate": {"hostile_threshold": 2},
    }
}


# ── Treasury Fit Tests ───────────────────────────────────────

class TestTreasuryFitClassification:

    def test_supportive_negative_sb_corr_tlt_outperforming(self):
        """Negative SB correlation + TLT outperforming SHY → Supportive."""
        result = classify_treasury_fit(
            sb_corr=-0.30,
            tlt_vs_shy_20d=0.02,
            yield_10y_20d_change=-0.05,
        )
        assert result == TreasuryFit.SUPPORTIVE

    def test_adverse_positive_sb_corr_tlt_underperforming(self):
        """Positive SB correlation + TLT underperforming SHY → Adverse."""
        result = classify_treasury_fit(
            sb_corr=0.30,
            tlt_vs_shy_20d=-0.02,
            yield_10y_20d_change=0.05,
        )
        assert result == TreasuryFit.ADVERSE

    def test_mixed_neutral_sb_corr(self):
        """Neutral SB correlation → Mixed regardless of other signals."""
        result = classify_treasury_fit(
            sb_corr=0.0,
            tlt_vs_shy_20d=0.05,
            yield_10y_20d_change=-0.3,
        )
        assert result == TreasuryFit.MIXED

    def test_mixed_conflicting_signals(self):
        """Negative SB correlation but TLT underperforming and yields rising → Mixed."""
        result = classify_treasury_fit(
            sb_corr=-0.30,
            tlt_vs_shy_20d=-0.02,
            yield_10y_20d_change=0.05,
        )
        assert result == TreasuryFit.MIXED


# ── Shock Type Tests ─────────────────────────────────────────

class TestTreasuryShockTypeClassification:

    def test_growth_scare_yields_falling_tlt_up(self):
        """Yields falling sharply + TLT outperforming → Growth Scare."""
        result = classify_shock_type(
            yield_10y_20d_change=-0.35,
            tlt_vs_shy_20d=0.03,
            move_level=120.0,
        )
        assert result == TreasuryShockType.GROWTH_SCARE

    def test_inflation_shock_yields_rising_tlt_down(self):
        """Yields rising sharply + TLT underperforming → Inflation/Term Premium."""
        result = classify_shock_type(
            yield_10y_20d_change=0.35,
            tlt_vs_shy_20d=-0.03,
            move_level=130.0,
        )
        assert result == TreasuryShockType.INFLATION_TERM_PREMIUM

    def test_no_shock_yields_flat(self):
        """Yields roughly flat → No shock."""
        result = classify_shock_type(
            yield_10y_20d_change=0.05,
            tlt_vs_shy_20d=0.001,
            move_level=100.0,
        )
        assert result == TreasuryShockType.NONE


# ── Defensive Vehicle Tests ──────────────────────────────────

class TestDefensiveVehicle:

    def test_supportive_growth_scare_tlt(self):
        """Supportive + Growth Scare → TLT (max duration)."""
        result = select_defensive_vehicle(
            TreasuryFit.SUPPORTIVE, TreasuryShockType.GROWTH_SCARE,
        )
        assert result == DefensiveVehicle.TLT

    def test_supportive_no_shock_ief(self):
        """Supportive + no shock → IEF (intermediate duration)."""
        result = select_defensive_vehicle(
            TreasuryFit.SUPPORTIVE, TreasuryShockType.NONE,
        )
        assert result == DefensiveVehicle.IEF

    def test_mixed_any_shy(self):
        """Mixed fit → SHY regardless of shock type."""
        for shock in TreasuryShockType:
            result = select_defensive_vehicle(TreasuryFit.MIXED, shock)
            assert result == DefensiveVehicle.SHY, f"Failed for shock={shock}"

    def test_adverse_inflation_tip_outperforming(self):
        """Adverse + inflation shock + TIP outperforming → TIP."""
        result = select_defensive_vehicle(
            TreasuryFit.ADVERSE,
            TreasuryShockType.INFLATION_TERM_PREMIUM,
            tip_outperforming=True,
        )
        assert result == DefensiveVehicle.TIP

    def test_adverse_default_bil(self):
        """Adverse + no special conditions → BIL (cash equivalent)."""
        result = select_defensive_vehicle(
            TreasuryFit.ADVERSE, TreasuryShockType.NONE,
        )
        assert result == DefensiveVehicle.BIL


# ── Gate Watch Tests ─────────────────────────────────────────

class TestGateWatch:

    def test_gate_watch_on_when_many_near_threshold(self):
        """Gate watch fires when >50% of signals are near their next-worse threshold."""
        signals = [
            # VIX at 18: NORMAL, but within 20% of 20 threshold (gap=20, close_dist=4, 18 >= 16)
            _make_regime_signal("vix", 18.0, SignalLevel.NORMAL),
            # Breadth at 0.15: NORMAL, gap=1.0, close_dist=0.2, 0.15 <= 0.0+0.2 → near
            _make_regime_signal("breadth", 0.15, SignalLevel.NORMAL),
            # Credit at -0.3: NORMAL, gap=1.0, close_dist=0.2, -0.3 <= -0.5+0.2=-0.3 → near
            _make_regime_signal("credit", -0.3, SignalLevel.NORMAL),
        ]
        result = compute_gate_watch(signals, DEFAULT_SETTINGS)
        assert result is True

    def test_gate_watch_off_when_comfortable(self):
        """Gate watch does not fire when signals are well away from thresholds."""
        signals = [
            # VIX at 12: NORMAL, well below 16 (threshold-20%)
            _make_regime_signal("vix", 12.0, SignalLevel.NORMAL),
            # Breadth at 1.5: NORMAL, well above 0.2
            _make_regime_signal("breadth", 1.5, SignalLevel.NORMAL),
            # Credit at 0.5: NORMAL, well above -0.3
            _make_regime_signal("credit", 0.5, SignalLevel.NORMAL),
        ]
        result = compute_gate_watch(signals, DEFAULT_SETTINGS)
        assert result is False


# ── SB Correlation Tests ─────────────────────────────────────

class TestSBCorrelation:

    def test_insufficient_data_returns_zero(self):
        """Short series → 0.0."""
        tlt = pd.Series([100.0, 101.0])
        spy = pd.Series([200.0, 201.0])
        result = compute_sb_correlation(tlt, spy, window=21)
        assert result == 0.0

    def test_returns_float(self):
        """Sufficient data returns a float in [-1, 1]."""
        np.random.seed(42)
        n = 60
        dates = pd.bdate_range(end="2026-03-18", periods=n)
        tlt = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, n)), index=dates)
        spy = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, n)), index=dates)
        result = compute_sb_correlation(tlt, spy, window=21)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0


# ── Integration Tests ────────────────────────────────────────

class TestComputeTreasuryContext:

    def test_compute_treasury_context_returns_reading(self):
        """Main entry point returns a TreasuryContextReading with all fields populated."""
        prices = _make_prices_df(n=60)
        signals = [
            _make_regime_signal("vix", 15.0, SignalLevel.NORMAL),
            _make_regime_signal("breadth", 0.5, SignalLevel.NORMAL),
            _make_regime_signal("credit", 0.0, SignalLevel.NORMAL),
        ]
        result = compute_treasury_context(
            prices=prices,
            regime_signals=signals,
            regime_state=RegimeState.NORMAL,
            settings=DEFAULT_SETTINGS,
            move_level=95.0,
        )
        assert isinstance(result, TreasuryContextReading)
        assert isinstance(result.treasury_fit, TreasuryFit)
        assert isinstance(result.shock_type, TreasuryShockType)
        assert isinstance(result.defensive_vehicle, DefensiveVehicle)
        assert isinstance(result.sb_correlation, float)
        assert isinstance(result.gate_watch, bool)
        assert len(result.description) > 0
        assert result.move_level == 95.0

    def test_missing_tlt_returns_mixed_default(self):
        """DataFrame without TLT column → default Mixed/SHY reading."""
        dates = pd.bdate_range(end="2026-03-18", periods=30)
        prices = pd.DataFrame({
            "SPY": np.linspace(400, 410, 30),
            "SHY": np.linspace(80, 81, 30),
        }, index=dates)
        signals = [
            _make_regime_signal("vix", 15.0, SignalLevel.NORMAL),
        ]
        result = compute_treasury_context(
            prices=prices,
            regime_signals=signals,
            regime_state=RegimeState.NORMAL,
            settings=DEFAULT_SETTINGS,
        )
        assert result.treasury_fit == TreasuryFit.MIXED
        assert result.defensive_vehicle == DefensiveVehicle.SHY
        assert result.sb_correlation == 0.0

    def test_cash_hurdle_from_tbill(self):
        """tbill_3m value flows through to cash_hurdle."""
        prices = _make_prices_df(n=60)
        signals = [
            _make_regime_signal("vix", 15.0, SignalLevel.NORMAL),
        ]
        result = compute_treasury_context(
            prices=prices,
            regime_signals=signals,
            regime_state=RegimeState.NORMAL,
            settings=DEFAULT_SETTINGS,
            tbill_3m=4.35,
        )
        assert result.cash_hurdle == 4.35

    def test_cash_hurdle_default_zero(self):
        """No tbill_3m → cash_hurdle defaults to 0.0."""
        prices = _make_prices_df(n=60)
        signals = []
        result = compute_treasury_context(
            prices=prices,
            regime_signals=signals,
            regime_state=RegimeState.NORMAL,
            settings=DEFAULT_SETTINGS,
        )
        assert result.cash_hurdle == 0.0

    def test_yield_10y_series_used(self):
        """Passing a yield_10y series computes a non-zero 20d change."""
        prices = _make_prices_df(n=60)
        dates = pd.bdate_range(end="2026-03-18", periods=60)
        # Yields falling from 4.5% to 4.0%
        yield_10y = pd.Series(np.linspace(4.5, 4.0, 60), index=dates)
        signals = []
        result = compute_treasury_context(
            prices=prices,
            regime_signals=signals,
            regime_state=RegimeState.NORMAL,
            settings=DEFAULT_SETTINGS,
            yield_10y=yield_10y,
        )
        # 20-day change should be negative (yields fell)
        assert result.yield_10y_20d_change < 0

    def test_missing_tlt_with_tbill_preserves_cash_hurdle(self):
        """Even when TLT is missing, cash_hurdle from tbill_3m is preserved."""
        dates = pd.bdate_range(end="2026-03-18", periods=30)
        prices = pd.DataFrame({
            "SPY": np.linspace(400, 410, 30),
        }, index=dates)
        result = compute_treasury_context(
            prices=prices,
            regime_signals=[],
            regime_state=RegimeState.NORMAL,
            settings=DEFAULT_SETTINGS,
            tbill_3m=5.25,
        )
        assert result.cash_hurdle == 5.25
        assert result.treasury_fit == TreasuryFit.MIXED
