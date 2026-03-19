"""
Integration tests for regime gate — live yfinance data.
"""
import pytest
import yaml
import numpy as np
from engine.schemas import RegimeState, RegimeAssessment
from engine.regime_gate import classify_regime_from_data
from engine.breadth import compute_breadth
from engine.normalizer import compute_zscore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_data():
    from data.fetcher import fetch_all
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)
    config = {**settings, **universe}
    return fetch_all(config), settings


class TestRegimeLive:

    def test_regime_returns_valid_state(self, live_data):
        """Fetch live data → regime returns one of NORMAL/FRAGILE/HOSTILE."""
        data, settings = live_data
        vix = data["vix"].iloc[-1] if len(data["vix"]) > 0 else 20.0
        vix3m = data["vix3m"].iloc[-1] if len(data["vix3m"]) > 0 else 20.0

        breadth = compute_breadth(data["prices"])
        credit = data["credit"]
        hyg = credit["hyg_close"] if "hyg_close" in credit.columns else None
        lqd = credit["lqd_close"] if "lqd_close" in credit.columns else None
        if hyg is not None and lqd is not None:
            ratio = hyg / lqd
            credit_z = compute_zscore(ratio.iloc[-1], ratio.dropna()) if len(ratio.dropna()) > 2 else 0.0
        else:
            credit_z = 0.0

        result = classify_regime_from_data(
            vix_current=vix,
            vix3m_current=vix3m,
            breadth_zscore=breadth.rsp_spy_ratio_zscore if not np.isnan(breadth.rsp_spy_ratio_zscore) else 0.0,
            credit_zscore=credit_z,
            thresholds=settings["regime"],
        )
        assert result.state in (RegimeState.NORMAL, RegimeState.FRAGILE, RegimeState.HOSTILE)

    def test_all_signals_have_values(self, live_data):
        """No signal should have NaN raw_value when live data is available."""
        data, settings = live_data
        vix = data["vix"].iloc[-1] if len(data["vix"]) > 0 else 20.0
        vix3m = data["vix3m"].iloc[-1] if len(data["vix3m"]) > 0 else 20.0

        result = classify_regime_from_data(
            vix_current=vix, vix3m_current=vix3m,
            breadth_zscore=0.0, credit_zscore=0.0,
            thresholds=settings["regime"],
        )
        for sig in result.signals:
            assert not np.isnan(sig.raw_value), f"Signal {sig.name} has NaN value"

    def test_vix_is_reasonable(self, live_data):
        """VIX value between 5 and 90."""
        data, _ = live_data
        if len(data["vix"]) > 0:
            vix = data["vix"].iloc[-1]
            assert 5 < vix < 90, f"VIX={vix} seems unreasonable"

    def test_term_structure_ratio_reasonable(self, live_data):
        """VIX/VIX3M ratio between 0.5 and 2.0."""
        data, _ = live_data
        if len(data["vix"]) > 0 and len(data["vix3m"]) > 0:
            vix = data["vix"].iloc[-1]
            vix3m = data["vix3m"].iloc[-1]
            if vix3m > 0:
                ratio = vix / vix3m
                assert 0.5 < ratio < 2.0, f"Term structure ratio={ratio:.2f}"

    def test_explanation_is_nonempty(self, live_data):
        """Regime explanation string is non-empty."""
        data, settings = live_data
        vix = data["vix"].iloc[-1] if len(data["vix"]) > 0 else 20.0
        vix3m = data["vix3m"].iloc[-1] if len(data["vix3m"]) > 0 else 20.0
        result = classify_regime_from_data(
            vix_current=vix, vix3m_current=vix3m,
            breadth_zscore=0.0, credit_zscore=0.0,
            thresholds=settings["regime"],
        )
        assert len(result.explanation) > 10
