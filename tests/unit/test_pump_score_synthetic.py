"""
Pump Score unit tests — synthetic data only.
Tests pillar weighting, score range, delta computation, 5d avg smoothing.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import PumpScoreReading
from engine.pump_score import (
    compute_pump_score,
    compute_pump_scores_all,
)


SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}
WEIGHTS = {"rs_weight": 0.40, "participation_weight": 0.30, "flow_weight": 0.30}


class TestPumpScoreComputation:

    def test_pillar_weights_sum_to_one(self):
        """0.40 + 0.30 + 0.30 = 1.0"""
        total = WEIGHTS["rs_weight"] + WEIGHTS["participation_weight"] + WEIGHTS["flow_weight"]
        assert abs(total - 1.0) < 1e-10

    def test_score_range_zero_to_one(self):
        """Pump score always in [0, 1]."""
        # Test across a range of pillar inputs
        for rs in [0, 25, 50, 75, 100]:
            for part in [0, 25, 50, 75, 100]:
                for flow in [0, 25, 50, 75, 100]:
                    score = compute_pump_score(rs, part, flow, WEIGHTS)
                    assert 0.0 <= score <= 1.0, f"score={score} for rs={rs}, part={part}, flow={flow}"

    def test_all_pillars_max_score_is_one(self):
        """All pillars at 100 → score = 1.0."""
        score = compute_pump_score(100.0, 100.0, 100.0, WEIGHTS)
        assert abs(score - 1.0) < 1e-10

    def test_all_pillars_zero_score_is_zero(self):
        """All pillars at 0 → score = 0.0."""
        score = compute_pump_score(0.0, 0.0, 0.0, WEIGHTS)
        assert abs(score) < 1e-10

    def test_strong_rs_raises_score(self):
        """High RS pillar → higher pump score than low RS pillar, other pillars equal."""
        high_rs = compute_pump_score(90.0, 50.0, 50.0, WEIGHTS)
        low_rs = compute_pump_score(10.0, 50.0, 50.0, WEIGHTS)
        assert high_rs > low_rs

    def test_rs_pillar_weight_is_40_pct(self):
        """RS pillar contributes 40% of the score."""
        # Only RS at 100, others at 0 → score = 0.40
        score = compute_pump_score(100.0, 0.0, 0.0, WEIGHTS)
        assert abs(score - 0.40) < 1e-10

    def test_participation_pillar_weight_is_30_pct(self):
        """Participation pillar contributes 30% of the score."""
        score = compute_pump_score(0.0, 100.0, 0.0, WEIGHTS)
        assert abs(score - 0.30) < 1e-10

    def test_flow_pillar_weight_is_30_pct(self):
        """Flow pillar contributes 30% of the score."""
        score = compute_pump_score(0.0, 0.0, 100.0, WEIGHTS)
        assert abs(score - 0.30) < 1e-10

    def test_midpoint_pillars(self):
        """All pillars at 50 → score = 0.50."""
        score = compute_pump_score(50.0, 50.0, 50.0, WEIGHTS)
        assert abs(score - 0.50) < 1e-10


class TestPumpDelta:

    def test_pump_delta_positive_when_improving(self):
        """Score increasing session over session → positive delta."""
        # Session 1: low pillars, Session 2: higher pillars
        scores = [0.30, 0.35, 0.42, 0.50, 0.55]
        readings = _build_readings_from_scores("XLK", scores)
        assert readings[-1].pump_delta > 0

    def test_pump_delta_negative_when_declining(self):
        """Score decreasing → negative delta."""
        scores = [0.80, 0.75, 0.68, 0.60, 0.55]
        readings = _build_readings_from_scores("XLK", scores)
        assert readings[-1].pump_delta < 0

    def test_pump_delta_zero_when_flat(self):
        """Score unchanged → delta = 0."""
        scores = [0.50, 0.50, 0.50, 0.50, 0.50]
        readings = _build_readings_from_scores("XLK", scores)
        assert abs(readings[-1].pump_delta) < 1e-10

    def test_pump_delta_5d_avg_smooths(self):
        """5d average of delta is smoother than raw delta."""
        # Volatile deltas: alternating big positive and negative
        scores = [0.50, 0.60, 0.45, 0.65, 0.40, 0.70, 0.35]
        readings = _build_readings_from_scores("XLK", scores)
        # The 5d avg should have smaller magnitude than at least some raw deltas
        raw_deltas = [readings[i].pump_delta for i in range(1, len(readings))]
        max_raw_abs = max(abs(d) for d in raw_deltas)
        assert abs(readings[-1].pump_delta_5d_avg) < max_raw_abs

    def test_first_session_delta_is_zero(self):
        """First session has no prior → delta = 0."""
        scores = [0.50]
        readings = _build_readings_from_scores("XLK", scores)
        assert readings[0].pump_delta == 0.0


class TestScoreDeltaPrinciple:
    """Core system principle: delta matters as much as level."""

    def test_high_score_negative_delta_is_exhaustion_signal(self):
        """Score = 0.85, delta = -0.03 → should flag concern."""
        scores = [0.90, 0.88, 0.85]
        readings = _build_readings_from_scores("XLK", scores)
        last = readings[-1]
        assert last.pump_score == pytest.approx(0.85)
        assert last.pump_delta < 0

    def test_mid_score_positive_delta_is_broadening_signal(self):
        """Score = 0.55, delta = +0.04 → should flag opportunity."""
        scores = [0.47, 0.51, 0.55]
        readings = _build_readings_from_scores("XLK", scores)
        last = readings[-1]
        assert last.pump_score == pytest.approx(0.55)
        assert last.pump_delta > 0

    def test_low_score_positive_delta_is_accumulation(self):
        """Score = 0.30, delta = +0.02 → early stage."""
        scores = [0.26, 0.28, 0.30]
        readings = _build_readings_from_scores("XLK", scores)
        last = readings[-1]
        assert last.pump_score == pytest.approx(0.30)
        assert last.pump_delta > 0


class TestComputeAllSectors:

    def test_returns_list_of_pump_score_readings(self):
        """compute_pump_scores_all returns list[PumpScoreReading]."""
        pillar_data = _make_pillar_data(n_sessions=5)
        results = compute_pump_scores_all(pillar_data, SECTOR_NAMES, WEIGHTS)
        assert len(results) == 11
        for r in results:
            assert isinstance(r, PumpScoreReading)

    def test_all_tickers_present(self):
        """Every sector ticker appears in output."""
        pillar_data = _make_pillar_data(n_sessions=5)
        results = compute_pump_scores_all(pillar_data, SECTOR_NAMES, WEIGHTS)
        tickers = {r.ticker for r in results}
        assert tickers == set(SECTOR_NAMES.keys())

    def test_scores_in_range(self):
        """All scores in [0, 1]."""
        pillar_data = _make_pillar_data(n_sessions=5)
        results = compute_pump_scores_all(pillar_data, SECTOR_NAMES, WEIGHTS)
        for r in results:
            assert 0.0 <= r.pump_score <= 1.0

    def test_names_populated(self):
        """Sector names populated correctly."""
        pillar_data = _make_pillar_data(n_sessions=5)
        results = compute_pump_scores_all(pillar_data, SECTOR_NAMES, WEIGHTS)
        xlk = next(r for r in results if r.ticker == "XLK")
        assert xlk.name == "Technology"

    def test_stronger_pillars_higher_score(self):
        """Sector with higher pillars gets higher pump score."""
        pillar_data = _make_pillar_data(n_sessions=5)
        # Override XLK to be high, XLE to be low
        for s in range(5):
            pillar_data["XLK"][s] = (90.0, 80.0, 85.0)
            pillar_data["XLE"][s] = (10.0, 15.0, 12.0)
        results = compute_pump_scores_all(pillar_data, SECTOR_NAMES, WEIGHTS)
        xlk = next(r for r in results if r.ticker == "XLK")
        xle = next(r for r in results if r.ticker == "XLE")
        assert xlk.pump_score > xle.pump_score


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def _build_readings_from_scores(ticker: str, scores: list[float]) -> list[PumpScoreReading]:
    """Build a sequence of PumpScoreReadings from pre-computed scores to test delta logic."""
    from engine.pump_score import build_readings_from_score_history
    return build_readings_from_score_history(ticker, SECTOR_NAMES.get(ticker, ticker), scores)


def _make_pillar_data(n_sessions: int = 5) -> dict[str, list[tuple[float, float, float]]]:
    """
    Create pillar data: {ticker: [(rs, participation, flow), ...]} for n sessions.
    Each session has 3 pillar values (0-100).
    """
    np.random.seed(88)
    data = {}
    for ticker in SECTOR_NAMES:
        sessions = []
        for _ in range(n_sessions):
            rs = np.random.uniform(20, 80)
            part = np.random.uniform(20, 80)
            flow = np.random.uniform(20, 80)
            sessions.append((rs, part, flow))
        data[ticker] = sessions
    return data
