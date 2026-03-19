"""
Turnover Filter unit tests.
"""
import pytest
from engine.schemas import TurnoverCheck, AnalysisState
from engine.turnover_filter import check_turnover, find_rotation_candidates

SETTINGS = {
    "min_delta_advantage": 0.08,
    "min_persistence_sessions": 3,
    "exempt_states": ["Exhaustion", "Ambiguous"],
}


class TestTurnoverFilter:

    def test_clear_advantage_passes(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.10] * 5, "XLK": [-0.02] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert result.passes_filter is True
        assert result.delta_advantage > 0.08

    def test_marginal_advantage_blocked(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.05] * 5, "XLK": [0.02] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert result.passes_filter is False

    def test_short_persistence_blocked(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.01, 0.01, 0.15, 0.15], "XLK": [0.01] * 4},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        # Only 2 sessions of clear advantage
        assert result.passes_filter is False

    def test_exhaustion_exempt(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.04] * 5, "XLK": [0.01] * 5},
            current_states={"XLK": AnalysisState.EXHAUSTION},
            settings=SETTINGS,
        )
        assert result.passes_filter is True
        assert result.current_state_exempt is True

    def test_ambiguous_exempt(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.02] * 5, "XLK": [0.01] * 5},
            current_states={"XLK": AnalysisState.AMBIGUOUS},
            settings=SETTINGS,
        )
        assert result.passes_filter is True
        assert result.current_state_exempt is True

    def test_broadening_not_exempt(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.05] * 5, "XLK": [0.02] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert result.current_state_exempt is False
        assert result.passes_filter is False  # 0.03 < 0.08

    def test_exact_threshold_passes(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.10] * 3, "XLK": [0.02] * 3},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        # advantage = 0.08, persistence = 3 → exactly meets threshold
        assert result.passes_filter is True

    def test_negative_advantage_blocked(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.01] * 5, "XLK": [0.05] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert result.passes_filter is False
        assert result.delta_advantage < 0

    def test_reason_string_populated(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.10] * 5, "XLK": [0.01] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert len(result.reason) > 0

    def test_returns_turnover_check(self):
        result = check_turnover(
            "XLV", "XLK",
            pump_deltas={"XLV": [0.10] * 5, "XLK": [0.01] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert isinstance(result, TurnoverCheck)


class TestFindRotationCandidates:

    def test_finds_best_candidate(self):
        results = find_rotation_candidates(
            current_holdings=["XLK"],
            all_groups=["XLK", "XLV", "XLE"],
            pump_deltas={"XLK": [-0.01] * 5, "XLV": [0.10] * 5, "XLE": [0.12] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert len(results) >= 1
        # Best candidate sorted first
        assert results[0].candidate_ticker == "XLE"

    def test_no_candidates_when_all_marginal(self):
        results = find_rotation_candidates(
            current_holdings=["XLK"],
            all_groups=["XLK", "XLV", "XLE"],
            pump_deltas={"XLK": [0.05] * 5, "XLV": [0.06] * 5, "XLE": [0.07] * 5},
            current_states={"XLK": AnalysisState.BROADENING},
            settings=SETTINGS,
        )
        assert len(results) == 0
