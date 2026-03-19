"""
Test that the 5-state model and color coding are correct everywhere.
This is the source of truth for state names and their colors.
"""
import pytest
from engine.schemas import AnalysisState
from dashboard.components.style_utils import STATE_COLORS, STATE_BAR_COLORS, MOMENTUM_COLORS, color_row_by_state
import pandas as pd


class TestFiveStateModel:
    """Verify the 5-state enum is exactly right."""

    def test_exactly_five_states(self):
        assert len(AnalysisState) == 5

    def test_overt_dump_exists(self):
        assert AnalysisState.OVERT_DUMP.value == "Overt Dump"

    def test_distribution_exists(self):
        assert AnalysisState.DISTRIBUTION.value == "Distribution"

    def test_ambiguous_exists(self):
        assert AnalysisState.AMBIGUOUS.value == "Ambiguous"

    def test_accumulation_exists(self):
        assert AnalysisState.ACCUMULATION.value == "Accumulation"

    def test_overt_pump_exists(self):
        assert AnalysisState.OVERT_PUMP.value == "Overt Pump"

    def test_no_broadening(self):
        """Broadening was removed in the 5-state model."""
        assert not hasattr(AnalysisState, "BROADENING")

    def test_no_exhaustion(self):
        """Exhaustion was renamed to Distribution."""
        assert not hasattr(AnalysisState, "EXHAUSTION")

    def test_no_rotation(self):
        """Rotation was renamed to Overt Dump."""
        assert not hasattr(AnalysisState, "ROTATION")


class TestStateColors:
    """Verify every state has colors defined in all color maps."""

    def test_all_states_in_state_colors(self):
        for state in AnalysisState:
            assert state.value in STATE_COLORS, f"Missing {state.value} in STATE_COLORS"

    def test_all_states_in_bar_colors(self):
        for state in AnalysisState:
            assert state.value in STATE_BAR_COLORS, f"Missing {state.value} in STATE_BAR_COLORS"

    def test_all_states_in_momentum_colors(self):
        for state in AnalysisState:
            assert state.value in MOMENTUM_COLORS, f"Missing {state.value} in MOMENTUM_COLORS"

    def test_overt_dump_is_deep_red(self):
        bg, fg = STATE_COLORS["Overt Dump"]
        assert "1d" in bg  # #7f1d1d contains "1d"

    def test_distribution_is_light_red(self):
        bg, fg = STATE_COLORS["Distribution"]
        assert bg != ""

    def test_ambiguous_has_no_color(self):
        bg, fg = STATE_COLORS["Ambiguous"]
        assert bg == "" and fg == ""

    def test_accumulation_is_light_green(self):
        bg, fg = STATE_COLORS["Accumulation"]
        assert "4e3b" in bg  # #064e3b

    def test_overt_pump_is_deep_green(self):
        bg, fg = STATE_COLORS["Overt Pump"]
        assert "2e16" in bg  # #052e16


class TestColorRowFunction:
    """Test that color_row_by_state applies the right colors."""

    def test_overt_pump_gets_colored(self):
        row = pd.Series({"Ticker": "XLE", "State": "Overt Pump"})
        styles = color_row_by_state(row)
        assert all("background-color" in s for s in styles)
        assert all("#052e16" in s for s in styles)

    def test_overt_dump_gets_colored(self):
        row = pd.Series({"Ticker": "XLB", "State": "Overt Dump"})
        styles = color_row_by_state(row)
        assert all("#7f1d1d" in s for s in styles)

    def test_ambiguous_no_color(self):
        row = pd.Series({"Ticker": "XLI", "State": "Ambiguous"})
        styles = color_row_by_state(row)
        assert all(s == "" for s in styles)

    def test_unknown_state_no_color(self):
        row = pd.Series({"Ticker": "XLI", "State": "SomethingElse"})
        styles = color_row_by_state(row)
        assert all(s == "" for s in styles)

    def test_missing_state_no_crash(self):
        row = pd.Series({"Ticker": "XLI"})
        styles = color_row_by_state(row)
        assert all(s == "" for s in styles)
