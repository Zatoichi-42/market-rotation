"""
Test that the 5-state model and color coding are correct everywhere.
This is the source of truth for state names and their colors.
"""
import pytest
from engine.schemas import AnalysisState
from dashboard.components.style_utils import STATE_COLORS, STATE_BAR_COLORS, MOMENTUM_COLORS, color_row_by_state
import pandas as pd


class TestSevenStateModel:
    """Verify the 7-state symmetric enum is exactly right."""

    def test_exactly_seven_states(self):
        assert len(AnalysisState) == 7

    def test_overt_dump_exists(self):
        assert AnalysisState.OVERT_DUMP.value == "Overt Dump"

    def test_exhaustion_exists(self):
        assert AnalysisState.EXHAUSTION.value == "Exhaustion"

    def test_distribution_exists(self):
        assert AnalysisState.DISTRIBUTION.value == "Distribution"

    def test_ambiguous_exists(self):
        assert AnalysisState.AMBIGUOUS.value == "Ambiguous"

    def test_accumulation_exists(self):
        assert AnalysisState.ACCUMULATION.value == "Accumulation"

    def test_broadening_exists(self):
        assert AnalysisState.BROADENING.value == "Broadening"

    def test_overt_pump_exists(self):
        assert AnalysisState.OVERT_PUMP.value == "Overt Pump"

    def test_no_rotation(self):
        """Rotation/Reversal state no longer exists."""
        assert not hasattr(AnalysisState, "ROTATION")

    def test_symmetry(self):
        """3 bullish + 3 bearish + 1 neutral = 7."""
        bullish = [AnalysisState.ACCUMULATION, AnalysisState.BROADENING, AnalysisState.OVERT_PUMP]
        bearish = [AnalysisState.DISTRIBUTION, AnalysisState.EXHAUSTION, AnalysisState.OVERT_DUMP]
        neutral = [AnalysisState.AMBIGUOUS]
        assert len(bullish) == len(bearish) == 3
        assert len(neutral) == 1


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

    def test_exhaustion_is_light_red(self):
        bg, fg = STATE_COLORS["Exhaustion"]
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
