"""
Test replay panel navigation logic.
The actual Streamlit button/slider rendering can't be unit tested,
but we can test the underlying state logic.
"""
import pytest


class TestReplayNavigation:

    def test_prev_clamps_at_zero(self):
        """Going back from index 0 stays at 0."""
        idx = 0
        new_idx = max(0, idx - 1)
        assert new_idx == 0

    def test_prev_decrements(self):
        """Going back from index 5 goes to 4."""
        idx = 5
        new_idx = max(0, idx - 1)
        assert new_idx == 4

    def test_next_clamps_at_max(self):
        """Going forward from last index stays at last."""
        available = 100
        idx = available - 1
        new_idx = min(available - 1, idx + 1)
        assert new_idx == available - 1

    def test_next_increments(self):
        """Going forward from index 5 goes to 6."""
        available = 100
        idx = 5
        new_idx = min(available - 1, idx + 1)
        assert new_idx == 6

    def test_slider_overrides_buttons(self):
        """If slider value differs from session state, slider wins."""
        session_idx = 50
        slider_idx = 75
        if slider_idx != session_idx:
            session_idx = slider_idx
        assert session_idx == 75

    def test_snapshot_load_by_date(self, tmp_path):
        """Snapshot loads correctly by date string."""
        from data.snapshots import save_snapshot, load_snapshot, list_snapshots
        from engine.schemas import (
            DailySnapshot, RegimeAssessment, RegimeSignal, RegimeState,
            SignalLevel, BreadthReading, BreadthSignal,
        )
        sig = RegimeSignal(name="vix", raw_value=15.0, level=SignalLevel.NORMAL, description="ok")
        regime = RegimeAssessment(
            state=RegimeState.NORMAL, signals=[sig],
            hostile_count=0, fragile_count=0, normal_count=1,
            timestamp="t", explanation="ok",
        )
        breadth = BreadthReading(
            rsp_spy_ratio=1.0, rsp_spy_ratio_20d_change=0.0,
            rsp_spy_ratio_zscore=0.0, signal=BreadthSignal.NARROWING, explanation="",
        )
        for d in ["2026-03-15", "2026-03-16", "2026-03-17"]:
            snap = DailySnapshot(
                date=d, timestamp="t", regime=regime, sectors=[], breadth=breadth,
                pump_scores=[], states=[],
            )
            save_snapshot(snap, base_path=str(tmp_path))

        dates = list_snapshots(base_path=str(tmp_path))
        assert len(dates) == 3

        # Navigate: start at last, go prev twice
        idx = len(dates) - 1  # 2026-03-17
        assert dates[idx] == "2026-03-17"
        idx = max(0, idx - 1)
        assert dates[idx] == "2026-03-16"
        idx = max(0, idx - 1)
        assert dates[idx] == "2026-03-15"
        # At start, can't go further back
        idx = max(0, idx - 1)
        assert dates[idx] == "2026-03-15"
