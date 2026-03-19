"""
Snapshot save/load unit tests — synthetic data, uses tmp_path.
"""
import pytest
from engine.schemas import (
    RegimeState, SignalLevel, RegimeSignal, RegimeAssessment,
    RSReading, BreadthSignal, BreadthReading,
    PumpScoreReading, AnalysisState, TransitionPressure, StateClassification,
    DailySnapshot,
)
from data.snapshots import save_snapshot, load_snapshot, list_snapshots, load_snapshot_range


def _make_snapshot(date: str = "2026-03-18") -> DailySnapshot:
    sig = RegimeSignal(name="vix", raw_value=15.0, level=SignalLevel.NORMAL, description="ok")
    regime = RegimeAssessment(
        state=RegimeState.NORMAL, signals=[sig],
        hostile_count=0, fragile_count=0, normal_count=1,
        timestamp="2026-03-18T10:00:00", explanation="All clear",
    )
    breadth = BreadthReading(
        rsp_spy_ratio=1.02, rsp_spy_ratio_20d_change=0.005,
        rsp_spy_ratio_zscore=0.5, signal=BreadthSignal.HEALTHY,
        explanation="Healthy breadth",
    )
    sector = RSReading(
        ticker="XLK", name="Technology",
        rs_5d=1.2, rs_20d=2.5, rs_60d=4.0,
        rs_slope=0.3, rs_rank=1, rs_rank_change=2, rs_composite=85.0,
    )
    pump = PumpScoreReading(
        ticker="XLK", name="Technology",
        rs_pillar=80.0, participation_pillar=70.0, flow_pillar=75.0,
        pump_score=0.75, pump_delta=0.03, pump_delta_5d_avg=0.02,
    )
    state = StateClassification(
        ticker="XLK", name="Technology",
        state=AnalysisState.BROADENING, confidence=72,
        sessions_in_state=6, transition_pressure=TransitionPressure.UP,
        prior_state=AnalysisState.ACCUMULATION, state_changed=False,
        explanation="Broadening confirmed",
    )
    return DailySnapshot(
        date=date, timestamp="2026-03-18T10:00:00",
        regime=regime, sectors=[sector], breadth=breadth,
        pump_scores=[pump], states=[state],
    )


class TestSnapshotRoundTrip:

    def test_save_and_load(self, tmp_path):
        snap = _make_snapshot()
        save_snapshot(snap, base_path=str(tmp_path))
        loaded = load_snapshot("2026-03-18", base_path=str(tmp_path))
        assert loaded.date == snap.date
        assert loaded.regime.state == RegimeState.NORMAL
        assert loaded.sectors[0].ticker == "XLK"
        assert loaded.breadth.signal == BreadthSignal.HEALTHY
        assert loaded.pump_scores[0].pump_score == 0.75
        assert loaded.states[0].state == AnalysisState.BROADENING
        assert loaded.states[0].prior_state == AnalysisState.ACCUMULATION

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_snapshot("1999-01-01", base_path=str(tmp_path))

    def test_list_snapshots(self, tmp_path):
        save_snapshot(_make_snapshot("2026-03-16"), base_path=str(tmp_path))
        save_snapshot(_make_snapshot("2026-03-17"), base_path=str(tmp_path))
        save_snapshot(_make_snapshot("2026-03-18"), base_path=str(tmp_path))
        dates = list_snapshots(base_path=str(tmp_path))
        assert dates == ["2026-03-16", "2026-03-17", "2026-03-18"]

    def test_load_range(self, tmp_path):
        for d in ["2026-03-14", "2026-03-15", "2026-03-16", "2026-03-17", "2026-03-18"]:
            save_snapshot(_make_snapshot(d), base_path=str(tmp_path))
        snaps = load_snapshot_range("2026-03-15", "2026-03-17", base_path=str(tmp_path))
        assert len(snaps) == 3
        assert snaps[0].date == "2026-03-15"
        assert snaps[-1].date == "2026-03-17"

    def test_empty_directory(self, tmp_path):
        assert list_snapshots(base_path=str(tmp_path)) == []

    def test_prior_state_none_roundtrip(self, tmp_path):
        snap = _make_snapshot()
        snap.states[0].prior_state = None
        save_snapshot(snap, base_path=str(tmp_path))
        loaded = load_snapshot("2026-03-18", base_path=str(tmp_path))
        assert loaded.states[0].prior_state is None
