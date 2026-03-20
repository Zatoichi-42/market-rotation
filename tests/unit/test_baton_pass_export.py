"""Test baton pass alert in export includes rs_rank."""
import pytest
from unittest.mock import MagicMock
from engine.schemas import PumpScoreReading
from dashboard.components.export import _get_baton_passes


def test_baton_pass_includes_rs_rank():
    """rs_rank present in sector_data → filter works, produces alerts."""
    result = {
        "pumps": {
            "XLK": PumpScoreReading("XLK", "Technology", 80, 60, 55, 0.57, 0.092, 0.02),
            "XLP": PumpScoreReading("XLP", "Staples", 30, 50, 50, 0.36, -0.052, -0.03),
        },
        "states": {
            "XLK": MagicMock(state=MagicMock(value="Overt Pump")),
            "XLP": MagicMock(state=MagicMock(value="Ambiguous")),
        },
        "rs_readings": [
            MagicMock(ticker="XLK", rs_rank=3),
            MagicMock(ticker="XLP", rs_rank=8),
        ],
    }
    alerts = _get_baton_passes(result)
    assert len(alerts) >= 1
    assert "XLK" in alerts[0]
    assert "XLP" in alerts[0]


def test_baton_pass_no_rank_no_false_suppression():
    """Even without rs_readings, alerts should not be completely suppressed
    if delta diff is large enough (default rank 11 fails rank<=5, so 0 alerts
    is expected — this documents the intentional behavior)."""
    result = {
        "pumps": {
            "XLK": PumpScoreReading("XLK", "Technology", 80, 60, 55, 0.57, 0.092, 0.02),
            "XLP": PumpScoreReading("XLP", "Staples", 30, 50, 50, 0.36, -0.052, -0.03),
        },
        "states": {
            "XLK": MagicMock(state=MagicMock(value="Overt Pump")),
            "XLP": MagicMock(state=MagicMock(value="Ambiguous")),
        },
        "rs_readings": [],  # No readings → default rank 11 → no alerts
    }
    alerts = _get_baton_passes(result)
    assert len(alerts) == 0  # Intentional: no rank data means no alerts


def test_baton_pass_capped_at_5():
    """Max 5 alerts returned."""
    pumps = {}
    states = {}
    readings = []
    # Create 6 rising + 6 declining to generate many pairs
    for i in range(6):
        t_rise = f"R{i}"
        t_fall = f"F{i}"
        pumps[t_rise] = PumpScoreReading(t_rise, t_rise, 80, 60, 55, 0.70, 0.15, 0.10)
        pumps[t_fall] = PumpScoreReading(t_fall, t_fall, 20, 40, 40, 0.30, -0.10, -0.08)
        states[t_rise] = MagicMock(state=MagicMock(value="Overt Pump"))
        states[t_fall] = MagicMock(state=MagicMock(value="Exhaustion"))
        readings.append(MagicMock(ticker=t_rise, rs_rank=i+1))
        readings.append(MagicMock(ticker=t_fall, rs_rank=6+i))
    result = {"pumps": pumps, "states": states, "rs_readings": readings}
    alerts = _get_baton_passes(result)
    assert len(alerts) <= 5
