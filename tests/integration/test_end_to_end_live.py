"""
Full pipeline integration test — fetches live data and runs every engine module.
"""
import time
import pytest
import yaml
import numpy as np
from engine.schemas import (
    RegimeState, RSReading, BreadthReading, PumpScoreReading,
    StateClassification, DailySnapshot, AnalysisState,
)
from engine.regime_gate import classify_regime_from_data
from engine.rs_scanner import compute_rs_readings
from engine.breadth import compute_breadth
from engine.normalizer import compute_zscore, percentile_rank
from engine.pump_score import compute_pump_score
from engine.state_classifier import classify_all_sectors
from data.snapshots import save_snapshot, load_snapshot

pytestmark = pytest.mark.integration

SECTOR_NAMES = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
}


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full pipeline once and cache for all tests."""
    from data.fetcher import fetch_all
    from datetime import datetime, timezone

    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/universe.yaml") as f:
        universe = yaml.safe_load(f)
    config = {**settings, **universe}

    start = time.time()

    # 1. Fetch
    data = fetch_all(config)
    prices = data["prices"]
    vix = data["vix"]
    vix3m = data["vix3m"]

    # 2. Regime
    breadth_reading = compute_breadth(prices)
    credit = data["credit"]
    hyg = credit["hyg_close"] if "hyg_close" in credit.columns else None
    lqd = credit["lqd_close"] if "lqd_close" in credit.columns else None
    if hyg is not None and lqd is not None:
        ratio = hyg / lqd
        credit_z = compute_zscore(ratio.iloc[-1], ratio.dropna()) if len(ratio.dropna()) > 2 else 0.0
    else:
        credit_z = 0.0

    vix_val = vix.iloc[-1] if len(vix) > 0 else 20.0
    vix3m_val = vix3m.iloc[-1] if len(vix3m) > 0 else 20.0
    bz = breadth_reading.rsp_spy_ratio_zscore
    if np.isnan(bz):
        bz = 0.0

    regime = classify_regime_from_data(
        vix_current=vix_val, vix3m_current=vix3m_val,
        breadth_zscore=bz, credit_zscore=credit_z,
        thresholds=settings["regime"],
    )

    # 3. RS Scanner
    rs_cfg = settings["rs"]
    rs_readings = compute_rs_readings(
        prices, SECTOR_NAMES,
        windows=rs_cfg["windows"],
        slope_window=rs_cfg["slope_window"],
        composite_weights=rs_cfg["composite_weights"],
    )

    # 4. Pump Scores (using RS composite as RS pillar, breadth as participation, 50 as flow placeholder)
    pump_weights = settings["pump_score"]
    pumps = {}
    for r in rs_readings:
        score = compute_pump_score(r.rs_composite, 50.0, 50.0, pump_weights)
        pumps[r.ticker] = PumpScoreReading(
            ticker=r.ticker, name=r.name,
            rs_pillar=r.rs_composite, participation_pillar=50.0, flow_pillar=50.0,
            pump_score=score, pump_delta=0.0, pump_delta_5d_avg=0.0,
        )

    # 5. State Classifier
    rs_ranks = {r.ticker: r.rs_rank for r in rs_readings}
    pump_scores_list = [p.pump_score for p in pumps.values()]
    pump_pcts = percentile_rank(
        __import__("pandas").Series({t: p.pump_score for t, p in pumps.items()})
    )
    states = classify_all_sectors(
        pumps=pumps, priors={}, regime=regime.state,
        rs_ranks=rs_ranks,
        pump_percentiles=pump_pcts.to_dict(),
        delta_histories={t: [0.0] for t in pumps},
        settings=settings["state"],
    )

    # 6. Snapshot
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot = DailySnapshot(
        date=now_str,
        timestamp=datetime.now(timezone.utc).isoformat(),
        regime=regime,
        sectors=rs_readings,
        breadth=breadth_reading,
        pump_scores=list(pumps.values()),
        states=list(states.values()),
    )

    elapsed = time.time() - start

    return {
        "snapshot": snapshot,
        "regime": regime,
        "rs_readings": rs_readings,
        "breadth": breadth_reading,
        "pumps": pumps,
        "states": states,
        "elapsed": elapsed,
        "data": data,
    }


class TestFullPipeline:

    def test_full_pipeline_completes_without_error(self, pipeline_result):
        """fetch → regime → RS → breadth → pump scores → states → snapshot."""
        assert pipeline_result["snapshot"] is not None

    def test_all_11_sectors_present_in_output(self, pipeline_result):
        """Every sector ETF appears in the final output."""
        tickers = {r.ticker for r in pipeline_result["rs_readings"]}
        assert tickers == set(SECTOR_NAMES.keys())

    def test_ranks_are_1_through_11(self, pipeline_result):
        """RS ranks are exactly {1, 2, ..., 11}."""
        ranks = sorted(r.rs_rank for r in pipeline_result["rs_readings"])
        assert ranks == list(range(1, 12))

    def test_snapshot_is_serializable(self, pipeline_result, tmp_path):
        """DailySnapshot can be saved to parquet and loaded back."""
        snap = pipeline_result["snapshot"]
        save_snapshot(snap, base_path=str(tmp_path))
        loaded = load_snapshot(snap.date, base_path=str(tmp_path))
        assert loaded.date == snap.date
        assert loaded.regime.state == snap.regime.state
        assert len(loaded.sectors) == len(snap.sectors)
        assert len(loaded.states) == len(snap.states)

    def test_pipeline_timing_under_60_seconds(self, pipeline_result):
        """Full pipeline from fetch to output < 60 seconds."""
        assert pipeline_result["elapsed"] < 60, f"Pipeline took {pipeline_result['elapsed']:.1f}s"

    def test_regime_is_valid(self, pipeline_result):
        assert pipeline_result["regime"].state in (RegimeState.NORMAL, RegimeState.FRAGILE, RegimeState.HOSTILE)

    def test_all_states_valid(self, pipeline_result):
        for ticker, sc in pipeline_result["states"].items():
            assert isinstance(sc, StateClassification)
            assert sc.state in AnalysisState
            assert 10 <= sc.confidence <= 95

    def test_breadth_reading_valid(self, pipeline_result):
        br = pipeline_result["breadth"]
        assert isinstance(br, BreadthReading)
        assert br.rsp_spy_ratio > 0
