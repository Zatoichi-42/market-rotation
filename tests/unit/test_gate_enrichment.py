"""
Collision tests for the MOVE/SB-corr enrichment architecture in engine/regime_gate.py.

The regime gate now has 6 base signals (VIX, term structure, breadth, credit, oil, correlation).
MOVE and SB-corr are NOT independent signals — they ENRICH existing pillars:
- MOVE >= 130 (enrich_threshold) -> worsens VIX signal by one level
- SB-corr >= 0.30 (enrich_threshold) -> worsens correlation signal by one level

MOVE and SB-corr are logged as informational entries (name contains "(info)") with
NORMAL level — they don't count in hostile/fragile tallies.
"""
import pytest
from engine.regime_gate import classify_regime_from_data

# Thresholds matching config/settings.yaml
SETTINGS = {
    "vix": {"normal_max": 20, "fragile_max": 30},
    "term_structure": {"contango_max": 0.95, "flat_max": 1.05},
    "breadth": {"normal_min_zscore": 0.0, "fragile_min_zscore": -1.0},
    "credit": {"normal_min_zscore": -0.5, "fragile_min_zscore": -1.5},
    "move": {"enrich_threshold": 130, "fragile_min": 110, "hostile_min": 130},
    "sb_correlation": {"enrich_threshold": 0.30, "fragile": 0.15, "hostile": 0.30},
    "correlation": {"window": 21, "zscore_window": 504, "fragile_zscore": 0.5, "hostile_zscore": 1.5, "absolute_hostile": 0.80},
    "gate": {"hostile_threshold": 2, "fragile_mixed": True},
}


class TestMoveEnrichment:
    def test_move_140_worsens_vix_normal_to_fragile(self):
        """VIX=18 (NORMAL) + MOVE=140 -> VIX worsened to FRAGILE."""
        r = classify_regime_from_data(18.0, 20.0, 0.5, 0.0, SETTINGS, move_level=140)
        vix_sig = [s for s in r.signals if s.name == "vix"][0]
        assert vix_sig.level.value == "FRAGILE"
        assert "MOVE enrichment" in vix_sig.description

    def test_move_140_worsens_vix_fragile_to_hostile(self):
        """VIX=25 (FRAGILE) + MOVE=140 -> VIX worsened to HOSTILE."""
        r = classify_regime_from_data(25.0, 28.0, 0.5, 0.0, SETTINGS, move_level=140)
        vix_sig = [s for s in r.signals if s.name == "vix"][0]
        assert vix_sig.level.value == "HOSTILE"

    def test_move_100_no_enrichment(self):
        """MOVE=100 (below 130 threshold) -> VIX unchanged."""
        r = classify_regime_from_data(18.0, 20.0, 0.5, 0.0, SETTINGS, move_level=100)
        vix_sig = [s for s in r.signals if s.name == "vix"][0]
        assert vix_sig.level.value == "NORMAL"
        assert "MOVE" not in vix_sig.description

    def test_move_hostile_vix_stays_hostile(self):
        """VIX=35 (already HOSTILE) + MOVE=140 -> stays HOSTILE (can't worsen further)."""
        r = classify_regime_from_data(35.0, 38.0, 0.5, 0.0, SETTINGS, move_level=140)
        vix_sig = [s for s in r.signals if s.name == "vix"][0]
        assert vix_sig.level.value == "HOSTILE"

    def test_move_logged_as_info(self):
        """MOVE appears as informational signal with NORMAL level."""
        r = classify_regime_from_data(18.0, 20.0, 0.5, 0.0, SETTINGS, move_level=120)
        info = [s for s in r.signals if "move" in s.name.lower() and "info" in s.name.lower()]
        assert len(info) == 1
        assert info[0].level.value == "NORMAL"


class TestSBCorrEnrichment:
    def test_sbcorr_040_worsens_corr_normal_to_fragile(self):
        """Equity corr NORMAL + SB-corr=0.40 -> correlation worsened to FRAGILE."""
        r = classify_regime_from_data(18.0, 20.0, 0.5, 0.0, SETTINGS,
                                       correlation_zscore=0.2, sb_correlation=0.40)
        corr_sig = [s for s in r.signals if s.name == "correlation"][0]
        assert corr_sig.level.value == "FRAGILE"
        assert "SB-corr enrichment" in corr_sig.description

    def test_sbcorr_010_no_enrichment(self):
        """SB-corr=0.10 (below 0.30) -> correlation unchanged."""
        r = classify_regime_from_data(18.0, 20.0, 0.5, 0.0, SETTINGS,
                                       correlation_zscore=0.2, sb_correlation=0.10)
        corr_sig = [s for s in r.signals if s.name == "correlation"][0]
        assert "SB-corr" not in corr_sig.description

    def test_sbcorr_logged_as_info(self):
        """SB-corr appears as informational signal."""
        r = classify_regime_from_data(18.0, 20.0, 0.5, 0.0, SETTINGS, sb_correlation=0.20)
        info = [s for s in r.signals if "sb_corr" in s.name.lower() and "info" in s.name.lower()]
        assert len(info) == 1


class TestEnrichedGateOutcome:
    def test_march20_scenario_is_fragile_not_hostile(self):
        """Today's exact values: oil HOSTILE, VIX FRAGILE, SB-corr 0.39.
        With enrichment (not independent vote), should be FRAGILE not HOSTILE."""
        r = classify_regime_from_data(
            vix_current=26.8, vix3m_current=28.0,
            breadth_zscore=-0.3, credit_zscore=0.1,
            thresholds=SETTINGS,
            oil_zscore=3.42,
            correlation_zscore=-0.30,
            sb_correlation=0.39,
            move_level=100.0,
        )
        # Only 1 HOSTILE (oil). SB-corr enriches correlation NORMAL->FRAGILE.
        # 1 hostile + fragiles -> FRAGILE, not HOSTILE
        assert r.state.value == "FRAGILE"

    def test_six_base_signals_only_count_for_gate(self):
        """Info signals don't count toward hostile/fragile tallies."""
        r = classify_regime_from_data(18.0, 20.0, 0.5, 0.0, SETTINGS,
                                       move_level=120, sb_correlation=0.20)
        # Both below enrichment thresholds, logged as info
        assert r.hostile_count == 0
        assert r.state.value == "NORMAL"
