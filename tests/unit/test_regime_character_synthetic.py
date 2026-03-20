"""
Regime Character classification tests -- synthetic data only.
"""
import pytest
from engine.schemas import RegimeState, RegimeCharacter, RegimeCharacterReading
from engine.regime_character import classify_regime_character
from engine.correlation import compute_cross_sector_dispersion


# ── Defaults shared across tests ─────────────────────────────

_DEFAULTS = dict(
    spy_20d_return=0.0,
    vix_level=18.0,
    vix_20d_change=0.0,
    breadth_zscore=0.0,
    breadth_zscore_change_5d=0.0,
    cross_sector_dispersion=0.01,
    correlation_zscore=0.0,
    credit_zscore=0.0,
    gold_divergence_active=False,
    gate_level=RegimeState.NORMAL,
    prior_character=None,
    sessions_in_prior=0,
    dispersion_history=None,
)


def _classify(**overrides) -> RegimeCharacterReading:
    kwargs = {**_DEFAULTS, **overrides}
    return classify_regime_character(**kwargs)


# ── Character classification tests ───────────────────────────


class TestCharacterClassification:
    """TEST-CHAR-01 through TEST-CHAR-06: First-match-wins rules."""

    def test_char01_trending_bull(self):
        """TEST-CHAR-01: SPY +3%, VIX 15, breadth 0.5, corr 0.2 -> TRENDING_BULL."""
        r = _classify(
            spy_20d_return=0.03,
            vix_level=15.0,
            breadth_zscore=0.5,
            correlation_zscore=0.2,
        )
        assert r.character == RegimeCharacter.TRENDING_BULL

    def test_char02_trending_bear(self):
        """TEST-CHAR-02: SPY -4%, VIX 25, breadth -0.5 -> TRENDING_BEAR."""
        r = _classify(
            spy_20d_return=-0.04,
            vix_level=25.0,
            breadth_zscore=-0.5,
        )
        assert r.character == RegimeCharacter.TRENDING_BEAR

    def test_char03_rotation(self):
        """TEST-CHAR-03: SPY flat, dispersion high vs history -> ROTATION."""
        r = _classify(
            spy_20d_return=0.005,
            cross_sector_dispersion=0.05,
            dispersion_history=[0.01, 0.02, 0.01, 0.02],
        )
        assert r.character == RegimeCharacter.ROTATION

    def test_char04_crisis_vix_correlation(self):
        """TEST-CHAR-04: VIX 40 + correlation 2.0 -> CRISIS."""
        r = _classify(
            vix_level=40.0,
            correlation_zscore=2.0,
        )
        assert r.character == RegimeCharacter.CRISIS

    def test_char05_recovery(self):
        """TEST-CHAR-05: Prior=CRISIS, VIX declining, breadth improving, SPY positive -> RECOVERY."""
        r = _classify(
            spy_20d_return=0.01,
            vix_level=20.0,
            vix_20d_change=-5.0,
            breadth_zscore_change_5d=0.3,
            prior_character=RegimeCharacter.CRISIS,
            sessions_in_prior=5,
        )
        assert r.character == RegimeCharacter.RECOVERY

    def test_char06_choppy(self):
        """TEST-CHAR-06: Mixed signals -> CHOPPY."""
        r = _classify(
            spy_20d_return=0.005,
            vix_level=18.0,
            breadth_zscore=0.1,
            correlation_zscore=0.3,
            cross_sector_dispersion=0.01,
            dispersion_history=[0.01, 0.02, 0.03, 0.04],
        )
        assert r.character == RegimeCharacter.CHOPPY


class TestPersistence:
    """TEST-CHAR-07: Persistence filter requires 3 sessions."""

    def test_char07_stays_as_prior_when_sessions_lt_3(self):
        """TEST-CHAR-07: New char differs from prior with sessions < 3 -> stays as prior."""
        # Prior is TRENDING_BULL with only 2 sessions -- new raw would be CHOPPY
        r = _classify(
            spy_20d_return=0.005,
            vix_level=18.0,
            breadth_zscore=0.1,
            correlation_zscore=0.3,
            cross_sector_dispersion=0.01,
            dispersion_history=[0.01, 0.02, 0.03, 0.04],
            prior_character=RegimeCharacter.TRENDING_BULL,
            sessions_in_prior=2,
        )
        assert r.character == RegimeCharacter.TRENDING_BULL

    def test_crisis_overrides_persistence(self):
        """CRISIS always takes effect immediately regardless of persistence."""
        r = _classify(
            vix_level=40.0,
            correlation_zscore=2.0,
            prior_character=RegimeCharacter.TRENDING_BULL,
            sessions_in_prior=1,
        )
        assert r.character == RegimeCharacter.CRISIS

    def test_transition_allowed_after_3_sessions(self):
        """After 3+ sessions in prior, transition to new character is allowed."""
        r = _classify(
            spy_20d_return=0.005,
            vix_level=18.0,
            breadth_zscore=0.1,
            correlation_zscore=0.3,
            cross_sector_dispersion=0.01,
            dispersion_history=[0.01, 0.02, 0.03, 0.04],
            prior_character=RegimeCharacter.TRENDING_BULL,
            sessions_in_prior=3,
        )
        assert r.character == RegimeCharacter.CHOPPY


class TestCrisisGateAlignment:
    """TEST-CHAR-09: HOSTILE gate -> CRISIS."""

    def test_char09_hostile_gate_crisis(self):
        """TEST-CHAR-09: gate_level=HOSTILE -> CRISIS regardless of other signals."""
        r = _classify(
            spy_20d_return=0.03,
            vix_level=15.0,
            breadth_zscore=0.5,
            gate_level=RegimeState.HOSTILE,
        )
        assert r.character == RegimeCharacter.CRISIS


class TestRotationCharacter:
    """TEST-CHAR-10: ROTATION character details."""

    def test_char10_rotation_no_history_high_dispersion(self):
        """ROTATION when no dispersion history and dispersion > 0.03."""
        r = _classify(
            spy_20d_return=0.005,
            cross_sector_dispersion=0.04,
            dispersion_history=None,
        )
        assert r.character == RegimeCharacter.ROTATION

    def test_rotation_no_history_low_dispersion(self):
        """Not ROTATION when no history and dispersion <= 0.03."""
        r = _classify(
            spy_20d_return=0.005,
            cross_sector_dispersion=0.02,
            dispersion_history=None,
        )
        assert r.character != RegimeCharacter.ROTATION


class TestConfidence:
    """TEST-CHAR-11, TEST-CHAR-12: Confidence adjustments from gate alignment."""

    def test_char11_bull_normal_gate_higher_confidence(self):
        """TEST-CHAR-11: TRENDING_BULL + NORMAL gate -> confidence boosted."""
        r = _classify(
            spy_20d_return=0.03,
            vix_level=15.0,
            breadth_zscore=0.5,
            correlation_zscore=0.2,
            gate_level=RegimeState.NORMAL,
        )
        assert r.character == RegimeCharacter.TRENDING_BULL
        # 60 base + 15 gate match = 75 (trends may add more)
        assert r.confidence >= 75

    def test_char12_choppy_fragile_gate_lower_confidence(self):
        """TEST-CHAR-12: CHOPPY + FRAGILE gate -> confidence reduced."""
        r = _classify(
            spy_20d_return=0.005,
            vix_level=18.0,
            breadth_zscore=0.1,
            correlation_zscore=0.3,
            cross_sector_dispersion=0.01,
            dispersion_history=[0.01, 0.02, 0.03, 0.04],
            gate_level=RegimeState.FRAGILE,
            prior_character=RegimeCharacter.CHOPPY,
            sessions_in_prior=5,
        )
        assert r.character == RegimeCharacter.CHOPPY
        # 60 base - 15 gate contradiction = 45
        assert r.confidence <= 60


class TestEnumCompleteness:
    """TEST-CHAR-14: All 6 characters are distinct enum values."""

    def test_char14_six_distinct_characters(self):
        assert len(RegimeCharacter) == 6
        values = [c.value for c in RegimeCharacter]
        assert len(set(values)) == 6


class TestTrends:
    """Breadth and VIX trend derivation."""

    def test_breadth_improving(self):
        r = _classify(breadth_zscore_change_5d=0.2)
        assert r.breadth_trend == "improving"

    def test_breadth_deteriorating(self):
        r = _classify(breadth_zscore_change_5d=-0.2)
        assert r.breadth_trend == "deteriorating"

    def test_breadth_stable(self):
        r = _classify(breadth_zscore_change_5d=0.05)
        assert r.breadth_trend == "stable"

    def test_vix_rising(self):
        r = _classify(vix_20d_change=3.0)
        assert r.vix_trend == "rising"

    def test_vix_declining(self):
        r = _classify(vix_20d_change=-3.0)
        assert r.vix_trend == "declining"

    def test_vix_stable(self):
        r = _classify(vix_20d_change=1.0)
        assert r.vix_trend == "stable"


class TestReadingFields:
    """Verify the returned dataclass has all expected fields."""

    def test_all_fields_populated(self):
        r = _classify(spy_20d_return=0.03, vix_level=15.0, breadth_zscore=0.5, correlation_zscore=0.2)
        assert isinstance(r, RegimeCharacterReading)
        assert isinstance(r.character, RegimeCharacter)
        assert isinstance(r.gate_level, RegimeState)
        assert isinstance(r.confidence, int)
        assert 0 <= r.confidence <= 100
        assert isinstance(r.spy_20d_return, float)
        assert isinstance(r.cross_sector_dispersion, float)
        assert r.breadth_trend in ("improving", "stable", "deteriorating")
        assert r.vix_trend in ("declining", "stable", "rising")
        assert isinstance(r.sessions_in_character, int)
        assert isinstance(r.description, str)
        assert len(r.description) > 0


# ── Cross-sector dispersion function tests ───────────────────


class TestCrossSectorDispersion:
    """Tests for compute_cross_sector_dispersion in correlation.py."""

    def test_eleven_values_positive(self):
        """11 sector returns -> positive float."""
        sector_returns = {
            "XLK": 0.05, "XLV": 0.02, "XLF": -0.01, "XLE": 0.08,
            "XLI": 0.03, "XLU": -0.02, "XLRE": 0.01, "XLC": 0.04,
            "XLY": -0.03, "XLP": 0.00, "XLB": 0.06,
        }
        result = compute_cross_sector_dispersion(sector_returns)
        assert isinstance(result, float)
        assert result > 0.0

    def test_two_values_returns_zero(self):
        """Fewer than 3 values -> 0.0."""
        result = compute_cross_sector_dispersion({"XLK": 0.05, "XLV": 0.02})
        assert result == 0.0

    def test_empty_returns_zero(self):
        """Empty dict -> 0.0."""
        result = compute_cross_sector_dispersion({})
        assert result == 0.0

    def test_three_values_ok(self):
        """Exactly 3 values -> non-zero (boundary)."""
        result = compute_cross_sector_dispersion({"XLK": 0.05, "XLV": 0.02, "XLF": -0.01})
        assert isinstance(result, float)
        assert result > 0.0
