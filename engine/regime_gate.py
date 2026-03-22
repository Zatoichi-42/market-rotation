"""
Regime Gate — Layer 1 of the Pump Rotation System.

Classifies the current market regime as NORMAL, FRAGILE, or HOSTILE
based on 6 signals: VIX, term structure, breadth, credit, oil, correlation.

Gate aggregation logic:
- >= hostile_threshold (default 2) HOSTILE signals → HOSTILE
- Any HOSTILE signal OR >= 2 FRAGILE signals → FRAGILE
- 1 FRAGILE + rest NORMAL → NORMAL
- All NORMAL → NORMAL

Modifiers (applied AFTER gate, can only tighten):
  A. Gold/VIX Divergence
  B. Gold/Silver Ratio
"""
import math
from datetime import datetime, timezone

from engine.schemas import (
    RegimeState, SignalLevel, RegimeSignal, RegimeAssessment,
)


def classify_signal(signal_name: str, value: float, thresholds: dict) -> RegimeSignal | None:
    """
    Classify a single signal into NORMAL / FRAGILE / HOSTILE.

    Returns None if value is NaN (signal excluded from gate).

    Boundary convention: exact boundary value goes to the WORSE bucket.
      - VIX >= normal_max → FRAGILE; VIX >= fragile_max → HOSTILE
      - term_structure >= contango_max → FRAGILE; >= flat_max → HOSTILE
      - breadth <= normal_min_zscore → FRAGILE; <= fragile_min_zscore → HOSTILE
      - credit <= normal_min_zscore → FRAGILE; <= fragile_min_zscore → HOSTILE
    """
    if math.isnan(value):
        return None

    if signal_name == "vix":
        cfg = thresholds["vix"]
        if value >= cfg["fragile_max"]:
            level = SignalLevel.HOSTILE
            desc = f"VIX at {value:.1f} (above {cfg['fragile_max']} hostile threshold)"
        elif value >= cfg["normal_max"]:
            level = SignalLevel.FRAGILE
            desc = f"VIX at {value:.1f} (between {cfg['normal_max']} and {cfg['fragile_max']})"
        else:
            level = SignalLevel.NORMAL
            desc = f"VIX at {value:.1f} (below {cfg['normal_max']} threshold)"

    elif signal_name == "term_structure":
        cfg = thresholds["term_structure"]
        if value >= cfg["flat_max"]:
            level = SignalLevel.HOSTILE
            desc = f"Term structure ratio {value:.2f} (backwardation, above {cfg['flat_max']})"
        elif value >= cfg["contango_max"]:
            level = SignalLevel.FRAGILE
            desc = f"Term structure ratio {value:.2f} (flat, between {cfg['contango_max']} and {cfg['flat_max']})"
        else:
            level = SignalLevel.NORMAL
            desc = f"Term structure ratio {value:.2f} (contango, below {cfg['contango_max']})"

    elif signal_name == "breadth":
        cfg = thresholds["breadth"]
        if value <= cfg["fragile_min_zscore"]:
            level = SignalLevel.HOSTILE
            desc = f"Breadth z-score {value:.2f} (collapsed, at or below {cfg['fragile_min_zscore']})"
        elif value <= cfg["normal_min_zscore"]:
            level = SignalLevel.FRAGILE
            desc = f"Breadth z-score {value:.2f} (narrowing, between {cfg['fragile_min_zscore']} and {cfg['normal_min_zscore']})"
        else:
            level = SignalLevel.NORMAL
            desc = f"Breadth z-score {value:.2f} (healthy, above {cfg['normal_min_zscore']})"

    elif signal_name == "credit":
        cfg = thresholds["credit"]
        if value <= cfg["fragile_min_zscore"]:
            level = SignalLevel.HOSTILE
            desc = f"Credit z-score {value:.2f} (crisis, at or below {cfg['fragile_min_zscore']})"
        elif value <= cfg["normal_min_zscore"]:
            level = SignalLevel.FRAGILE
            desc = f"Credit z-score {value:.2f} (stressed, between {cfg['fragile_min_zscore']} and {cfg['normal_min_zscore']})"
        else:
            level = SignalLevel.NORMAL
            desc = f"Credit z-score {value:.2f} (stable, above {cfg['normal_min_zscore']})"

    elif signal_name == "oil":
        # Oil z-score: high oil = inflationary pressure = regime stress
        if value >= 2.5:
            level = SignalLevel.HOSTILE
            desc = f"Oil z-score {value:.2f} (extreme, ≥2.5σ above mean)"
        elif value >= 1.5:
            level = SignalLevel.FRAGILE
            desc = f"Oil z-score {value:.2f} (elevated, ≥1.5σ above mean)"
        else:
            level = SignalLevel.NORMAL
            desc = f"Oil z-score {value:.2f} (normal range)"

    elif signal_name == "correlation":
        # Cross-sector correlation: high = diversification breakdown
        corr_cfg = thresholds.get("correlation", {})
        hostile_z = corr_cfg.get("hostile_zscore", 1.5)
        fragile_z = corr_cfg.get("fragile_zscore", 0.5)
        if value >= hostile_z:
            level = SignalLevel.HOSTILE
            desc = f"Correlation z-score {value:.2f} (sectors in lockstep, ≥{hostile_z}σ)"
        elif value >= fragile_z:
            level = SignalLevel.FRAGILE
            desc = f"Correlation z-score {value:.2f} (rising, ≥{fragile_z}σ)"
        else:
            level = SignalLevel.NORMAL
            desc = f"Correlation z-score {value:.2f} (healthy dispersion)"

    elif signal_name == "move":
        # MOVE index: bond market volatility — enriches Risk Appetite pillar
        move_cfg = thresholds.get("move", {})
        hostile_min = move_cfg.get("hostile_min", 130)
        fragile_min = move_cfg.get("fragile_min", 110)
        if value >= hostile_min:
            level = SignalLevel.HOSTILE
            desc = f"MOVE at {value:.0f} (extreme bond vol, ≥{hostile_min})"
        elif value >= fragile_min:
            level = SignalLevel.FRAGILE
            desc = f"MOVE at {value:.0f} (elevated bond vol, {fragile_min}-{hostile_min})"
        else:
            level = SignalLevel.NORMAL
            desc = f"MOVE at {value:.0f} (normal bond vol, <{fragile_min})"

    elif signal_name == "sb_correlation":
        # Stock-bond correlation: enriches Correlation Structure pillar
        # Positive SB-corr = stocks and bonds falling together = bad
        sb_cfg = thresholds.get("sb_correlation", {})
        hostile_thresh = sb_cfg.get("hostile", 0.30)
        fragile_thresh = sb_cfg.get("fragile", 0.15)
        if value >= hostile_thresh:
            level = SignalLevel.HOSTILE
            desc = f"SB-corr {value:+.2f} (stocks and bonds falling together, ≥{hostile_thresh})"
        elif value >= fragile_thresh:
            level = SignalLevel.FRAGILE
            desc = f"SB-corr {value:+.2f} (positive, hedging unreliable, ≥{fragile_thresh})"
        else:
            level = SignalLevel.NORMAL
            desc = f"SB-corr {value:+.2f} (negative/neutral, duration hedging works)"

    else:
        raise ValueError(f"Unknown signal name: {signal_name}")

    return RegimeSignal(name=signal_name, raw_value=value, level=level, description=desc)


def classify_regime(signals: list[RegimeSignal], thresholds: dict,
                    fred_hy_oas_value: float | None = None) -> RegimeAssessment:
    """
    Aggregate individual signals into a regime classification.

    Gate logic:
    - hostile_count >= hostile_threshold → HOSTILE
    - hostile_count >= 1 OR fragile_count >= 2 → FRAGILE
    - fragile_count <= 1 and hostile_count == 0 → NORMAL
    - No signals → NORMAL with warning
    """
    from engine.explain import explain_regime as _explain_regime

    now = datetime.now(timezone.utc).isoformat()

    if not signals:
        return RegimeAssessment(
            state=RegimeState.NORMAL,
            signals=[],
            hostile_count=0,
            fragile_count=0,
            normal_count=0,
            timestamp=now,
            explanation="NORMAL (default): No signals available — missing data.",
        )

    hostile_count = sum(1 for s in signals if s.level == SignalLevel.HOSTILE)
    fragile_count = sum(1 for s in signals if s.level == SignalLevel.FRAGILE)
    normal_count = sum(1 for s in signals if s.level == SignalLevel.NORMAL)

    hostile_threshold = thresholds.get("gate", {}).get("hostile_threshold", 2)

    if hostile_count >= hostile_threshold:
        state = RegimeState.HOSTILE
    elif hostile_count >= 1 or fragile_count >= 2:
        state = RegimeState.FRAGILE
    else:
        state = RegimeState.NORMAL

    # Build assessment first (with placeholder explanation), then generate explanation
    assessment = RegimeAssessment(
        state=state,
        signals=signals,
        hostile_count=hostile_count,
        fragile_count=fragile_count,
        normal_count=normal_count,
        timestamp=now,
        explanation="",
    )
    assessment.explanation = _explain_regime(assessment, fred_hy_oas_value=fred_hy_oas_value)

    return assessment


def classify_regime_from_data(
    vix_current: float,
    vix3m_current: float,
    breadth_zscore: float,
    credit_zscore: float,
    thresholds: dict,
    fred_hy_oas_value: float | None = None,
    oil_zscore: float = float("nan"),
    correlation_zscore: float = float("nan"),
    gold_silver_reading=None,
    gold_divergence_reading=None,
    move_level: float = float("nan"),
    sb_correlation: float = float("nan"),
) -> RegimeAssessment:
    """
    Convenience function: classify regime from raw values.
    Handles NaN gracefully by excluding those signals.

    6 signals: VIX, term structure, breadth, credit, oil, correlation.
    Modifiers applied after: gold/VIX divergence, gold/silver ratio.
    """
    # 6 base signals (v9 constitution: MOVE and SB-corr ENRICH existing pillars)
    signal_inputs = [
        ("vix", vix_current),
        ("term_structure", vix_current / vix3m_current if not math.isnan(vix3m_current) and vix3m_current != 0 else float("nan")),
        ("breadth", breadth_zscore),
        ("credit", credit_zscore),
        ("oil", oil_zscore),
        ("correlation", correlation_zscore),
    ]

    signals = []
    for name, value in signal_inputs:
        sig = classify_signal(name, value, thresholds)
        if sig is not None:
            signals.append(sig)

    # Enrichment: MOVE worsens VIX (Risk Appetite) by one level if extreme
    move_enrich = thresholds.get("move", {}).get("enrich_threshold", 130)
    if not math.isnan(move_level) and move_level >= move_enrich:
        for i, sig in enumerate(signals):
            if sig.name == "vix" and sig.level != SignalLevel.HOSTILE:
                new_level = SignalLevel.HOSTILE if sig.level == SignalLevel.FRAGILE else SignalLevel.FRAGILE
                signals[i] = RegimeSignal(
                    name="vix", raw_value=sig.raw_value, level=new_level,
                    description=f"{sig.description} [MOVE enrichment: {move_level:.0f} ≥ {move_enrich} → worsened to {new_level.value}]",
                )
                break

    # Enrichment: SB-corr worsens correlation by one level if extreme
    sb_enrich = thresholds.get("sb_correlation", {}).get("enrich_threshold", 0.30)
    if not math.isnan(sb_correlation) and sb_correlation >= sb_enrich:
        for i, sig in enumerate(signals):
            if sig.name == "correlation" and sig.level != SignalLevel.HOSTILE:
                new_level = SignalLevel.HOSTILE if sig.level == SignalLevel.FRAGILE else SignalLevel.FRAGILE
                signals[i] = RegimeSignal(
                    name="correlation", raw_value=sig.raw_value, level=new_level,
                    description=f"{sig.description} [SB-corr enrichment: {sb_correlation:+.2f} ≥ {sb_enrich} → worsened to {new_level.value}]",
                )
                break

    # Log MOVE and SB-corr as informational (no vote, for transparency)
    if not math.isnan(move_level):
        signals.append(RegimeSignal(name="move (info)", raw_value=move_level,
                                    level=SignalLevel.NORMAL,
                                    description=f"MOVE {move_level:.0f} (enrichment only, no independent vote)"))
    if not math.isnan(sb_correlation):
        signals.append(RegimeSignal(name="sb_corr (info)", raw_value=sb_correlation,
                                    level=SignalLevel.NORMAL,
                                    description=f"SB-corr {sb_correlation:+.2f} (enrichment only, no independent vote)"))

    assessment = classify_regime(signals, thresholds, fred_hy_oas_value=fred_hy_oas_value)

    # Determine if gold/VIX divergence is active
    gold_vix_divergence_active = False
    if gold_divergence_reading is not None:
        gold_vix_divergence_active = gold_divergence_reading.is_margin_call_regime

    # Apply gold/VIX divergence modifier (can only tighten)
    if gold_divergence_reading is not None:
        from engine.gold_divergence import apply_gold_divergence_modifier
        new_state, mod_exp = apply_gold_divergence_modifier(
            assessment.state, gold_divergence_reading,
        )
        if new_state != assessment.state:
            assessment.state = new_state
        if mod_exp:
            assessment.explanation += f" [Modifier: {mod_exp}]"

    # Apply gold/silver ratio modifier (can only tighten)
    if gold_silver_reading is not None:
        from engine.gold_silver_ratio import apply_gold_silver_modifier
        new_state, mod_exp = apply_gold_silver_modifier(
            assessment.state, gold_silver_reading,
            gold_vix_divergence_active=gold_vix_divergence_active,
        )
        if new_state != assessment.state:
            assessment.state = new_state
        if mod_exp:
            assessment.explanation += f" [Modifier: {mod_exp}]"

    return assessment


