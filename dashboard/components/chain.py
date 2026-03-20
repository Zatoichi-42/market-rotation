"""
Causal Chain Display — explanatory macro chain when 2+ signals are FRAGILE or worse.

This is NOT predictive — it helps the operator understand WHY the regime is deteriorating.
"""
import streamlit as st

from engine.schemas import SignalLevel, RegimeAssessment


def generate_causal_chain(
    regime: RegimeAssessment,
    gold_divergence_active: bool = False,
    correlation_reading=None,
) -> list[str]:
    """
    Generate text-based causal chain from elevated regime signals.

    Only generates when 2+ signals are FRAGILE or worse.
    Returns list of chain strings.
    """
    elevated = [s for s in regime.signals if s.level in (SignalLevel.FRAGILE, SignalLevel.HOSTILE)]
    if len(elevated) < 2:
        return []

    signal_names = {s.name for s in elevated}
    signal_levels = {s.name: s.level for s in elevated}
    chains = []

    # Oil + VIX chain
    if "oil" in signal_names and "vix" in signal_names:
        chains.append(
            "Oil shock ↑↑ → Inflation expectations ↑ → Rate cuts off table "
            "→ Growth/duration assets pressured ↓ → Energy benefiting ↑"
        )

    # Correlation + Breadth chain
    if "correlation" in signal_names and "breadth" in signal_names:
        chains.append(
            "Correlation spike ↑ → Diversification breaking down "
            "→ Sectors moving in lockstep → Rotation signals unreliable"
        )

    # Credit + VIX chain
    if "credit" in signal_names and "vix" in signal_names:
        chains.append(
            "Credit widening ↓ → Risk appetite declining "
            "→ Flight to quality → Defensive sectors favored"
        )

    # Breadth alone if HOSTILE
    if signal_levels.get("breadth") == SignalLevel.HOSTILE:
        chains.append(
            "Breadth collapsing ↓↓ → Market leadership narrowing "
            "→ Fragile index → Vulnerable to gap down"
        )

    # Oil + Correlation chain
    if "oil" in signal_names and "correlation" in signal_names:
        chains.append(
            "Oil ↑↑ + Correlation ↑ → Stagflation pressure "
            "→ Defensive rotation accelerates"
        )

    # Gold/VIX divergence chain
    if gold_divergence_active:
        chains.append(
            "Gold ↓ + Equities ↓ + VIX ↑ → Margin call regime "
            "→ Forced liquidation across ALL assets → Cash only safe haven"
        )

    return chains


def render_causal_chain(result: dict):
    """Render causal chain in the regime panel when conditions warrant."""
    regime = result["regime"]
    gd = result.get("gold_divergence_reading")
    corr = result.get("correlation_reading")

    gold_div_active = gd.is_margin_call_regime if gd else False

    chains = generate_causal_chain(
        regime,
        gold_divergence_active=gold_div_active,
        correlation_reading=corr,
    )

    if not chains:
        return

    st.subheader("Macro Chain")
    st.caption("Explanatory — not a trade signal. Shows why the regime gate is deteriorating.")

    for chain in chains:
        st.markdown(
            f"<div style='padding:6px 12px;background:rgba(0,0,0,0.2);"
            f"border-left:3px solid #ffa500;margin:4px 0;font-size:0.95em;'>"
            f"{chain}</div>",
            unsafe_allow_html=True,
        )
