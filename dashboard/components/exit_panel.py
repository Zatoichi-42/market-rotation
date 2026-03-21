"""Exit monitor panel — shows exit signal status per position."""
import streamlit as st
from engine.schemas import ExitAssessment, ExitUrgency

_URGENCY_STYLE = {
    ExitUrgency.WATCH: ("\U0001f441", "#f59e0b"),       # amber
    ExitUrgency.WARNING: ("\u26a0", "#ea580c"),          # orange
    ExitUrgency.ALERT: ("\U0001f534", "#dc2626"),        # red
    ExitUrgency.IMMEDIATE: ("\U0001f6a8", "#7f1d1d"),    # dark red
}

def exit_badge_html(assessment: ExitAssessment | None) -> str:
    """Return HTML for exit status badge."""
    if assessment is None or not assessment.signals:
        return '<span style="color:#6b7280;">—</span>'
    icon, color = _URGENCY_STYLE.get(assessment.urgency, ("", "#6b7280"))
    n = len(assessment.signals)
    return (f'<span style="color:{color};font-weight:bold;">'
            f'{icon} {n} sig — {assessment.recommendation}</span>')

def render_exit_panel(result: dict):
    """Render full exit monitor panel if there are any exit assessments."""
    exit_assessments = result.get("exit_assessments", {})
    if not exit_assessments:
        st.info("No open positions to monitor.")
        return

    # Build name lookup from trade states
    trade_states = result.get("trade_states", {})
    _names = {t: ts.name for t, ts in trade_states.items() if hasattr(ts, "name")}

    st.subheader("Exit Monitor")
    for ticker, ea in exit_assessments.items():
        if not ea.signals:
            continue
        icon, color = _URGENCY_STYLE.get(ea.urgency, ("", "#6b7280"))
        name = _names.get(ticker, "")
        label = f"{ticker} ({name})" if name else ticker
        st.markdown(
            f"**{label}** — {icon} {ea.recommendation} "
            f"({len(ea.signals)} signal{'s' if len(ea.signals) > 1 else ''})",
        )
        for sig in ea.signals:
            st.caption(f"  \u2022 {sig.signal_type.value}: {sig.description}")
