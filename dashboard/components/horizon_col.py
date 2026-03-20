"""Horizon pattern column rendering for sector/industry tables."""
import streamlit as st
from engine.schemas import HorizonPattern

# Pattern -> short display label + color
_PATTERN_DISPLAY = {
    HorizonPattern.FULL_CONFIRM: ("↑↑↑ Conf", "#16a34a"),     # green
    HorizonPattern.ROTATION_IN: ("↑↑↓ RotIn", "#2563eb"),    # blue
    HorizonPattern.ROTATION_OUT: ("↓↓↑ RotOut", "#ea580c"),   # orange
    HorizonPattern.FULL_REJECT: ("↓↓↓ Rej", "#dc2626"),      # red
    HorizonPattern.DEAD_CAT: ("↑↓↓ TRAP!", "#b91c1c"),       # dark red
    HorizonPattern.HEALTHY_DIP: ("↓↑↑ DIP", "#059669"),      # teal
    HorizonPattern.NO_PATTERN: ("— —", "#6b7280"),            # gray
}

def horizon_badge_html(pattern: HorizonPattern) -> str:
    """Return HTML span with colored badge for a horizon pattern."""
    label, color = _PATTERN_DISPLAY.get(pattern, ("—", "#6b7280"))
    return f'<span style="color:{color};font-weight:bold;font-size:12px;">{label}</span>'

def get_horizon_label(pattern: HorizonPattern) -> str:
    """Return plain text label for a horizon pattern."""
    label, _ = _PATTERN_DISPLAY.get(pattern, ("—", "#6b7280"))
    return label

def get_horizon_color(pattern: HorizonPattern) -> str:
    """Return color hex for a horizon pattern."""
    _, color = _PATTERN_DISPLAY.get(pattern, ("—", "#6b7280"))
    return color
