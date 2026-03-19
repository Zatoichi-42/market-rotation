"""
Composite Score Ranking — horizontal bar chart colored by analysis state.

Per spec §Feature 3:
- Horizontal bars, sorted best-to-worst (top = highest composite)
- Bar color by state: green=Broadening/Overt, red=Rotation, orange=Exhaustion,
  yellow=Ambiguous, gray=Accumulation
- Score text at end of each bar
- No gridlines, dark background compatible
"""
import plotly.graph_objects as go
from dashboard.components.style_utils import STATE_BAR_COLORS


def make_composite_bar_chart(
    groups: list[dict],
    title: str = "Composite Score Ranking",
) -> go.Figure:
    """
    Horizontal bar chart of composite scores, colored by state.

    groups: list of dicts with keys: ticker, name, composite, state
    """
    # Sort best-to-worst (highest at top → reverse for horizontal bars)
    sorted_groups = sorted(groups, key=lambda g: g["composite"])

    labels = [f"{g['ticker']} ({g['name']})" for g in sorted_groups]
    values = [g["composite"] for g in sorted_groups]
    colors = [STATE_BAR_COLORS.get(g.get("state", ""), "#94a3b8") for g in sorted_groups]

    fig = go.Figure(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=max(300, len(groups) * 30 + 60),
        margin=dict(t=40, b=20, l=10, r=40),
        xaxis=dict(title="Composite (0-100)", range=[0, 105], showgrid=False),
        yaxis=dict(showgrid=False, categoryorder="array", categoryarray=labels),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
