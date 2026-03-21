"""
RS Heat Map — diverging red/green heatmap with State column.

Per spec §Feature 4:
- Rows: sectors (then industries)
- Columns: 5d RS, 20d RS, 60d RS, (vs Parent for industries), State
- Color: diverging red→dark→green, centered on 0
- Cell text: RS value with 1 decimal
- State column: text, not heatmap-colored
- Sort: by 20d RS descending

Pattern reading:
  🟩🟩🟩 = Sustained leader (watch for exhaustion)
  🟥🟥🟩 = Former leader decaying (rotation OUT)
  🟩🟩🟥 = New leader emerging (rotation IN)
  🟥🟥🟥 = Persistent laggard
"""
import plotly.graph_objects as go
import numpy as np


def make_rs_heatmap(
    groups: list[dict],
    title: str = "RS Heat Map",
) -> go.Figure:
    """
    Build heatmap with RS values and State annotation.

    groups: list of dicts with keys:
        ticker, name, rs_5d, rs_20d, rs_60d,
        rs_vs_parent (None for sectors), state
    """
    if not groups:
        fig = go.Figure()
        fig.update_layout(title=title, height=100)
        return fig

    # Sort by 20d RS descending (leaders at top)
    sorted_g = sorted(groups, key=lambda g: g.get("rs_20d", 0), reverse=True)

    labels = [f"{g['ticker']} {g['name']}" for g in sorted_g]
    has_parent = any(g.get("rs_vs_parent") is not None for g in sorted_g)

    # Build column data
    has_1d = any(g.get("rs_1d") is not None and g.get("rs_1d", 0) != 0 for g in sorted_g)
    col_names = []
    if has_1d:
        col_names.append("1d RS")
    col_names.extend(["5d RS", "20d RS", "60d RS"])
    if has_parent:
        col_names.append("vs Parent")

    z = []
    text = []
    for g in sorted_g:
        row_z = []
        row_text = []
        if has_1d:
            v1d = g.get("rs_1d", 0)
            row_z.append(v1d * 100)
            row_text.append(f"{v1d*100:+.1f}%")
        row_z.extend([
            g.get("rs_5d", 0) * 100,
            g.get("rs_20d", 0) * 100,
            g.get("rs_60d", 0) * 100,
        ])
        row_text.extend([
            f"{g.get('rs_5d', 0)*100:+.1f}%",
            f"{g.get('rs_20d', 0)*100:+.1f}%",
            f"{g.get('rs_60d', 0)*100:+.1f}%",
        ])
        if has_parent:
            vp = g.get("rs_vs_parent")
            if vp is not None:
                row_z.append(vp * 100)
                row_text.append(f"{vp*100:+.1f}%")
            else:
                row_z.append(0)
                row_text.append("—")
        z.append(row_z)
        text.append(row_text)

    states = [g.get("state", "—") for g in sorted_g]

    # Build heatmap
    z_arr = np.array(z)
    max_abs = max(np.abs(z_arr).max(), 1.0)

    fig = go.Figure()

    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=z_arr,
        x=col_names,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=[
            [0.0, "#ef4444"],    # Red (negative)
            [0.5, "#1e293b"],    # Dark neutral (zero)
            [1.0, "#22c55e"],    # Green (positive)
        ],
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        showscale=False,  # Hide colorbar — values shown as text in cells
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))

    # Add State annotations on the right side
    for i, state in enumerate(states):
        fig.add_annotation(
            x=1.01, y=labels[i],
            xref="paper", yref="y",
            text=state,
            showarrow=False,
            font=dict(size=10, color="#94a3b8"),
            xanchor="left",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=max(300, len(groups) * 32 + 80),
        margin=dict(t=40, b=20, l=10, r=120),  # Extra right margin for State annotations
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(side="top", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
