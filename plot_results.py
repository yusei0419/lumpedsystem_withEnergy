"""
Render the simulation results as an interactive Plotly HTML.

Layout:
    - Two side-by-side D1-D2 phase diagrams (series, parallel)
    - Trajectories: black lines; initial points: open white circles;
      end points: green squares.
    - A slider over lambda values toggles which dataset is shown.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


_INIT_MARK = dict(color="white", size=8, symbol="circle",
                  line=dict(color="black", width=1.2))
_END_MARK  = dict(color="#2ca02c", size=9, symbol="square",
                  line=dict(color="black", width=0.8))
_LINE      = dict(color="black", width=1.0)


def _traces_for_dataset(records: list[dict[str, Any]], col: int) -> list[go.Scatter]:
    """Build trajectory + initial + end traces for one (mode, lambda) dataset."""
    traj_x, traj_y = [], []
    init_x, init_y = [], []
    end_x, end_y = [], []
    for r in records:
        d1 = r["D1"]; d2 = r["D2"]
        traj_x += d1 + [None]
        traj_y += d2 + [None]
        init_x.append(d1[0]); init_y.append(d2[0])
        end_x.append(d1[-1]); end_y.append(d2[-1])

    traces = [
        go.Scatter(x=traj_x, y=traj_y, mode="lines", line=_LINE,
                   hoverinfo="skip", showlegend=False, xaxis=f"x{col}", yaxis=f"y{col}"),
        go.Scatter(x=init_x, y=init_y, mode="markers", marker=_INIT_MARK,
                   name="initial", showlegend=(col == 1),
                   hovertemplate="D1_0=%{x:.2f}<br>D2_0=%{y:.2f}<extra></extra>",
                   xaxis=f"x{col}", yaxis=f"y{col}"),
        go.Scatter(x=end_x, y=end_y, mode="markers", marker=_END_MARK,
                   name="end", showlegend=(col == 1),
                   hovertemplate="D1_end=%{x:.3f}<br>D2_end=%{y:.3f}<extra></extra>",
                   xaxis=f"x{col}", yaxis=f"y{col}"),
    ]
    return traces


def build_phase_html(meta: dict[str, Any], out_path: str) -> None:
    lambda_values = meta["lambda_values"]
    results = meta["results"]
    params = meta["params"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Series  (force control, F0={params['F0']})",
            f"Parallel (force control, F_total=2*F0={2*params['F0']})",
        ),
        horizontal_spacing=0.12,
    )

    # Build all (lambda, mode) trace blocks; remember which traces belong to which lambda
    trace_blocks: list[tuple[float, list[int]]] = []  # (lam, [trace indices])
    for lam in lambda_values:
        idx_list = []
        # series -> col 1
        for tr in _traces_for_dataset(results["series"][f"{lam}"], col=1):
            fig.add_trace(tr, row=1, col=1)
            idx_list.append(len(fig.data) - 1)
        # parallel -> col 2
        for tr in _traces_for_dataset(results["parallel"][f"{lam}"], col=2):
            fig.add_trace(tr, row=1, col=2)
            idx_list.append(len(fig.data) - 1)
        trace_blocks.append((lam, idx_list))

    # Default: only first lambda visible
    n_total = len(fig.data)
    visibility_default = [False] * n_total
    for i in trace_blocks[0][1]:
        visibility_default[i] = True
    for i, vis in enumerate(visibility_default):
        fig.data[i].visible = vis

    # Slider
    steps = []
    for k, (lam, idxs) in enumerate(trace_blocks):
        vis = [False] * n_total
        for i in idxs:
            vis[i] = True
        steps.append(dict(
            method="update",
            args=[{"visible": vis},
                  {"title.text": _title(params, lam)}],
            label=f"λ={lam}",
        ))

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "lambda: "},
        pad={"t": 40},
        steps=steps,
    )]

    # Buttons (alternative way to switch lambda)
    buttons = []
    for k, (lam, idxs) in enumerate(trace_blocks):
        vis = [False] * n_total
        for i in idxs:
            vis[i] = True
        buttons.append(dict(
            label=f"λ = {lam}",
            method="update",
            args=[{"visible": vis},
                  {"title.text": _title(params, lam)}],
        ))

    fig.update_layout(
        title=_title(params, lambda_values[0]),
        sliders=sliders,
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.5, xanchor="center", y=1.13, yanchor="top",
            buttons=buttons, showactive=True,
        )],
        width=1100, height=620,
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=130, b=90),
    )

    for col in (1, 2):
        fig.update_xaxes(title_text="D1", range=[-0.02, 1.02],
                         showgrid=True, gridcolor="#eee",
                         zeroline=False, mirror=True, ticks="outside",
                         row=1, col=col)
        fig.update_yaxes(title_text="D2", range=[-0.02, 1.02],
                         showgrid=True, gridcolor="#eee",
                         zeroline=False, mirror=True, ticks="outside",
                         scaleanchor=f"x{col}", scaleratio=1.0,
                         row=1, col=col)

    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


def _title(params: dict[str, Any], lam: float) -> str:
    return (
        f"Two-bar bone remodeling — phase portrait (D1, D2) "
        f"&nbsp;|&nbsp; λ = {lam} "
        f"&nbsp;|&nbsp; α={params['alpha']}, b={params['b']}, "
        f"E0={params['E0']}, F0={params['F0']}"
    )
