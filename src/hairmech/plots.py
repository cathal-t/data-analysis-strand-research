from __future__ import annotations

from typing import Sequence, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.cm import tab20

from .io.config import Condition
from .util import (
    rgba,
    hex_to_rgb01,
    post_seg,          # returns two points for dotted helper
    yld_seg,           # returns two points for dashed helper
)

# ───────────────────────────── palette ──────────────────────────────
_SAFE: List[str] = [
    "#3E79F7", "#F75F00", "#8E44AD", "#2ECC71",
    "#F1C40F", "#16A085", "#E74C3C", "#34495E",
]


def _palette(n: int) -> List[str]:
    """Return *n* distinct colours, colour-blind-safe first, then tab20."""
    cols = _SAFE.copy()
    while len(cols) < n:
        i = len(cols)
        cols.append(rgba(tab20(i / 20)[:3]))
    return cols[:n]


# ────────────────────────── overlay figure ──────────────────────────
def make_overlay(
    curves_df: pd.DataFrame,
    conditions: Sequence[Condition],
) -> go.Figure:
    """
    Parameters
    ----------
    curves_df
        Tidy DataFrame with columns **Slot · Condition · Strain · Stress_MPa**.
        (Stress already converted to MPa.)
    conditions
        List from ``load_config`` – gives names & counts for legends/colours.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    palette = _palette(len(conditions))
    colour: Dict[str, str] = {c.name: palette[i] for i, c in enumerate(conditions)}

    fig = go.Figure()

    for slot, grp in curves_df.groupby("Slot"):
        cond = grp["Condition"].iat[0]

        # main curve --------------------------------------------------------
        fig.add_trace(
            go.Scattergl(
                x=grp["Strain"],
                y=grp["Stress_MPa"],
                mode="lines",
                line=dict(color=colour[cond], width=1),
                name=f"{cond} · slot {slot}",
                legendgroup=cond,
                hovertemplate=(
                    f"{cond}<br>slot {slot}<br>"
                    "ε=%{x:.3f}<br>σ=%{y:.2f} MPa"
                ),
            )
        )

        # helper lines ------------------------------------------------------
        proc = pd.DataFrame(
            {
                "strain": grp["Strain"],
                "stress_Pa": grp["Stress_MPa"] * 1e6,
            }
        )
        for seg_fn, dash in [(post_seg, "dot"), (yld_seg, "dash")]:
            pts = seg_fn(proc)
            if not pts:
                continue
            (x0, y0), (x1, y1) = pts
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0 / 1e6, y1 / 1e6],
                    mode="lines",
                    showlegend=False,
                    legendgroup=cond,
                    line=dict(color=colour[cond], width=2, dash=dash),
                    hoverinfo="skip",
                )
            )

    # layout ---------------------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=60, r=220, t=110, b=50),
        xaxis=dict(title="Engineering strain ε"),
        yaxis=dict(title="Engineering stress (MPa)", ticksuffix=" MPa"),
        title=dict(
            text="<b>Stress–strain overlay</b>",
            x=0.005, y=0.97, xanchor="left", yanchor="top",
        ),
        annotations=[
            dict(
                text=f"{len(conditions)} conditions · "
                     f"{curves_df['Slot'].nunique()} fibres",
                x=0.005, y=0.915, xanchor="left", yanchor="top",
                showarrow=False, font=dict(size=12, color="#444"),
            )
        ],
        legend=dict(
            title="Curve traces",
            y=1, x=1.02, yanchor="top", xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=1,
            tracegroupgap=5,
        ),
    )
    return fig


# ───────────────────── 3 × 2 violin-grid figure ──────────────────────
_METRICS: Dict[str, str] = {
    "Elastic_Modulus_GPa":     "Elastic modulus (GPa)",
    "Yield_Gradient_MPa_perc": "Yield-grad. (MPa / %ε)",
    "Post_Gradient_MPa_perc":  "Post-grad. (MPa / %ε)",
    "Break_Stress_MPa":        "Break stress (MPa)",
    "Break_Strain_%":          "Break strain (%)",
}


def make_violin_grid(
    summary_df: pd.DataFrame,
    conditions: Sequence[Condition],
) -> go.Figure:
    """
    Parameters
    ----------
    summary_df
        Per-slot summary table from ``analysis.build_summary``.
    conditions
        Same list – used for legend colouring.
    """
    palette = _palette(len(conditions))
    colour = {c.name: palette[i] for i, c in enumerate(conditions)}

    N_ROWS, N_COLS = 3, 2
    cat = make_subplots(
        rows=N_ROWS,
        cols=N_COLS,
        subplot_titles=list(_METRICS.values()),
        horizontal_spacing=0.11,
        vertical_spacing=0.13,
        shared_xaxes=True,
    )

    # violin traces --------------------------------------------------------
    for idx, (key, nice) in enumerate(_METRICS.items()):
        r, c = divmod(idx, N_COLS)
        r, c = r + 1, c + 1
        for cond in conditions:
            vals = summary_df[summary_df["Condition"] == cond.name][key]
            cat.add_trace(
                go.Violin(
                    y=vals,
                    name=cond.name,
                    legendgroup=cond.name,
                    showlegend=False,
                    line_color=colour[cond.name],
                    fillcolor=rgba(hex_to_rgb01(colour[cond.name][1:]), 0.18),
                    meanline_visible=True,
                    points="all",
                    width=0.6,
                    pointpos=0,
                ),
                row=r,
                col=c,
            )
        cat.update_yaxes(title_text=nice, row=r, col=c)
        cat.update_xaxes(visible=False, row=r, col=c)

    # consolidated legend (markers only) ----------------------------------
    for cond in conditions:
        cat.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=colour[cond.name]),
                name=cond.name,
                legendgroup=cond.name,
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # layout --------------------------------------------------------------
    cat.update_layout(
        template="plotly_white",
        height=900,
        width=1050,
        margin=dict(t=100, l=70, r=170, b=70),
        title=dict(
            text="<b>Mechanical metric distributions</b>",
            x=0.5, y=0.98, xanchor="center", font=dict(size=18),
        ),
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=1,
            itemsizing="constant",
            tracegroupgap=4,
        ),
    )
    return cat