from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.cm import tab20
from matplotlib.colors import to_hex

from .analysis import METRIC_LABELS
from .io.config import Condition
from .util import (
    rgba,
    post_seg,          # returns two points for dotted helper
    yld_seg,           # returns two points for dashed helper
)

# ───────────────────────────── palette ──────────────────────────────
_SAFE_HEX: List[str] = [
    "#3E79F7", "#F75F00", "#8E44AD", "#2ECC71",
    "#F1C40F", "#16A085", "#E74C3C", "#34495E",
]


def _palette_hex(n: int) -> List[str]:
    """Return *n* distinct colours, colour-blind-safe first, then tab20."""
    cols = _SAFE_HEX.copy()
    while len(cols) < n:
        idx = len(cols)
        cols.append(to_hex(tab20(idx / 20)))
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
    palette = _palette_hex(len(conditions))
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




# ────────────────────────────────────────────────────────────────────
# Make 3 × 2 violin grid — identical to the original notebook
def make_violin_grid(
    summary_df: pd.DataFrame,
    conds: list[Condition],
    *,
    stacked: bool = False,
    legend_labels: dict[str, str] | None = None,
    legend_position: str = "bottom",
) -> go.Figure:
    """
    Grid of metric distributions – visually identical to the Jupyter notebook by default.

    Parameters
    ----------
    summary_df
        Tidy DataFrame of metrics with a ``Condition`` column matching ``conds``.
    conds
        Ordered list of conditions (legend order follows this).
    stacked
        When ``True``, render one metric per row (single-column layout).
    legend_labels
        Optional mapping to override legend labels per condition name.
    legend_position
        Where to position the legend. ``"bottom"`` matches the original notebook;
        ``"right"`` moves it to a vertical block to the right of the plots.
    """
    # ---- palette identical to notebook ------------------------------------
    metrics = METRIC_LABELS
    n_rows, n_cols = (len(metrics), 1) if stacked else (3, 2)

    cond_names = [c.name for c in conds]
    palette = _palette_hex(len(cond_names))
    color_lut = dict(zip(cond_names, palette))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(metrics.values()),
        horizontal_spacing=0.11 if not stacked else 0.08,
        vertical_spacing=0.13 if not stacked else 0.09,
        shared_xaxes=True,
    )

    # ---- violin traces ----------------------------------------------------
    for idx, (key, ttl) in enumerate(metrics.items()):
        r, c = divmod(idx, n_cols)
        r += 1
        c += 1
        for cond in cond_names:
            vals = summary_df[summary_df["Condition"] == cond][key]
            fig.add_trace(
                go.Violin(
                    y=vals,
                    name=cond,
                    legendgroup=cond,
                    showlegend=False,
                    line_color=color_lut[cond],
                    fillcolor=rgba(color_lut[cond], 0.18),
                    meanline_visible=True,
                    points="all",
                    width=0.6,
                    pointpos=0,
                ),
                row=r,
                col=c,
            )
        fig.update_yaxes(title_text=ttl, row=r, col=c)
        fig.update_xaxes(visible=False, row=r, col=c)

    # ---- consolidated legend (markers only) ------------------------------
    for cond, col_hex in color_lut.items():
        legend_name = legend_labels.get(cond, cond) if legend_labels else cond
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=col_hex),
                name=legend_name,
                legendgroup=cond,
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # ---- layout identical to notebook ------------------------------------
    control_name = next(c.name for c in conds if c.is_control)
    height = 300 * n_rows if stacked else 950
    width = 1250 if legend_position == "right" else 1150
    margin_bottom = 230 if stacked else 180
    margin_right = 220 if legend_position == "right" else 40
    legend_y = -0.22 if stacked else -0.18

    legend_cfg = dict(
        title="Conditions",
        bgcolor="rgba(255,255,255,0.92)",
        borderwidth=1,
        itemsizing="constant",
        itemwidth=180,
        tracegroupgap=10,
        font=dict(size=12),
    )

    if legend_position == "right":
        margin_bottom = 120 if stacked else 120
        legend_cfg.update(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        )
    else:
        legend_cfg.update(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
        )
    fig.update_layout(
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(t=90, l=70, r=margin_right, b=margin_bottom),
        title=dict(
            text=f"<b>Mechanical metric distributions</b><br>"
                 f"<sup>Control = {control_name}</sup>",
            x=0.5,
            y=0.985,
            xanchor="center",
            font=dict(size=18),
        ),
        legend=legend_cfg,
    )

    return fig
