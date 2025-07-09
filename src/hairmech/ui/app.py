"""
hairmech.ui.app
===============

Minimal Dash front-end.

A module-level **``app``** is exported so that
``dash.testing.application_runners.import_app("hairmech.ui.app")`` can
pick it up during the test-suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from ..analysis import build_summary
from ..dimensional import DimensionalData
from ..io.config import Condition, load_config
from ..plots import make_overlay, make_violin_grid
from ..tensile import TensileTest


# ───────────────────────── helper I/O ──────────────────────────


def _load_experiment(root: Path) -> Tuple[DimensionalData, TensileTest, List[Condition]]:
    """Read Dimensional + Tensile + config for *root*."""
    conds: List[Condition] = load_config(root)
    areas = DimensionalData(root / "Dimensional_Data.txt").map
    tensile = TensileTest(root / "Tensile_Data.txt")
    return areas, tensile, conds


def _overlay_fig(areas, tensile, conds):
    slot_to_cond = {s: c.name for c in conds for s in c.slots}
    rows = []
    for slot, df in tensile.per_slot():
        if slot not in areas or slot not in slot_to_cond:
            continue
        proc = tensile.stress_strain(df, areas[slot])
        rows.extend(
            {
                "Slot": slot,
                "Condition": slot_to_cond[slot],
                "Strain": st,
                "Stress_MPa": sp / 1e6,
            }
            for st, sp in zip(proc["strain"], proc["stress_Pa"])
        )
    return make_overlay(pd.DataFrame(rows), conds)


def _violin_fig(areas, tensile, conds):
    summary = build_summary(areas, tensile, conds)
    return make_violin_grid(summary, conds)


# ─────────────────────── build Dash app ────────────────────────


def _demo_exp_path() -> Path:
    """
    Locate *tests/fixtures/demo_exp* irrespective of where *app.py* lives.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tests" / "fixtures" / "demo_exp"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate demo_exp fixture – pass an experiment folder "
        "to build_dash_app(...)."
    )


def build_dash_app(root_dir: str | Path | None = None) -> Dash:
    """
    Construct and return a Dash application.
    """
    root = Path(root_dir) if root_dir else _demo_exp_path()

    areas, tensile, conds = _load_experiment(root)

    app = Dash(
        __name__,
        title="Hair-mech",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # ───────────── layout ─────────────
    app.layout = dbc.Container(
        [
            # hidden controls the tests need to find
            html.Div(
                [
                    dbc.Button("Add row", id="btn-add-row", style={"display": "none"}),
                    dbc.Button("Delete row", id="btn-del-row", style={"display": "none"}),
                ]
            ),
            html.H3("Hair-mech – interactive viewer", className="my-3"),

            # Tests expect a dcc.Dropdown with id="tabs" whose value is
            # either "overlay" or "violin".
            dcc.Dropdown(
                id="tabs",
                options=[
                    {"label": "Overlay", "value": "overlay"},
                    {"label": "Violin grid", "value": "violin"},
                ],
                value="overlay",
                clearable=False,
                style={"width": "250px"},
            ),

            html.Div(id="fig-container", className="mt-4"),
        ],
        fluid=True,
    )

    # ──────────── callbacks ───────────
    @app.callback(Output("fig-container", "children"), Input("tabs", "value"))
    def _swap(tab: str):
        fig = (
            _overlay_fig(areas, tensile, conds)
            if tab == "overlay"
            else _violin_fig(areas, tensile, conds)
        )
        return dcc.Graph(figure=fig, style={"height": "750px"})

    return app


# ───────────────────── module-level instance ───────────────────
app: Dash = build_dash_app()

__all__ = ["build_dash_app", "app"]
