"""
hairmech.ui.app
===============

Dash front-end for interactive hair-mechanical analysis.

Update
------
* *Control?* column is now a **click-to-toggle tick box** (✓ = control).
* Only one row can hold the tick; clicking another row moves it.
"""

from __future__ import annotations

import base64
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..analysis import build_summary
from ..dimensional import DimensionalData
from ..io.config import Condition, ConfigError, load_config
from ..plots import make_overlay, make_violin_grid
from ..tensile import TensileTest

TICK = "✓"        # unicode tick mark
EMPTY = ""        # blank for non-control rows


# ───────────────────────── helper I/O ──────────────────────────
def _load_experiment(
    root: Path,
) -> Tuple[dict[int, float], TensileTest, List[Condition]]:
    conds = load_config(root)
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


# ───────────────────────── utilities ──────────────────────────
def _demo_exp_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "tests" / "fixtures" / "demo_exp"
        if cand.exists():
            return cand
    raise FileNotFoundError("Could not locate demo_exp fixture.")


def _b64_to_bytes(content: str) -> bytes:
    return base64.b64decode(content.partition(",")[2])


def _bytes_to_dim(raw: bytes) -> dict[int, float]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(raw)
        tmp.flush()
        return DimensionalData(Path(tmp.name)).map


def _bytes_to_ten(raw: bytes) -> TensileTest:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(raw)
        tmp.flush()
        return TensileTest(Path(tmp.name))


def _max_slot(areas: dict[int, float], tensile: TensileTest) -> int:
    return max(
        max(areas, default=0),
        max((slot for slot, _ in tensile.per_slot()), default=0),
    )


# ───────────────────── Dashboard factory ──────────────────────
def build_dash_app(root_dir: str | Path | None = None) -> Dash:
    root = Path(root_dir) if root_dir else _demo_exp_path()
    default_areas, default_tensile, default_conds = _load_experiment(root)

    app = Dash(
        __name__,
        title="Hair-mech",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # ───────────── layout ─────────────
    app.layout = dbc.Container(
        [
            html.H3("Hair-mech – interactive viewer", className="my-3"),
            # uploads
            dbc.Row(
                [
                    dbc.Col(dcc.Upload(id="upload-dim",
                                       children=dbc.Button("Upload Dimensional_Data.txt"),
                                       multiple=False),
                            width="auto"),
                    dbc.Col(dcc.Upload(id="upload-ten",
                                       children=dbc.Button("Upload Tensile_Data.txt"),
                                       multiple=False),
                            width="auto"),
                ],
                className="mb-3",
            ),
            # table control buttons
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Add condition", id="btn-add", color="secondary"), width="auto"),
                    dbc.Col(dbc.Button("Delete selected", id="btn-del", color="danger"), width="auto"),
                ],
                className="mb-2",
            ),
            # editable table
            dash_table.DataTable(
                id="cond-table",
                columns=[
                    dict(name="Condition name", id="name",       editable=True, type="text"),
                    dict(name="Slot start",     id="slot_start", editable=True, type="numeric"),
                    dict(name="Slot end",       id="slot_end",   editable=True, type="numeric"),
                    dict(name="Control?",       id="is_control", editable=False,
                         type="text", presentation="markdown"),
                ],
                data=[{
                    "name": "Condition 1",
                    "slot_start": 1,
                    "slot_end": 1,
                    "is_control": TICK,
                }],
                editable=True,
                row_selectable="multi",
                row_deletable=True,
                style_table={"overflowX": "auto"},
                style_data_conditional=[
                    {
                        "if": {"column_id": "is_control"},
                        "textAlign": "center",
                    }
                ],
            ),
            dbc.Button("Apply & plot", id="btn-apply", color="primary", className="my-3"),
            # figure selector
            dcc.Dropdown(
                id="tabs",
                options=[{"label": "Overlay", "value": "overlay"},
                         {"label": "Violin grid", "value": "violin"}],
                value="overlay",
                clearable=False,
                style={"width": "250px"},
            ),
            html.Div(id="fig-container", className="mt-4"),
            # caches
            dcc.Store(id="exp-data"),
        ],
        fluid=True,
    )

    # ─────────────── callbacks ───────────────
    # auto-populate table after uploads
    @app.callback(
        Output("cond-table", "data", allow_duplicate=True),
        Input("upload-dim", "contents"),
        Input("upload-ten", "contents"),
        prevent_initial_call=True,
    )
    def _prime_table(dim_b64, ten_b64):
        if not dim_b64 or not ten_b64:
            raise PreventUpdate
        areas = _bytes_to_dim(_b64_to_bytes(dim_b64))
        tensile = _bytes_to_ten(_b64_to_bytes(ten_b64))
        return [{
            "name": "Condition 1",
            "slot_start": 1,
            "slot_end": _max_slot(areas, tensile),
            "is_control": TICK,
        }]

    # add blank row
    @app.callback(
        Output("cond-table", "data", allow_duplicate=True),
        Input("btn-add", "n_clicks"),
        State("cond-table", "data"),
        prevent_initial_call=True,
    )
    def _add_row(_, rows):
        rows.append({"name": "", "slot_start": "", "slot_end": "", "is_control": EMPTY})
        return rows

    # delete selected rows
    @app.callback(
        Output("cond-table", "data", allow_duplicate=True),
        Input("btn-del", "n_clicks"),
        State("cond-table", "data"),
        State("cond-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def _del_rows(_, rows, sel):
        if not sel:
            raise PreventUpdate
        return [row for i, row in enumerate(rows) if i not in sel]

    # toggle tick on single click
    @app.callback(
        Output("cond-table", "data"),
        Input("cond-table", "active_cell"),
        State("cond-table", "data"),
        prevent_initial_call=True,
    )
    def _toggle_tick(cell, rows):
        if not cell or cell.get("column_id") != "is_control":
            raise PreventUpdate
        # remove tick everywhere, add to clicked row
        for r in rows:
            r["is_control"] = EMPTY
        rows[cell["row"]]["is_control"] = TICK
        return rows

    # validate + cache
    @app.callback(
        Output("exp-data", "data"),
        Input("btn-apply", "n_clicks"),
        State("cond-table", "data"),
        State("upload-dim", "contents"),
        State("upload-ten", "contents"),
        prevent_initial_call=True,
    )
    def _cache(_, rows, dim_b64, ten_b64):
        if not dim_b64 or not ten_b64:
            raise PreventUpdate

        conds, seen = [], set()
        for row in rows:
            name = (row["name"] or "").strip() or "(unnamed)"
            try:
                s0, s1 = int(row["slot_start"]), int(row["slot_end"])
            except ValueError:
                raise ConfigError(f"{name}: slot numbers must be integers")
            if s0 < 1 or s1 < 1 or s0 > s1:
                raise ConfigError(f"{name}: invalid slot range {s0}–{s1}")
            rng = range(s0, s1 + 1)
            if seen & set(rng):
                raise ConfigError(f"{name}: overlapping slot ranges")
            seen.update(rng)
            conds.append(
                Condition(name=name, slots=list(rng), is_control=row["is_control"] == TICK)
            )

        if sum(c.is_control for c in conds) != 1:
            raise ConfigError("Exactly one row must have the ✓ tick (control)")

        return json.dumps({
            "dim_b64": dim_b64,
            "ten_b64": ten_b64,
            "conds": [asdict(c) for c in conds],
        })

    # draw figure
    @app.callback(
        Output("fig-container", "children"),
        Input("tabs", "value"),
        Input("exp-data", "data"),
    )
    def _draw(tab, payload):
        if payload:
            data = json.loads(payload)
            areas = _bytes_to_dim(_b64_to_bytes(data["dim_b64"]))
            tensile = _bytes_to_ten(_b64_to_bytes(data["ten_b64"]))
            conds = [Condition(**c) for c in data["conds"]]
        else:
            areas, tensile, conds = default_areas, default_tensile, default_conds

        fig = _overlay_fig(areas, tensile, conds) if tab == "overlay" else _violin_fig(areas, tensile, conds)
        return dcc.Graph(figure=fig, style={"height": "750px"})

    return app


# module-level instance
app: Dash = build_dash_app()
__all__ = ["build_dash_app", "app"]
