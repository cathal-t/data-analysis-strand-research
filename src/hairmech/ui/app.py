"""
hairmech.ui.app
===============

Dash front-end for interactive hair-mechanical analysis.

What’s new (July 2025)
----------------------
* **Control?** column is a click-to-toggle ✓ tick (only one row allowed).
* Two filename inputs + buttons let the user download:
  • metrics.xlsx  – per-slot summary (build_summary)  
  • stats.xlsx    – Welch-t & effect sizes (build_stats → long_to_wide)
  The files are generated in memory and streamed via dcc.Download.
* A simple `run_hairmech.bat` can call:  `python -m hairmech.ui.app`
  to launch the server and open the browser.
"""

from __future__ import annotations

import base64
import json
import tempfile
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..analysis import build_summary, build_stats, long_to_wide
from ..dimensional import DimensionalData
from ..io.config import Condition, ConfigError, load_config
from ..plots import make_overlay, make_violin_grid
from ..tensile import TensileTest

TICK = "✓"
EMPTY = ""

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


def _to_excel_bytes(df_dict: dict[str, pd.DataFrame]) -> bytes:
    """Helper: write several DataFrames to one Excel file → bytes."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xls:
        for sheet, df in df_dict.items():
            df.to_excel(xls, sheet_name=sheet, index=False)
    buf.seek(0)
    return buf.getvalue()


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

            # editable DataTable
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
                style_data_conditional=[{
                    "if": {"column_id": "is_control"},
                    "textAlign": "center",
                }],
            ),

            dbc.Button("Apply & plot", id="btn-apply", color="primary", className="my-3"),

            # download controls ------------------------------------------------
            dbc.Row(
                [
                    dbc.Col(dbc.Input(id="metrics-name", value="metrics.xlsx",
                                      placeholder="metrics.xlsx", type="text", size="md"), width="auto"),
                    dbc.Col(dbc.Button("Download Metrics", id="btn-dl-metrics", color="info"), width="auto"),
                    dbc.Col(width=2),  # spacer
                    dbc.Col(dbc.Input(id="stats-name", value="stats.xlsx",
                                      placeholder="stats.xlsx", type="text", size="md"), width="auto"),
                    dbc.Col(dbc.Button("Download Stats", id="btn-dl-stats", color="info"), width="auto"),
                ],
                className="align-items-center mb-3",
            ),
            dcc.Download(id="dl-metrics"),
            dcc.Download(id="dl-stats"),

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

            dcc.Store(id="exp-data"),
        ],
        fluid=True,
    )

    # ─────────────── callbacks ───────────────
    # auto-fill table
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

    # add / delete rows
    @app.callback(
        Output("cond-table", "data", allow_duplicate=True),
        Input("btn-add", "n_clicks"),
        State("cond-table", "data"),
        prevent_initial_call=True,
    )
    def _add_row(_, rows):
        rows.append({"name": "", "slot_start": "", "slot_end": "", "is_control": EMPTY})
        return rows

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

    # toggle tick
    @app.callback(
        Output("cond-table", "data"),
        Input("cond-table", "active_cell"),
        State("cond-table", "data"),
        prevent_initial_call=True,
    )
    def _toggle_tick(cell, rows):
        if not cell or cell.get("column_id") != "is_control":
            raise PreventUpdate
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

    # build figure
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

    # download metrics
    @app.callback(
        Output("dl-metrics", "data"),
        Input("btn-dl-metrics", "n_clicks"),
        State("metrics-name", "value"),
        State("exp-data", "data"),
        prevent_initial_call=True,
    )
    def _download_metrics(_, fname, payload):
        if not payload:
            raise PreventUpdate
        data = json.loads(payload)
        areas = _bytes_to_dim(_b64_to_bytes(data["dim_b64"]))
        tensile = _bytes_to_ten(_b64_to_bytes(data["ten_b64"]))
        conds = [Condition(**c) for c in data["conds"]]
        df = build_summary(areas, tensile, conds)
        bytes_out = _to_excel_bytes({"Metrics": df})
        return dcc.send_bytes(bytes_out, fname or "metrics.xlsx")

    # download stats
    @app.callback(
        Output("dl-stats", "data"),
        Input("btn-dl-stats", "n_clicks"),
        State("stats-name", "value"),
        State("exp-data", "data"),
        prevent_initial_call=True,
    )
    def _download_stats(_, fname, payload):
        if not payload:
            raise PreventUpdate

        data = json.loads(payload)
        areas   = _bytes_to_dim(_b64_to_bytes(data["dim_b64"]))
        tensile = _bytes_to_ten(_b64_to_bytes(data["ten_b64"]))
        conds   = [Condition(**c) for c in data["conds"]]

        # ➊ build per-slot summary
        summary = build_summary(areas, tensile, conds)

        # ➋ metric columns = everything except Slot / Condition
        from collections import OrderedDict
        metric_cols = [c for c in summary.columns if c not in ("Slot", "Condition")]
        metrics_od  = OrderedDict((c, c.replace("_", " ")) for c in metric_cols)

        # ➌ Welch-t / Cohen-d table
        long  = build_stats(summary, conds, metrics_od)
        wide  = long_to_wide(long)

        bytes_out = _to_excel_bytes({"Stats": wide})
        return dcc.send_bytes(bytes_out, fname or "stats.xlsx")
        return app


# module-level instance (importable by tests or BAT launcher)
app: Dash = build_dash_app()
__all__ = ["build_dash_app", "app"]
