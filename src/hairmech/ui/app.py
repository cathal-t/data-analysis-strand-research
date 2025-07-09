"""
hairmech.ui.app
===============

Interactive Dash front-end for hair-mechanical analysis.

Changes in this version
-----------------------
* Bootstrap cards for a cleaner, compact layout
* Upload buttons turn green and show the filename on success
* “Add condition” / “Delete selected” automatically rebalance slot
  ranges and name rows “Condition N”
* All previous functionality is preserved
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
from collections import OrderedDict
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

# ────────────── constants ──────────────
TICK = "✓"
EMPTY = ""
DEFAULT_RE = re.compile(r"Condition\s+\d+", re.I)

# ────────────── helper I/O ──────────────
def _load_experiment(root: Path) -> Tuple[dict[int, float], TensileTest, List[Condition]]:
    conds = load_config(root)
    areas = DimensionalData(root / "Dimensional_Data.txt").map
    tensile = TensileTest(root / "Tensile_Data.txt")
    return areas, tensile, conds


def _overlay_fig(areas, tensile, conds):
    slot_to_cond = {slot: c.name for c in conds for slot in c.slots}
    rows = []
    for slot, df in tensile.per_slot():
        if slot not in areas or slot not in slot_to_cond:
            continue
        proc = tensile.stress_strain(df, areas[slot])
        for st, sp in zip(proc["strain"], proc["stress_Pa"]):
            rows.append(
                {
                    "Slot": slot,
                    "Condition": slot_to_cond[slot],
                    "Strain": st,
                    "Stress_MPa": sp / 1e6,
                }
            )
    return make_overlay(pd.DataFrame(rows), conds)


def _violin_fig(areas, tensile, conds):
    summary = build_summary(areas, tensile, conds)
    return make_violin_grid(summary, conds)


def _demo_exp_path() -> Path:
    """Locate bundled demo data so the app can load without uploads."""
    here = Path(__file__).resolve()
    for p in here.parents:
        demo = p / "tests" / "fixtures" / "demo_exp"
        if demo.exists():
            return demo
    raise FileNotFoundError("demo_exp fixture not found")


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


def _to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Write multiple DataFrames to an in-memory Excel file.
    Keeps multi-level columns if present.
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xls:
        for sheet, df in sheets.items():
            df.to_excel(xls, sheet_name=sheet, index=isinstance(df.columns, pd.MultiIndex))
    buf.seek(0)
    return buf.getvalue()


# ────────────── slot rebalance ──────────────
def _rebalance_rows(rows: List[dict]):
    """
    Evenly re-split the total slot range across all rows in-place and
    set default names where appropriate.
    """
    total = max(int(r.get("slot_end") or 0) for r in rows)
    if total == 0:
        return
    base, extra = divmod(total, len(rows))
    start = 1
    for i, r in enumerate(rows):
        span = base + (extra if i == len(rows) - 1 else 0)
        r["slot_start"], r["slot_end"] = start, start + span - 1
        if not r.get("name") or DEFAULT_RE.fullmatch(str(r["name"])):
            r["name"] = f"Condition {i + 1}"
        start += span


# ───────────────────── Dash app ─────────────────────
def build_dash_app(root_dir: str | Path | None = None) -> Dash:
    root = Path(root_dir) if root_dir else _demo_exp_path()
    default_areas, default_tensile, default_conds = _load_experiment(root)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Hair-mech",
        suppress_callback_exceptions=True,
    )

    # ═══════════ LAYOUT ═══════════
    header = dbc.Row(dbc.Col(html.H3("Strand Research - Data Analysis"), width="auto"),
                     className="mt-2 mb-3")

    # Upload card
    upload_card = dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Upload(
                                id="upload-dim",
                                children=dbc.Button("Upload Dimensional Data",
                                                    color="primary", id="btn-dim"),
                                multiple=False,
                            ),
                            html.Small(id="dim-msg", className="text-muted"),
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        [
                            dcc.Upload(
                                id="upload-ten",
                                children=dbc.Button("Upload Tensile Data",
                                                    color="primary", id="btn-ten"),
                                multiple=False,
                            ),
                            html.Small(id="ten-msg", className="text-muted"),
                        ],
                        width="auto",
                    ),
                ],
                className="g-3 flex-wrap",
            )
        ),
        className="mb-3 shadow-sm",
    )

    # Condition editor
    cond_ctrls = dbc.Row(
        [
            dbc.Col(dbc.Button("Add condition", id="btn-add", color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Delete selected", id="btn-del", color="danger"), width="auto", className="ms-auto"),
        ],
        className="g-2 flex-wrap",
    )

    cond_table = dash_table.DataTable(
        id="cond-table",
        columns=[
            dict(name="Condition name", id="name", editable=True, type="text"),
            dict(name="Slot start", id="slot_start", editable=True, type="numeric"),
            dict(name="Slot end", id="slot_end", editable=True, type="numeric"),
            dict(name="Control?", id="is_control", editable=False,
                 type="text", presentation="markdown"),
        ],
        data=[{"name": "Condition 1", "slot_start": 1, "slot_end": 1, "is_control": TICK}],
        editable=True,
        row_selectable="multi",
        row_deletable=True,
        style_table={"overflowX": "auto", "minWidth": "100%"},
        style_header={"backgroundColor": "#f7f7f9"},
        style_data_conditional=[{"if": {"column_id": "is_control"}, "textAlign": "center"}],
    )

    cond_card = dbc.Card(
        [dbc.CardHeader("Conditions"), dbc.CardBody([cond_ctrls, cond_table])],
        className="mb-3 shadow-sm",
    )

    # Actions card
    apply_btn = dbc.Button("Apply & plot", id="btn-apply", color="primary")
    view_dd = dcc.Dropdown(
        id="tabs",
        options=[{"label": "Overlay", "value": "overlay"},
                 {"label": "Violin grid", "value": "violin"}],
        value="overlay",
        clearable=False,
        style={"width": "180px"},
    )

    download_row = dbc.Row(
        [
            dbc.Col(dbc.Input(id="metrics-name", value="metrics.xlsx", type="text"), width="auto"),
            dbc.Col(dbc.Button("Download Metrics", id="btn-dl-metrics", color="info"), width="auto"),
            dbc.Col(width=2),
            dbc.Col(dbc.Input(id="stats-name", value="stats.xlsx", type="text"), width="auto"),
            dbc.Col(dbc.Button("Download Stats", id="btn-dl-stats", color="info"), width="auto"),
        ],
        className="g-2 flex-wrap",
    )

    actions_card = dbc.Card(
        dbc.CardBody([apply_btn, html.Span(className="mx-2"), view_dd, html.Hr(), download_row]),
        className="mb-3 shadow-sm",
    )

    # Figure card
    fig_card = dbc.Card(dbc.CardBody(html.Div(id="fig-container")), className="shadow-sm")

    # Main container
    app.layout = dbc.Container(
        [
            header,
            upload_card,
            cond_card,
            actions_card,
            fig_card,
            # hidden stores / downloads
            dcc.Download(id="dl-metrics"),
            dcc.Download(id="dl-stats"),
            dcc.Store(id="exp-data"),
        ],
        fluid=True,
        style={"maxWidth": "1400px"},
    )

    # ═══════════ CALLBACKS ═══════════
    # Upload status colour + caption
    @app.callback(
        Output("btn-dim", "color"), Output("dim-msg", "children"),
        Input("upload-dim", "contents"), State("upload-dim", "filename"),
        prevent_initial_call=True,
    )
    def _dim_status(contents, filename):
        return ("success", f"Loaded: {filename}") if contents else ("primary", "")

    @app.callback(
        Output("btn-ten", "color"), Output("ten-msg", "children"),
        Input("upload-ten", "contents"), State("upload-ten", "filename"),
        prevent_initial_call=True,
    )
    def _ten_status(contents, filename):
        return ("success", f"Loaded: {filename}") if contents else ("primary", "")

    # Prime table once both files uploaded
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
        total = _max_slot(areas, tensile)
        return [{"name": "Condition 1", "slot_start": 1, "slot_end": total, "is_control": TICK}]

    # Add / delete rows with automatic rebalance
    @app.callback(
        Output("cond-table", "data", allow_duplicate=True),
        Input("btn-add", "n_clicks"),
        State("cond-table", "data"),
        prevent_initial_call=True,
    )
    def _add_row(_, rows):
        rows.append({"name": "", "slot_start": "", "slot_end": "", "is_control": EMPTY})
        _rebalance_rows(rows)
        return rows

    @app.callback(
        Output("cond-table", "data", allow_duplicate=True),
        Input("btn-del", "n_clicks"),
        State("cond-table", "data"),
        State("cond-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def _del_rows(_, rows, selected):
        if not selected:
            raise PreventUpdate
        rows = [r for i, r in enumerate(rows) if i not in selected]
        _rebalance_rows(rows)
        return rows

    # Toggle ✓
    @app.callback(
        Output("cond-table", "data"),
        Input("cond-table", "active_cell"),
        State("cond-table", "data"),
        prevent_initial_call=True,
    )
    def _toggle_tick(cell, rows):
        if not cell or cell["column_id"] != "is_control":
            raise PreventUpdate
        for r in rows:
            r["is_control"] = EMPTY
        rows[cell["row"]]["is_control"] = TICK
        return rows

    # Validate & cache experiment
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
        for r in rows:
            name = (r["name"] or "").strip() or "(unnamed)"
            try:
                s0, s1 = int(r["slot_start"]), int(r["slot_end"])
            except ValueError:
                raise ConfigError(f"{name}: slot numbers must be integers")
            if s0 < 1 or s1 < 1 or s0 > s1:
                raise ConfigError(f"{name}: invalid slot range {s0}–{s1}")
            rng = range(s0, s1 + 1)
            if seen & set(rng):
                raise ConfigError(f"{name}: overlapping slot ranges")
            seen.update(rng)
            conds.append(Condition(name, list(rng), r["is_control"] == TICK))

        if sum(c.is_control for c in conds) != 1:
            raise ConfigError("Mark exactly one ✓ row as control")

        return json.dumps(
            {"dim_b64": dim_b64, "ten_b64": ten_b64, "conds": [asdict(c) for c in conds]}
        )

    # Plot
    @app.callback(
        Output("fig-container", "children"),
        Input("tabs", "value"),
        Input("exp-data", "data"),
    )
    def _draw(tab, payload):
        if payload:
            d = json.loads(payload)
            areas = _bytes_to_dim(_b64_to_bytes(d["dim_b64"]))
            tensile = _bytes_to_ten(_b64_to_bytes(d["ten_b64"]))
            conds = [Condition(**c) for c in d["conds"]]
        else:
            areas, tensile, conds = default_areas, default_tensile, default_conds

        fig = _overlay_fig(areas, tensile, conds) if tab == "overlay" else _violin_fig(areas, tensile, conds)
        return dcc.Graph(figure=fig, style={"height": "750px"})

    # Download metrics
    @app.callback(
        Output("dl-metrics", "data"),
        Input("btn-dl-metrics", "n_clicks"),
        State("metrics-name", "value"),
        State("exp-data", "data"),
        prevent_initial_call=True,
    )
    def _dl_metrics(_, fname, payload):
        if not payload:
            raise PreventUpdate
        d = json.loads(payload)
        areas = _bytes_to_dim(_b64_to_bytes(d["dim_b64"]))
        tensile = _bytes_to_ten(_b64_to_bytes(d["ten_b64"]))
        conds = [Condition(**c) for c in d["conds"]]
        df = build_summary(areas, tensile, conds)
        return dcc.send_bytes(_to_excel_bytes({"Metrics": df}), fname or "metrics.xlsx")

    # Download stats
    @app.callback(
        Output("dl-stats", "data"),
        Input("btn-dl-stats", "n_clicks"),
        State("stats-name", "value"),
        State("exp-data", "data"),
        prevent_initial_call=True,
    )
    def _dl_stats(_, fname, payload):
        if not payload:
            raise PreventUpdate
        d = json.loads(payload)
        areas = _bytes_to_dim(_b64_to_bytes(d["dim_b64"]))
        tensile = _bytes_to_ten(_b64_to_bytes(d["ten_b64"]))
        conds = [Condition(**c) for c in d["conds"]]

        summary = build_summary(areas, tensile, conds)
        metrics_od = OrderedDict(
            (c, c.replace("_", " ")) for c in summary.columns if c not in ("Slot", "Condition")
        )
        long = build_stats(summary, conds, metrics_od)
        control_name = next(c.name for c in conds if c.is_control)
        wide = long_to_wide(long, summary, control_name, metrics_od)

        return dcc.send_bytes(_to_excel_bytes({"Stats": wide}), fname or "stats.xlsx")

    return app


# module-level instance (picked up by unit tests)
app: Dash = build_dash_app()
__all__ = ["build_dash_app", "app"]
