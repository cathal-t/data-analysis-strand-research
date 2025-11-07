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
import platform
import re
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analysis import build_summary, build_stats, long_to_wide
from ..dimensioncleaning import parse_dimensional_export
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


def _looks_like_absolute(path_str: str) -> bool:
    """Return True if the given path string appears to be absolute on any platform."""

    for pure_cls in (PureWindowsPath, PurePosixPath):
        try:
            pure_path = pure_cls(path_str)
        except Exception:
            continue
        if pure_path.is_absolute():
            return True
    return False


def _infer_original_dir(filename: str) -> Path | None:
    """Best-effort inference of the original directory of an uploaded file."""

    if not filename:
        return None

    fakepath_tokens = {"c:/fakepath", "c\\fakepath", "c:/fakepath/", "c\\fakepath\\"}

    # Try common path flavours. If they yield a meaningful parent directory, use it.
    for pure_cls in (PureWindowsPath, PurePosixPath):
        try:
            pure_path = pure_cls(filename)
        except Exception:
            continue

        parent = pure_path.parent
        parent_str = str(parent)
        if not parent or parent_str in ("", ".", "\\", "/"):
            continue

        if parent_str.lower().replace("\\", "/") in fakepath_tokens:
            continue

        if not parent.is_absolute():
            continue

        return Path(parent_str)

    return None


def _parse_export_directory(value: str | None) -> Path | None:
    """Parse a user-provided export directory, ensuring it is absolute if provided."""

    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    fakepath_tokens = {"c:/fakepath", "c\\fakepath", "c:/fakepath/", "c\\fakepath\\"}

    for pure_cls in (PureWindowsPath, PurePosixPath):
        try:
            pure_path = pure_cls(value)
        except Exception:
            continue

        if not pure_path.is_absolute():
            continue

        parent_str = str(pure_path)
        if parent_str.lower().replace("\\", "/") in fakepath_tokens:
            continue

        return Path(parent_str)

    raise ValueError("Please provide an absolute export directory path.")


def _store_uvc_file(raw: bytes, filename: str) -> tuple[Path, Path | None]:
    """Persist uploaded UVC bytes to a temporary directory.

    Returns the path to the stored copy and, if inferrable, the original parent
    directory of the uploaded file so that downstream steps can mirror outputs
    alongside the source data.
    """

    safe_name = Path(filename).name or "uploaded.uvc"
    tmp_dir = Path(tempfile.mkdtemp(prefix="uvc-upload-"))
    target = tmp_dir / safe_name
    target.write_bytes(raw)

    return target, _infer_original_dir(filename)


def _resolve_export_target(
    uvc_path: Path,
    original_dir: Path | None,
    preferred_dir: Path | None,
) -> tuple[Path, Path]:
    candidates = [preferred_dir, original_dir]
    output_parent = None
    for candidate in candidates:
        if candidate is None:
            continue
        if _looks_like_absolute(str(candidate)):
            output_parent = candidate
            break

    if output_parent is None:
        output_parent = uvc_path.parent

    output_file = (output_parent / uvc_path.name).with_suffix(".txt")
    return output_file, output_parent


def _run_dimensional_export(
    uvc_path: Path,
    original_dir: Path | None = None,
    preferred_dir: Path | None = None,
) -> tuple[bool, str]:
    """Invoke UvWin4 to export dimensional data for the provided UVC file."""

    if platform.system() != "Windows":
        return False, "Dimensional export is only available on Windows installations."

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")
    if not exe_path.exists():
        return False, f"UvWin executable not found at '{exe_path}'."

    output_file, output_parent = _resolve_export_target(
        uvc_path, original_dir, preferred_dir
    )

    try:
        output_parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Unable to prepare export directory '{output_parent}': {exc}"

    uvc_abs = uvc_path.resolve()
    output_abs = output_file.resolve()

    print(
        f"Dimensional export debug - UVC path: {uvc_abs}, export path: {output_abs}"
    )

    cmd = [
        str(exe_path),
        "-export",
        "dimensional",
        "-i",
        str(uvc_abs),
        "-o",
        str(output_abs),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(uvc_path.parent),
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return False, f"UvWin executable not found at '{exe_path}'."
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        details = stderr or stdout or str(exc)
        return False, f"Dimensional export failed: {details}"

    if output_file.exists():
        return True, f"Export complete. Output saved to: {output_file}"

    message = result.stdout.strip() or "Dimensional export completed."
    return True, message


def _max_slot(areas: dict[int, float], tensile: TensileTest) -> int:
    return max(
        max(areas, default=0),
        max((slot for slot, _ in tensile.per_slot()), default=0),
    )


def _to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Write multiple DataFrames to an in-memory Excel file.

    Behavior aligned with the standalone script:
    - 'Metrics' sheet: plain export with index=True.
    - 'Stats'  sheet: styled export (freeze panes, header fill, widths,
      number formats, conditional formatting for p-values).
    """
    buf = BytesIO()

    def _col_letter(idx: int) -> str:
        s = ""
        while idx >= 0:
            s = chr(idx % 26 + 65) + s
            idx = idx // 26 - 1
        return s

    with pd.ExcelWriter(buf, engine="xlsxwriter") as xls:
        for sheet, df in sheets.items():
            sheet_lower = sheet.lower()

            # Always write with index=True (script behavior).
            df.to_excel(xls, sheet_name=sheet, index=True)

            if sheet_lower != "stats":
                # 'Metrics' (and any other sheets) remain unstyled like the script's Metrics export.
                continue

            # ── styling block for 'Stats' sheet (replicates script) ──
            ws   = xls.sheets[sheet]
            book = xls.book

            hdr  = book.add_format({"bold": True, "bg_color": "#dfe6e9",
                                    "border": 1, "align": "center"})
            f_txt = book.add_format({"text_wrap": True})
            f_int = book.add_format({"num_format": "0", "align": "center"})
            f_num = book.add_format({"num_format": "0.00"})
            f_pct = book.add_format({"num_format": "0.0%"})
            f_p   = book.add_format({"num_format": "0.000"})
            f_sig = book.add_format({"bg_color": "#c8e6c9"})

            # Freeze header rows & first two columns (like script)
            ws.freeze_panes(2, 2)
            ws.set_row(0, None, hdr)
            ws.set_row(1, None, hdr)
            # Pandas sometimes leaves a spacer row right after headers in MultiIndex exports;
            # keep parity with the script that hides row 3 (0-based index 2).
            try:
                ws.set_row(2, 0, None, {"hidden": True})
            except Exception:
                pass

            # Column sizing: first column is the index (condition names)
            ws.set_column(0, 0, 32, f_txt)  # Condition
            # Second column should be ("", "N") in your wide table
            ws.set_column(1, 1, 6,  f_int)  # N

            # Work out stat-specific formats for remaining columns
            # We rely on the MultiIndex structure: (Metric, Stat)
            # Pandas writes the index in col 0 and the first data column at col 1.
            # Our wide dataframe had an inserted ("", "N") at position 0 in the script;
            # here, we just mirror the formatting starting at col 2.
            start_col = 2
            # Try to get columns from the dataframe to map stat names to excel columns.
            try:
                cols = list(df.columns)
            except Exception:
                cols = []

            # Apply per-column widths/formats & conditional formatting for 'p'
            for col_i, col_key in enumerate(cols[1:], start=start_col):  # skip the first data col ("", "N")
                stat_name = ""
                if isinstance(col_key, tuple) and len(col_key) >= 2:
                    stat_name = str(col_key[1]).strip().lower()
                else:
                    # Fallback: if not a tuple (unexpected), treat it generically
                    stat_name = str(col_key).strip().lower()

                if   stat_name == "test mean":
                    width, fmt = 14, f_num
                elif stat_name == "% change" or stat_name == "% change".lower():
                    width, fmt = 12, f_pct
                elif stat_name == "p":
                    width, fmt = 12, f_p
                else:
                    width, fmt = 12, f_num

                ws.set_column(col_i, col_i, width, fmt)

                # Conditional shading for p-values < 0.05
                if stat_name == "p":
                    # Data starts at row 4 (1-based) in the script; replicate the same window.
                    # Translate to 1-based Excel addresses using our helper.
                    first_row_1based = 4
                    last_row_1based  = 4 + len(df.index) - 1
                    let = _col_letter(col_i)
                    ws.conditional_format(
                        first_row_1based - 1, col_i, last_row_1based - 1, col_i,
                        {"type": "formula",
                         "criteria": f'=AND(ISNUMBER({let}{first_row_1based}),{let}{first_row_1based}<0.05)',
                         "format": f_sig}
                    )

    buf.seek(0)
    return buf.getvalue()


def _make_dimensional_record_fig(
    record_id: int, df: pd.DataFrame, slice_cols: list[str]
) -> go.Figure:
    cols = [c for c in slice_cols if c in df.columns]
    if not cols:
        fig = go.Figure()
        fig.update_layout(title=f"Record {record_id}")
        return fig

    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
    x_vals = df["N"].tolist() if "N" in df.columns else list(range(1, len(df) + 1))

    stacked = df[cols].stack(dropna=True)
    y_range: list[float] | None = None
    if not stacked.empty:
        y_min = float(stacked.min())
        y_max = float(stacked.max())
        if y_max <= y_min:
            padding = max(abs(y_max), 1.0) * 0.05
            y_min -= padding
            y_max += padding
        else:
            padding = (y_max - y_min) * 0.05
            y_min -= padding
            y_max += padding
        y_range = [y_min, y_max]

    for idx, col in enumerate(cols, start=1):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df[col],
                mode="lines",
                name=col,
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text="N", row=1, col=idx)
        if y_range:
            fig.update_yaxes(title_text="Value", row=1, col=idx, range=y_range)
        else:
            fig.update_yaxes(title_text="Value", row=1, col=idx)

    fig.update_layout(
        title=f"Record {record_id}",
        height=320,
        margin=dict(t=80, l=40, r=20, b=40),
    )
    return fig


def _make_slice_error_table(df: pd.DataFrame, slice_cols: list[str]) -> html.Div:
    cols = [c for c in slice_cols if c in df.columns]
    if not cols:
        return html.Div()

    min_vals: dict[str, float] = {}
    max_vals: dict[str, float] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        series = series.dropna()
        if series.empty:
            continue
        min_vals[col] = float(series.min())
        max_vals[col] = float(series.max())

    if not min_vals or not max_vals:
        return html.Div()

    record_min = min(min_vals.values())
    record_max = max(max_vals.values())

    def _coeff(val: float | None, ref: float | None) -> float | None:
        if val is None or ref is None:
            return None
        if ref == 0:
            return None
        return abs(val - ref) / abs(ref) * 100.0

    def _fmt(value: float | None, suffix: str = "") -> str:
        if value is None or pd.isna(value):
            return "–"
        return f"{value:.2f}{suffix}"

    def _style_pct(value: float | None) -> dict:
        if value is None:
            return {}
        if value > 10:
            return {"color": "#c53030", "fontWeight": "600"}
        return {}

    header = html.Thead(
        html.Tr(
            [
                html.Th("Slice"),
                html.Th("Min"),
                html.Th("Max"),
                html.Th("Min coeff. error"),
                html.Th("Max coeff. error"),
            ]
        )
    )

    rows = []
    for col in cols:
        min_val = min_vals.get(col)
        max_val = max_vals.get(col)
        min_coeff = _coeff(min_val, record_min)
        max_coeff = _coeff(max_val, record_max)
        rows.append(
            html.Tr(
                [
                    html.Th(col, scope="row"),
                    html.Td(_fmt(min_val)),
                    html.Td(_fmt(max_val)),
                    html.Td(_fmt(min_coeff, suffix="%"), style=_style_pct(min_coeff)),
                    html.Td(_fmt(max_coeff, suffix="%"), style=_style_pct(max_coeff)),
                ]
            )
        )

    body = html.Tbody(rows)

    return html.Div(
        [
            html.H6("Slice extremes", className="mt-4"),
            dbc.Table([header, body], bordered=True, size="sm", className="mb-0"),
        ]
    )


def _build_dimensional_plot_children(
    records: dict[int, pd.DataFrame], slice_cols: list[str]
) -> list:
    if not records:
        return [
            dbc.Alert(
                "Export succeeded but no record data was found in the file.",
                color="warning",
                className="mt-3",
            )
        ]

    children = []
    for record_id in sorted(records):
        df = records[record_id]
        if df.empty:
            continue
        fig = _make_dimensional_record_fig(record_id, df, slice_cols)
        error_table = _make_slice_error_table(df, slice_cols)
        children.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        dcc.Graph(figure=fig, config={"displaylogo": False}),
                        error_table,
                    ]
                ),
                className="mb-4 shadow-sm",
            )
        )

    if not children:
        return [
            dbc.Alert(
                "Export succeeded but no plottable records were detected.",
                color="warning",
                className="mt-3",
            )
        ]

    return children


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

    def _header() -> dbc.Row:
        return dbc.Row(
            dbc.Col(html.H3("Strand Research - Data Analysis"), width="auto"),
            className="mt-2 mb-3",
        )

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

    analysis_layout = dbc.Container(
        [
            _header(),
            upload_card,
            cond_card,
            actions_card,
            fig_card,
            dcc.Download(id="dl-metrics"),
            dcc.Download(id="dl-stats"),
            dcc.Store(id="exp-data"),
        ],
        fluid=True,
        style={"maxWidth": "1400px"},
    )

    dim_clean_layout = dbc.Container(
        [
            _header(),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Upload Dimensional UVC File", className="card-title"),
                        html.P(
                            "Drag and drop or click to browse for a .uvc file. The file will be processed "
                            "using the UvWin dimensional export tool.",
                            className="text-muted",
                        ),
                        dcc.Upload(
                            id="upload-dim-cleaning",
                            accept=".uvc",
                            multiple=False,
                            children=html.Div(
                                [
                                    "Drag and drop or ",
                                    html.A("browse", className="fw-semibold"),
                                    " for a .uvc file",
                                ],
                                className="py-4",
                            ),
                            style={
                                "width": "100%",
                                "height": "120px",
                                "lineHeight": "120px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "backgroundColor": "#fafafa",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Label("Export directory (optional)", html_for="dim-export-dir"),
                                dbc.Input(
                                    id="dim-export-dir",
                                    type="text",
                                    placeholder="e.g. C:/Data/Exports",
                                ),
                                dbc.FormText(
                                    "Provide a folder where dimensional exports should be saved."
                                    " Leave blank to use the upload's original folder when available.",
                                ),
                            ],
                            className="mt-3",
                        ),
                        dbc.Alert(id="dim-cleaning-alert", is_open=False, className="mt-3"),
                        html.Div(id="dim-cleaning-plots", className="mt-4"),
                    ]
                ),
                className="shadow-sm",
            ),
        ],
        fluid=True,
        style={"maxWidth": "1100px"},
    )

    landing_intro = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Welcome to Strand Research Analytics", className="card-title"),
                html.P(
                    "Select a workflow to get started. Additional modules will be available soon.",
                    className="text-muted",
                ),
                dbc.Stack(
                    [
                        dbc.Button(
                            "Data Cleaning",
                            id="btn-landing-cleaning",
                            color="secondary",
                            className="mx-auto",
                            style={"width": "70%"},
                        ),
                        html.Div(
                            dbc.Stack(
                                [
                                    dbc.Button(
                                        "Dimensional Cleaning",
                                        id="btn-landing-dim-cleaning",
                                        color="light",
                                        className="mx-auto",
                                        style={"width": "100%"},
                                    ),
                                    dbc.Button(
                                        "Tensile Cleaning",
                                        id="btn-landing-ten-cleaning",
                                        color="light",
                                        className="mx-auto",
                                        style={"width": "100%"},
                                    ),
                                ],
                                gap=2,
                                className="mx-auto",
                                style={"width": "70%"},
                            ),
                            id="cleaning-subbuttons",
                            style={"display": "none"},
                        ),
                        dbc.Button(
                            "Dimensional & Tensile Analysis",
                            id="btn-landing-analysis",
                            color="primary",
                            href="/analysis",
                            className="mx-auto",
                            style={"width": "70%"},
                        ),
                        dbc.Button(
                            "Multiple Cassette Analysis (coming soon)",
                            id="btn-landing-cross",
                            color="secondary",
                            disabled=True,
                            className="mx-auto",
                            style={"width": "70%"},
                        ),
                    ],
                    gap=3,
                ),
            ]
        ),
        className="shadow-sm",
    )

    landing_layout = dbc.Container(
        [
            _header(),
            landing_intro,
        ],
        fluid=True,
        style={"maxWidth": "900px"},
    )

    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            html.Div(id="page-content"),
        ]
    )

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def _render_page(pathname: str):
        if pathname == "/analysis":
            return analysis_layout
        if pathname == "/dimensional-cleaning":
            return dim_clean_layout
        return landing_layout

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("btn-landing-dim-cleaning", "n_clicks"),
        prevent_initial_call=True,
    )
    def _go_dim_cleaning(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return "/dimensional-cleaning"

    @app.callback(Output("cleaning-subbuttons", "style"), Input("btn-landing-cleaning", "n_clicks"))
    def _toggle_cleaning_subbuttons(n_clicks):
        hidden = {"display": "none"}
        shown = {"display": "block"}
        return shown if n_clicks else hidden

    @app.callback(
        Output("dim-cleaning-alert", "children"),
        Output("dim-cleaning-alert", "color"),
        Output("dim-cleaning-alert", "is_open"),
        Output("dim-cleaning-plots", "children"),
        Input("upload-dim-cleaning", "contents"),
        State("upload-dim-cleaning", "filename"),
        State("dim-export-dir", "value"),
        prevent_initial_call=True,
    )
    def _process_dimensional_cleaning(contents, filename, preferred_dir):
        if not contents or not filename:
            raise PreventUpdate

        raw = _b64_to_bytes(contents)
        uvc_path, original_dir = _store_uvc_file(raw, filename)

        try:
            preferred_path = _parse_export_directory(preferred_dir)
        except ValueError as exc:
            return str(exc), "danger", True

        export_path, _ = _resolve_export_target(
            uvc_path, original_dir=original_dir, preferred_dir=preferred_path
        )

        success, message = _run_dimensional_export(
            uvc_path, original_dir=original_dir, preferred_dir=preferred_path
        )

        plots: list = []
        if success and export_path.exists():
            try:
                records, slice_cols = parse_dimensional_export(export_path)
            except Exception as exc:
                plots = [
                    dbc.Alert(
                        f"Export succeeded but the output could not be parsed: {exc}",
                        color="danger",
                        className="mt-3",
                    )
                ]
            else:
                plots = _build_dimensional_plot_children(records, slice_cols)

        return message, ("success" if success else "danger"), True, plots

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
