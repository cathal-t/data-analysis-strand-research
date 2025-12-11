"""Dash components for the Multiple Cassette Analysis workflow.

This page mirrors the look and feel of the Dimensional & Tensile Analysis
workflow but works from previously generated ``*_metrics.xlsx`` exports.
Users can upload one or many Metrics files, choose which conditions to plot,
and download combined statistics for the selected conditions.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Iterable

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate

from ..analysis import METRIC_LABELS, build_stats, long_to_wide
from ..io.config import Condition
from ..plots import make_violin_grid


def _b64_to_bytes(content: str) -> bytes:
    return base64.b64decode(content.partition(",")[2])


def _render_plot_component(fig, *, class_name: str | None = None, alt: str = "Plot"):
    return html.Div(
        dcc.Graph(
            figure=fig,
            className=class_name,
            config={"displaylogo": False},
        ),
        role="img",
        **{"aria-label": alt},
    )


def _condition_label(filename: str, condition: str) -> str:
    stem = Path(filename).stem
    return f"{condition} ({stem})"


def _strip_source(label: str) -> str:
    if " (" in label and label.endswith(")"):
        return label.rsplit(" (", 1)[0]
    return label


def _load_metrics_df(raw: bytes) -> pd.DataFrame:
    buf = BytesIO(raw)
    try:
        df = pd.read_excel(buf, sheet_name="Metrics")
    except ValueError:
        buf.seek(0)
        df = pd.read_excel(buf, sheet_name=0)

    if "Condition" not in df.columns:
        raise ValueError("Metrics sheet must include a 'Condition' column.")

    missing_cols = [col for col in METRIC_LABELS if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Metrics sheet is missing columns: {missing}")

    if "Slot" in df.columns:
        df = df.set_index("Slot")

    return df


def _merge_uploaded_files(
    current_files: list[dict] | None, new_files: list[dict]
) -> list[dict]:
    """Merge newly uploaded Metrics files with any already stored ones.

    Previously uploaded files are preserved by filename, and any re-upload of
    the same filename replaces the stored entry while retaining its original
    position in the list. New filenames are appended to the end.
    """

    merged: dict[str, dict] = {}

    for entry in current_files or []:
        merged.setdefault(entry["name"], entry)

    for entry in new_files:
        merged[entry["name"]] = entry

    return list(merged.values())


def _build_summary(
    files: list[dict], selections: Iterable[str], control_label: str | None
):
    frames: list[pd.DataFrame] = []
    cond_labels: list[str] = []
    selection_lookup = set(selections or [])

    for idx, entry in enumerate(files):
        df = pd.DataFrame(entry["records"])
        available = {c for c in df["Condition"].unique()}
        chosen = {
            val.split("::", 1)[1]
            for val in selection_lookup
            if val.startswith(f"{idx}::")
        }
        for cond in sorted(available & chosen):
            label = _condition_label(entry["name"], cond)
            cond_labels.append(label)
            subset = df[df["Condition"] == cond].copy()
            subset["Condition"] = label
            frames.append(subset)

    if not frames:
        raise ValueError("Select at least one condition to plot.")

    if not control_label:
        raise ValueError("Select a control condition for statistics.")

    summary_df = pd.concat(frames, ignore_index=False).reset_index(drop=True)

    conds = [
        Condition(name=label, slots=(i + 1,), is_control=label == control_label)
        for i, label in enumerate(dict.fromkeys(cond_labels))
    ]

    if not any(c.is_control for c in conds):
        raise ValueError("The chosen control must be among the selected conditions.")

    return summary_df, conds


def register_multi_cassette_page(app: dash.Dash):
    header = dbc.Row(
        dbc.Col(html.H3("Multiple Cassette Analysis"), width="auto"),
        className="mt-2 mb-3",
    )

    upload_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Upload Metrics", className="card-title"),
                html.P(
                    "Upload one or more *_metrics.xlsx files produced by the Dimensional & Tensile Analysis workflow.",
                    className="text-muted",
                ),
                dcc.Upload(
                    id="mc-upload",
                    accept=".xlsx",
                    multiple=True,
                    children=dbc.Button("Upload Metrics", color="primary"),
                ),
                html.Small(
                    "All uploaded files must contain a 'Metrics' sheet with Condition and metric columns.",
                    className="text-muted",
                ),
                dbc.Alert(id="mc-upload-alert", is_open=False, className="mt-3"),
            ]
        ),
        className="mb-3 shadow-sm",
    )

    selection_card = dbc.Card(
        [
            dbc.CardHeader("Conditions"),
            dbc.CardBody(
                [
                    html.P(
                        "Choose which conditions from each Metrics file should be included.",
                        className="text-muted",
                    ),
                    html.Div(id="mc-condition-selectors"),
                    html.Hr(),
                    dbc.Label("Control condition"),
                    dcc.Dropdown(id="mc-control", placeholder="Select control", clearable=False),
                ]
            ),
        ],
        className="mb-3 shadow-sm",
    )

    actions_card = dbc.Card(
        dbc.CardBody(
            [
                dbc.Button("Plot selected", id="mc-plot", color="primary"),
                html.Span(className="mx-2"),
                dbc.Input(id="mc-stats-name", value="stats.xlsx", type="text", style={"width": "200px"}),
                html.Span(className="mx-2"),
                dbc.Button("Download Stats", id="mc-download-stats", color="info"),
                dbc.Alert(id="mc-plot-alert", is_open=False, className="mt-3"),
            ]
        ),
        className="mb-3 shadow-sm",
    )

    figure_card = dbc.Card(dbc.CardBody(html.Div(id="mc-figure")), className="shadow-sm")

    layout = dbc.Container(
        [
            header,
            upload_card,
            selection_card,
            actions_card,
            figure_card,
            dcc.Store(id="mc-files-store"),
            dcc.Store(id="mc-summary-store"),
            dcc.Download(id="mc-dl-stats"),
        ],
        fluid=True,
        style={"maxWidth": "1400px"},
    )

    @app.callback(
        Output("mc-files-store", "data"),
        Output("mc-upload-alert", "children"),
        Output("mc-upload-alert", "color"),
        Output("mc-upload-alert", "is_open"),
        Input("mc-upload", "contents"),
        State("mc-upload", "filename"),
        State("mc-files-store", "data"),
        prevent_initial_call=True,
    )
    def _ingest_metrics(contents, filenames, existing_files):
        if not contents or not filenames:
            raise PreventUpdate

        parsed: list[dict] = []
        for content, fname in zip(contents, filenames):
            try:
                df = _load_metrics_df(_b64_to_bytes(content))
            except Exception as exc:  # pragma: no cover - file validation
                return None, f"Unable to read '{fname}': {exc}", "danger", True

            parsed.append(
                {
                    "name": fname,
                    "records": df.reset_index().to_dict("records"),
                }
            )

        merged_files = _merge_uploaded_files(existing_files, parsed)
        msg = (
            f"Loaded {len(parsed)} Metrics file{'s' if len(parsed) > 1 else ''}. "
            f"{len(merged_files)} total in session."
        )
        return merged_files, msg, "success", True

    @app.callback(
        Output("mc-condition-selectors", "children"),
        Output("mc-control", "options"),
        Output("mc-control", "value"),
        Input("mc-files-store", "data"),
        prevent_initial_call=True,
    )
    def _render_selectors(files_data):
        if not files_data:
            raise PreventUpdate

        selectors = []
        control_options: list[dict] = []

        for idx, entry in enumerate(files_data):
            df = pd.DataFrame(entry["records"])
            counts = df["Condition"].value_counts().to_dict()
            options = [
                {
                    "label": f"{cond} ({counts.get(cond, 0)} fibres)",
                    "value": f"{idx}::{cond}",
                }
                for cond in sorted(counts)
            ]
            selectors.append(
                dbc.Card(
                    [
                        dbc.CardHeader(entry["name"]),
                        dbc.CardBody(
                            dbc.Checklist(
                                options=options,
                                value=[opt["value"] for opt in options],
                                id={"type": "mc-conditions", "index": idx},
                                inline=True,
                            )
                        ),
                    ],
                    className="mb-3",
                )
            )
            for opt in options:
                control_options.append(
                    {
                        "label": _condition_label(entry["name"], opt["value"].split("::", 1)[1]),
                        "value": _condition_label(entry["name"], opt["value"].split("::", 1)[1]),
                    }
                )

        control_value = control_options[0]["value"] if control_options else None
        return selectors, control_options, control_value

    @app.callback(
        Output("mc-control", "options", allow_duplicate=True),
        Output("mc-control", "value", allow_duplicate=True),
        Input({"type": "mc-conditions", "index": ALL}, "value"),
        State("mc-files-store", "data"),
        State("mc-control", "value"),
        prevent_initial_call=True,
    )
    def _sync_control(selection_values, files_data, current_control):
        if not files_data:
            raise PreventUpdate

        all_selected: list[str] = []
        for idx, values in enumerate(selection_values or []):
            for raw in values or []:
                cond_name = raw.split("::", 1)[1]
                all_selected.append(_condition_label(files_data[idx]["name"], cond_name))

        options = [{"label": val, "value": val} for val in sorted(set(all_selected))]
        control_value = current_control if current_control in all_selected else (options[0]["value"] if options else None)
        return options, control_value

    @app.callback(
        Output("mc-figure", "children"),
        Output("mc-summary-store", "data"),
        Output("mc-plot-alert", "children"),
        Output("mc-plot-alert", "color"),
        Output("mc-plot-alert", "is_open"),
        Input("mc-plot", "n_clicks"),
        State({"type": "mc-conditions", "index": ALL}, "value"),
        State("mc-control", "value"),
        State("mc-files-store", "data"),
        prevent_initial_call=True,
    )
    def _plot(_, selections, control_value, files_data):
        if not files_data:
            raise PreventUpdate

        try:
            flat_selections = [val for group in (selections or []) for val in (group or [])]
            summary_df, conds = _build_summary(files_data, flat_selections, control_value)
            legend_labels = {cond.name: _strip_source(cond.name) for cond in conds}
            fig = make_violin_grid(
                summary_df,
                conds,
                stacked=True,
                legend_labels=legend_labels,
            )
        except Exception as exc:  # pragma: no cover - user feedback
            return [], None, str(exc), "danger", True

        store_payload = {
            "summary": summary_df.to_dict("records"),
            "conditions": [asdict(c) for c in conds],
        }
        graph = _render_plot_component(fig, class_name="mb-4", alt="Violin grid")
        return graph, store_payload, "", "success", False

    @app.callback(
        Output("mc-dl-stats", "data"),
        Input("mc-download-stats", "n_clicks"),
        State("mc-summary-store", "data"),
        State("mc-stats-name", "value"),
        prevent_initial_call=True,
    )
    def _download_stats(_, summary_payload, fname):
        if not summary_payload:
            raise PreventUpdate

        summary_df = pd.DataFrame(summary_payload["summary"])
        conds = [Condition(**c) for c in summary_payload["conditions"]]
        metrics_od = METRIC_LABELS
        long = build_stats(summary_df, conds, metrics_od)
        control_name = next(c.name for c in conds if c.is_control)
        wide = long_to_wide(long, summary_df, control_name, metrics_od)

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xls:
            wide.to_excel(xls, sheet_name="Stats")

        return dcc.send_bytes(buf.getvalue(), fname or "stats.xlsx")

    return layout


__all__ = ["register_multi_cassette_page"]
