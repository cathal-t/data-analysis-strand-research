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
import logging
import math
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
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.development.base_component import Component
from dash.dependencies import ALL, MATCH, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analysis import build_summary, build_stats, long_to_wide
from ..dimensioncleaning import parse_dimensional_export
from ..dimensional import DimensionalData
from ..io.config import Condition, ConfigError, load_config
from ..plots import make_overlay, make_violin_grid
from .excel import to_excel_bytes
from ..tensile import TensileTest
from .multi_cassette import register_multi_cassette_page

# ────────────── constants ──────────────
TICK = "✓"
EMPTY = ""
DEFAULT_RE = re.compile(r"Condition\s+\d+", re.I)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _set_log_level(enabled: bool) -> None:
    level = logging.DEBUG if enabled else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    logger.setLevel(level)

# ────────────── helper I/O ──────────────
def _load_experiment(root: Path) -> Tuple[DimensionalData, TensileTest, List[Condition]]:
    conds = load_config(root)
    dim_data = DimensionalData(root / "Dimensional_Data.txt")
    tensile = TensileTest(root / "Tensile_Data.txt")
    return dim_data, tensile, conds


def _overlay_fig(areas, tensile, conds):
    slot_to_cond = {slot: c.name for c in conds for slot in c.slots}
    rows = []
    for slot, df in tensile.per_slot():
        if slot not in areas or slot not in slot_to_cond:
            continue
        record_ids = (
            df["Record"].dropna().unique().astype(int).tolist()
            if "Record" in df.columns
            else []
        )
        record_label = ", ".join(str(r) for r in sorted(record_ids)) if record_ids else "?"
        record_value = record_ids[0] if len(record_ids) == 1 else None
        proc = tensile.stress_strain(df, areas[slot])
        for st, sp in zip(proc["strain"], proc["stress_Pa"]):
            rows.append(
                {
                    "Slot": slot,
                    "Record": record_value,
                    "Record_Label": record_label,
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


def _bytes_to_dim(
    raw: bytes,
    removed: dict[int, set[int]] | None = None,
    slot_map: dict[int, int] | None = None,
) -> DimensionalData:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(raw)
        tmp.flush()
        return DimensionalData(Path(tmp.name), removed_slices=removed, slot_map=slot_map)


def _bytes_to_ten(raw: bytes) -> TensileTest:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(raw)
        tmp.flush()
        return TensileTest(Path(tmp.name))


def _parse_tensile_ascii(path: Path) -> pd.DataFrame:
    """Parse the Dia-Stron ``_tensile`` ASCII export into a tidy DataFrame."""

    def _parse_int(text: str) -> int | None:
        try:
            return int(float(text.strip()))
        except (TypeError, ValueError):
            return None

    def _parse_float(text: str) -> float | None:
        try:
            return float(text.strip())
        except (TypeError, ValueError):
            return None

    path = Path(path)
    rows: list[dict[str, object]] = []
    slot_number: int | None = None
    n_points: int | None = None
    remaining = 0
    reading_data = False
    skip_units = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("ASCII Export File Version"):
            slot_number = None
            n_points = None
            remaining = 0
            reading_data = False
            skip_units = False
            continue

        if line.startswith("Sample / Slot Number:"):
            slot_number = _parse_int(line.split(":", 1)[1])
            continue

        if line.startswith("Number of Points:"):
            n_points = _parse_int(line.split(":", 1)[1])
            continue

        header_tokens = raw_line.strip().split()
        if header_tokens[:6] == ["Record", "Index", "Position", "Strain", "Time", "Force"]:
            skip_units = True
            continue

        if skip_units:
            skip_units = False
            reading_data = True
            remaining = n_points or 0
            continue

        if reading_data and remaining > 0:
            parts = line.split()
            if len(parts) < 6 or slot_number is None:
                remaining -= 1
                continue

            record = _parse_int(parts[0])
            index = _parse_int(parts[1])
            position = _parse_float(parts[2])
            strain = _parse_float(parts[3])
            time = _parse_float(parts[4])
            force = _parse_float(parts[5])

            if None not in (record, index, position, strain, time, force):
                rows.append(
                    {
                        "Slot": int(slot_number),
                        "Record": int(record),
                        "Index": int(index),
                        "Position_um": float(position),
                        "Strain_pct": float(strain),
                        "Time_s": float(time),
                        "Force_N": float(force),
                    }
                )

            remaining -= 1
            if remaining <= 0:
                reading_data = False

    if not rows:
        return pd.DataFrame(
            columns=[
                "Slot",
                "Record",
                "Index",
                "Position_um",
                "Strain_pct",
                "Time_s",
                "Force_N",
            ]
        )

    return pd.DataFrame(rows)


def _trim_gmf_pivot(export_df: pd.DataFrame) -> pd.DataFrame:
    """Trim early strain rows and re-zero each record around a GMF pivot."""

    if export_df.empty:
        return export_df.copy()

    trimmed_groups: list[pd.DataFrame] = []
    strain_tol = 1e-9
    gmf_tol = 1e-12

    for _, group in export_df.groupby("Record", sort=False):
        working = group.reset_index(drop=True).copy()
        if working.empty:
            continue

        strain_series = pd.to_numeric(working["% Strain"], errors="coerce")
        gmf_series = pd.to_numeric(working["gmf"], errors="coerce")

        valid_mask = strain_series.notna() & gmf_series.notna()
        if not valid_mask.all():
            working = working.loc[valid_mask].reset_index(drop=True)
            strain_series = strain_series.loc[valid_mask].reset_index(drop=True)
            gmf_series = gmf_series.loc[valid_mask].reset_index(drop=True)

        if working.empty:
            continue

        scale = 100.0 if strain_series.max(skipna=True) > 1.0 + strain_tol else 1.0
        strain_fraction = strain_series / scale
        window_indices = [
            idx for idx, value in enumerate(strain_fraction) if value <= 0.10 + strain_tol
        ]

        if not window_indices:
            trimmed_groups.append(working)
            continue

        gmf_window = gmf_series.iloc[window_indices].to_numpy(dtype=float)
        future_mins = np.empty(len(gmf_window), dtype=float)
        future_mins[-1] = math.inf
        for pos in range(len(gmf_window) - 2, -1, -1):
            future_mins[pos] = min(future_mins[pos + 1], gmf_window[pos + 1])

        pivot_position: int | None = None
        for offset, group_index in enumerate(window_indices):
            if gmf_window[offset] < future_mins[offset] - gmf_tol:
                pivot_position = group_index
                break

        if pivot_position is None:
            trimmed_groups.append(working)
            continue

        pivot_strain = working.loc[pivot_position, "% Strain"]
        trimmed = working.iloc[pivot_position:].copy()
        trimmed.loc[:, "% Strain"] = trimmed["% Strain"] - pivot_strain
        trimmed_groups.append(trimmed)

    if not trimmed_groups:
        return export_df.iloc[0:0].copy()

    result = pd.concat(trimmed_groups, ignore_index=True)
    try:
        result["Record"] = result["Record"].astype(export_df["Record"].dtype)
    except Exception:  # pragma: no cover - fallback to numeric type
        result["Record"] = pd.to_numeric(result["Record"], errors="coerce")
    return result


def _build_tensile_force_figure(df: pd.DataFrame) -> go.Figure:
    """Create lean force–strain subplots for each Slot/Record combination."""

    grouped = list(
        df.sort_values(["Slot", "Record", "Index"]).groupby(["Slot", "Record"], sort=True)
    )

    if not grouped:
        fig = go.Figure()
        fig.update_layout(template="simple_white")
        fig.update_xaxes(title="Strain (%)")
        fig.update_yaxes(title="Force (N)")
        return fig

    n_curves = len(grouped)
    n_cols = 5
    n_rows = math.ceil(n_curves / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=[
            f"Slot {slot} · Record {record}" for (slot, record), _ in grouped
        ]
        + ["" for _ in range(n_rows * n_cols - n_curves)],
    )

    for idx, ((slot, record), grp) in enumerate(grouped):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        custom = np.stack(
            (grp["Index"].to_numpy(), grp["Time_s"].to_numpy(), grp["Position_um"].to_numpy()),
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=grp["Strain_pct"],
                y=grp["Force_N"],
                mode="lines",
                line={"width": 1.4},
                name=f"Slot {slot} · Record {record}",
                showlegend=False,
                customdata=custom,
                hovertemplate=(
                    "Index: %{customdata[0]}<br>Strain: %{x:.3f}%<br>"
                    "Force: %{y:.3f} N<br>Time: %{customdata[1]:.3f} s<br>"
                    "Position: %{customdata[2]:.3f} µm<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    # Apply axis titles to the outer plots only for clarity.
    for col in range(1, min(n_cols, n_curves) + 1):
        fig.update_xaxes(title_text="Strain (%)", row=n_rows, col=col)
    for row in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Force (N)", row=row, col=1)

    fig.update_layout(
        template="simple_white",
        height=280 * n_rows,
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
    )

    return fig


def _parse_removed_slices_csv(raw: bytes) -> list[dict[str, object]]:
    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        raise ValueError("Unable to read removed slices CSV") from exc

    if df.empty:
        return []

    lower_map = {col.strip().lower(): col for col in df.columns}
    record_col = lower_map.get("record")
    slot_col = lower_map.get("slot")
    removed_col = next(
        (orig for key, orig in lower_map.items() if "removed" in key and "slice" in key),
        None,
    )

    if removed_col is None:
        raise ValueError("Column 'Removed slice(s)' not found")
    if record_col is None and slot_col is None:
        raise ValueError("CSV must include a 'Record' or 'Slot' column")

    def _coerce_int(value) -> int | None:
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        if value is None or str(value).strip() == "":
            return None
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None

    slice_re = re.compile(r"\d+")
    aggregated: dict[int, dict[str, object]] = {}

    for _, row in df.iterrows():
        record_val = _coerce_int(row[record_col]) if record_col else None
        slot_val = _coerce_int(row[slot_col]) if slot_col else None
        removed_val = row.get(removed_col)
        if removed_val is None:
            continue
        digits = slice_re.findall(str(removed_val))
        if not digits:
            continue
        slices = {int(d) for d in digits}
        slot_key = slot_val if slot_val is not None else record_val
        if slot_key is None:
            continue
        entry = aggregated.setdefault(
            int(slot_key),
            {
                "slot": int(slot_key),
                "records": set(),
                "slot_from_csv": None,
                "slices": set(),
            },
        )
        entry["slices"].update(slices)
        if record_val is not None:
            entry["records"].add(int(record_val))
        if slot_val is not None:
            entry["slot_from_csv"] = int(slot_val)

    result: list[dict[str, object]] = []
    for slot, data in sorted(aggregated.items()):
        slices = sorted(data["slices"])
        if not slices:
            continue
        records = sorted(data.get("records", set()))
        slot_from_csv = data.get("slot_from_csv")
        result.append(
            {
                "slot": int(slot),
                "records": records,
                "slot_from_csv": int(slot_from_csv) if slot_from_csv is not None else None,
                "slices": slices,
            }
        )

    return result


def _entries_to_removal_map(entries: list[dict[str, object]]) -> dict[int, set[int]]:
    removal_map: dict[int, set[int]] = {}
    for entry in entries:
        slot = entry.get("slot")
        slices = entry.get("slices")
        if slot is None or not slices:
            continue
        removal_map.setdefault(int(slot), set()).update(int(s) for s in slices)
    return removal_map


def _build_removed_feedback_data(
    entries: list[dict[str, object]], dim_data: DimensionalData
) -> dict[str, object] | None:
    if not entries:
        return None

    items: list[dict[str, object]] = []
    entries_by_slot = {int(entry["slot"]): entry for entry in entries if "slot" in entry}

    for slot, entry in sorted(entries_by_slot.items()):
        requested = [int(s) for s in entry.get("slices", [])]
        records = [int(r) for r in entry.get("records", [])]
        items.append(
            {
                "slot": slot,
                "records": records,
                "slot_from_csv": entry.get("slot_from_csv"),
                "requested": requested,
                "applied": dim_data.removed_applied.get(slot, []),
                "missing": dim_data.removed_missing_slices.get(slot, []),
                "dropped": slot in dim_data.removed_empty_slots,
                "absent": slot in dim_data.removed_missing_slots,
            }
        )

    extra_missing = [
        slot for slot in dim_data.removed_missing_slots if slot not in entries_by_slot
    ]

    return {"items": items, "extra_missing_slots": extra_missing}


def _render_removed_feedback(feedback: dict[str, object] | None) -> list:
    if not feedback:
        return []

    items = feedback.get("items") or []
    if not items:
        return []

    applied_items: list = []
    warning_items: list = []

    for item in items:
        slot = item.get("slot")
        records = item.get("records") or []
        label_parts = []
        if records:
            rec_label = ", ".join(str(r) for r in records)
            label_parts.append(f"Record {rec_label}")
        if slot is not None:
            label_parts.append(f"Slot {slot}")
        label = " ".join(label_parts) if label_parts else f"Slot {slot}"

        applied = item.get("applied") or []
        if applied:
            applied_text = ", ".join(f"Slice {s}" for s in applied)
            text = f"{label}: removed {applied_text}."
        else:
            text = f"{label}: no matching slices from the summary were found."
        if item.get("dropped"):
            text += " Slot excluded from analysis because no slices remained."
        applied_items.append(html.Li(text))

        missing = item.get("missing") or []
        if missing:
            missing_text = ", ".join(f"Slice {s}" for s in missing)
            warning_items.append(
                html.Li(f"{label}: slices not found in dimensional file – {missing_text}.")
            )
        if item.get("absent"):
            warning_items.append(html.Li(f"{label}: slot not present in dimensional file."))

    extra_missing = feedback.get("extra_missing_slots") or []
    for slot in extra_missing:
        warning_items.append(html.Li(f"Slot {slot}: summary references a slot not in the dimensional file."))

    alerts: list = []
    if applied_items:
        alerts.append(
            dbc.Alert(
                [
                    html.H6("Removed slices applied", className="mb-2"),
                    html.Ul(applied_items, className="mb-0"),
                ],
                color="info",
                className="mb-3",
            )
        )
    if warning_items:
        alerts.append(
            dbc.Alert(
                [
                    html.H6("Removed slice warnings", className="mb-2"),
                    html.Ul(warning_items, className="mb-0"),
                ],
                color="warning",
                className="mb-3",
            )
        )

    return alerts


def _render_gpdsr_feedback(feedback: dict[str, object] | None) -> list:
    if not feedback:
        return []

    warnings = feedback.get("warnings") or []
    infos = feedback.get("infos") or []

    alerts: list = []
    if infos:
        alerts.append(
            dbc.Alert(
                [html.H6("GPDSR info", className="mb-2"), html.Ul([html.Li(msg) for msg in infos], className="mb-0")],
                color="info",
                className="mb-3",
            )
        )
    if warnings:
        alerts.append(
            dbc.Alert(
                [html.H6("GPDSR warnings", className="mb-2"), html.Ul([html.Li(msg) for msg in warnings], className="mb-0")],
                color="warning",
                className="mb-3",
            )
        )
    return alerts


def _parse_gpdsr_mapping(gpdsr_path: Path) -> tuple[pd.DataFrame, list[int]]:
    """Parse a *_gpdsr.txt file into a Record→Slot mapping.

    Returns a tuple of (DataFrame, deduped_slots). The DataFrame contains at
    least ``Record`` and ``Slot`` integer columns sorted by Record ascending.
    ``deduped_slots`` lists any slots that required de-duplication.
    """

    text = gpdsr_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    idx = 0
    # Drop leading metadata lines.
    while idx < len(lines) and (
        lines[idx].startswith("Summary Data Ver:")
        or lines[idx].startswith("Source File:")
    ):
        idx += 1

    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx >= len(lines):
        raise ValueError("GPDSR file is missing a header row")

    headers = [h.strip() for h in lines[idx].split("\t")]
    idx += 1

    rows: list[dict[str, object]] = []
    for line in lines[idx:]:
        if not line.strip():
            continue

        fields = line.split("\t")
        if len(fields) < len(headers):
            fields += [""] * (len(headers) - len(fields))
        elif len(fields) > len(headers):
            fields = fields[: len(headers)]

        row: dict[str, object] = {}
        for header, value in zip(headers, fields):
            row[header] = value.strip()

        record_value = row.get("Record")
        if record_value in (None, ""):
            continue

        try:
            row["Record"] = int(float(str(record_value)))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid Record value '{record_value}'") from exc

        sample_value = row.get("Sample")
        if sample_value not in (None, ""):
            try:
                row["Sample"] = int(float(str(sample_value)))
            except (TypeError, ValueError):
                pass

        description = str(row.get("Description", ""))
        slot_match = re.search(r"\bSlot\s+(\d+)\b", description)
        if not slot_match:
            desc_lower = description.strip().lower()
            if re.search(r"\btag\b", desc_lower):
                # Some files include metadata rows describing tagged slices. These do not
                # correspond to a physical slot and should be ignored when building the
                # Record→Slot mapping.
                continue
            logger.warning(
                "Skipping GPDSR row with unrecognized description: %s", description
            )
            continue
        row["Slot"] = int(slot_match.group(1))

        cycle_match = re.search(r"Cycle:\s*(\d+)", description)
        if cycle_match:
            row["Cycle"] = int(cycle_match.group(1))

        for header, value in list(row.items()):
            if header in {"Record", "Sample", "Slot", "Cycle", "Description"}:
                continue
            if isinstance(value, str):
                if value == "":
                    row[header] = pd.NA
                    continue
                try:
                    row[header] = float(value)
                except ValueError:
                    pass

        rows.append(row)

    if not rows:
        raise ValueError("GPDSR file contained no data rows")

    df = pd.DataFrame(rows)
    if df["Record"].duplicated().any():
        dupes = sorted(df.loc[df["Record"].duplicated(), "Record"].unique())
        raise ValueError(f"Duplicate Record values in GPDSR file: {dupes}")

    deduped_slots: list[int] = []
    if df["Slot"].duplicated().any():
        deduped_slots = sorted(df.loc[df["Slot"].duplicated(), "Slot"].unique())
        df = df.sort_values("Record").drop_duplicates("Slot", keep="last")

    if df["Slot"].duplicated().any():
        dupes = sorted(df.loc[df["Slot"].duplicated(), "Slot"].unique())
        raise ValueError(
            f"Duplicate Slot values remain after de-duplication: {dupes}"
        )

    df = df.sort_values("Record").reset_index(drop=True)
    return df, deduped_slots


class _SlotMappedTensile(TensileTest):
    """Lightweight TensileTest wrapper that groups by Slot instead of Record."""

    def __init__(self, base: TensileTest, df: pd.DataFrame):
        self.df = df
        self.is_mpa = base.is_mpa
        self.mode = getattr(base, "mode", None)

    def per_slot(self):  # type: ignore[override]
        for slot, grp in self.df.groupby("Slot", sort=True):
            yield int(slot), grp.reset_index(drop=True)


def _build_gpdsr_mapping(gpdsr_b64: str) -> tuple[list[tuple[int, int]], list[int]]:
    raw_bytes = _b64_to_bytes(gpdsr_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix="_gpdsr.txt") as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        df, deduped_slots = _parse_gpdsr_mapping(Path(tmp.name))

    mapping = [(int(row.Record), int(row.Slot)) for row in df.itertuples(index=False)]
    return mapping, deduped_slots


def _remap_tensile_slots(
    tensile: TensileTest, slot_map: dict[int, int] | None
) -> tuple[TensileTest, list[int]]:
    df = tensile.df.copy()
    df["Slot"] = df["Record"].map(slot_map) if slot_map else df["Record"]

    unmapped: list[int] = []
    if slot_map:
        missing_mask = df["Slot"].isna()
        if missing_mask.any():
            unmapped = sorted(df.loc[missing_mask, "Record"].dropna().unique().astype(int))
            df.loc[missing_mask, "Slot"] = df.loc[missing_mask, "Record"]

    df["Slot"] = pd.to_numeric(df["Slot"], errors="coerce")
    try:
        df["Slot"] = df["Slot"].astype(int)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    df = df.dropna(subset=["Slot"])

    mapped = _SlotMappedTensile(tensile, df)
    return mapped, unmapped


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


def _parse_export_directory(value: str | None) -> Path:
    """Parse a user-provided export directory, ensuring it is absolute and present."""

    if value is None:
        raise ValueError("Please provide an absolute export directory path.")

    value = value.strip()
    if not value:
        raise ValueError("Please provide an absolute export directory path.")

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


def _run_uvwin_export(
    uvc_path: Path,
    original_dir: Path | None = None,
    preferred_dir: Path | None = None,
    modes: tuple[str, ...] | list[str] | None = None,
) -> tuple[bool, str, dict[str, Path]]:
    """Invoke UvWin4 to export data for the provided UVC file.

    Parameters
    ----------
    uvc_path:
        Path to the source UVC file.
    original_dir / preferred_dir:
        Optional directories used to resolve the export destination.
    modes:
        Iterable of export modes to request from UvWin. Supported values are
        ``"dimensional"``, ``"gpdsr"``, and ``"tensile"``. When ``None`` both
        dimensional and GPDSR exports are generated.
    """

    if platform.system() != "Windows":
        return (
            False,
            "UvWin exports are only available on Windows installations.",
            {},
        )

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")
    if not exe_path.exists():
        return False, f"UvWin executable not found at '{exe_path}'.", {}

    output_file, output_parent = _resolve_export_target(
        uvc_path, original_dir, preferred_dir
    )

    gpdsr_output = output_file.with_name(
        f"{output_file.stem}_gpdsr{output_file.suffix}"
    )
    ascii_output = output_file.with_name(
        f"{uvc_path.stem}_ascii{output_file.suffix}"
    )

    try:
        output_parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return (
            False,
            f"Unable to prepare export directory '{output_parent}': {exc}",
            {},
        )

    uvc_abs = uvc_path.resolve()
    available_exports: dict[str, Path] = {
        "dimensional": output_file,
        "gpdsr": gpdsr_output,
        "tensile": ascii_output,
    }

    if modes is None:
        modes_to_run = tuple(available_exports.keys())
    else:
        normalized: list[str] = []
        for mode in modes:
            if mode not in available_exports:
                return False, f"Unsupported export mode '{mode}'.", {}
            if mode not in normalized:
                normalized.append(mode)
        if not normalized:
            return False, "No export modes were requested.", {}
        modes_to_run = tuple(normalized)

    exports = [(mode, available_exports[mode]) for mode in modes_to_run]

    messages: list[str] = []
    produced: dict[str, Path] = {}

    for export_name, export_path in exports:
        export_abs = export_path.resolve()

        print(
            "UvWin export debug - UVC path: "
            f"{uvc_abs}, export type: {export_name}, export path: {export_abs}"
        )

        cmd = [
            str(exe_path),
            "-export",
            export_name,
            "-i",
            str(uvc_abs),
            "-o",
            str(export_abs),
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
            return False, f"UvWin executable not found at '{exe_path}'.", {}
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            details = stderr or stdout or str(exc)
            return False, f"{export_name.capitalize()} export failed: {details}", {}

        messages.append(result.stdout.strip())
        if export_path.exists():
            produced[export_name] = export_path.resolve()

    if produced:
        produced_values = list(produced.values())
        if len(produced_values) == 1:
            message = f"Export complete. Output saved to: {produced_values[0]}"
        elif len(produced_values) == 2:
            message = (
                "Export complete. Outputs saved to: "
                f"{produced_values[0]} and {produced_values[1]}"
            )
        else:
            produced_text = ", ".join(str(path) for path in produced_values)
            message = f"Export complete. Outputs saved to: {produced_text}"
    else:
        fallback = next((msg for msg in messages if msg), None)
        message = fallback or "Export completed."

    return True, message, produced


def _max_slot(areas: dict[int, float], tensile: TensileTest) -> int:
    return max(
        max(areas, default=0),
        max((slot for slot, _ in tensile.per_slot()), default=0),
    )


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

    max_points = 1000
    step = max(1, len(df) // max_points) if len(df) else 1

    values = df[cols].to_numpy(dtype=float, na_value=np.nan)

    y_range: list[float] | None = None
    if values.size:
        with np.errstate(all="ignore"):
            y_min = float(np.nanmin(values))
            y_max = float(np.nanmax(values))
        if np.isfinite(y_min) and np.isfinite(y_max):
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
        y_full = df[col].to_numpy(dtype=float, na_value=np.nan)
        x_sample = x_vals[::step]
        y_sample = y_full[::step]

        if x_vals and (not x_sample or x_sample[-1] != x_vals[-1]):
            x_sample = list(x_sample) + [x_vals[-1]]
            y_sample = list(y_sample) + [y_full[-1]]

        fig.add_trace(
            go.Scatter(
                x=x_sample,
                y=y_sample,
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


def _render_plot_component(
    fig: go.Figure,
    graph_height: int | None = None,
    class_name: str | None = None,
    style: dict | None = None,
    alt: str = "Plot image",
    graph_config: dict | None = None,
    width: int | None = None,
):
    merged_style = dict(style or {})
    effective_height = graph_height
    if effective_height is None:
        layout_height = getattr(fig.layout, "height", None)
        if isinstance(layout_height, (int, float)):
            effective_height = int(layout_height)

    if effective_height and "height" not in merged_style:
        merged_style["height"] = f"{effective_height}px"

    graph_component = dcc.Graph(
        figure=fig,
        className=class_name,
        style=merged_style,
        config=graph_config or {"displaylogo": False},
    )

    return graph_component


def _compute_slice_extremes(df: pd.DataFrame, slice_cols: list[str]) -> list[dict[str, float]]:
    cols = [c for c in slice_cols if c in df.columns]
    stats: list[dict[str, float]] = []
    for col in cols:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        series = series.dropna()
        if series.empty:
            continue
        stats.append({"slice": col, "min": float(series.min()), "max": float(series.max())})
    return stats


def _render_slice_error_table(
    record_id: int, stats: list[dict[str, float]], removed: list[str] | None
) -> dbc.Table:
    removed = set(removed or [])

    def _fmt(value: float | None, suffix: str = "") -> str:
        if value is None or pd.isna(value):
            return "–"
        return f"{value:.2f}{suffix}"

    def _coeff(val: float | None, ref: float | None) -> float | None:
        if val is None or ref is None:
            return None
        if ref == 0:
            return None
        return abs(val - ref) / abs(ref) * 100.0

    def _style_pct(value: float | None) -> dict:
        if value is None:
            return {}
        if value > 10:
            return {"color": "#c53030", "fontWeight": "600"}
        return {}

    included = [s for s in stats if s["slice"] not in removed]

    def _avg_without_current(values: list[tuple[str, float]], current: str) -> float | None:
        peers = [val for name, val in values if name != current]
        if not peers:
            return None
        return sum(peers) / len(peers)

    included_mins = [
        (s["slice"], s["min"])
        for s in included
        if s.get("min") is not None and not pd.isna(s["min"])
    ]
    included_maxs = [
        (s["slice"], s["max"])
        for s in included
        if s.get("max") is not None and not pd.isna(s["max"])
    ]

    header = html.Thead(
        html.Tr(
            [
                html.Th("Slice"),
                html.Th("Min"),
                html.Th("Max"),
                html.Th("Min coeff. error"),
                html.Th("Max coeff. error"),
                html.Th("Remove slice"),
            ]
        )
    )

    rows = []
    for s in stats:
        slice_name = s["slice"]
        min_val = s["min"]
        max_val = s["max"]
        min_ref = _avg_without_current(included_mins, slice_name)
        max_ref = _avg_without_current(included_maxs, slice_name)
        min_coeff = _coeff(min_val, min_ref)
        max_coeff = _coeff(max_val, max_ref)
        is_removed = slice_name in removed
        button_label = "Restore slice" if is_removed else "Remove slice"
        button_color = "secondary" if is_removed else "danger"
        button_outline = is_removed
        row_style = {"opacity": 0.6} if is_removed else {}
        rows.append(
            html.Tr(
                [
                    html.Th(slice_name, scope="row"),
                    html.Td(_fmt(min_val)),
                    html.Td(_fmt(max_val)),
                    html.Td(_fmt(min_coeff, suffix="%"), style=_style_pct(min_coeff)),
                    html.Td(_fmt(max_coeff, suffix="%"), style=_style_pct(max_coeff)),
                    html.Td(
                        dbc.Button(
                            button_label,
                            id={
                                "type": "dim-slice-toggle",
                                "record": record_id,
                                "slice": slice_name,
                            },
                            color=button_color,
                            outline=button_outline,
                            size="sm",
                        )
                    ),
                ],
                style=row_style,
            )
        )

    body = html.Tbody(rows)

    return dbc.Table([header, body], bordered=True, size="sm", className="mb-0")


def _make_slice_error_table(record_id: int, df: pd.DataFrame, slice_cols: list[str]) -> html.Div:
    stats = _compute_slice_extremes(df, slice_cols)
    if not stats:
        return html.Div()

    return html.Div(
        [
            dcc.Store(id={"type": "dim-slice-data", "record": record_id}, data=stats),
            dcc.Store(id={"type": "dim-slice-store", "record": record_id}, data=[]),
            html.H6("Slice extremes", className="mt-4"),
            html.Div(
                _render_slice_error_table(record_id, stats, []),
                id={"type": "dim-slice-table", "record": record_id},
            ),
        ]
    )


def _build_dimensional_plot_children(
    records: dict[int, pd.DataFrame],
    slice_cols: list[str],
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
        error_table = _make_slice_error_table(record_id, df, slice_cols)
        children.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        _render_plot_component(
                            fig,
                            class_name="mb-3",
                            alt=f"Record {record_id} slice plot",
                        ),
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


def _serialise_dimensional_records(
    records: dict[int, pd.DataFrame], slice_cols: list[str]
) -> dict[str, object]:
    return {
        "slice_cols": list(slice_cols),
        "records": [
            {
                "record_id": int(record_id),
                "rows": df.to_dict(orient="records"),
            }
            for record_id, df in records.items()
        ],
    }


def _deserialise_dimensional_records(
    payload: dict[str, object] | None,
) -> tuple[dict[int, pd.DataFrame], list[str]]:
    if not payload:
        return {}, []

    slice_cols = [str(col) for col in payload.get("slice_cols", [])]
    records: dict[int, pd.DataFrame] = {}
    for entry in payload.get("records", []):
        try:
            record_id = int(entry.get("record_id"))
        except (TypeError, ValueError):
            continue
        rows = entry.get("rows") or []
        records[record_id] = pd.DataFrame(rows)

    return records, slice_cols


def _tensile_slot_records(tensile: TensileTest) -> tuple[dict[int, list[int]], str]:
    df = getattr(tensile, "df", None)
    if df is None or df.empty:
        return {}, ""

    slot_col = "Slot" if "Slot" in df.columns else "Record"
    slot_note = (
        "Tensile slots parsed from ASCII export; mappings show Slot → Record values from the file."
        if slot_col == "Slot"
        else (
            "Legacy tensile export lacks slot numbers; assuming tensile slots align with the Record column (mapping cannot be guaranteed)."
        )
    )
    record_col = "Record" if "Record" in df.columns else slot_col

    records_by_slot: dict[int, list[int]] = {}
    for slot, grp in df.groupby(slot_col, sort=True):
        try:
            slot_int = int(slot)
        except (TypeError, ValueError):
            continue
        rec_series = grp[record_col] if record_col in grp else pd.Series(dtype=int)
        try:
            records = (
                rec_series.dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        except Exception:
            records = []
        records_by_slot[slot_int] = sorted(records)
    return records_by_slot, slot_note


def _render_slot_alignment(dim_data: DimensionalData, tensile: TensileTest) -> html.Div:
    dim_records = getattr(dim_data, "slot_records", {}) or {}
    ten_records, ten_slot_note = _tensile_slot_records(tensile)
    all_slots = sorted(set(dim_records) | set(ten_records))

    if not all_slots:
        return dbc.Alert(
            "No slots detected in dimensional or tensile data.",
            color="warning",
            className="mb-0",
        )

    header = html.Thead(
        html.Tr(
            [
                html.Th("Slot"),
                html.Th("Dimensional Record(s)"),
                html.Th("Tensile Slot → Record(s)"),
            ]
        )
    )
    rows = []
    for slot in all_slots:
        dim_vals = dim_records.get(slot, [])
        ten_vals = ten_records.get(slot, [])
        rows.append(
            html.Tr(
                [
                    html.Td(slot),
                    html.Td(", ".join(str(r) for r in dim_vals) or "—"),
                    html.Td(
                        ", ".join(f"{slot} → {rec}" for rec in ten_vals) if ten_vals else "—"
                    ),
                ]
            )
        )
    body = html.Tbody(rows)
    table = dbc.Table([header, body], bordered=True, hover=True, size="sm", className="mb-0")

    card_children: list[Component] = [
        html.H5("Slot / Record alignment", className="mb-3"),
    ]
    if ten_slot_note:
        card_children.append(html.Small(ten_slot_note, className="text-muted d-block mb-2"))
    card_children.append(table)

    return dbc.Card(
        dbc.CardBody(card_children),
        className="mb-4 shadow-sm",
    )


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
    default_dim, default_tensile, default_conds = _load_experiment(root)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Hair-mech",
        suppress_callback_exceptions=True,
    )

    multi_cassette_layout = register_multi_cassette_page(app)

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
                        html.Div(
                            [
                                dbc.Label("Export directory", html_for="dim-export-dir"),
                                dbc.Input(
                                    id="dim-export-dir",
                                    type="text",
                                    placeholder="e.g. C:/Data/Exports",
                                    required=True,
                                ),
                                dbc.FormText(
                                    "Provide the absolute folder path where dimensional exports should be saved.",
                                ),
                            ],
                            className="mb-3",
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
                            className="mb-3",
                        ),
                        dcc.Store(id="dim-cleaning-data"),
                        dcc.Store(id="dim-export-directory"),
                        dcc.Loading(
                            id="dim-cleaning-loading",
                            type="default",
                            custom_spinner=html.Div(
                                [
                                    dbc.Spinner(color="primary", size="md"),
                                    html.Div(
                                        "Loading...",
                                        className="fw-semibold text-primary mt-2",
                                    ),
                                ],
                                className="d-flex flex-column align-items-center py-3",
                            ),
                            children=[
                                dbc.Alert(id="dim-cleaning-alert", is_open=False, className="mt-3"),
                                html.Div(id="dim-cleaning-plots", className="mt-4"),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Generate List of Remove Slices",
                                            id="dim-generate-remove-list",
                                            color="primary",
                                            size="lg",
                                            className="w-100",
                                        ),
                                        html.Div(id="dim-removed-summary-message", className="mt-3"),
                                        html.Div(id="dim-removed-summary-table", className="mt-3"),
                                    ],
                                    id="dim-removed-summary-container",
                                    className="mt-4",
                                    style={"display": "none"},
                                ),
                            ],
                        ),
                    ]
                ),
                className="shadow-sm",
            ),
        ],
        fluid=True,
        style={"maxWidth": "1100px"},
    )

    ten_clean_layout = dbc.Container(
        [
            _header(),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Upload Tensile UVC File", className="card-title"),
                        html.P(
                            "Provide the absolute export directory and a .uvc file to run the UvWin tensile export.",
                            className="text-muted",
                        ),
                        html.Div(
                            [
                                dbc.Label("Export directory", html_for="ten-export-dir"),
                                dbc.Input(
                                    id="ten-export-dir",
                                    type="text",
                                    placeholder="e.g. C:/Data/Exports",
                                    required=True,
                                ),
                                dbc.FormText(
                                    "Tensile force-curve text files will be saved to this folder with '_tensile' appended to the filename.",
                                ),
                            ],
                            className="mb-3",
                        ),
                        dcc.Upload(
                            id="upload-ten-cleaning",
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
                            className="mb-3",
                        ),
                        dcc.Store(id="ten-cleaning-export"),
                        dcc.Loading(
                            id="ten-cleaning-loading",
                            type="default",
                            custom_spinner=html.Div(
                                [
                                    dbc.Spinner(color="primary", size="md"),
                                    html.Div(
                                        "Loading...",
                                        className="fw-semibold text-primary mt-2",
                                    ),
                                ],
                                className="d-flex flex-column align-items-center py-3",
                            ),
                            children=[
                                dbc.Alert(id="ten-cleaning-alert", is_open=False, className="mt-3"),
                                html.Div(id="ten-cleaning-result", className="mt-4"),
                                dbc.Button(
                                    "Generate Tensile_Data.txt",
                                    id="ten-cleaning-generate",
                                    color="primary",
                                    className="mt-3",
                                    disabled=True,
                                ),
                                dbc.Alert(
                                    id="ten-cleaning-export-alert",
                                    is_open=False,
                                    className="mt-3",
                                    style={"display": "none"},
                                ),
                            ],
                        ),
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
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Checklist(
                                id="debug-toggle",
                                options=[{"label": " Enable debug logging", "value": "debug"}],
                                value=[],
                                switch=True,
                                className="mb-0",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Span(
                                "Debug logging is off. Enable to print debug statements to the server console.",
                                id="debug-toggle-status",
                                className="text-muted",
                            )
                        ),
                    ],
                    className="align-items-center g-2",
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
                            "Multiple Cassette Analysis",
                            id="btn-landing-cross",
                            color="info",
                            href="/multiple-cassette",
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
        if pathname == "/multiple-cassette":
            return multi_cassette_layout
        if pathname == "/dimensional-cleaning":
            return dim_clean_layout
        if pathname == "/tensile-cleaning":
            return ten_clean_layout
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

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("btn-landing-ten-cleaning", "n_clicks"),
        prevent_initial_call=True,
    )
    def _go_ten_cleaning(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return "/tensile-cleaning"

    @app.callback(Output("cleaning-subbuttons", "style"), Input("btn-landing-cleaning", "n_clicks"))
    def _toggle_cleaning_subbuttons(n_clicks):
        hidden = {"display": "none"}
        shown = {"display": "block"}
        return shown if n_clicks else hidden

    @app.callback(
        Output("debug-toggle-status", "children"),
        Input("debug-toggle", "value"),
        prevent_initial_call=True,
    )
    def _toggle_debug_logging(values):
        enabled = "debug" in (values or [])
        _set_log_level(enabled)
        status = (
            "Debug logging is on. Check the server console for detailed output."
            if enabled
            else "Debug logging is off. Enable to print debug statements to the server console."
        )
        if enabled:
            logger.debug("Debug logging enabled via home page toggle.")
        else:
            logger.info("Debug logging disabled via home page toggle.")
        return status

    @app.callback(
        Output("dim-cleaning-alert", "children"),
        Output("dim-cleaning-alert", "color"),
        Output("dim-cleaning-alert", "is_open"),
        Output("dim-cleaning-plots", "children"),
        Output("dim-cleaning-data", "data"),
        Output("dim-export-directory", "data"),
        Output("dim-removed-summary-container", "style"),
        Input("upload-dim-cleaning", "contents"),
        Input("dim-export-dir", "value"),
        State("upload-dim-cleaning", "filename"),
        prevent_initial_call=True,
    )
    def _process_dimensional_cleaning(contents, preferred_dir, filename):
        if not contents or not filename:
            raise PreventUpdate

        try:
            preferred_path = _parse_export_directory(preferred_dir)
        except ValueError as exc:
            return str(exc), "danger", True, [], None, None, {"display": "none"}

        raw = _b64_to_bytes(contents)
        uvc_path, original_dir = _store_uvc_file(raw, filename)

        export_path, _ = _resolve_export_target(
            uvc_path, original_dir=original_dir, preferred_dir=preferred_path
        )

        success, message, produced_paths = _run_uvwin_export(
            uvc_path,
            original_dir=original_dir,
            preferred_dir=preferred_path,
            modes=("dimensional",),
        )

        plots: list = []
        records: dict[int, pd.DataFrame] | None = None
        slice_cols: list[str] = []
        expected_gpdsr_path = export_path.with_name(
            f"{export_path.stem}_gpdsr{export_path.suffix}"
        )
        export_dir_data: dict[str, object] | None = None
        serialized_data: dict[str, object] | None = None
        summary_style = {"display": "none"}

        dimensional_path = produced_paths.get("dimensional", export_path)

        if success:
            export_dir_data = {
                "directory": str(export_path.parent),
                "output": str(dimensional_path),
                "gpdsr": str(
                    produced_paths.get("gpdsr", expected_gpdsr_path)
                ),
                "uvc_path": str(uvc_path),
                "original_dir": str(original_dir) if original_dir else None,
                "preferred_dir": str(preferred_path) if preferred_path else None,
                "exports": {
                    mode: str(path) for mode, path in produced_paths.items()
                },
            }

        if success and dimensional_path.exists():
            try:
                records, slice_cols = parse_dimensional_export(dimensional_path)
            except Exception as exc:
                plots = [
                    dbc.Alert(
                        f"Export succeeded but the output could not be parsed: {exc}",
                        color="danger",
                        className="mt-3",
                    )
                ]
            else:
                serialized_data = _serialise_dimensional_records(records, slice_cols)
                plots = _build_dimensional_plot_children(records, slice_cols)
                if records:
                    summary_style = {"display": "block"}

        return (
            message,
            ("success" if success else "danger"),
            True,
            plots,
            serialized_data,
            export_dir_data,
            summary_style,
        )

    @app.callback(
        Output("ten-cleaning-alert", "children"),
        Output("ten-cleaning-alert", "color"),
        Output("ten-cleaning-alert", "is_open"),
        Output("ten-cleaning-result", "children"),
        Output("ten-cleaning-export", "data"),
        Output("ten-cleaning-generate", "disabled"),
        Output("ten-cleaning-export-alert", "children", allow_duplicate=True),
        Output("ten-cleaning-export-alert", "color", allow_duplicate=True),
        Output("ten-cleaning-export-alert", "is_open", allow_duplicate=True),
        Output("ten-cleaning-export-alert", "style", allow_duplicate=True),
        Input("upload-ten-cleaning", "contents"),
        State("upload-ten-cleaning", "filename"),
        State("ten-export-dir", "value"),
        prevent_initial_call=True,
    )
    def _process_tensile_cleaning(contents, filename, preferred_dir):
        if not contents or not filename:
            raise PreventUpdate

        try:
            preferred_path = _parse_export_directory(preferred_dir)
        except ValueError as exc:
            return (
                str(exc),
                "danger",
                True,
                [],
                None,
                True,
                None,
                "info",
                False,
                {"display": "none"},
            )

        raw = _b64_to_bytes(contents)
        uvc_path, original_dir = _store_uvc_file(raw, filename)

        export_path, _ = _resolve_export_target(
            uvc_path, original_dir=original_dir, preferred_dir=preferred_path
        )
        expected_tensile_path = export_path.with_name(
            f"{uvc_path.stem}_tensile{export_path.suffix}"
        )

        success, message, produced_paths = _run_uvwin_export(
            uvc_path,
            original_dir=original_dir,
            preferred_dir=preferred_path,
            modes=("tensile",),
        )

        output_path = produced_paths.get("tensile", expected_tensile_path)

        result_children: list = []
        store_payload: dict[str, object] | None = None
        if success:
            result_children = [
                html.H5("Tensile export output", className="mb-2"),
                html.P(
                    [
                        "Force-curve data saved to ",
                        html.Code(str(output_path)),
                        ".",
                    ],
                    className="mb-2",
                ),
            ]

            if output_path.exists():
                try:
                    tensile_df = _parse_tensile_ascii(output_path)
                except Exception as exc:  # pragma: no cover - defensive
                    result_children.append(
                        dbc.Alert(
                            f"Tensile output parsed with an error: {exc}",
                            color="danger",
                            className="mb-0",
                        )
                    )
                else:
                    if tensile_df.empty:
                        result_children.append(
                            dbc.Alert(
                                "No force/strain data were found in the tensile export.",
                                color="warning",
                                className="mb-0",
                            )
                        )
                    else:
                        store_rows: list[dict[str, object]] = []
                        for row in tensile_df[
                            ["Slot", "Record", "Index", "Strain_pct", "Force_N"]
                        ].to_dict("records"):
                            try:
                                slot_val = int(row.get("Slot"))
                            except (TypeError, ValueError):
                                continue
                            try:
                                record_val = int(row.get("Record"))
                            except (TypeError, ValueError):
                                record_val = None
                            try:
                                index_val = int(row.get("Index"))
                            except (TypeError, ValueError):
                                index_val = None
                            try:
                                strain_val = float(row.get("Strain_pct"))
                            except (TypeError, ValueError):
                                strain_val = None
                            try:
                                force_val = float(row.get("Force_N"))
                            except (TypeError, ValueError):
                                force_val = None

                            store_rows.append(
                                {
                                    "Slot": slot_val,
                                    "Record": record_val,
                                    "Index": index_val,
                                    "Strain_pct": strain_val,
                                    "Force_N": force_val,
                                }
                            )

                        if store_rows:
                            store_payload = {
                                "export_directory": str(output_path.parent),
                                "tensile_path": str(output_path),
                                "uvc_path": str(uvc_path),
                                "data": store_rows,
                            }
                        result_children.append(
                            _render_plot_component(
                                _build_tensile_force_figure(tensile_df),
                                graph_config={"displaylogo": False},
                                alt="Force versus strain curves",
                            )
                        )
            else:
                result_children.append(
                    dbc.Alert(
                        "Expected tensile export file could not be located for plotting.",
                        color="warning",
                        className="mb-0",
                    )
                )

        button_disabled = store_payload is None

        return (
            message,
            ("success" if success else "danger"),
            True,
            result_children,
            store_payload,
            button_disabled,
            None,
            "info",
            False,
            {"display": "none"},
        )

    @app.callback(
        Output("ten-cleaning-export-alert", "children", allow_duplicate=True),
        Output("ten-cleaning-export-alert", "color", allow_duplicate=True),
        Output("ten-cleaning-export-alert", "is_open", allow_duplicate=True),
        Output("ten-cleaning-export-alert", "style", allow_duplicate=True),
        Input("ten-cleaning-generate", "n_clicks"),
        State("ten-cleaning-export", "data"),
        prevent_initial_call=True,
    )
    def _generate_tensile_export(n_clicks, stored_payload):
        if not n_clicks:
            raise PreventUpdate

        if not stored_payload or not stored_payload.get("data"):
            return (
                "No tensile data are available to export.",
                "warning",
                True,
                {},
            )

        try:
            df = pd.DataFrame(stored_payload.get("data", []))
        except Exception as exc:  # pragma: no cover - defensive
            return (
                f"Unable to reconstruct tensile data: {exc}",
                "danger",
                True,
                {},
            )

        required_cols = {"Slot", "Strain_pct", "Force_N"}
        missing = required_cols - set(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            return (
                f"Tensile data are missing required fields: {missing_list}.",
                "danger",
                True,
                {},
            )

        if {"Slot", "Index"}.issubset(df.columns):
            df = df.sort_values(["Slot", "Index"]).reset_index(drop=True)
        elif {"Slot", "Record", "Index"}.issubset(df.columns):
            df = df.sort_values(["Slot", "Record", "Index"]).reset_index(drop=True)

        export_directory = stored_payload.get("export_directory")
        if not export_directory:
            return (
                "Export directory information is unavailable.",
                "danger",
                True,
                {},
            )

        export_dir_path = Path(export_directory)
        try:
            export_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            return (
                f"Unable to access export directory '{export_dir_path}': {exc}",
                "danger",
                True,
                {},
            )

        output_path = export_dir_path / "Tensile_Data.txt"
        uvc_source = stored_payload.get("uvc_path")

        try:
            record = pd.to_numeric(df["Slot"], errors="coerce")
            strain = pd.to_numeric(df["Strain_pct"], errors="coerce")
            force_n = pd.to_numeric(df["Force_N"], errors="coerce")
        except Exception as exc:  # pragma: no cover - defensive
            return (
                f"Unable to normalise tensile data: {exc}",
                "danger",
                True,
                {},
            )

        export_df = pd.DataFrame(
            {
                "Record": record,
                "% Strain": strain,
                "gmf": force_n / TensileTest.GF_TO_N,
            }
        )
        export_df = export_df.dropna(subset=["Record", "% Strain", "gmf"])

        if export_df.empty:
            return (
                "No valid tensile rows remain after processing.",
                "warning",
                True,
                {},
            )

        try:
            export_df["Record"] = export_df["Record"].round().astype(int)
        except Exception as exc:  # pragma: no cover - defensive
            return (
                f"Unable to coerce slot identifiers: {exc}",
                "danger",
                True,
                {},
            )

        if "Index" in df.columns:
            export_df = export_df.assign(_Index=pd.to_numeric(df["Index"], errors="coerce"))
            export_df = export_df.sort_values(["Record", "_Index"])
            export_df = export_df.drop(columns=["_Index"])
        else:
            export_df = export_df.sort_values(["Record", "% Strain"])

        export_df = _trim_gmf_pivot(export_df)

        if export_df.empty:
            return (
                "No valid tensile rows remain after trimming initial strain data.",
                "warning",
                True,
                {},
            )

        export_df = export_df.sort_values(["Record", "% Strain"]).reset_index(drop=True)

        header_lines = ["Tensile Data Report Version: 1.0"]
        if uvc_source:
            header_lines.append(f"Source File: {uvc_source}")
        else:
            header_lines.append("Source File:")

        try:
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(header_lines) + "\n")
                export_df.to_csv(
                    handle,
                    sep="\t",
                    index=False,
                    lineterminator="\n",
                )
        except Exception as exc:  # pragma: no cover - defensive
            return (
                f"Unable to write Tensile_Data.txt: {exc}",
                "danger",
                True,
                {},
            )

        return (
            f"Tensile_Data.txt saved to {output_path}.",
            "success",
            True,
            {},
        )

    @app.callback(
        Output({"type": "dim-slice-store", "record": MATCH}, "data"),
        Output({"type": "dim-slice-table", "record": MATCH}, "children"),
        Input({"type": "dim-slice-toggle", "record": MATCH, "slice": ALL}, "n_clicks"),
        State({"type": "dim-slice-store", "record": MATCH}, "data"),
        State({"type": "dim-slice-data", "record": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def _toggle_dim_slice(_, removed, stats):
        ctx = dash.callback_context
        triggered = ctx.triggered_id if ctx.triggered_id else None
        if not triggered:
            raise PreventUpdate
        if not stats:
            raise PreventUpdate

        slice_name = triggered.get("slice")
        record_id = triggered.get("record")
        if slice_name is None or record_id is None:
            raise PreventUpdate

        removed_list = list(removed or [])
        if slice_name in removed_list:
            removed_list = [s for s in removed_list if s != slice_name]
        else:
            removed_list.append(slice_name)

        table = _render_slice_error_table(record_id, stats, removed_list)
        return removed_list, table

    @app.callback(
        Output("dim-removed-summary-message", "children"),
        Output("dim-removed-summary-table", "children"),
        Output("dim-export-directory", "data", allow_duplicate=True),
        Input("dim-generate-remove-list", "n_clicks"),
        State({"type": "dim-slice-store", "record": ALL}, "data"),
        State({"type": "dim-slice-store", "record": ALL}, "id"),
        State({"type": "dim-slice-data", "record": ALL}, "data"),
        State("dim-export-directory", "data"),
        prevent_initial_call=True,
    )
    def _generate_removed_slice_summary(
        n_clicks, removed_lists, store_ids, slice_stats_lists, export_directory
    ):
        if not n_clicks:
            raise PreventUpdate

        if not store_ids:
            message = dbc.Alert(
                "No dimensional records are currently available.",
                color="warning",
                className="mb-0",
            )
            return message, None, export_directory

        summary_rows = []
        removed_by_record: dict[int, set[str]] = {}
        stats_by_record: dict[int, list[dict[str, float]]] = {}

        removed_lists = removed_lists or [None] * len(store_ids)
        slice_stats_lists = slice_stats_lists or [None] * len(store_ids)

        for store_id, removed, stats in zip(
            store_ids, removed_lists, slice_stats_lists
        ):
            record_id = store_id.get("record") if isinstance(store_id, dict) else None
            if record_id is None:
                continue
            try:
                record_int = int(record_id)
            except (TypeError, ValueError):
                continue

            removed_list = [str(s) for s in (removed or [])]
            removed_display = ", ".join(removed_list) if removed_list else "None"
            summary_rows.append({"Record": record_int, "Removed slice(s)": removed_display})
            removed_by_record[record_int] = set(removed_list)

            if isinstance(stats, list):
                valid_stats: list[dict[str, float]] = []
                for entry in stats:
                    if isinstance(entry, dict):
                        valid_stats.append(entry)
                if valid_stats:
                    stats_by_record[record_int] = valid_stats

        if not summary_rows:
            message = dbc.Alert(
                "No slice removals have been selected.",
                color="info",
                className="mb-0",
            )
            return message, None, export_directory

        summary_df = pd.DataFrame(summary_rows).sort_values("Record").reset_index(drop=True)

        store_payload: dict[str, object] | None = None
        exports_map: dict[str, str] = {}
        if isinstance(export_directory, dict):
            store_payload = {k: v for k, v in export_directory.items()}
            existing_exports = store_payload.get("exports")
            if isinstance(existing_exports, dict):
                exports_map = {k: str(v) for k, v in existing_exports.items()}
            store_payload["exports"] = exports_map
        elif isinstance(export_directory, str):
            store_payload = {"directory": export_directory, "exports": exports_map}

        def _path_from_store(value: object | None) -> Path | None:
            if isinstance(value, Path):
                return value
            if isinstance(value, str) and value.strip():
                return Path(value)
            return None

        export_dir_path = _path_from_store(store_payload.get("directory")) if store_payload else None
        output_path = _path_from_store(store_payload.get("output")) if store_payload else None
        gpdsr_path = _path_from_store(store_payload.get("gpdsr")) if store_payload else None

        expected_gpdsr: Path | None = None
        if output_path is not None:
            expected_gpdsr = output_path.with_name(
                f"{output_path.stem}_gpdsr{output_path.suffix}"
            )
            if gpdsr_path is None:
                gpdsr_path = expected_gpdsr
        if export_dir_path is None and output_path is not None:
            export_dir_path = output_path.parent

        if gpdsr_path is None and export_dir_path is not None:
            candidates = sorted(export_dir_path.glob("*_gpdsr*.txt"))
            expected_name = expected_gpdsr.name if expected_gpdsr is not None else None
            if expected_name:
                for candidate in candidates:
                    if candidate.name == expected_name:
                        gpdsr_path = candidate
                        break
            if gpdsr_path is None and len(candidates) == 1:
                gpdsr_path = candidates[0]

        if store_payload is not None:
            if gpdsr_path is not None:
                store_payload["gpdsr"] = str(gpdsr_path)
                if gpdsr_path.exists():
                    exports_map.setdefault("gpdsr", str(gpdsr_path))
            elif expected_gpdsr is not None and "gpdsr" not in store_payload:
                store_payload["gpdsr"] = str(expected_gpdsr)
        gpdsr_alerts: list = []
        slot_lookup: dict[int, int] = {}

        def _append_reexport_alert(message: str, success: bool, produced_gpdsr: bool) -> None:
            color = "success" if success and produced_gpdsr else ("warning" if success else "danger")
            gpdsr_alerts.append(
                dbc.Alert(
                    message,
                    color=color,
                    className="mb-2",
                )
            )

        needs_gpdsr = gpdsr_path is None or not gpdsr_path.exists()
        if needs_gpdsr and store_payload is not None:
            uvc_source = _path_from_store(store_payload.get("uvc_path"))
            original_source = _path_from_store(store_payload.get("original_dir"))
            preferred_source = _path_from_store(store_payload.get("preferred_dir"))
            if preferred_source is None:
                preferred_source = export_dir_path
            if uvc_source is not None:
                success, export_message, produced = _run_uvwin_export(
                    uvc_source,
                    original_dir=original_source,
                    preferred_dir=preferred_source,
                    modes=("gpdsr",),
                )
                produced_gpdsr = "gpdsr" in produced
                _append_reexport_alert(export_message, success, produced_gpdsr)
                if success and produced_gpdsr:
                    gpdsr_path = produced["gpdsr"]
                    exports_map["gpdsr"] = str(gpdsr_path)
                    if store_payload is not None:
                        store_payload["gpdsr"] = str(gpdsr_path)
                needs_gpdsr = gpdsr_path is None or not (gpdsr_path and gpdsr_path.exists())
            else:
                gpdsr_alerts.append(
                    dbc.Alert(
                        "GPDSR file could not be regenerated because the UVC file location is unavailable.",
                        color="warning",
                        className="mb-2",
                    )
                )
        elif needs_gpdsr and store_payload is None:
            gpdsr_alerts.append(
                dbc.Alert(
                    "GPDSR file information is unavailable. Slot numbers may be incomplete.",
                    color="warning",
                    className="mb-2",
                )
            )

        if needs_gpdsr:
            if gpdsr_path is None:
                if export_dir_path is not None:
                    gpdsr_alerts.append(
                        dbc.Alert(
                            "Slot numbers could not be determined because the GPDSR file was not located.",
                            color="warning",
                            className="mb-2",
                        )
                    )
            elif not gpdsr_path.exists():
                gpdsr_alerts.append(
                    dbc.Alert(
                        f"GPDSR file '{gpdsr_path}' was not found. Slot numbers will be omitted.",
                        color="warning",
                        className="mb-2",
                    )
                )
        else:
            try:
                gpdsr_df, deduped_slots = _parse_gpdsr_mapping(gpdsr_path)
            except Exception as exc:
                gpdsr_alerts.append(
                    dbc.Alert(
                        f"Unable to parse GPDSR file '{gpdsr_path}': {exc}",
                        color="warning",
                        className="mb-2",
                    )
                )
            else:
                slot_lookup = dict(zip(gpdsr_df["Record"], gpdsr_df["Slot"]))
                if deduped_slots:
                    slot_list = ", ".join(str(slot) for slot in deduped_slots)
                    gpdsr_alerts.append(
                        dbc.Alert(
                            "Duplicate slot entries detected in the GPDSR file. "
                            f"Keeping the highest Record for slots: {slot_list}.",
                            color="info",
                            className="mb-2",
                        )
                    )

        summary_df["Slot"] = summary_df["Record"].map(slot_lookup)
        try:
            summary_df["Slot"] = summary_df["Slot"].astype("Int64")
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

        summary_df = summary_df[["Record", "Slot", "Removed slice(s)"]]

        table_header = html.Thead(
            html.Tr([html.Th("Record"), html.Th("Slot"), html.Th("Removed slice(s)")])
        )
        table_rows = []
        for row in summary_df.to_dict("records"):
            slot_value = row.get("Slot")
            slot_display = "Unknown" if pd.isna(slot_value) else str(int(slot_value))
            table_rows.append(
                html.Tr(
                    [
                        html.Td(row["Record"]),
                        html.Td(slot_display),
                        html.Td(row["Removed slice(s)"]),
                    ]
                )
            )
        table_body = html.Tbody(table_rows)
        table = dbc.Table([table_header, table_body], bordered=True, hover=True, className="mb-0")

        alerts: list = []
        alerts.extend(gpdsr_alerts)

        if export_dir_path is None:
            csv_message = dbc.Alert(
                "Summary generated, but no export directory is available for saving the CSV.",
                color="warning",
                className="mb-0",
            )
        else:
            csv_path = export_dir_path / "removed_slices_summary.csv"
            try:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                summary_df.to_csv(csv_path, index=False)
            except Exception as exc:  # pragma: no cover - filesystem errors are user specific
                csv_message = dbc.Alert(
                    f"Unable to save removed slices summary to '{csv_path}': {exc}",
                    color="danger",
                    className="mb-0",
                )
            else:
                csv_message = dbc.Alert(
                    f"Removed slices summary saved to '{csv_path}'.",
                    color="success",
                    className="mb-0",
                )

        alerts.append(csv_message)

        autodim_message = None
        if export_dir_path is not None:
            autodim_path = export_dir_path / "removed_slices_summary_autodimensional.txt"

            autodim_lines = [
                "Auto-generated dimensional summary",
                "Record\tSlot\tDescription\tSlice No.\tCross-Sectional Area\tMax Diameter\tMin Diameter",
            ]

            autodim_rows = []
            for record, stats in sorted(stats_by_record.items()):
                removed_set = removed_by_record.get(record, set())
                slot_series = summary_df.loc[
                    summary_df["Record"] == record, "Slot"
                ]
                slot_value = slot_series.iloc[0] if not slot_series.empty else None
                try:
                    slot_number = int(slot_value) if pd.notna(slot_value) else int(record)
                except (TypeError, ValueError):
                    slot_number = int(record)

                for idx, entry in enumerate(stats, start=1):
                    slice_name = str(entry.get("slice", idx))
                    if slice_name in removed_set:
                        continue

                    min_val = entry.get("min")
                    max_val = entry.get("max")
                    try:
                        min_f = float(min_val)
                        max_f = float(max_val)
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(min_f) or pd.isna(max_f):
                        continue

                    slice_digits = re.findall(r"\d+", slice_name)
                    slice_no = int(slice_digits[-1]) if slice_digits else idx

                    area = float(np.pi * (min_f / 2.0) * (max_f / 2.0))
                    autodim_rows.append(
                        (
                            record,
                            slot_number,
                            slice_no,
                            area,
                            max_f,
                            min_f,
                        )
                    )

            if autodim_rows:
                for row in autodim_rows:
                    record, slot_number, slice_no, area, max_f, min_f = row
                    description = f"Slot {slot_number} : Auto-generated"
                    autodim_lines.append(
                        "\t".join(
                            [
                                str(record),
                                str(slot_number),
                                description,
                                str(slice_no),
                                f"{area:.4f}",
                                f"{max_f:.4f}",
                                f"{min_f:.4f}",
                            ]
                        )
                    )

                autodim_text = "\n".join(autodim_lines) + "\n"
                try:
                    autodim_path.write_text(autodim_text)
                except Exception as exc:  # pragma: no cover - filesystem errors
                    autodim_message = dbc.Alert(
                        f"Unable to save autodimensional summary to '{autodim_path}': {exc}",
                        color="danger",
                        className="mb-0",
                    )
                else:
                    autodim_message = dbc.Alert(
                        f"Autodimensional summary saved to '{autodim_path}'.",
                        color="success",
                        className="mb-0",
                    )
                    exports_map.setdefault("autodimensional", str(autodim_path))
                    if store_payload is not None:
                        store_payload["autodimensional"] = str(autodim_path)
            else:
                autodim_message = dbc.Alert(
                    "Autodimensional summary could not be generated because no slice statistics were available.",
                    color="warning",
                    className="mb-0",
                )

        if autodim_message is not None:
            alerts.append(autodim_message)

        updated_store = store_payload if store_payload is not None else export_directory

        return (
            html.Div(alerts),
            dbc.Card(dbc.CardBody(table), className="shadow-sm"),
            updated_store,
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
        dim_data = _bytes_to_dim(_b64_to_bytes(dim_b64))
        areas = dim_data.map
        tensile_raw = _bytes_to_ten(_b64_to_bytes(ten_b64))
        tensile, _ = _remap_tensile_slots(tensile_raw, None)
        total = _max_slot(dim_data.map, tensile)
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

        dim_bytes = _b64_to_bytes(dim_b64)
        dim_data = _bytes_to_dim(dim_bytes)

        payload: dict[str, object] = {
            "dim_b64": dim_b64,
            "ten_b64": ten_b64,
            "conds": [asdict(c) for c in conds],
        }

        return json.dumps(payload)

    def _analysis_inputs(payload: str | None):
        gpdsr_feedback = {"warnings": [], "infos": []}

        if not payload:
            mapped_tensile, _ = _remap_tensile_slots(default_tensile, None)
            return default_dim, mapped_tensile, default_conds, None, None

        data = json.loads(payload)
        removed_info = data.get("removed") or {}
        entries = removed_info.get("entries") or []
        removal_map = _entries_to_removal_map(entries)

        gpdsr_info = data.get("gpdsr") or {}
        mapping_data = gpdsr_info.get("mapping")
        slot_map: dict[int, int] | None = None
        if isinstance(mapping_data, dict):
            try:
                slot_map = {int(rec): int(slot) for rec, slot in mapping_data.items()}
            except (TypeError, ValueError):  # pragma: no cover - defensive
                slot_map = None
        elif isinstance(mapping_data, list):
            slot_map = {}
            for entry in mapping_data:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                try:
                    rec_val = int(entry[0])
                    slot_val = int(entry[1])
                except (TypeError, ValueError):
                    continue
                slot_map[rec_val] = slot_val
            if not slot_map:
                slot_map = None

        dim_data = _bytes_to_dim(
            _b64_to_bytes(data["dim_b64"]),
            removal_map if removal_map else None,
            slot_map=slot_map,
        )
        feedback = removed_info.get("feedback")
        if entries and not feedback:
            feedback = _build_removed_feedback_data(entries, dim_data)

        deduped_slots = gpdsr_info.get("deduped_slots") or []
        if deduped_slots:
            slot_list = ", ".join(str(s) for s in deduped_slots)
            gpdsr_feedback["infos"].append(
                "Duplicate slot entries detected in GPDSR file. Keeping the highest Record "
                f"for slots: {slot_list}."
            )

        tensile_raw = _bytes_to_ten(_b64_to_bytes(data["ten_b64"]))
        tensile, _ = _remap_tensile_slots(tensile_raw, None)

        conds = [Condition(**c) for c in data["conds"]]

        if not gpdsr_feedback["warnings"] and not gpdsr_feedback["infos"]:
            gpdsr_feedback = None

        return dim_data, tensile, conds, feedback, gpdsr_feedback

    # Plot
    @app.callback(
        Output("fig-container", "children"),
        Input("tabs", "value"),
        Input("exp-data", "data"),
    )
    def _draw(tab, payload):
        dim_data, tensile, conds, feedback, gpdsr_feedback = _analysis_inputs(payload)
        fig = (
            _overlay_fig(dim_data.map, tensile, conds)
            if tab == "overlay"
            else _violin_fig(dim_data.map, tensile, conds)
        )
        layout_height = getattr(fig.layout, "height", None)
        graph_height = 750
        if isinstance(layout_height, (int, float)):
            graph_height = max(graph_height, int(layout_height))

        children = [
            _render_plot_component(
                fig,
                graph_height=graph_height,
                class_name="mb-5",
                style={"marginBottom": "8rem"},
                alt="Overlay plot" if tab == "overlay" else "Violin grid",
            )
        ]
        alerts = _render_removed_feedback(feedback)
        if alerts:
            children.extend(alerts)
        gpdsr_alerts = _render_gpdsr_feedback(gpdsr_feedback)
        if gpdsr_alerts:
            children.extend(gpdsr_alerts)
        children.append(_render_slot_alignment(dim_data, tensile))
        return html.Div(children)

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
        dim_data, tensile, conds, _, _ = _analysis_inputs(payload)
        df = build_summary(dim_data.map, tensile, conds)
        return dcc.send_bytes(to_excel_bytes({"Metrics": df}), fname or "metrics.xlsx")

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
        dim_data, tensile, conds, _, _ = _analysis_inputs(payload)

        summary = build_summary(dim_data.map, tensile, conds)
        metrics_od = OrderedDict(
            (c, c.replace("_", " "))
            for c in summary.columns
            if c not in ("Slot", "Condition", "Record")
        )
        long = build_stats(summary, conds, metrics_od)
        control_name = next(c.name for c in conds if c.is_control)
        condition_order = [c.name for c in conds]
        wide = long_to_wide(
            long,
            summary,
            control_name,
            metrics_od,
            condition_order=condition_order,
        )

        return dcc.send_bytes(to_excel_bytes({"Stats": wide}), fname or "stats.xlsx")

    return app


# module-level instance (picked up by unit tests)
app: Dash = build_dash_app()
__all__ = ["build_dash_app", "app"]
