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
from dash.dependencies import ALL, MATCH, Input, Output, State
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


def _bytes_to_dim(
    raw: bytes, removed: dict[int, set[int]] | None = None
) -> DimensionalData:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(raw)
        tmp.flush()
        return DimensionalData(Path(tmp.name), removed_slices=removed)


def _bytes_to_ten(raw: bytes) -> TensileTest:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(raw)
        tmp.flush()
        return TensileTest(Path(tmp.name))


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
            raise ValueError(
                f"Unable to determine Slot from description '{description}'"
            )
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

    gpdsr_output = output_file.with_name(
        f"{output_file.stem}_gpdsr{output_file.suffix}"
    )

    try:
        output_parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Unable to prepare export directory '{output_parent}': {exc}"

    uvc_abs = uvc_path.resolve()
    exports = [
        ("dimensional", output_file),
        ("gpdsr", gpdsr_output),
    ]

    messages: list[str] = []

    for export_name, export_path in exports:
        export_abs = export_path.resolve()

        print(
            "Dimensional export debug - UVC path: "
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
            return False, f"UvWin executable not found at '{exe_path}'."
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            details = stderr or stdout or str(exc)
            return False, f"{export_name.capitalize()} export failed: {details}"

        messages.append(result.stdout.strip())

    if output_file.exists() and gpdsr_output.exists():
        return (
            True,
            f"Export complete. Outputs saved to: {output_file} and {gpdsr_output}",
        )

    fallback = next((msg for msg in messages if msg), None)
    return True, fallback or "Dimensional export completed."


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


def _compute_slice_extremes(df: pd.DataFrame, slice_cols: list[str]) -> list[dict[str, float]]:
    cols = [c for c in slice_cols if c in df.columns]
    stats: list[dict[str, float]] = []
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
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
        error_table = _make_slice_error_table(record_id, df, slice_cols)
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
                    dbc.Col(
                        [
                            dcc.Upload(
                                id="upload-removed",
                                accept=".csv",
                                children=dbc.Button(
                                    "Upload Removed Slices", color="primary", id="btn-removed"
                                ),
                                multiple=False,
                            ),
                            html.Small(id="removed-msg", className="text-muted"),
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
                        dcc.Store(id="dim-export-directory"),
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
        Output("dim-export-directory", "data"),
        Output("dim-removed-summary-container", "style"),
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
            return str(exc), "danger", True, [], None, {"display": "none"}

        export_path, _ = _resolve_export_target(
            uvc_path, original_dir=original_dir, preferred_dir=preferred_path
        )

        success, message = _run_dimensional_export(
            uvc_path, original_dir=original_dir, preferred_dir=preferred_path
        )

        plots: list = []
        records: dict[int, pd.DataFrame] | None = None
        slice_cols: list[str] = []
        gpdsr_path = export_path.with_name(
            f"{export_path.stem}_gpdsr{export_path.suffix}"
        )
        export_dir_data: dict[str, str] | None = None
        summary_style = {"display": "none"}

        if export_path.exists():
            export_dir_data = {
                "directory": str(export_path.parent),
                "output": str(export_path),
                "gpdsr": str(gpdsr_path),
            }

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
                if records:
                    summary_style = {"display": "block"}

        if not success:
            export_dir_data = None

        return (
            message,
            ("success" if success else "danger"),
            True,
            plots,
            export_dir_data,
            summary_style,
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
        Input("dim-generate-remove-list", "n_clicks"),
        State({"type": "dim-slice-store", "record": ALL}, "data"),
        State({"type": "dim-slice-store", "record": ALL}, "id"),
        State("dim-export-directory", "data"),
        prevent_initial_call=True,
    )
    def _generate_removed_slice_summary(n_clicks, removed_lists, store_ids, export_directory):
        if not n_clicks:
            raise PreventUpdate

        if not store_ids:
            message = dbc.Alert(
                "No dimensional records are currently available.",
                color="warning",
                className="mb-0",
            )
            return message, None

        summary_rows = []
        removed_lists = removed_lists or [None] * len(store_ids)
        for store_id, removed in zip(store_ids, removed_lists):
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

        if not summary_rows:
            message = dbc.Alert(
                "No slice removals have been selected.",
                color="info",
                className="mb-0",
            )
            return message, None

        summary_df = pd.DataFrame(summary_rows).sort_values("Record").reset_index(drop=True)

        export_info = export_directory
        export_dir_path: Path | None = None
        output_path: Path | None = None
        gpdsr_path: Path | None = None

        if isinstance(export_info, dict):
            dir_value = export_info.get("directory")
            if dir_value:
                export_dir_path = Path(dir_value)
            output_value = export_info.get("output")
            if output_value:
                output_path = Path(output_value)
            gpdsr_value = export_info.get("gpdsr")
            if gpdsr_value:
                gpdsr_path = Path(gpdsr_value)
            elif output_path is not None:
                gpdsr_path = output_path.with_name(
                    f"{output_path.stem}_gpdsr{output_path.suffix}"
                )
        elif isinstance(export_info, str):
            export_dir_path = Path(export_info)

        if export_dir_path is None and output_path is not None:
            export_dir_path = output_path.parent

        if gpdsr_path is None and export_dir_path is not None:
            candidates = sorted(export_dir_path.glob("*_gpdsr*.txt"))
            expected_name = None
            if output_path is not None:
                expected_name = f"{output_path.stem}_gpdsr{output_path.suffix}"
            if expected_name:
                for candidate in candidates:
                    if candidate.name == expected_name:
                        gpdsr_path = candidate
                        break
            if gpdsr_path is None and len(candidates) == 1:
                gpdsr_path = candidates[0]

        gpdsr_alerts: list = []
        slot_lookup: dict[int, int] = {}

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

        return html.Div(alerts), dbc.Card(dbc.CardBody(table), className="shadow-sm")

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

    @app.callback(
        Output("btn-removed", "color"),
        Output("removed-msg", "children"),
        Input("upload-removed", "contents"),
        State("upload-removed", "filename"),
        prevent_initial_call=True,
    )
    def _removed_status(contents, filename):
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
        State("upload-removed", "contents"),
        State("upload-removed", "filename"),
        prevent_initial_call=True,
    )
    def _cache(_, rows, dim_b64, ten_b64, removed_b64, removed_name):
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

        removal_entries: list[dict[str, object]] = []
        removal_map: dict[int, set[int]] | None = None
        removal_feedback: dict[str, object] | None = None

        if removed_b64:
            try:
                removal_entries = _parse_removed_slices_csv(_b64_to_bytes(removed_b64))
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
            removal_map = _entries_to_removal_map(removal_entries)

        dim_bytes = _b64_to_bytes(dim_b64)
        dim_data = _bytes_to_dim(dim_bytes, removal_map if removal_map else None)
        if removal_entries:
            removal_feedback = _build_removed_feedback_data(removal_entries, dim_data)

        payload: dict[str, object] = {
            "dim_b64": dim_b64,
            "ten_b64": ten_b64,
            "conds": [asdict(c) for c in conds],
        }

        if removed_b64:
            payload["removed"] = {
                "entries": removal_entries,
                "feedback": removal_feedback,
                "filename": removed_name,
            }

        return json.dumps(payload)

    def _analysis_inputs(payload: str | None):
        if not payload:
            return default_areas, default_tensile, default_conds, None

        data = json.loads(payload)
        removed_info = data.get("removed") or {}
        entries = removed_info.get("entries") or []
        removal_map = _entries_to_removal_map(entries)
        dim_data = _bytes_to_dim(
            _b64_to_bytes(data["dim_b64"]), removal_map if removal_map else None
        )
        feedback = removed_info.get("feedback")
        if entries and not feedback:
            feedback = _build_removed_feedback_data(entries, dim_data)

        tensile = _bytes_to_ten(_b64_to_bytes(data["ten_b64"]))
        conds = [Condition(**c) for c in data["conds"]]

        return dim_data.map, tensile, conds, feedback

    # Plot
    @app.callback(
        Output("fig-container", "children"),
        Input("tabs", "value"),
        Input("exp-data", "data"),
    )
    def _draw(tab, payload):
        areas, tensile, conds, feedback = _analysis_inputs(payload)
        fig = (
            _overlay_fig(areas, tensile, conds)
            if tab == "overlay"
            else _violin_fig(areas, tensile, conds)
        )
        children = [
            dcc.Graph(
                figure=fig,
                style={"height": "750px", "marginBottom": "8rem"},
                className="mb-5",
            )
        ]
        alerts = _render_removed_feedback(feedback)
        if alerts:
            children.extend(alerts)
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
        areas, tensile, conds, _ = _analysis_inputs(payload)
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
        areas, tensile, conds, _ = _analysis_inputs(payload)

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
