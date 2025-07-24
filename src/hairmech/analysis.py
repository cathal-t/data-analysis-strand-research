"""
High-level analysis helpers
===========================

* **build_summary** – per-slot mechanical metrics
* **build_stats**   – Welch-t / Cohen-d long table
* **long_to_wide**  – pivot to the 2-level “wide” table for Excel export
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind  # noqa: WPS433

from .io.config import Condition
from .tensile import TensileTest
from .util import post_gradient

# ────────────────────────── internal helpers ──────────────────────────


def _slots_to_condition_map(conditions: List[Condition]) -> Dict[int, str]:
    """Expand each Condition’s slot range → {slot_num: condition_name}."""
    return {slot: c.name for c in conditions for slot in c.slots}


def _control_name(conditions: List[Condition]) -> str:
    """Return the (single) control name, validated in load_config()."""
    ctrls = [c.name for c in conditions if c.is_control]
    if len(ctrls) != 1:  # pragma: no cover
        raise ValueError("Exactly one control condition is required")
    return ctrls[0]


# ─────────────────────────────── summary ──────────────────────────────


def build_summary(
    areas_map: Dict[int, float],
    tensile: TensileTest,
    conditions: List[Condition],
) -> pd.DataFrame:
    """
    Iterate over all slots, compute metrics, and return a tidy DataFrame
    indexed by **Slot**.
    """
    slot_to_cond = _slots_to_condition_map(conditions)
    rows: list[dict] = []

    for slot, df_raw in tensile.per_slot():
        if slot not in areas_map or slot not in slot_to_cond:
            continue

        df_proc = tensile.stress_strain(df_raw, areas_map[slot])
        if df_proc.empty:
            continue

        uts, bs, bp, E = TensileTest.metrics(df_proc)
        rows.append(
            {
                "Slot": slot,
                "Condition": slot_to_cond[slot],
                "UTS_MPa": uts,
                "Break_Stress_MPa": bs,
                "Break_Strain_%": bp,
                "Elastic_Modulus_GPa": E,
                "Post_Gradient_MPa_perc": post_gradient(df_proc),
                "Yield_Gradient_MPa_perc": TensileTest.yield_gradient(df_proc),
            }
        )

    return pd.DataFrame(rows).set_index("Slot")


# ─────────────────────────────── stats (long) ─────────────────────────


def build_stats(
    summary_df: pd.DataFrame,
    conditions: List["Condition"],
    metrics: "OrderedDict[str, str]",
) -> pd.DataFrame:
    """
    Welch-t vs control, Cohen-d, % change – returned in a **long / tidy**
    layout.  ``metrics`` **must be an OrderedDict** whose key is the
    column name in *summary_df* and whose value is the pretty label.  The
    order of that OrderedDict drives the column order in the wide export.
    """
    control = _control_name(conditions)
    rows: list[list] = []

    for col, nice in metrics.items():
        ctrl_vals = summary_df.loc[summary_df["Condition"] == control, col].dropna()
        ctrl_mean = ctrl_vals.mean()

        for cond in summary_df["Condition"].unique():
            test_vals = summary_df.loc[summary_df["Condition"] == cond, col].dropna()

            if cond == control:
                rows.append(
                    [nice, cond, ctrl_mean, ctrl_mean, np.nan, np.nan, np.nan, "-"]
                )
                continue

            if ctrl_vals.empty or test_vals.empty:
                continue

            t, p = ttest_ind(ctrl_vals, test_vals, equal_var=False)
            pooled = np.sqrt(
                (
                    (len(ctrl_vals) - 1) * ctrl_vals.std(ddof=1) ** 2
                    + (len(test_vals) - 1) * test_vals.std(ddof=1) ** 2
                )
                / (len(ctrl_vals) + len(test_vals) - 2)
            )
            d = abs((ctrl_mean - test_vals.mean()) / pooled) if pooled else np.nan
            eff = (
                "Large"
                if d >= 0.8
                else "Medium"
                if d >= 0.5
                else "Small"
                if d >= 0.2
                else "Negligible"
            )
            rows.append([nice, cond, ctrl_mean, test_vals.mean(), t, p, d, eff])

    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Condition",
            "Control_Mean",
            "Test_Mean",
            "t",
            "p",
            "d",
            "Effect",
        ],
    )


def long_to_wide(
    stats_long: pd.DataFrame,
    summary_df: pd.DataFrame,
    control_name: str,
    metrics: "OrderedDict[str, str] | None" = None,
) -> pd.DataFrame:
    wanted_stats = ["Test_Mean", "% Change", "p", "d", "Effect"]

    ctrl_means = (
        stats_long[stats_long["Condition"] == control_name]
        .set_index("Metric")["Control_Mean"]
    )
    stats_long["% Change"] = (
        (stats_long["Test_Mean"] - stats_long["Metric"].map(ctrl_means))
        / stats_long["Metric"].map(ctrl_means)
    ).fillna(0)

    wide = (
        stats_long.set_index(["Condition", "Metric"])[wanted_stats]
        .unstack("Metric")
        .swaplevel(axis=1)
    )

    if metrics is not None:
        metric_order = [metrics[col] for col in metrics]
        sub_order = wanted_stats
        ordered_cols: list[tuple] = []
        for met in metric_order:
            for sub in sub_order:
                tup = (met, sub)
                if tup in wide.columns:
                    ordered_cols.append(tup)
        wide = wide[ordered_cols]

    n_sizes = summary_df.reset_index().groupby("Condition").size()
    wide.insert(0, ("", "N"), n_sizes.reindex(wide.index).astype("Int64"))

    if control_name in wide.index:
        wide = wide.reindex(
            [control_name] + [idx for idx in wide.index if idx != control_name]
        )

    return wide


# ──────────────── Excel Export with Formatting ───────────────────

def export_formatted_excel(wide: pd.DataFrame):
    try:
        import xlsxwriter
        engine = "xlsxwriter"
    except ModuleNotFoundError:
        engine = None

    with pd.ExcelWriter(ROOT / f"{ROOT.name}_stats.xlsx", engine=engine) as xl:
        wide.to_excel(xl, sheet_name="Stats", index=True)

        if engine == "xlsxwriter":
            ws = xl.sheets["Stats"]
            book = xl.book

            hdr = book.add_format({
                "bold": True, "bg_color": "#dfe6e9",
                "border": 1, "align": "center"
            })
            f_txt = book.add_format({"text_wrap": True})
            f_int = book.add_format({"num_format": "0", "align": "center"})
            f_num = book.add_format({"num_format": "0.00"})
            f_pct = book.add_format({"num_format": "0.0%"})
            f_p = book.add_format({"num_format": "0.000"})
            f_sig = book.add_format({"bg_color": "#c8e6c9"})

            ws.freeze_panes(2, 2)
            ws.set_row(0, None, hdr)
            ws.set_row(1, None, hdr)
            ws.set_row(2, 0, None, {'hidden': True})

            ws.set_column(0, 0, 32, f_txt)   # Condition
            ws.set_column(1, 1, 6, f_int)    # N

            def col_letter(idx):
                s = ""
                while idx >= 0:
                    s = chr(idx % 26 + 65) + s
                    idx = idx // 26 - 1
                return s

            for col_i, (met, stat) in enumerate(wide.columns[1:], start=2):
                if stat == "Test_Mean":
                    width, fmt = 14, f_num
                elif stat == "% Change":
                    width, fmt = 12, f_pct
                elif stat == "p":
                    width, fmt = 12, f_p
                else:
                    width, fmt = 12, f_num
                ws.set_column(col_i, col_i, width, fmt)

                if stat == "p":
                    first, last = 4, 4 + len(wide) - 1
                    let = col_letter(col_i)
                    ws.conditional_format(
                        first - 1, col_i, last - 1, col_i,
                        {"type": "formula",
                         "criteria": f'=AND(ISNUMBER({let}{first}),{let}{first}<0.05)',
                         "format": f_sig})
