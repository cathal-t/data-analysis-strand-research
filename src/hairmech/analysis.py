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
    layout with Excel-friendly formatting and structure.
    """
    control = _control_name(conditions)
    rows_tbl = []

    for col, nice in metrics.items():
        ctrl_vals = summary_df.loc[summary_df["Condition"] == control, col].dropna()
        ctrl_mean = ctrl_vals.mean()

        for cond in summary_df["Condition"].unique():
            if cond == control:
                continue

            test_vals = summary_df.loc[summary_df["Condition"] == cond, col].dropna()

            if len(ctrl_vals) and len(test_vals):
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
                rows_tbl.append([
                    nice, cond,
                    f"{ctrl_mean:.2f}", f"{test_vals.mean():.2f}",
                    f"{t:.2f}", f"{p:.3g}", f"{d:.2f}", eff
                ])

    # Define column names to match Excel export
    stat_hdr = [
        "Metric", "Test condition",
        f"{control} mean", "Test mean",
        "t", "p", "|d|", "Effect"
    ]

    stats_long = pd.DataFrame(rows_tbl, columns=stat_hdr)

    # ---- add control rows ---------------------------------------------
    ctrl_rows = []
    for metric in stats_long["Metric"].unique():
        base = stats_long.loc[stats_long["Metric"] == metric, f"{control} mean"].iat[0]
        ctrl_rows.append({
            "Metric": metric, "Test condition": control,
            f"{control} mean": base, "Test mean": base,
            "t": np.nan, "p": np.nan, "|d|": np.nan, "Effect": "-",
        })
    stats_long = pd.concat([stats_long, pd.DataFrame(ctrl_rows)], ignore_index=True)

    # ---- numeric cast -------------------------------------------------
    num_cols = ["Test mean", f"{control} mean", "t", "p", "|d|"]
    stats_long[num_cols] = stats_long[num_cols].apply(pd.to_numeric, errors="coerce")

    # ---- % change vs control ------------------------------------------
    ctrl_means = stats_long.groupby("Metric")[f"{control} mean"].first()
    stats_long["% Change"] = (
        (stats_long["Test mean"] - stats_long["Metric"].map(ctrl_means))
        / stats_long["Metric"].map(ctrl_means)
    ).fillna(0)

    return stats_long


def long_to_wide(
    stats_long: pd.DataFrame,
    summary_df: pd.DataFrame,
    control_name: str,
    metrics: "OrderedDict[str, str] | None" = None,
) -> pd.DataFrame:
    """
    Pivot *stats_long* to the 2-level wide layout expected by Excel,
    with tidy formatting, column ordering, and % change.
    """
    wanted_stats = ["Test mean", "% Change", "p", "|d|", "Effect"]

    wide = (
        stats_long
        .set_index(["Test condition", "Metric"])[wanted_stats]
        .unstack("Metric")
        .swaplevel(axis=1)
    )

    # ---- sort columns in tidy order -----------------------------------
    if metrics is not None:
        metric_order = list(metrics.values())
    else:
        metric_order = list(dict.fromkeys(stats_long["Metric"]))

    sub_order = ["% Change", "Effect", "Test mean", "p", "|d|"]
    ordered_cols = []
    for met in metric_order:
        for sub in sub_order:
            tup = (met, sub)
            if tup in wide.columns:
                ordered_cols.append(tup)

    # ---- insert N -----------------------------------------------------
    n_sizes = summary_df.reset_index().groupby("Condition").size()
    wide.insert(0, ("", "N"), n_sizes.reindex(wide.index).astype("Int64"))

    # Final column order
    wide = wide[[("", "N")] + ordered_cols]

    # Clean up index
    wide.index.name = None

    # ---- reorder to match input order ---------------------------------
    if metrics is not None:
        all_conditions = list(summary_df["Condition"].unique())
        if control_name in all_conditions:
            all_conditions.remove(control_name)
            all_conditions = [control_name] + all_conditions
        wide = wide.reindex(all_conditions)

    return wide

