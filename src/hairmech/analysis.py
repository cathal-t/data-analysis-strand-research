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

# Canonical mapping of metric column names → human-friendly labels used across
# CLI exports, figures, and UI components.  Keep this in sync with the columns
# emitted by ``build_summary``.
METRIC_LABELS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("Elastic_Modulus_GPa", "Elastic modulus (GPa)"),
        ("Yield_Gradient_MPa_perc", "Yield-grad. (MPa / %ε)"),
        ("Post_Gradient_MPa_perc", "Post-grad. (MPa / %ε)"),
        ("Break_Stress_MPa", "Break stress (MPa)"),
        ("Break_Strain_%", "Break strain (%)"),
    ]
)

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
    conditions: List[Condition],
    metrics: "OrderedDict[str, str]",
) -> pd.DataFrame:
    """
    Welch-t vs control, Cohen-d, % change – returned in a **long / tidy**
    layout.  ``metrics`` **must be an OrderedDict** whose key is the
    column name in *summary_df* and whose value is the pretty label.  The
    order of that OrderedDict drives the column order in the wide export.
    """
    control = _control_name(conditions)
    condition_order = [c.name for c in conditions]
    rows: list[list] = []

    for col, nice in metrics.items():
        ctrl_vals = summary_df.loc[summary_df["Condition"] == control, col].dropna()
        ctrl_mean = ctrl_vals.mean()

        for cond in condition_order:
            test_vals = summary_df.loc[summary_df["Condition"] == cond, col].dropna()

            # always emit a row for the control itself
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


# ──────────────────────────────── to wide ─────────────────────────────


def long_to_wide(
    stats_long: pd.DataFrame,
    summary_df: pd.DataFrame,
    control_name: str,
    metrics: "OrderedDict[str, str] | None" = None,
    condition_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pivot *stats_long* to the 2-level wide layout expected by Excel.

    *metrics* (if given) provides the desired **order** of metric
    blocks; otherwise the first-appearance order in *stats_long* is used.
    *condition_order* (if given) determines the row order, otherwise the
    control is placed first followed by first-appearance order.
    """
    wanted_stats = ["Test_Mean", "% Change", "p", "d", "Effect"]

    # ---- % change vs control -----------------------------------------
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

    # ---- metric block ordering ---------------------------------------
    if metrics is not None:
        metric_order = [metrics[col] for col in metrics]  # nice labels in order
        sub_order = wanted_stats
        # produce a new column order list
        ordered_cols: list[tuple] = []
        for met in metric_order:
            for sub in sub_order:
                tup = (met, sub)
                if tup in wide.columns:
                    ordered_cols.append(tup)
        wide = wide[ordered_cols]

    # ---- insert N -----------------------------------------------------
    n_sizes = summary_df.reset_index().groupby("Condition").size()
    wide.insert(0, ("", "N"), n_sizes.reindex(wide.index).astype("Int64"))

    # ---- control row first -------------------------------------------
    if condition_order:
        ordered_rows = [cond for cond in condition_order if cond in wide.index]
        ordered_rows.extend(idx for idx in wide.index if idx not in ordered_rows)
        wide = wide.reindex(ordered_rows)
    elif control_name in wide.index:
        wide = wide.reindex(
            [control_name] + [idx for idx in wide.index if idx != control_name]
        )

    return wide
