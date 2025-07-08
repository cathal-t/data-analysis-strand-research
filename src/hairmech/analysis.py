"""
High-level analysis helpers:

* build_summary  – per-slot mechanical metrics
* build_stats    – Welch-t / Cohen-d long table
* long_to_wide   – pivot to the 2-level “wide” table for Excel export
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind  # noqa: WPS433

from .io.config import Condition
from .util import post_gradient
from .tensile import TensileTest


# ───────────────────────────────── internal helpers ──────────────────
def _slots_to_condition_map(conditions: List[Condition]) -> Dict[int, str]:
    return {s: c.name for c in conditions for s in c.slots}


def _control_name(conditions: List[Condition]) -> str:
    ctrls = [c.name for c in conditions if c.is_control]
    if len(ctrls) != 1:  # should already be validated in load_config
        raise ValueError("Exactly one control condition is required")
    return ctrls[0]


# ─────────────────────────────────── summary ──────────────────────────
def build_summary(
    areas_map: Dict[int, float],
    tensile: TensileTest,
    conditions: List[Condition],
) -> pd.DataFrame:
    """
    Iterate over all slots, compute metrics, return tidy DataFrame.
    """
    slot_to_cond = _slots_to_condition_map(conditions)

    rows = []
    for slot, df_raw in tensile.per_slot():
        if slot not in areas_map or slot not in slot_to_cond:
            continue

        df = tensile.stress_strain(df_raw, areas_map[slot])
        if df.empty:
            continue

        uts, bs, bp, E = TensileTest.metrics(df)
        rows.append(
            {
                "Slot": slot,
                "Condition": slot_to_cond[slot],
                "UTS_MPa": uts,
                "Break_Stress_MPa": bs,
                "Break_Strain_%": bp,
                "Elastic_Modulus_GPa": E,
                "Post_Gradient_MPa_perc": post_gradient(df),
                "Yield_Gradient_MPa_perc": TensileTest.yield_gradient(df),
            }
        )

    return pd.DataFrame(rows).set_index("Slot")


# ─────────────────────────────────── stats (long) ─────────────────────
def build_stats(
    summary_df: pd.DataFrame,
    conditions: List[Condition],
    metrics: Dict[str, str],
) -> pd.DataFrame:
    """
    Welch-t vs control, effect size, % change – long (“tidy”) layout.
    """
    control = _control_name(conditions)
    rows = []

    for key, nice in metrics.items():
        ctrl_vals = summary_df[summary_df["Condition"] == control][key].dropna()
        ctrl_mean = ctrl_vals.mean()

        for cond in summary_df["Condition"].unique():
            test_vals = summary_df[summary_df["Condition"] == cond][key].dropna()

            if cond == control:  # keep explicit control row
                rows.append(
                    [nice, cond, ctrl_mean, ctrl_mean, np.nan, np.nan, np.nan, "-"]
                )
                continue

            if ctrl_vals.empty or test_vals.empty:
                continue

            t, p = ttest_ind(ctrl_vals, test_vals, equal_var=False)
            pooled = np.sqrt(
                ((len(ctrl_vals) - 1) * ctrl_vals.std(ddof=1) ** 2
                 + (len(test_vals) - 1) * test_vals.std(ddof=1) ** 2)
                / (len(ctrl_vals) + len(test_vals) - 2)
            )
            d = abs((ctrl_mean - test_vals.mean()) / pooled) if pooled else np.nan
            eff = (
                "Large" if d >= 0.8 else
                "Medium" if d >= 0.5 else
                "Small" if d >= 0.2 else
                "Negligible"
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


# ──────────────────────────────── pivot to wide ───────────────────────
def long_to_wide(
    stats_long: pd.DataFrame,
    summary_df: pd.DataFrame,
    control: str,
) -> pd.DataFrame:
    """
    Convert *stats_long* to the 2-level wide table used for Excel export.

    Adds an “N” column with per-condition fibre counts.
    """
    wanted_stats = ["Test_Mean", "% Change", "p", "d", "Effect"]

    # % change ----- (compute here so we can reuse in tests & Excel)
    ctrl_means = (
        stats_long[stats_long["Condition"] == control]
        .set_index("Metric")["Control_Mean"]
    )
    stats_long["% Change"] = (
        (stats_long["Test_Mean"] - stats_long["Metric"].map(ctrl_means))
        / stats_long["Metric"].map(ctrl_means)
    ).fillna(0)

    wide = (
        stats_long
        .set_index(["Condition", "Metric"])[wanted_stats]
        .unstack("Metric")
        .swaplevel(axis=1)
        .sort_index(axis=1, level=0)
    )

    # insert N sizes
    n_sizes = summary_df.reset_index().groupby("Condition").size()
    wide.insert(0, ("", "N"), n_sizes.reindex(wide.index).astype("Int64"))

    # control first for readability
    if control in wide.index:
        wide = wide.reindex([control] + [idx for idx in wide.index if idx != control])

    return wide
