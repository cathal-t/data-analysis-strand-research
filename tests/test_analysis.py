"""
Quick happy-path tests for build_summary / build_stats using tiny synthetic data
(no large fixture files needed).
"""
from pathlib import Path

import pandas as pd

from hairmech.analysis import build_summary, build_stats
from hairmech.io.config import Condition
from hairmech.tensile import TensileTest


def _dummy_tensile() -> TensileTest:
    """
    Build a TensileTest object from an in-memory tiny CSV-like string.
    Slots 1 and 2 have three rows each.
    """
    from io import StringIO

    txt = StringIO(
        "Record\t% Strain\tgmf\n"
        "1\t0.0\t10\n"
        "1\t0.02\t20\n"
        "1\t0.04\t30\n"
        "2\t0.0\t15\n"
        "2\t0.02\t25\n"
        "2\t0.04\t35\n"
    )
    path = Path(__file__).with_suffix(".tmp.txt")
    path.write_text(txt.getvalue())
    tt = TensileTest(path)
    path.unlink()
    return tt


def test_build_stats_small():
    # ---------- dummy summary table ---------------------------------
    summary = pd.DataFrame(
        {
            "Slot": [1, 2, 3, 4],
            "Condition": ["A", "A", "B", "B"],
            "UTS_MPa": [100, 102, 110, 112],
        }
    ).set_index("Slot")

    conds = [
        Condition(name="A", slots={1, 2}, is_control=True),
        Condition(name="B", slots={3, 4}, is_control=False),
    ]

    stats = build_stats(summary_df=summary,
                        conditions=conds,
                        metrics={"UTS_MPa": "UTS"})

    assert {"A", "B"} <= set(stats["Condition"])
    assert "UTS" in stats["Metric"].unique()


def test_build_summary_with_conditions():
    # two slots, each with an arbitrary mean area
    areas = {1: 50_000.0, 2: 52_000.0}
    conds = [
        Condition(name="CTRL", slots={1}, is_control=True),
        Condition(name="TEST", slots={2}, is_control=False),
    ]

    summary = build_summary(
        areas_map=areas,
        tensile=_dummy_tensile(),
        conditions=conds,
    )

    # Expect exactly two rows (slot 1 & 2)
    assert summary.shape[0] == 2
    assert set(summary["Condition"]) == {"CTRL", "TEST"}
    # Columns added by metrics helpers exist
    for col in [
        "UTS_MPa",
        "Break_Stress_MPa",
        "Break_Strain_%",
        "Elastic_Modulus_GPa",
    ]:
        assert col in summary.columns
