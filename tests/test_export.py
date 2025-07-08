import pandas as pd
from pathlib import Path
from hairmech.io.export import save_metrics, save_stats_wide


def test_export_files_exist(tmp_path: Path):
    # minimal metrics frame
    m_df = pd.DataFrame({"A": [1.0]}).set_index(pd.Index([1], name="Slot"))
    metrics_p = tmp_path / "metrics.xlsx"
    save_metrics(m_df, metrics_p)
    assert metrics_p.exists() and metrics_p.stat().st_size > 0

    # simple wide frame (multi-index cols)
    wide = (
        pd.DataFrame(
            {
                ("", "N"): [2],
                ("UTS", "Test mean"): [100],
                ("UTS", "% Change"): [0.1],
                ("UTS", "p"): [0.04],
                ("UTS", "|d|"): [0.5],
                ("UTS", "Effect"): ["Medium"],
            },
            index=pd.Index(["CTRL"], name=None),
        )
        .sort_index(axis=1)
    )
    stats_p = tmp_path / "stats.xlsx"
    save_stats_wide(wide, control="CTRL", path=stats_p)
    assert stats_p.exists() and stats_p.stat().st_size > 0
