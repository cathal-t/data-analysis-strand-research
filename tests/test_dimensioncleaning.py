from __future__ import annotations

import pandas as pd

from hairmech.dimensioncleaning import parse_dimensional_export
from hairmech.ui import app


def test_parse_dimensional_export_tracks_union_of_slice_columns(tmp_path):
    export = tmp_path / "dimensional.txt"
    export.write_text(
        "\n".join(
            [
                "Record\tN\tSlice 1\tSlice 2\tSlice 3\tSlice 4\tSlice 5",
                "1\t1\t1\t2\t3\t4\t5",
                "Record\tN\tSlice 1\tSlice 2\tSlice 3\tSlice 4\tSlice 5",
                "2\t1\t10\t20\t30\t40\t50",
                "Record\tN\tSlice 1\tSlice 2",
                "3\t1\t100\t200",
            ]
        )
    )

    records, slice_cols = parse_dimensional_export(export)

    assert slice_cols == ["Slice 1", "Slice 2", "Slice 3", "Slice 4", "Slice 5"]
    assert list(records[3].columns) == [
        "Record",
        "N",
        "Slice 1",
        "Slice 2",
        "Slice 3",
        "Slice 4",
        "Slice 5",
    ]
    assert pd.isna(records[3].loc[0, "Slice 3"])
    assert pd.isna(records[3].loc[0, "Slice 4"])
    assert pd.isna(records[3].loc[0, "Slice 5"])


def test_dimensional_plot_helpers_allow_missing_slice_columns():
    records = {
        1: pd.DataFrame({"Record": [1], "N": [1], "Slice 1": [1.0], "Slice 2": [2.0]}),
        2: pd.DataFrame({"Record": [2], "N": [1], "Slice 1": [3.0]}),
    }
    slice_cols = ["Slice 1", "Slice 2"]

    fig = app._make_dimensional_record_fig(2, records[2], slice_cols)
    stats = app._compute_slice_extremes(records[2], slice_cols)
    children = app._build_dimensional_plot_children(records, slice_cols)

    assert len(fig.data) == 1
    assert stats == [{"slice": "Slice 1", "min": 3.0, "max": 3.0}]
    assert len(children) == 2
