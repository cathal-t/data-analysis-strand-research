"""Utilities for working with dimensional cleaning exports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DimensionalRecords = Dict[int, pd.DataFrame]


def _finalise_block(
    records: DimensionalRecords,
    rows: List[List[str]],
    block_slice_cols: List[str],
    all_slice_cols: List[str],
) -> None:
    """Convert a collected record block into a dataframe and store it."""

    if not rows:
        return

    cols = ["Record", "N", *block_slice_cols]
    df = pd.DataFrame(rows, columns=cols)

    for col in all_slice_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[["Record", "N", *all_slice_cols]]

    # Coerce numeric types, dropping rows that do not contain valid Record / N values.
    df["Record"] = pd.to_numeric(df["Record"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df = df.dropna(subset=["Record", "N"])

    if df.empty:
        return

    df["Record"] = df["Record"].astype(int)
    df["N"] = df["N"].astype(int)

    for col in all_slice_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Individual blocks are expected to contain a single record id, but guard just in case.
    for record_id, record_df in df.groupby("Record"):
        records[int(record_id)] = record_df.reset_index(drop=True)


def parse_dimensional_export(
    path: str | Path,
) -> Tuple[DimensionalRecords, List[str]]:
    """Parse an "All Dimensional Data" export into per-record dataframes.

    Parameters
    ----------
    path:
        Location of the exported ``.txt`` file produced by UvWin.

    Returns
    -------
    (records, slice_columns)
        ``records`` is a mapping of record id to a dataframe with columns
        ``["Record", "N", "Slice 1", ...]``. ``slice_columns`` provides the
        ordered list of slice column names that were detected in the export.
    """

    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("ascii", errors="replace")

    records: DimensionalRecords = {}
    block_slice_cols: List[str] = []
    all_slice_cols: List[str] = []
    current_rows: List[List[str]] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            continue

        if line.startswith("Record\tN\tSlice"):
            # Finalise previous block before starting a new one.
            _finalise_block(records, current_rows, block_slice_cols, all_slice_cols)
            current_rows = []

            cols = line.split("\t")
            block_slice_cols = [c.strip() for c in cols[2:] if c.strip()]
            for slice_col in block_slice_cols:
                if slice_col not in all_slice_cols:
                    all_slice_cols.append(slice_col)
            continue

        if not block_slice_cols:
            continue

        if line.startswith("Mean:") or line.startswith("SD:"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            continue

        row = [parts[0], parts[1]]
        row.extend(parts[2 : 2 + len(block_slice_cols)])
        current_rows.append(row)

    _finalise_block(records, current_rows, block_slice_cols, all_slice_cols)

    return records, all_slice_cols


__all__ = ["parse_dimensional_export", "DimensionalRecords"]
