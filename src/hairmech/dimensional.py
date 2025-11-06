from __future__ import annotations  # optional but nice for type hints

import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

class DimensionalData:
    """
    Read a Dia-Stron dimensional *.txt* file and expose

        self.map  -> { slot_number : mean_cross_sectional_area_µm² }

    Changes from the original version
    ---------------------------------
    • Uses the **Description** column (e.g. “Slot 23 : …”) to obtain the
      slot number; ignores the 'Record' column entirely.  
    • Still scans the whole file – reruns appended after record 100 are read.
    """

    _AREA_RE  = re.compile(r"cross.*area", re.I)         # find area column
    _DESC_RE  = re.compile(r"^slot\s+(\d+)\b", re.I)     # “Slot 17 : …”
    _NUM_RE   = re.compile(r"^\d+(\.\d+)?$")             # basic numeric test

    def __init__(self, path: Path):
        lines = path.read_text().splitlines()

        # 1 ── locate the first header line
        hdr_idx = next(i for i, l in enumerate(lines) if l.startswith("Record"))
        header  = [c.strip() for c in lines[hdr_idx].split("\t")]

        # 2 ── index of area & description columns
        try:
            area_idx = next(i for i, c in enumerate(header)
                            if self._AREA_RE.search(c))
        except StopIteration as exc:
            raise ValueError("Area column not found") from exc

        try:
            desc_idx = header.index("Description")
        except ValueError as exc:
            raise ValueError("Description column not found") from exc

        logger.debug(
            "Area col = %s · Description col = %s", area_idx, desc_idx
        )

        # 3 ── parse every data row
        slot_vals: Dict[int, List[float]] = {}

        for row in lines[hdr_idx + 1:]:
            if not row.strip() or row.startswith("Record"):
                continue                                  # skip blank / header
            parts = row.split("\t")
            if len(parts) <= max(area_idx, desc_idx):
                continue

            desc = parts[desc_idx].strip()
            m = self._DESC_RE.match(desc)
            if not m:
                continue                                  # no “Slot N : ...”
            slot = int(m.group(1))

            area_str = parts[area_idx].strip()
            if not self._NUM_RE.match(area_str):
                continue
            area_val = float(area_str)

            slot_vals.setdefault(slot, []).append(area_val)

        if not slot_vals:
            raise ValueError("No numeric area entries parsed")

        # 4 ── build the map + debug print
        self.map: Dict[int, float] = {
            s: float(np.mean(vals)) for s, vals in slot_vals.items()
        }

        df_debug = (
            pd.Series(self.map, name="Mean_Area_µm²")
              .rename_axis("Slot")
              .reset_index()
              .sort_values("Slot")
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Dimensional – slots parsed: %s", len(self.map))
            logger.debug(
                "First 10 rows:\n%s",
                df_debug.head(100).to_string(index=False),
            )
