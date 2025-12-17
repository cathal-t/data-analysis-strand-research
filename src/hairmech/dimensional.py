from __future__ import annotations  # optional but nice for type hints

import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Set

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

    def __init__(
        self,
        path: Path,
        removed_slices: Mapping[int, Sequence[int]] | None = None,
        slot_map: Mapping[int, int] | None = None,
    ):
        lines = path.read_text().splitlines()

        # 1 ── locate the first header line
        hdr_idx = next(i for i, l in enumerate(lines) if l.startswith("Record"))
        header  = [c.strip() for c in lines[hdr_idx].split("\t")]
        rec_idx = header.index("Record")

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
        all_slices: Dict[int, Set[int]] = {}
        slot_records: Dict[int, Set[int]] = {}
        removed_lookup: Dict[int, Set[int]] = {
            int(slot): {int(s) for s in slices}
            for slot, slices in (removed_slices or {}).items()
        }
        removed_applied: MutableMapping[int, Set[int]] = {
            slot: set() for slot in removed_lookup
        }

        try:
            slice_idx = header.index("Slice No.")
        except ValueError:
            slice_idx = next(
                (i for i, col in enumerate(header) if col.strip().lower() == "slice no."),
                None,
            )

        for row in lines[hdr_idx + 1:]:
            if not row.strip() or row.startswith("Record"):
                continue                                  # skip blank / header
            parts = row.split("\t")
            if len(parts) <= max(area_idx, desc_idx):
                continue

            record_val: int | None = None
            if rec_idx < len(parts):
                try:
                    record_val = int(float(parts[rec_idx].strip()))
                except ValueError:
                    record_val = None

            desc = parts[desc_idx].strip()
            m = self._DESC_RE.match(desc)
            slot_from_desc = int(m.group(1)) if m else None

            slot: int | None = None
            if slot_map is not None and record_val is not None:
                slot = slot_map.get(record_val)
            if slot is None:
                slot = slot_from_desc
            if slot is None:
                continue

            area_str = parts[area_idx].strip()
            if not self._NUM_RE.match(area_str):
                continue
            area_val = float(area_str)

            slice_number: int | None = None
            if slice_idx is not None and slice_idx < len(parts):
                slice_raw = parts[slice_idx].strip()
                if slice_raw.isdigit():
                    slice_number = int(slice_raw)

            all_slices.setdefault(slot, set())
            if slice_number is not None:
                all_slices[slot].add(slice_number)

            slot_vals.setdefault(slot, [])
            if (
                slice_number is not None
                and slot in removed_lookup
                and slice_number in removed_lookup[slot]
            ):
                removed_applied.setdefault(slot, set()).add(slice_number)
                continue

            if record_val is not None:
                slot_records.setdefault(slot, set()).add(record_val)
            slot_vals[slot].append(area_val)

        if not slot_vals:
            raise ValueError("No numeric area entries parsed")

        # 4 ── build the map + debug print
        self.map: Dict[int, float] = {
            s: float(np.mean(vals))
            for s, vals in slot_vals.items()
            if vals
        }

        requested_slots = set(removed_lookup)
        present_slots = set(all_slices)
        self.removed_applied: Dict[int, List[int]] = {
            slot: sorted(vals)
            for slot, vals in removed_applied.items()
            if vals
        }
        self.removed_missing_slices: Dict[int, List[int]] = {}
        for slot, requested in removed_lookup.items():
            applied = removed_applied.get(slot, set())
            missing = sorted(requested - applied)
            if missing:
                self.removed_missing_slices[slot] = missing

        self.removed_missing_slots: List[int] = sorted(
            slot for slot in requested_slots if slot not in present_slots
        )
        self.removed_empty_slots: List[int] = sorted(
            slot
            for slot, vals in slot_vals.items()
            if not vals and slot in requested_slots
        )

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

        self.slot_records: Dict[int, List[int]] = {
            slot: sorted(records) for slot, records in slot_records.items()
        }
