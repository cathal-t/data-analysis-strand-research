from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class TensileTest:
    """
    Wrapper for Dia-Stron tensile files (legacy flat table + new ASCII blocks).

    Supports:
      • Legacy flat export:
          - Single table with columns like: Record, % Strain (or Strain (%)), gmf
            OR a stress column in MPa (various spellings).
      • New ASCII export (repeated blocks):
          - A metadata line 'Sample / Slot Number: <int>'
          - Then a table: Record, Index, Position, Strain, Time, Force
            where Force is in Newtons and Strain is in %.

    Public API:
      - self.df with columns: Record (int), Strain_pct (float), raw_stress (float)
      - self.is_mpa flag (True when the legacy file provides MPa directly)
      - per_slot()      – yields (slot, df_slot) with Record==Slot for new files
      - stress_strain() – converts to engineering strain and stress (Pa)
      - metrics(), yield_gradient()
    """

    GF_TO_N = 0.00981

    # regex to detect MPa columns like: "Stress", "Stress (MPa)", "Stress_MPa", etc.
    _MPA_COL_RE = re.compile(r"(?:^|\W)mpa(?:$|\W)|^stress$|^stress\s*\(.*mpa.*\)$", re.I)

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, path: Path):
        # Try UTF-8 with BOM first; fall back to UTF-16LE (common in instrument exports)
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception:
            text = path.read_text(errors="ignore")

        if "Sample / Slot Number" in text:
            self._init_from_ascii_blocks(text)
        else:
            self._init_from_legacy(text)

    # ------------------------------------------------------------------ #
    # Legacy reader                                                      #
    # ------------------------------------------------------------------ #
    def _init_from_legacy(self, text: str):
        lines = text.splitlines()

        header_line = None
        header_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("Record"):
                header_line = ln
                header_idx = i
                break

        if header_line is None:  # pragma: no cover
            raise ValueError("Header line starting with 'Record' not found")

        # Detect delimiter for legacy table
        sep_char = "," if ("," in header_line and "\t" not in header_line) else "\t"
        header = [c.strip() for c in re.split(r"[,\t]", header_line) if c.strip()]

        # Collect body: include subsequent lines; repeated headers will be dropped later
        body = []
        for ln in lines[header_idx + 1:]:
            if not ln.strip():
                continue
            body.append(ln)

        df = pd.read_csv(
            StringIO("\n".join(body)),
            sep=sep_char,
            names=header,
            engine="python",
            on_bad_lines="skip",
        )
        df.rename(columns=lambda c: c.strip(), inplace=True)

        # ---- detect/normalize strain column -----------------------------
        strain_col = next((c for c in df.columns if re.search(r"strain", c, re.I)), None)
        if strain_col is None:
            raise ValueError("Strain column not found (legacy mode).")
        df.rename(columns={strain_col: "Strain_pct"}, inplace=True)

        # ---- detect stress/force column --------------------------------
        col_lut = {c.lower(): c for c in df.columns}

        # 1) MPa style column?
        mpa_cols = [c for c in df.columns if self._MPA_COL_RE.search(c)]
        if mpa_cols:
            stress_col = mpa_cols[0]
            self.mode = "MPA"
            self.is_mpa = True
        # 2) gram-force?
        elif "gmf" in col_lut:
            stress_col = col_lut["gmf"]
            self.mode = "GMF"
            self.is_mpa = False
        # 3) Force (N) as last resort
        elif "force" in col_lut:
            stress_col = col_lut["force"]
            self.mode = "N"
            self.is_mpa = False
        else:
            raise ValueError("No stress/force column found (legacy mode).")

        logger.debug("Tensile units (legacy): %s (column '%s')", self.mode, stress_col)

        # ---- clean numeric data ----------------------------------------
        df.rename(columns={stress_col: "raw_stress"}, inplace=True)

        if "Record" not in df.columns:
            raise ValueError("Column 'Record' not found (legacy mode).")

        df["Record"] = pd.to_numeric(df["Record"], errors="coerce")
        df["Strain_pct"] = pd.to_numeric(df["Strain_pct"], errors="coerce")
        df["raw_stress"] = pd.to_numeric(df["raw_stress"], errors="coerce")

        self.df = (
            df.dropna(subset=["Record", "Strain_pct", "raw_stress"])
            .astype({"Record": int})
        )

    # ------------------------------------------------------------------ #
    # New ASCII blocks reader (delimiter-agnostic: comma or tab)         #
    # ------------------------------------------------------------------ #
    def _init_from_ascii_blocks(self, text: str):
        lines = text.splitlines()

        # accept comma or tab separators in the data header
        sep_pattern = r"[,\t]"
        data_hdr_re = re.compile(
            rf"^\s*Record\s*{sep_pattern}\s*Index\s*{sep_pattern}\s*Position\s*{sep_pattern}\s*Strain\s*{sep_pattern}\s*Time\s*{sep_pattern}\s*Force\b",
            re.I,
        )

        def split_fields(s: str) -> list[str]:
            return [p.strip() for p in re.split(sep_pattern, s.rstrip())]

        # locate each sample block
        slot_idxs = [i for i, l in enumerate(lines) if "Sample / Slot Number" in l]
        records = []

        for si in slot_idxs:
            # Lines like: "Sample / Slot Number:,2" or "Sample / Slot Number:\t2" or "Sample / Slot Number: 2"
            m = re.search(r"Sample\s*/\s*Slot\s*Number\s*:\D*?(\d+)", lines[si])
            if not m:
                continue
            slot = int(m.group(1))

            # find the table header following this block
            try:
                h = next(idx for idx in range(si + 1, len(lines)) if data_hdr_re.match(lines[idx]))
            except StopIteration:
                continue

            # units row expected at h+1 (e.g., ", , µm, %, s, N") – skip it
            j = h + 2
            while j < len(lines):
                row = lines[j].strip()
                # stop at blank or next header/block
                if (
                    not row
                    or "Sample / Slot Number" in row
                    or data_hdr_re.match(row)
                    or row.startswith("ASCII Export File Version")
                ):
                    break

                parts = split_fields(lines[j])
                if len(parts) >= 6:
                    # parts: Record, Index, Position(µm), Strain(%), Time(s), Force(N)
                    try:
                        strain_pct = float(parts[3])
                        force_N = float(parts[5])
                    except ValueError:
                        j += 1
                        continue

                    records.append(
                        {
                            "Record": slot,  # downstream grouping key (aligns to slot)
                            "Slot": slot,
                            "Strain_pct": strain_pct,
                            "raw_stress": force_N,  # Newtons in ASCII export
                        }
                    )
                j += 1

        if not records:
            raise ValueError("No data rows parsed from ASCII blocks (checked comma and tab formats).")

        self.df = pd.DataFrame.from_records(records)
        self.mode = "N"  # Force in Newtons
        self.is_mpa = False
        logger.debug(
            "Tensile (ASCII blocks): slots=%s · rows=%d",
            sorted(self.df["Record"].unique()),
            len(self.df),
        )

    # ------------------------------------------------------------------ #
    # public helpers                                                     #
    # ------------------------------------------------------------------ #
    def per_slot(self):
        """Yield (slot_number, DataFrame) pairs in ascending slot order."""
        for slot, grp in self.df.groupby("Record", sort=True):
            yield slot, grp.reset_index(drop=True)

    def stress_strain(self, df_slot: pd.DataFrame, area_um2: float) -> pd.DataFrame:
        """Convert raw slot data → engineering strain + true stress (Pa)."""
        out = df_slot.copy()
        out["strain"] = out["Strain_pct"] / 100

        if getattr(self, "mode", None) == "MPA":
            out["stress_Pa"] = out["raw_stress"] * 1_000_000
        elif getattr(self, "mode", None) == "N":
            out["stress_Pa"] = out["raw_stress"] / (area_um2 * 1e-12)
        else:
            out["stress_Pa"] = (
                out["raw_stress"] * self.GF_TO_N / (area_um2 * 1e-12)
            )
        return out

    # ------------------------------------------------------------------ #
    # static metrics                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
        """Return UTS, break stress, break strain, Young’s modulus."""
        s = df["stress_Pa"].to_numpy()
        e = df["strain"].to_numpy()

        idx = s.argmax()
        uts = s[idx]  # Pa
        brk_stress = s[idx]  # Pa (same as UTS in this file format)
        brk_strain = e[idx] * 100  # %
        # Fit Young's modulus over 0.2%–0.8% strain (0.002–0.008 in decimal strain)
        linear_mask = (e >= 0.002) & (e <= 0.008)
        E = (
            np.polyfit(e[linear_mask], s[linear_mask], 1)[0]
            if linear_mask.sum() > 1
            else np.nan
        )
        return uts / 1e6, brk_stress / 1e6, brk_strain, E / 1e9

    @staticmethod
    def yield_gradient(df: pd.DataFrame, low_pct: float = 7.0, high_pct: float = 16.0) -> float:
        """
        Slope (MPa / % strain) between `low_pct` and `high_pct` strain.
        """
        if high_pct <= low_pct:
            raise ValueError("high_pct must be greater than low_pct")

        e_pct = df["strain"].to_numpy() * 100
        s_pa = df["stress_Pa"].to_numpy()
        if e_pct.size < 2:
            return np.nan
        # ensure ascending order before interpolation
        idx = np.argsort(e_pct)
        e_pct = e_pct[idx]
        s_pa = s_pa[idx]

        if e_pct.min() > low_pct or e_pct.max() < high_pct:
            return np.nan

        σ_low = np.interp(low_pct, e_pct, s_pa)
        σ_high = np.interp(high_pct, e_pct, s_pa)

        return ((σ_high - σ_low) / 1e6) / (high_pct - low_pct)
