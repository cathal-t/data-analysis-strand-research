from __future__ import annotations

from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

class TensileTest:
    """
    Wrapper for Dia-Stron tensile .txt files.

    The file can contain either:
      • raw gram-force   (column 'gmf')
      • engineering MPa  (any header in MPa_HEADERS)

    Provides
    --------
    self.df        – cleaned full DataFrame
    per_slot()     – generator yielding (slot, df_slot)
    stress_strain  – convert df_slot → with strain + stress_Pa columns
    metrics()      – static helper for UTS, break values, Young’s modulus
    """

    GF_TO_N     = 0.00981
    MPa_HEADERS = {"mpa", "stress", "stress_mpa"}

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, path: Path):
        raw_lines = path.read_text().splitlines()

        header: list[str] | None = None
        body:   list[str] = []
        for ln in raw_lines:
            if ln.startswith("Record"):
                header = [c.strip() for c in ln.split("\t") if c.strip()]
            elif header and ln.strip():
                body.append(ln)

        if header is None:  # pragma: no cover
            raise ValueError("Header line starting with 'Record' not found")

        df = pd.read_csv(StringIO("\n".join(body)), sep="\t", names=header)

        # ---- normalise columns --------------------------------------------
        df.rename(columns=lambda c: c.strip(), inplace=True)
        if "% Strain" in df.columns:
            df.rename(columns={"% Strain": "Strain_pct"}, inplace=True)

        # ---- detect stress units ------------------------------------------
        col_lut  = {c.lower(): c for c in df.columns}
        mpa_cols = [orig for low, orig in col_lut.items()
                    if low in self.MPa_HEADERS]
        self.is_mpa = bool(mpa_cols)
        stress_col  = mpa_cols[0] if self.is_mpa else "gmf"
        print(f"[DEBUG] Tensile units: "
              f"{'MPa' if self.is_mpa else 'gmf'} (column '{stress_col}')")

        # ---- clean numeric data -------------------------------------------
        df.rename(columns={stress_col: "raw_stress"}, inplace=True)
        df["Record"]     = pd.to_numeric(df["Record"],     errors="coerce")
        df["Strain_pct"] = pd.to_numeric(df["Strain_pct"], errors="coerce")
        df["raw_stress"] = pd.to_numeric(df["raw_stress"], errors="coerce")

        self.df = (
            df.dropna(subset=["Record", "Strain_pct", "raw_stress"])
              .astype({"Record": int})
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
        if self.is_mpa:
            out["stress_Pa"] = out["raw_stress"] * 1_000_000
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
        uts          = s[idx]              # Pa
        brk_stress   = s[idx]              # Pa (same as UTS in this file format)
        brk_strain   = e[idx] * 100        # %
        # Fit Young's modulus over 0.2%–0.8% strain (0.002–0.008 in decimal strain)
        linear_mask = (e >= 0.002) & (e <= 0.008)
        E = (
            np.polyfit(e[linear_mask], s[linear_mask], 1)[0]
            if linear_mask.sum() > 1 else np.nan
        )
        return uts / 1e6, brk_stress / 1e6, brk_strain, E / 1e9

    @staticmethod
    def yield_gradient(df: pd.DataFrame,
                       low_pct: float = 7.0,
                       high_pct: float = 16.0) -> float:
        """
        Slope (MPa / % strain) between `low_pct` and `high_pct` strain.
        """
        if high_pct <= low_pct:
            raise ValueError("high_pct must be greater than low_pct")

        e_pct = df["strain"].to_numpy() * 100
        s_pa  = df["stress_Pa"].to_numpy()
        if e_pct.size < 2:
            return np.nan
        # ensure ascending order before interpolation
        idx   = np.argsort(e_pct)
        e_pct = e_pct[idx]
        s_pa  = s_pa[idx]

        if e_pct.min() > low_pct or e_pct.max() < high_pct:
            return np.nan

        σ_low  = np.interp(low_pct,  e_pct, s_pa)
        σ_high = np.interp(high_pct, e_pct, s_pa)

        return ((σ_high - σ_low) / 1e6) / (high_pct - low_pct)
