"""Utility helpers for colours and simple curve metrics."""

from __future__ import annotations

from pathlib import Path  # (useful for future helpers)
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb  # <- the real colour converter


# ──────────────────────────────────────────────────────────────────────
#  Colour helpers
# ──────────────────────────────────────────────────────────────────────
def rgba(mat_color: str | Sequence[float], alpha: float = 1.0) -> str:
    """
    Convert a Matplotlib-style colour (hex or RGB-tuple) to a Plotly-style
    string: "rgba(r,g,b,a)" with r,g,b in 0-255 and a 0-1.

    Examples
    --------
    >>> rgba("#2a9d8f", 0.3)
    'rgba(42,157,143,0.3)'
    >>> rgba((0.5, 0.8, 1.0), 1)
    'rgba(128,204,255,1)'
    """
    r, g, b = (np.array(to_rgb(mat_color)) * 255).astype(int)
    return f"rgba({r},{g},{b},{alpha})"


def hex_to_rgb01(hex_str: str) -> Tuple[float, float, float]:
    """'#RRGGBB' → (r,g,b) each 0–1."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) / 255 for i in (0, 2, 4))


# ──────────────────────────────────────────────────────────────────────
#  Curve-metric helpers
# ──────────────────────────────────────────────────────────────────────
def post_gradient(proc_df: pd.DataFrame, delta_pct: float = 8.0) -> float:
    """
    Slope (MPa / % strain) from break point back by `delta_pct` % strain.

    Expects columns:
    * ``strain`` in 0–1 range
    * ``stress_Pa`` in Pascals
    """
    s = proc_df["stress_Pa"].to_numpy()
    e = proc_df["strain"].to_numpy()
    if len(s) < 3:
        return np.nan

    idx_break = s.argmax()
    brk_strain_pct = e[idx_break] * 100
    target_pct = brk_strain_pct - delta_pct
    if target_pct < 0:
        return np.nan

    anchor_idx = np.where(e * 100 <= target_pct)[0]
    if anchor_idx.size == 0:
        return np.nan
    anchor_idx = anchor_idx[-1]

    dσ_MPa = (s[idx_break] - s[anchor_idx]) / 1e6
    dε_pct = brk_strain_pct - e[anchor_idx] * 100
    return dσ_MPa / dε_pct if dε_pct else np.nan


def post_seg(df: pd.DataFrame, delta_pct: float = 8.0):
    """Return two (strain, stress_Pa) points for the post-gradient helper line."""
    s, e = df["stress_Pa"].to_numpy(), df["strain"].to_numpy()
    idx_brk = s.argmax()
    tgt = e[idx_brk] * 100 - delta_pct
    idx = np.where(e * 100 <= tgt)[0]
    if idx.size == 0:
        return None
    return (e[idx[-1]], s[idx[-1]]), (e[idx_brk], s[idx_brk])


def yld_seg(df: pd.DataFrame, low: float = 7.0, high: float = 16.0):
    """Return two (strain, stress_Pa) points for the yield-gradient helper line."""
    e, s = df["strain"].to_numpy(), df["stress_Pa"].to_numpy()
    idx = np.argsort(e)
    e_pct, s_sort = e[idx] * 100, s[idx]
    if e_pct.min() > low or e_pct.max() < high:
        return None
    σ_low = np.interp(low, e_pct, s_sort)
    σ_high = np.interp(high, e_pct, s_sort)
    return (low / 100, σ_low), (high / 100, σ_high)
