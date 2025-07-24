"""
Excel exporters that reproduce the exact formatting of the original notebook.

Public API
~~~~~~~~~~
save_metrics(df, path)        – plain per-slot metrics sheet
save_stats_wide(wide, ctrl, path) – wide Welch-t table with styling

Both functions are silent; they overwrite *path* if it exists.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


# ────────────────────────────────────────────────────────────────────
def save_metrics(df: pd.DataFrame, path: Path) -> None:
    """
    Write *Metrics.xlsx* (no fancy styling in the notebook – just a sheet).
    """
    path = Path(path)
    df.to_excel(path, sheet_name="Metrics", engine="xlsxwriter")
    # nothing else to style


# ────────────────────────────────────────────────────────────────────
def save_stats_wide(
    wide: pd.DataFrame,
    control: str,
    path: Path,
) -> None:
    """
    Apply the same xlsxwriter cosmetics the notebook used:

    * frozen header rows & first two columns
    * grey header fill
    * width & number-format per sub-metric
    * conditional green shading for p < 0.05
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # fall back to openpyxl if xlsxwriter absent (formats will be lost)
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ModuleNotFoundError:
        engine = None

    with pd.ExcelWriter(path, engine=engine) as xl:
        wide.to_excel(xl, sheet_name="Stats", index=True)

        if engine != "xlsxwriter":
            return  # cannot style

        # ---------- styling ------------------------------------------------
        ws = xl.sheets["Stats"]
        book = xl.book

        hdr  = book.add_format({"bold": True, "bg_color": "#dfe6e9",
                                "border": 1, "align": "center"})
        f_txt = book.add_format({"text_wrap": True})
        f_int = book.add_format({"num_format": "0", "align": "center"})
        f_num = book.add_format({"num_format": "0.00"})
        f_pct = book.add_format({"num_format": "0.0%"})
        f_p   = book.add_format({"num_format": "0.000"})
        f_sig = book.add_format({"bg_color": "#c8e6c9"})

        # freeze header rows & first two cols
        ws.freeze_panes(2, 2)
        ws.set_row(0, None, hdr)
        ws.set_row(1, None, hdr)
        ws.set_row(2, 0, None, {'hidden': True})

        ws.set_column(0, 0, 32, f_txt)   # Condition
        ws.set_column(1, 1, 6,  f_int)   # N

        # helper to convert idx → Excel letter(s)
        def col_letter(idx):
            s = ""
            while idx >= 0:
                s = chr(idx % 26 + 65) + s
                idx = idx // 26 - 1
            return s

        # column formats + p-value shading
        for col_i, (met, stat) in enumerate(wide.columns[1:], start=2):
            if   stat == "Test mean": width, fmt = 14, f_num
            elif stat == "% Change":  width, fmt = 12, f_pct
            elif stat == "p":         width, fmt = 12, f_p
            else:                     width, fmt = 12, f_num
            ws.set_column(col_i, col_i, width, fmt)

            if stat == "p":
                first, last = 4, 4 + len(wide) - 1
                let = col_letter(col_i)
                ws.conditional_format(
                    first-1, col_i, last-1, col_i,
                    {"type": "formula",
                     "criteria": f'=AND(ISNUMBER({let}{first}),{let}{first}<0.05)',
                     "format": f_sig})
print("[INFO] Metrics →", ROOT / f"test_metrics.xlsx")
print("[INFO] Stats   →", ROOT / f"test_stats.xlsx")
