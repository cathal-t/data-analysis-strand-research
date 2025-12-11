from __future__ import annotations

from io import BytesIO

import pandas as pd


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write multiple DataFrames to an in-memory Excel file.

    Behavior aligned with the standalone script:
    - 'Metrics' sheet: plain export with index=True.
    - 'Stats'  sheet: styled export (freeze panes, header fill, widths,
      number formats, conditional formatting for p-values).
    """
    buf = BytesIO()

    def _col_letter(idx: int) -> str:
        s = ""
        while idx >= 0:
            s = chr(idx % 26 + 65) + s
            idx = idx // 26 - 1
        return s

    with pd.ExcelWriter(buf, engine="xlsxwriter") as xls:
        for sheet, df in sheets.items():
            sheet_lower = sheet.lower()

            # Always write with index=True (script behavior).
            df.to_excel(xls, sheet_name=sheet, index=True)

            if sheet_lower != "stats":
                # 'Metrics' (and any other sheets) remain unstyled like the script's Metrics export.
                continue

            # ── styling block for 'Stats' sheet (replicates script) ──
            ws = xls.sheets[sheet]
            book = xls.book

            hdr = book.add_format(
                {
                    "bold": True,
                    "bg_color": "#dfe6e9",
                    "border": 1,
                    "align": "center",
                }
            )
            f_txt = book.add_format({"text_wrap": True})
            f_int = book.add_format({"num_format": "0", "align": "center"})
            f_num = book.add_format({"num_format": "0.00"})
            f_pct = book.add_format({"num_format": "0.0%"})
            f_p = book.add_format({"num_format": "0.000"})
            f_sig = book.add_format({"bg_color": "#c8e6c9"})

            # Freeze header rows & first two columns (like script)
            ws.freeze_panes(2, 2)
            ws.set_row(0, None, hdr)
            ws.set_row(1, None, hdr)
            # Pandas sometimes leaves a spacer row right after headers in MultiIndex exports;
            # keep parity with the script that hides row 3 (0-based index 2).
            try:
                ws.set_row(2, 0, None, {"hidden": True})
            except Exception:
                pass

            # Column sizing: first column is the index (condition names)
            ws.set_column(0, 0, 32, f_txt)  # Condition
            # Second column should be ("", "N") in your wide table
            ws.set_column(1, 1, 6, f_int)  # N

            # Work out stat-specific formats for remaining columns
            # We rely on the MultiIndex structure: (Metric, Stat)
            # Pandas writes the index in col 0 and the first data column at col 1.
            # Our wide dataframe had an inserted ("", "N") at position 0 in the script;
            # here, we just mirror the formatting starting at col 2.
            start_col = 2
            # Try to get columns from the dataframe to map stat names to excel columns.
            try:
                cols = list(df.columns)
            except Exception:
                cols = []

            # Apply per-column widths/formats & conditional formatting for 'p'
            for col_i, col_key in enumerate(cols[1:], start=start_col):  # skip the first data col ("", "N")
                stat_name = ""
                if isinstance(col_key, tuple) and len(col_key) >= 2:
                    stat_name = str(col_key[1]).strip().lower()
                else:
                    # Fallback: if not a tuple (unexpected), treat it generically
                    stat_name = str(col_key).strip().lower()

                if stat_name == "test mean":
                    width, fmt = 14, f_num
                elif stat_name == "% change" or stat_name == "% change".lower():
                    width, fmt = 12, f_pct
                elif stat_name == "p":
                    width, fmt = 12, f_p
                else:
                    width, fmt = 12, f_num

                ws.set_column(col_i, col_i, width, fmt)

                # Conditional shading for p-values < 0.05
                if stat_name == "p":
                    # Data starts at row 4 (1-based) in the script; replicate the same window.
                    # Translate to 1-based Excel addresses using our helper.
                    first_row_1based = 4
                    last_row_1based = 4 + len(df.index) - 1
                    let = _col_letter(col_i)
                    ws.conditional_format(
                        first_row_1based - 1,
                        col_i,
                        last_row_1based - 1,
                        col_i,
                        {
                            "type": "formula",
                            "criteria": f'=AND(ISNUMBER({let}{first_row_1based}),{let}{first_row_1based}<0.05)',
                            "format": f_sig,
                        },
                    )

    buf.seek(0)
    return buf.getvalue()


__all__ = ["to_excel_bytes"]
