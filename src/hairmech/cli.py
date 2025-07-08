"""
hairmech.cli
============

Command-line interface for the Hair-mechanics toolkit.

Usage
-----

# basic – writes results/ under the input folder
poetry run hairmech run -i <experiment_folder>

# explicit output dir
poetry run hairmech run -i <in> -o <out_dir>

Future sub-commands
-------------------
• ``hairmech dash``   – interactive Dash GUI (to be added).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
import click
import pandas as pd

from .io.config import load_config
from .dimensional import DimensionalData
from .tensile import TensileTest
from .analysis import build_summary, build_stats
from .io.export import save_metrics, save_stats_wide
from .plots import make_overlay, make_violin_grid


# ───────────────────────── click setup ─────────────────────────
@click.group()
def cli() -> None:  # pragma: no cover
    """Hair-mech command-line entry point."""
    pass


# ----------------------------------------------------------------
@cli.command()
@click.option(
    "-i",
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Experiment folder (contains Dimensional_Data.txt, Tensile_Data.txt, "
    "and config.yml).",
)
@click.option(
    "-o",
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory (default: <input>/results).",
)
def run(input_dir: Path, out_dir: Path | None) -> None:
    """
    Batch-run the full pipeline: read TXT files, compute metrics & stats,
    generate plots, and write Excel/HTML to *out_dir*.
    """
    try:
        # 1 ── load user config ------------------------------------------------
        conds = load_config(input_dir)  # list[Condition]
        control_name = next(c.name for c in conds if c.is_control)
        slot_to_cond = {s: c.name for c in conds for s in c.slots}

        # 2 ── read raw files ---------------------------------------------------
        areas = DimensionalData(input_dir / "Dimensional_Data.txt").map
        tensile = TensileTest(input_dir / "Tensile_Data.txt")

        # 3 ── metrics + tidy curves -------------------------------------------
        summary = build_summary(areas, tensile, slot_to_cond)

        curves_rows = []
        for slot, df_raw in tensile.per_slot():
            if slot not in areas or slot not in slot_to_cond:
                continue
            proc = tensile.stress_strain(df_raw, areas[slot])
            curves_rows.extend(
                {
                    "Slot": slot,
                    "Condition": slot_to_cond[slot],
                    "Strain": st,
                    "Stress_MPa": sp / 1e6,
                }
                for st, sp in zip(proc["strain"], proc["stress_Pa"])
            )
        curves_df = pd.DataFrame(curves_rows)

        # 4 ── stats-wide table -------------------------------------------------
        METRICS = {
            "Elastic_Modulus_GPa": "Elastic modulus (GPa)",
            "Yield_Gradient_MPa_perc": "Yield-grad. (MPa / %ε)",
            "Post_Gradient_MPa_perc": "Post-grad. (MPa / %ε)",
            "Break_Stress_MPa": "Break stress (MPa)",
            "Break_Strain_%": "Break strain (%)",
            "UTS_MPa": "UTS",
        }
        stats_wide = build_stats(summary, control_name, METRICS)

        # 5 ── output dir -------------------------------------------------------
        out_dir = out_dir or input_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 6 ── Excel files ------------------------------------------------------
        save_metrics(summary, out_dir / "metrics.xlsx")
        save_stats_wide(stats_wide, control_name, out_dir / "stats.xlsx")
        click.echo(f"✓ Excel written → {out_dir}")

        # 7 ── plots ------------------------------------------------------------
        fig_overlay = make_overlay(curves_df, conds)
        fig_violin = make_violin_grid(summary, conds)
        fig_overlay.write_html(out_dir / "overlay.html", include_plotlyjs="cdn")
        fig_violin.write_html(out_dir / "violin_grid.html", include_plotlyjs="cdn")
        click.echo(f"✓ Plots written → {out_dir}")

    except Exception as exc:  # pragma: no cover
        click.echo("✖ Unhandled error - see traceback below", err=True)
        traceback.print_exception(exc, file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------
def _main():  # pragma: no cover
    """Entry-point wrapper for ``python -m hairmech``."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    _main()
