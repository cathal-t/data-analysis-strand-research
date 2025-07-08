from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List

import click
import pandas as pd

from .io.config import load_config, ConfigError
from .dimensional import DimensionalData
from .tensile import TensileTest
from .analysis import build_summary, build_stats, long_to_wide
from .io.export import save_metrics, save_stats_wide
from .plots import make_overlay, make_violin_grid
from collections import OrderedDict


# ───────────────────────── click root ─────────────────────────
@click.group()
def cli() -> None:  # pragma: no cover
    """Hair-mech command-line entry point."""
    pass


# ─────────────────────────── run ──────────────────────────────
@cli.command()
@click.option(
    "-i",
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Experiment folder (Dimensional_Data.txt, Tensile_Data.txt, config.yml)",
)
@click.option(
    "-o",
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory (default: <input>/results)",
)
def run(input_dir: Path, out_dir: Path | None) -> None:
    """
    Run the full pipeline and write Excel + Plotly HTML to *out_dir*.
    """
    try:
        # 1 ── user config ----------------------------------------------------
        conds = load_config(input_dir)           # list[Condition]
        control_name = next(c.name for c in conds if c.is_control)
        slot_to_cond = {s: c.name for c in conds for s in c.slots}

        # 2 ── raw input ------------------------------------------------------
        areas = DimensionalData(input_dir / "Dimensional_Data.txt").map
        tensile = TensileTest(input_dir / "Tensile_Data.txt")

        # 3 ── per-slot metrics & tidy curves --------------------------------
        # NOTE: build_summary now takes the Condition list, not a dict
        summary = build_summary(areas, tensile, conds)

        curves_rows: List[dict] = []
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

        # 4 ── wide stats table ----------------------------------------------
        METRICS: "OrderedDict[str, str]" = OrderedDict([
            ("Elastic_Modulus_GPa",      "Elastic modulus (GPa)"),
            ("Yield_Gradient_MPa_perc",  "Yield-grad. (MPa / %ε)"),
            ("Post_Gradient_MPa_perc",   "Post-grad. (MPa / %ε)"),
            ("Break_Stress_MPa",         "Break stress (MPa)"),
            ("Break_Strain_%",           "Break strain (%)"),
            # ("UTS_MPa",               "UTS"),   # ← left commented-out to drop
        ])
        stats_long = build_stats(summary, conds, METRICS)
        stats_wide = long_to_wide(stats_long, summary, control_name, METRICS)

        # 5 ── output dir -----------------------------------------------------
        out_dir = out_dir or input_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 6 ── Excel ----------------------------------------------------------
        save_metrics(summary, out_dir / "metrics.xlsx")
        save_stats_wide(stats_wide, control_name, out_dir / "stats.xlsx")
        click.echo(f"✓ Excel written → {out_dir}")

        # 7 ── Plotly HTML ----------------------------------------------------
        make_overlay(curves_df, conds).write_html(
            out_dir / "overlay.html", include_plotlyjs="cdn"
        )
        make_violin_grid(summary, conds).write_html(
            out_dir / "violin_grid.html", include_plotlyjs="cdn"
        )
        click.echo(f"✓ Plots written → {out_dir}")

    # ---------- friendly error messages ------------------------------------
    except FileNotFoundError:
        click.echo(f"config.yml not found in {input_dir}", err=True)
        sys.exit(1)
    except ConfigError as err:
        click.echo(str(err), err=True)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        click.echo("✖ Unhandled error – see traceback below", err=True)
        traceback.print_exception(exc, file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------
def _main() -> None:  # pragma: no cover
    """Entry-point for `python -m hairmech`."""
    cli()


# legacy alias so tools/tests looking for cli.main still work
main = _main  # type: ignore


if __name__ == "__main__":  # pragma: no cover
    _main()