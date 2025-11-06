from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List

import click
import pandas as pd

from .analysis import METRIC_LABELS, build_stats, build_summary, long_to_wide
from .dimensional import DimensionalData
from .io.config import ConfigError, load_config
from .io.export import save_metrics, save_stats_wide
from .plots import make_overlay, make_violin_grid
from .tensile import TensileTest
from .ui import build_dash_app

# ───────────────────────── click root ─────────────────────────
@click.group()
def cli() -> None:  # pragma: no cover
    """Hair-mech command-line entry point."""
    pass  # noqa: WPS420


# ─────────────────────────── run ──────────────────────────────
@cli.command(help="Run full analysis on an experiment folder.")
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
    """CLI pipeline – mirrors notebook steps."""
    try:
        conds = load_config(input_dir)
        control_name = next(c.name for c in conds if c.is_control)
        slot_to_cond = {s: c.name for c in conds for s in c.slots}

        areas = DimensionalData(input_dir / "Dimensional_Data.txt").map
        tensile = TensileTest(input_dir / "Tensile_Data.txt")

        summary = build_summary(areas, tensile, conds)

        stats_long = build_stats(summary, conds, METRIC_LABELS)
        stats_wide = long_to_wide(stats_long, summary, control_name, METRIC_LABELS)

        out_dir = out_dir or input_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        save_metrics(summary, out_dir / "metrics.xlsx")
        save_stats_wide(stats_wide, control_name, out_dir / "stats.xlsx")
        click.echo(f"✓ Excel written → {out_dir}")

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

        make_overlay(curves_df, conds).write_html(
            out_dir / "overlay.html", include_plotlyjs="cdn"
        )
        make_violin_grid(summary, conds).write_html(
            out_dir / "violin_grid.html", include_plotlyjs="cdn"
        )
        click.echo(f"✓ Plots written → {out_dir}")

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


# ────────────────────────── serve (Dash) ──────────────────────
@cli.command(help="Launch Dash UI on http://127.0.0.1:8050")
def serve() -> None:  # pragma: no cover
    app = build_dash_app()
    app.run_server("127.0.0.1", 8050, debug=True)


# -----------------------------------------------------------------
def _main() -> None:  # pragma: no cover
    """Entry-point for `python -m hairmech`."""
    cli()


main = _main  # legacy alias

if __name__ == "__main__":  # pragma: no cover
    _main()
