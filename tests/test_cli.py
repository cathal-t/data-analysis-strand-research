import subprocess, sys, shutil
from pathlib import Path

from click.testing import CliRunner

from hairmech.cli import cli

def test_cli_run_requires_config(tmp_path: Path):
    exp = tmp_path / "demo"
    exp.mkdir()
    (exp / "Dimensional_Data.txt").write_text("")
    (exp / "Tensile_Data.txt").write_text("")

    runner = CliRunner()
    result = runner.invoke(cli, ["run", "-i", str(exp)])
    assert result.exit_code != 0
    assert "config.yml not found" in result.output


def test_cli_run_demo_fixture(tmp_path: Path):
    demo = Path(__file__).parent / "fixtures" / "demo_exp"
    out_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "-i", str(demo), "-o", str(out_dir)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "KeyError" not in result.output
    assert (out_dir / "stats.xlsx").exists()
    assert (out_dir / "metrics.xlsx").exists()
