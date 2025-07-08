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
