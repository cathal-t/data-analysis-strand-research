from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hairmech.ui import app


def test_infer_original_dir_windows_path():
    filename = (
        "G:\\Other computers\\Dia-Stron Machine Laptop\\Dia-Stron Lab Data\\"
        "Dia-Stron Experiments\\Ohad\\Foundational Experiments\\20251028 Wet AT\\"
        "20251028 Wet AT.uvc"
    )

    inferred = app._infer_original_dir(filename)

    assert inferred == Path(
        "G:\\Other computers\\Dia-Stron Machine Laptop\\Dia-Stron Lab Data\\"
        "Dia-Stron Experiments\\Ohad\\Foundational Experiments\\20251028 Wet AT"
    )


@pytest.mark.parametrize(
    "filename",
    [
        r"C:\\fakepath\\20251028 Wet AT.uvc",
        "20251028 Wet AT.uvc",
        "relative/path/20251028 Wet AT.uvc",
    ],
)
def test_infer_original_dir_rejects_non_absolute_paths(filename: str):
    assert app._infer_original_dir(filename) is None


def _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds):
    real_exists = app.Path.exists

    def fake_exists(self):
        normalized = str(self).replace("\\", "/")
        if normalized.lower() == str(exe_path).replace("\\", "/").lower():
            return True
        return real_exists(self)

    monkeypatch.setattr(app.Path, "exists", fake_exists, raising=False)
    monkeypatch.setattr(app.platform, "system", lambda: "Windows")

    def fake_run(cmd, cwd, capture_output, text, check):
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("exported")
        captured_cmds.append({"cmd": cmd, "cwd": cwd})
        return SimpleNamespace(stdout="done", stderr="")

    monkeypatch.setattr(app.subprocess, "run", fake_run)


def test_parse_export_directory_accepts_absolute_path():
    path = app._parse_export_directory("/tmp/exports")
    assert path == Path("/tmp/exports")


def test_parse_export_directory_rejects_relative_path():
    with pytest.raises(ValueError):
        app._parse_export_directory("relative/path")


def test_run_dimensional_export_uses_uvc_directory_by_default(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []
    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message = app._run_dimensional_export(uvc_path)

    expected_output = uvc_path.with_suffix(".txt").resolve()
    expected_gpdsr = expected_output.with_name(
        f"{expected_output.stem}_gpdsr{expected_output.suffix}"
    )
    assert success is True
    assert expected_output.exists()
    assert expected_gpdsr.exists()
    assert (
        message
        == f"Export complete. Outputs saved to: {expected_output} and {expected_gpdsr}"
    )
    assert len(captured_cmds) == 2
    for captured in captured_cmds:
        assert captured["cwd"] == str(uvc_path.parent)
        assert Path(captured["cmd"][captured["cmd"].index("-i") + 1]) == uvc_path.resolve()
    assert expected_output == Path(
        captured_cmds[0]["cmd"][captured_cmds[0]["cmd"].index("-o") + 1]
    )
    assert expected_gpdsr == Path(
        captured_cmds[1]["cmd"][captured_cmds[1]["cmd"].index("-o") + 1]
    )


def test_run_dimensional_export_prefers_original_dir(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    original_dir = tmp_path / "expected"
    original_dir.mkdir()

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message = app._run_dimensional_export(uvc_path, original_dir)

    expected_output = (original_dir / uvc_path.name).with_suffix(".txt").resolve()
    expected_gpdsr = expected_output.with_name(
        f"{expected_output.stem}_gpdsr{expected_output.suffix}"
    )
    assert success is True
    assert expected_output.exists()
    assert expected_gpdsr.exists()
    assert (
        message
        == f"Export complete. Outputs saved to: {expected_output} and {expected_gpdsr}"
    )
    assert len(captured_cmds) == 2
    for captured in captured_cmds:
        assert captured["cwd"] == str(uvc_path.parent)
        assert Path(captured["cmd"][captured["cmd"].index("-i") + 1]) == uvc_path.resolve()
    assert expected_output == Path(
        captured_cmds[0]["cmd"][captured_cmds[0]["cmd"].index("-o") + 1]
    )
    assert expected_gpdsr == Path(
        captured_cmds[1]["cmd"][captured_cmds[1]["cmd"].index("-o") + 1]
    )


def test_run_dimensional_export_creates_missing_original_dir(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    original_dir = tmp_path / "nested" / "expected"

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message = app._run_dimensional_export(uvc_path, original_dir)

    expected_output = (original_dir / uvc_path.name).with_suffix(".txt").resolve()
    expected_gpdsr = expected_output.with_name(
        f"{expected_output.stem}_gpdsr{expected_output.suffix}"
    )
    assert success is True
    assert expected_output.exists()
    assert expected_gpdsr.exists()
    assert (
        message
        == f"Export complete. Outputs saved to: {expected_output} and {expected_gpdsr}"
    )
    assert len(captured_cmds) == 2
    for captured in captured_cmds:
        assert captured["cwd"] == str(uvc_path.parent)
        assert Path(captured["cmd"][captured["cmd"].index("-i") + 1]) == uvc_path.resolve()
    assert expected_output == Path(
        captured_cmds[0]["cmd"][captured_cmds[0]["cmd"].index("-o") + 1]
    )
    assert expected_gpdsr == Path(
        captured_cmds[1]["cmd"][captured_cmds[1]["cmd"].index("-o") + 1]
    )


def test_run_dimensional_export_prefers_user_directory(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    preferred_dir = tmp_path / "preferred"

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message = app._run_dimensional_export(
        uvc_path, original_dir=None, preferred_dir=preferred_dir
    )

    expected_output = (preferred_dir / uvc_path.name).with_suffix(".txt").resolve()
    expected_gpdsr = expected_output.with_name(
        f"{expected_output.stem}_gpdsr{expected_output.suffix}"
    )
    assert success is True
    assert expected_output.exists()
    assert expected_gpdsr.exists()
    assert (
        message
        == f"Export complete. Outputs saved to: {expected_output} and {expected_gpdsr}"
    )
    assert len(captured_cmds) == 2
    for captured in captured_cmds:
        assert captured["cwd"] == str(uvc_path.parent)
        assert Path(captured["cmd"][captured["cmd"].index("-i") + 1]) == uvc_path.resolve()
    assert expected_output == Path(
        captured_cmds[0]["cmd"][captured_cmds[0]["cmd"].index("-o") + 1]
    )
    assert expected_gpdsr == Path(
        captured_cmds[1]["cmd"][captured_cmds[1]["cmd"].index("-o") + 1]
    )
