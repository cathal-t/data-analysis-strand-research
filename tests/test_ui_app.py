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


def _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmd):
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
        captured_cmd["cmd"] = cmd
        captured_cmd["cwd"] = cwd
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

    captured_cmd = {}
    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmd)

    success, message = app._run_dimensional_export(uvc_path)

    expected_output = uvc_path.with_suffix(".txt").resolve()
    assert success is True
    assert expected_output.exists()
    assert message == f"Export complete. Output saved to: {expected_output}"
    assert captured_cmd["cwd"] == str(uvc_path.parent)
    assert expected_output == Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-o") + 1])
    assert Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-i") + 1]) == uvc_path.resolve()


def test_run_dimensional_export_prefers_original_dir(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    original_dir = tmp_path / "expected"
    original_dir.mkdir()

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmd = {}

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmd)

    success, message = app._run_dimensional_export(uvc_path, original_dir)

    expected_output = (original_dir / uvc_path.name).with_suffix(".txt").resolve()
    assert success is True
    assert expected_output.exists()
    assert message == f"Export complete. Output saved to: {expected_output}"
    assert captured_cmd["cwd"] == str(uvc_path.parent)
    assert expected_output == Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-o") + 1])
    assert Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-i") + 1]) == uvc_path.resolve()


def test_run_dimensional_export_creates_missing_original_dir(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    original_dir = tmp_path / "nested" / "expected"

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmd = {}

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmd)

    success, message = app._run_dimensional_export(uvc_path, original_dir)

    expected_output = (original_dir / uvc_path.name).with_suffix(".txt").resolve()
    assert success is True
    assert expected_output.exists()
    assert message == f"Export complete. Output saved to: {expected_output}"
    assert captured_cmd["cwd"] == str(uvc_path.parent)
    assert expected_output == Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-o") + 1])
    assert Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-i") + 1]) == uvc_path.resolve()


def test_run_dimensional_export_prefers_user_directory(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    preferred_dir = tmp_path / "preferred"

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmd = {}

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmd)

    success, message = app._run_dimensional_export(
        uvc_path, original_dir=None, preferred_dir=preferred_dir
    )

    expected_output = (preferred_dir / uvc_path.name).with_suffix(".txt").resolve()
    assert success is True
    assert expected_output.exists()
    assert message == f"Export complete. Output saved to: {expected_output}"
    assert captured_cmd["cwd"] == str(uvc_path.parent)
    assert expected_output == Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-o") + 1])
    assert Path(captured_cmd["cmd"][captured_cmd["cmd"].index("-i") + 1]) == uvc_path.resolve()
