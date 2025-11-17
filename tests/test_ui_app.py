from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hairmech.ui import app


def test_trim_gmf_pivot_resets_origin_and_trims_rows():
    export_df = pd.DataFrame(
        {
            "Record": [1, 1, 1, 1, 2, 2, 2, 2],
            "% Strain": [0.02, 0.05, 0.08, 0.12, 0.0, 5.0, 9.0, 12.0],
            "gmf": [5.0, 3.0, 4.0, 6.0, 10.0, 8.0, 9.0, 7.0],
        }
    )
    export_df = export_df.sort_values(["Record", "% Strain"]).reset_index(drop=True)

    trimmed = app._trim_gmf_pivot(export_df)

    expected = pd.DataFrame(
        {
            "Record": [1, 1, 1, 2, 2, 2],
            "% Strain": [0.0, 0.03, 0.07, 0.0, 4.0, 7.0],
            "gmf": [3.0, 4.0, 6.0, 8.0, 9.0, 7.0],
        }
    )

    pd.testing.assert_frame_equal(
        trimmed.reset_index(drop=True),
        expected,
        check_exact=False,
        atol=1e-12,
    )


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


@pytest.mark.parametrize("value", [None, "", "   "])
def test_parse_export_directory_requires_value(value):
    with pytest.raises(ValueError):
        app._parse_export_directory(value)


def test_parse_gpdsr_mapping_deduplicates_by_slot(tmp_path):
    gpdsr_path = tmp_path / "example_gpdsr.txt"
    gpdsr_path.write_text(
        "\n".join(
            [
                "Summary Data Ver: 1.0",
                "Source File: demo",
                "Record\tSample\tDescription",
                "1\t1\tSlot 5 Cycle: 1",
                "4\t2\tSlot 5 Cycle: 2",
                "5\t3\tSlot 7",
            ]
        )
    )

    mapping, deduped = app._parse_gpdsr_mapping(gpdsr_path)

    assert list(mapping["Record"]) == [4, 5]
    assert list(mapping["Slot"]) == [5, 7]
    assert deduped == [5]


def test_parse_gpdsr_mapping_ignores_unmapped_descriptions(tmp_path, caplog):
    gpdsr_path = tmp_path / "unmapped_gpdsr.txt"
    gpdsr_path.write_text(
        "\n".join(
            [
                "Summary Data Ver: 1.0",
                "Source File: demo",
                "Record\tSample\tDescription",
                "1\t1\tSlot 5 Cycle: 1",
                "2\t2\tNew Record",
                "3\t3\tSlot 7",
            ]
        )
    )

    with caplog.at_level("WARNING"):
        mapping, deduped = app._parse_gpdsr_mapping(gpdsr_path)

    assert list(mapping["Record"]) == [1, 3]
    assert list(mapping["Slot"]) == [5, 7]
    assert deduped == []
    assert any("unrecognized description" in msg for msg in caplog.messages)


def test_parse_gpdsr_mapping_skips_tagged_rows(tmp_path):
    gpdsr_path = tmp_path / "tagged_gpdsr.txt"
    gpdsr_path.write_text(
        "\n".join(
            [
                "Summary Data Ver: 1.0",
                "Source File: demo",
                "Record\tSample\tDescription",
                "1\t1\tTag",
                "2\t2\tSlot 6",
            ]
        )
    )

    mapping, deduped = app._parse_gpdsr_mapping(gpdsr_path)

    assert list(mapping["Record"]) == [2]
    assert list(mapping["Slot"]) == [6]
    assert deduped == []


def test_parse_gpdsr_mapping_errors_on_duplicate_records(tmp_path):
    gpdsr_path = tmp_path / "bad_gpdsr.txt"
    gpdsr_path.write_text(
        "\n".join(
            [
                "Summary Data Ver: 1.0",
                "Source File: demo",
                "Record\tSample\tDescription",
                "1\t1\tSlot 5 Cycle: 1",
                "1\t2\tSlot 6 Cycle: 1",
            ]
        )
    )

    with pytest.raises(ValueError, match="Duplicate Record values"):
        app._parse_gpdsr_mapping(gpdsr_path)


def test_run_dimensional_export_uses_uvc_directory_by_default(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []
    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message, produced = app._run_dimensional_export(uvc_path)

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
    assert produced == {
        "dimensional": expected_output,
        "gpdsr": expected_gpdsr,
    }
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

    success, message, produced = app._run_dimensional_export(uvc_path, original_dir)

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
    assert produced == {
        "dimensional": expected_output,
        "gpdsr": expected_gpdsr,
    }
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


def test_make_dimensional_record_fig_preserves_padding():
    df = pd.DataFrame(
        {
            "N": [1, 2, 3, 4],
            "Slice 1": [1.0, 2.0, np.nan, 4.0],
            "Slice 2": [2.0, 2.5, 3.5, 4.0],
        }
    )

    fig = app._make_dimensional_record_fig(7, df, ["Slice 1", "Slice 2"])

    stacked = df[["Slice 1", "Slice 2"]].stack(future_stack=True).dropna()
    y_min = float(stacked.min())
    y_max = float(stacked.max())
    padding = (y_max - y_min) * 0.05
    expected_range = [y_min - padding, y_max + padding]

    assert fig.layout.yaxis.range == pytest.approx(expected_range)
    assert fig.layout.yaxis2.range == pytest.approx(expected_range)


def test_make_dimensional_record_fig_handles_flat_series_padding():
    df = pd.DataFrame(
        {
            "Slice A": [3.0, 3.0, 3.0],
            "Slice B": [np.nan, np.nan, np.nan],
        }
    )

    fig = app._make_dimensional_record_fig(5, df, ["Slice A", "Slice B"])

    padding = max(abs(3.0), 1.0) * 0.05
    expected_range = [3.0 - padding, 3.0 + padding]

    assert fig.layout.yaxis.range == pytest.approx(expected_range)


def test_compute_slice_extremes_numeric_only():
    df = pd.DataFrame(
        {
            "Slice 1": [1.0, np.nan, 4.0],
            "Slice 2": [2.5, 2.5, 2.5],
            "Notes": ["low", "med", "high"],
        }
    )

    stats = app._compute_slice_extremes(df, ["Slice 1", "Slice 2", "Notes"])

    assert stats == [
        {"slice": "Slice 1", "min": 1.0, "max": 4.0},
        {"slice": "Slice 2", "min": 2.5, "max": 2.5},
    ]


def test_run_dimensional_export_creates_missing_original_dir(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    original_dir = tmp_path / "nested" / "expected"

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message, produced = app._run_dimensional_export(uvc_path, original_dir)

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
    assert produced == {
        "dimensional": expected_output,
        "gpdsr": expected_gpdsr,
    }
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

    success, message, produced = app._run_dimensional_export(
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
    assert produced == {
        "dimensional": expected_output,
        "gpdsr": expected_gpdsr,
    }


def test_run_dimensional_export_supports_dimensional_only(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message, produced = app._run_dimensional_export(
        uvc_path, modes=("dimensional",)
    )

    expected_output = uvc_path.with_suffix(".txt").resolve()
    expected_gpdsr = expected_output.with_name(
        f"{expected_output.stem}_gpdsr{expected_output.suffix}"
    )

    assert success is True
    assert expected_output.exists()
    assert not expected_gpdsr.exists()
    assert message == f"Export complete. Output saved to: {expected_output}"
    assert produced == {"dimensional": expected_output}
    assert len(captured_cmds) == 1
    captured = captured_cmds[0]
    assert captured["cwd"] == str(uvc_path.parent)
    assert captured["cmd"][captured["cmd"].index("-export") + 1] == "dimensional"
    assert Path(captured["cmd"][captured["cmd"].index("-o") + 1]) == expected_output


def test_run_dimensional_export_supports_gpdsr_only(monkeypatch, tmp_path):
    uvc_path = tmp_path / "input.uvc"
    uvc_path.write_text("dummy")

    exe_path = Path("C:/Program Files (x86)/UvWin4/UvWin.exe")

    captured_cmds = []

    _setup_fake_export(monkeypatch, uvc_path, exe_path, captured_cmds)

    success, message, produced = app._run_dimensional_export(
        uvc_path, modes=("gpdsr",)
    )

    expected_output = uvc_path.with_suffix(".txt").resolve()
    expected_gpdsr = expected_output.with_name(
        f"{expected_output.stem}_gpdsr{expected_output.suffix}"
    )

    assert success is True
    assert expected_gpdsr.exists()
    assert message == f"Export complete. Output saved to: {expected_gpdsr}"
    assert produced == {"gpdsr": expected_gpdsr}
    assert len(captured_cmds) == 1
    captured = captured_cmds[0]
    assert captured["cwd"] == str(uvc_path.parent)
    assert captured["cmd"][captured["cmd"].index("-export") + 1] == "gpdsr"
    assert Path(captured["cmd"][captured["cmd"].index("-o") + 1]) == expected_gpdsr
