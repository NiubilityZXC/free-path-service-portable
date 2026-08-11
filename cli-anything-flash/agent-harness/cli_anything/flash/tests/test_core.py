from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from cli_anything.flash.core import freepath
from cli_anything.flash.core.resolution import summarize_resolution, update_resolution
from cli_anything.flash.core.auth import verify_password
from cli_anything.flash.core.doctor import inspect_project
from cli_anything.flash.core.project import create_project, parse_flash_par, set_params
from cli_anything.flash.core.session import Session
from cli_anything.flash.core.export import plot_table, table_summary
from cli_anything.flash.utils import accelerator


def test_parse_flash_par(tmp_path: Path):
    par = tmp_path / "flash.par"
    par.write_text(
        'useRadTrans = .true. # comment\nname = "abc"\ncount = 3\n# ignored = 1\n',
        encoding="utf-8",
    )
    rows = parse_flash_par(par)
    assert [row["name"] for row in rows] == ["useRadTrans", "name", "count"]
    assert rows[0]["value"] is True
    assert rows[1]["value"] == "abc"
    assert rows[2]["value"] == 3


def test_set_params_preserves_backup(tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    (project / "flash.par").write_text("a = 1 # keep\n", encoding="utf-8")
    result = set_params(project, {"a": "2", "newParam": "hello"}, allow_new=True)
    text = (project / "flash.par").read_text(encoding="utf-8")
    assert "a = 2 # keep" in text
    assert 'newParam = "hello"' in text
    assert Path(result["backup"]).exists()


def test_create_project_light_ignores_outputs(tmp_path: Path):
    root = tmp_path
    source = root / "base"
    source.mkdir()
    (source / "flash.par").write_text("a = 1\n", encoding="utf-8")
    (source / "result_hdf5_plt_cnt_0000").write_text("heavy", encoding="utf-8")
    result = create_project(root, "new_project", "base", "light")
    dest = Path(result["created"])
    assert (dest / "flash.par").exists()
    assert not (dest / "result_hdf5_plt_cnt_0000").exists()


def test_resolution_summary_and_update(tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    (project / "flash.par").write_text(
        "\n".join(
            [
                "iGridSize = 128",
                "jGridSize = 64",
                "kGridSize = 1",
                "iProcs = 2",
                "jProcs = 1",
                "kProcs = 1",
                "meshCopyCount = 1",
                "xmin = 0.0",
                "xmax = 0.2",
                "ymin = 0.0",
                "ymax = 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = summarize_resolution(project)
    assert summary["derived"]["total_cells"] == 8192
    assert summary["derived"]["dimensions"] == 2
    assert summary["derived"]["mpi_ranks"] == 2
    assert summary["derived"]["spacings"]["dx"] == pytest.approx(0.2 / 128)
    assert summary["derived"]["spacings"]["dy"] == pytest.approx(0.1 / 64)

    result = update_resolution(
        project,
        {"iGridSize": 256, "jGridSize": 128, "iProcs": 1, "jProcs": 1, "kProcs": 1, "meshCopyCount": 1},
    )
    assert Path(result["backup"]).exists()
    assert result["resolution"]["derived"]["total_cells"] == 32768
    assert result["resolution"]["derived"]["mpi_ranks"] == 1


def test_resolution_rejects_invalid_domain(tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    (project / "flash.par").write_text(
        "iGridSize = 8\njGridSize = 8\nkGridSize = 1\nxmin = 0.0\nxmax = 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="xmax must be greater than xmin"):
        update_resolution(project, {"xmax": -1.0})


def test_table_summary_and_plot(tmp_path: Path):
    table = tmp_path / "data.dat"
    table.write_text("# t y\n0 1\n1 3\n2 9\n", encoding="utf-8")
    summary = table_summary(table)
    assert summary["rows"] == 3
    out = tmp_path / "plot.png"
    result = plot_table(table, out, 0, 1, width=4.0, height=3.0, dpi=100)
    assert out.exists()
    assert result["rows"] == 3
    assert result["figure"]["pixels"] == [400, 300]
    with Image.open(out) as image:
        assert image.size == (400, 300)


def test_session_undo_redo(tmp_path: Path):
    session = Session(path=tmp_path / "session.json")
    session.snapshot("before", {"a": 1})
    item = session.undo()
    assert item["label"] == "before"
    item2 = session.redo()
    assert item2["data"] == {"a": 1}


def test_password_hash_accepts_only_expected_secret():
    password = os.environ.get("FLASH_SKILL_PASSWORD")
    if not password:
        pytest.skip("FLASH_SKILL_PASSWORD is required for auth verification")
    assert verify_password(password) is True
    assert verify_password("wrong-password") is False
    assert verify_password(None) is False


def test_doctor_detects_log_errors_and_interprets(tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    (project / "flash.par").write_text("run_comment = \"doctor\"\n", encoding="utf-8")
    (project / "ZPinch_2D.log").write_text("step 1\nFATAL not converged\n", encoding="utf-8")
    (project / "trajectory.dat").write_text("# t x\n0 1\n1 2\n", encoding="utf-8")
    payload = inspect_project(project)
    messages = " ".join(issue["message"] for issue in payload["issues"])
    assert "flash4 executable is missing" in messages
    assert "Error-like lines" in messages
    assert payload["tables"]
    assert payload["interpretation"]
    assert payload["next_steps"]


def test_freepath_batch_payload_uses_web_endpoint(monkeypatch):
    calls = []

    def fake_request(method, url, payload=None, timeout=30.0):
        calls.append({"method": method, "url": url, "payload": payload, "timeout": timeout})
        return {"ok": True, "summary": {"rows_total": 1}}

    monkeypatch.setattr(freepath, "_request_json", fake_request)
    result = freepath.batch("13 -1.870 1.771 1.755", mode="hybrid", model_elements="13", default_z=None)
    assert result["ok"] is True
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"].endswith("/api/predict/batch")
    assert calls[0]["payload"]["predictionMode"] == "hybrid"
    assert calls[0]["payload"]["modelElements"] == "13"


def test_accelerator_shell_command_quotes_environment_and_arguments():
    command = accelerator.shell_command(
        {
            "environment": {"OMP_NUM_THREADS": "32", "OMP_PLACES": "core list"},
            "argv": ["mpirun", "-np", "2", "/tmp/project with space/flash4.gpu"],
        }
    )
    assert "OMP_NUM_THREADS=32" in command
    assert "OMP_PLACES='core list'" in command
    assert "'/tmp/project with space/flash4.gpu'" in command
