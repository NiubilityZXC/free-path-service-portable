from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


def _resolve_cli(name):
    """Resolve installed CLI command; falls back to python -m for dev."""

    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    print("[_resolve_cli] Falling back to python -m cli_anything.flash")
    return [sys.executable, "-m", "cli_anything.flash"]


def _root() -> Path:
    for item in [
        os.environ.get("FLASH_ROOT"),
        Path(__file__).resolve().parents[5] / "FLASH4.8",
    ]:
        if item and Path(item).exists():
            return Path(item)
    raise RuntimeError("FLASH root not available")


class TestCLISubprocess:
    CLI_BASE = _resolve_cli("cli-anything-flash")
    PASSWORD = os.environ.get("FLASH_SKILL_PASSWORD")

    def _run(self, args, check=True, env=None):
        result = subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
            env=env,
        )
        return result

    def _json(self, args):
        if not self.PASSWORD:
            raise RuntimeError("FLASH_SKILL_PASSWORD is required for authenticated E2E tests")
        result = self._run(["--json", "--password", self.PASSWORD] + args)
        return json.loads(result.stdout)

    def test_help(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "Operate FLASH" in result.stdout

    def test_auth_rejects_missing_and_wrong_password(self):
        clean_env = os.environ.copy()
        clean_env.pop("FLASH_SKILL_PASSWORD", None)
        missing = self._run(["--json", "discover"], check=False, env=clean_env)
        assert missing.returncode != 0
        missing_payload = json.loads(missing.stdout)
        assert missing_payload["ok"] is False
        wrong = self._run(["--json", "--password", "wrong-password", "discover"], check=False)
        assert wrong.returncode != 0
        wrong_payload = json.loads(wrong.stdout)
        assert wrong_payload["ok"] is False
        good = self._json(["auth", "check"])
        assert good["authenticated"] is True

    def test_server_commands_reject_without_password(self):
        clean_env = os.environ.copy()
        clean_env.pop("FLASH_SKILL_PASSWORD", None)
        cases = [
            ["discover"],
            ["--project", "improved_MRT", "status"],
            ["project", "list"],
            ["--project", "improved_MRT", "param", "get", "opacity_useFreePathModel"],
            ["--project", "improved_MRT", "build", "--timeout", "1"],
            ["--project", "improved_MRT", "run", "status"],
            ["--project", "improved_MRT", "command", "run", "pwd"],
            ["--project", "improved_MRT", "file", "list"],
            ["--project", "improved_MRT", "plot", "hdf5", "ZPinch_2D_forced_hdf5_plt_cnt_0000", "--variable", "dens"],
            ["--project", "improved_MRT", "resolution", "show"],
            ["--project", "improved_MRT", "doctor"],
            ["freepath", "control"],
        ]
        for args in cases:
            result = self._run(["--json"] + args, check=False, env=clean_env)
            assert result.returncode != 0, args
            payload = json.loads(result.stdout)
            assert payload["ok"] is False, args
            assert payload["type"] == "FlashAuthError", args

    def test_discover_and_status(self):
        payload = self._json(["discover"])
        assert payload["ok"] is True
        assert payload["projects"]
        assert Path(payload["root"]).resolve() == _root().resolve()
        status = self._json(["--project", "improved_MRT", "status"])
        assert status["ok"] is True
        assert status["has_flash_par"] is True

    def test_param_list_and_get(self):
        payload = self._json(["--project", "improved_MRT", "param", "list", "--search", "opacity_freePath"])
        assert payload["ok"] is True
        names = {row["name"] for row in payload["parameters"]}
        assert "opacity_freePathUrl" in names
        single = self._json(["--project", "improved_MRT", "param", "get", "opacity_useFreePathModel"])
        assert single["ok"] is True

    def test_command_run(self):
        payload = self._json(["--project", "improved_MRT", "command", "run", "pwd && test -f flash.par"])
        assert payload["ok"] is True
        assert payload["returncode"] == 0

    def test_temp_project_edit_run_stop_and_cleanup(self):
        root = _root()
        name = f"agent_skill_tmp_{int(time.time())}"
        try:
            created = self._json(["project", "create", name, "--template", "ZPinch2D_UG"])
            assert created["ok"] is True
            before = self._json(["--project", name, "resolution", "show"])
            assert before["ok"] is True
            assert "total_cells" in before["resolution"]["derived"]
            resolution = self._json(
                [
                    "--project",
                    name,
                    "resolution",
                    "set",
                    "--i-grid-size",
                    "64",
                    "--j-grid-size",
                    "64",
                    "--k-grid-size",
                    "1",
                    "--i-procs",
                    "1",
                    "--j-procs",
                    "1",
                    "--k-procs",
                    "1",
                    "--mesh-copy-count",
                    "1",
                ]
            )
            assert resolution["ok"] is True
            assert resolution["resolution"]["derived"]["total_cells"] == 4096
            assert resolution["resolution"]["derived"]["mpi_ranks"] == 1
            changed = self._json(["--project", name, "param", "set", "run_comment=agent_skill_test", "--allow-new"])
            assert changed["ok"] is True
            viewed = self._json(["--project", name, "file", "view", "flash.par", "--lines", "20"])
            assert viewed["ok"] is True
            started = self._json(["--project", name, "run", "start", "sleep 30"])
            assert started["ok"] is True
            status = self._json(["--project", name, "run", "status"])
            assert status["process"]["running"] is True
            stopped = self._json(["--project", name, "run", "stop"])
            assert stopped["ok"] is True
        finally:
            target = root / name
            if target.exists():
                shutil.rmtree(target)

    def test_plot_outputs_when_samples_exist(self):
        root = _root()
        hdf5 = root / "improved_MRT" / "ZPinch_2D_forced_hdf5_plt_cnt_0000"
        if hdf5.exists():
            hdf5_out = Path("/tmp/cli_anything_flash_hdf5_resolution.png")
            hdf5_out.unlink(missing_ok=True)
            payload = self._json(
                [
                    "--project",
                    "improved_MRT",
                    "plot",
                    "hdf5",
                    hdf5.name,
                    "--variable",
                    "dens",
                    "--width",
                    "4",
                    "--height",
                    "3",
                    "--dpi",
                    "120",
                    "--output",
                    str(hdf5_out),
                ]
            )
            assert payload["ok"] is True
            assert Path(payload["output"]).exists()
            assert payload["figure"]["pixels"] == [480, 360]
            with Image.open(payload["output"]) as image:
                assert image.size == (480, 360)
        table = root / "improved_MRT" / "trajectory.dat"
        if table.exists():
            table_out = Path("/tmp/cli_anything_flash_table_resolution.png")
            table_out.unlink(missing_ok=True)
            payload = self._json(
                [
                    "--project",
                    "improved_MRT",
                    "plot",
                    "table",
                    "trajectory.dat",
                    "--x-column",
                    "0",
                    "--y-column",
                    "1",
                    "--width",
                    "5",
                    "--height",
                    "4",
                    "--dpi",
                    "100",
                    "--output",
                    str(table_out),
                ]
            )
            assert payload["ok"] is True
            assert Path(payload["output"]).exists()
            assert payload["figure"]["pixels"] == [500, 400]
            with Image.open(payload["output"]) as image:
                assert image.size == (500, 400)

    def test_clarify(self):
        result = self._run(["--json", "clarify", "--request", "帮我运行 FLASH 并看结果", "--no-auto-modify"])
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["questions"]
        assert payload["auto_modify"] is False

    def test_workflow_plan_and_doctor(self):
        plan = self._json(["workflow", "plan", "--request", "运行 FLASH 并检查结果", "--no-auto-modify"])
        assert plan["ok"] is True
        assert plan["workflow"]
        assert "subagent" in plan["subagent_review_prompt"].lower()
        doctor = self._json(["--project", "improved_MRT", "doctor", "--include-logs"])
        assert doctor["ok"] is True
        assert "interpretation" in doctor["doctor"]
        assert "next_steps" in doctor["doctor"]
