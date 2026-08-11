"""Real FLASH backend process wrappers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from cli_anything.flash.core.project import process_status


class FlashBackendError(RuntimeError):
    """Raised when a FLASH backend command fails."""


def run_shell(project_path: Path, command: str, timeout: float = 60.0) -> dict[str, Any]:
    """Run a shell command inside one FLASH project directory."""

    completed = subprocess.run(
        command,
        cwd=str(project_path),
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "cwd": str(project_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build(project_path: Path, jobs: int = 2, timeout: float = 3600.0) -> dict[str, Any]:
    """Run make in the project directory."""

    return run_shell(project_path, f"make -j{int(jobs)}", timeout=timeout)


def start(project_path: Path, command: str = "./flash4") -> dict[str, Any]:
    """Start a FLASH command in the background and store a PID file."""

    pid_file = project_path / ".agent_flash.pid"
    log_file = project_path / ".agent_flash.log"
    status = process_status(project_path)
    if status.get("running"):
        raise FlashBackendError(f"FLASH process already running with pid {status.get('pid')}")
    log_handle = log_file.open("ab")
    proc = subprocess.Popen(
        command,
        cwd=str(project_path),
        shell=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(0.2)
    return {"pid": proc.pid, "command": command, "log": str(log_file), "running": proc.poll() is None}


def stop(project_path: Path, grace: float = 3.0) -> dict[str, Any]:
    """Stop a background process started by this harness."""

    pid_file = project_path / ".agent_flash.pid"
    status = process_status(project_path)
    pid = status.get("pid")
    if not pid:
        return {"stopped": False, "message": "No PID file"}
    if not status.get("running"):
        pid_file.unlink(missing_ok=True)
        return {"stopped": False, "pid": pid, "message": "Process was not running"}
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return {"stopped": False, "pid": pid, "message": "Process disappeared"}
    deadline = time.time() + grace
    while time.time() < deadline:
        if not process_status(project_path).get("running"):
            pid_file.unlink(missing_ok=True)
            return {"stopped": True, "pid": pid, "signal": "TERM"}
        time.sleep(0.1)
    try:
        os.killpg(int(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_file.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid, "signal": "KILL"}
