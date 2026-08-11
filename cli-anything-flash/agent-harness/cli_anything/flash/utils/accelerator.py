"""Adapters for the FLASH CPU/GPU backend selector installed in FLASH_ROOT."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


class AcceleratorError(RuntimeError):
    """Raised when the FLASH accelerator helper fails."""


def _helper(root: Path) -> Path:
    path = root / "gpu/flash_backend.py"
    if not path.is_file():
        raise AcceleratorError(f"GPU backend helper not found: {path}")
    return path


def _run(root: Path, args: list[str], timeout: float = 60.0) -> dict[str, Any]:
    command = [str(_helper(root)), *args]
    completed = subprocess.run(
        command,
        cwd=str(root),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "accelerator command failed"
        raise AcceleratorError(message)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AcceleratorError(f"Invalid accelerator JSON: {exc}") from exc


def status(root: Path, project: Path) -> dict[str, Any]:
    return _run(root, ["status", project.name])


def build_gpu(root: Path, project: Path, jobs: int, timeout: float) -> dict[str, Any]:
    return _run(root, ["build-gpu", project.name, "--jobs", str(jobs)], timeout=timeout)


def launch_spec(
    root: Path,
    project: Path,
    backend: str,
    ranks: int | None,
    threads: int | None,
) -> dict[str, Any]:
    args = ["command", project.name, "--backend", backend]
    if ranks is not None:
        args.extend(["--ranks", str(ranks)])
    if threads is not None:
        args.extend(["--threads", str(threads)])
    return _run(root, args)


def shell_command(spec: dict[str, Any]) -> str:
    env = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in spec["environment"].items()
    )
    argv = " ".join(shlex.quote(str(item)) for item in spec["argv"])
    return f"{env} {argv}".strip()
