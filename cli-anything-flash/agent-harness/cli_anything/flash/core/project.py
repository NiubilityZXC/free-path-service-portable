"""Project and parameter helpers for FLASH run directories."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parents[5]
BUNDLE_FLASH_ROOT = str(BUNDLE_ROOT / "FLASH4.8")
DEFAULT_ROOTS = (BUNDLE_FLASH_ROOT,)
PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
PARAM_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*(?:[#].*)?)$")
HEAVY_IGNORE = shutil.ignore_patterns(
    "*.o",
    "*.mod",
    "*.log",
    "*.out",
    "*.err",
    "*.h5",
    "*plt*",
    "*chk*",
    ".agent_flash.pid",
    ".agent_flash.log",
    ".web_flash.pid",
    "web_flash_run.log",
    ".success",
)


class FlashProjectError(RuntimeError):
    """Raised for invalid FLASH project operations."""


def resolve_root(root: str | None = None) -> Path:
    """Return the FLASH root from an explicit path, env var, or common paths."""

    candidates = []
    if root:
        candidates.append(root)
    env_root = os.environ.get("FLASH_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(DEFAULT_ROOTS)
    for item in candidates:
        path = Path(item).expanduser()
        if path.exists() and path.is_dir():
            return path
    raise FlashProjectError(
        "Cannot find FLASH root. Pass --root or set FLASH_ROOT to the FLASH4.8 directory."
    )


def is_project_dir(path: Path) -> bool:
    """Return whether a directory looks like a FLASH run/build directory."""

    return path.is_dir() and (
        (path / "flash.par").exists()
        or (path / "flash4").exists()
        or (path / "Makefile").exists()
    )


def discover_projects(root: Path) -> list[dict[str, Any]]:
    """Discover immediate child directories that look like FLASH projects."""

    projects = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith(".") or not is_project_dir(child):
            continue
        files = 0
        try:
            files = sum(1 for p in child.iterdir() if p.is_file())
        except PermissionError:
            files = -1
        projects.append(
            {
                "name": child.name,
                "path": str(child),
                "has_flash_par": (child / "flash.par").exists(),
                "has_executable": os.access(child / "flash4", os.X_OK),
                "has_makefile": (child / "Makefile").exists(),
                "file_count": files,
            }
        )
    return projects


def resolve_project(root: Path, project: str | None, require: bool = True) -> Path | None:
    """Resolve a project name or path under the FLASH root."""

    if not project:
        if not require:
            return None
        raise FlashProjectError("Project is required. Use project list first, then pass --project NAME.")
    raw = Path(project).expanduser()
    path = raw if raw.is_absolute() else root / project
    logical_path = path
    real_path = path.resolve()
    try:
        real_path.relative_to(root.resolve())
    except ValueError as exc:
        raise FlashProjectError(f"Project must be inside FLASH root: {root}") from exc
    if require and not logical_path.exists():
        raise FlashProjectError(f"Project not found: {logical_path}")
    return logical_path


def project_status(root: Path, project: str | None = None) -> dict[str, Any]:
    """Return status for the root or one project."""

    if not project:
        projects = discover_projects(root)
        return {"root": str(root), "project_count": len(projects), "projects": projects}
    path = resolve_project(root, project)
    assert path is not None
    params = parse_flash_par(path / "flash.par") if (path / "flash.par").exists() else []
    pid_info = process_status(path)
    return {
        "root": str(root),
        "project": path.name,
        "path": str(path),
        "has_flash_par": (path / "flash.par").exists(),
        "parameter_count": len(params),
        "has_executable": os.access(path / "flash4", os.X_OK),
        "has_makefile": (path / "Makefile").exists(),
        "process": pid_info,
        "files": list_files(path, limit=200),
    }


def _strip_inline_comment(text: str) -> tuple[str, str]:
    in_quote: str | None = None
    for idx, char in enumerate(text):
        if char in {"'", '"'}:
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
        if char == "#" and in_quote is None:
            return text[:idx].rstrip(), text[idx:].rstrip()
    return text.rstrip(), ""


def parse_flash_par(path: Path) -> list[dict[str, Any]]:
    """Parse active `name = value` entries from a FLASH `flash.par` file."""

    if not path.exists():
        raise FlashProjectError(f"flash.par not found: {path}")
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = PARAM_RE.match(line)
        if not match or line.lstrip().startswith("#"):
            continue
        value_text, inline_comment = _strip_inline_comment(match.group(4))
        rows.append(
            {
                "line": lineno,
                "name": match.group(2),
                "value": parse_value(value_text.strip()),
                "raw_value": value_text.strip(),
                "comment": (inline_comment or match.group(5)).strip(),
                "raw": line,
            }
        )
    return rows


def get_param(project_path: Path, name: str) -> dict[str, Any]:
    """Return one parameter by exact name."""

    for row in parse_flash_par(project_path / "flash.par"):
        if row["name"] == name:
            return row
    raise FlashProjectError(f"Parameter not found in flash.par: {name}")


def parse_value(text: str) -> Any:
    """Parse a FLASH parameter value into a JSON-friendly value."""

    low = text.strip().lower()
    if low in {".true.", "true"}:
        return True
    if low in {".false.", "false"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if any(ch in text for ch in ".eEdD"):
            return float(text.replace("D", "E").replace("d", "e"))
        return int(text)
    except ValueError:
        return text


def format_value(value: str) -> str:
    """Format an incoming CLI value for `flash.par`."""

    text = str(value).strip()
    low = text.lower()
    if low in {"true", ".true.", "yes", "on"}:
        return ".true."
    if low in {"false", ".false.", "no", "off"}:
        return ".false."
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text
    try:
        float(text.replace("D", "E").replace("d", "e"))
        return text
    except ValueError:
        return '"' + text.replace('"', '\\"') + '"'


def set_params(project_path: Path, updates: dict[str, str], allow_new: bool = False) -> dict[str, Any]:
    """Set parameters in `flash.par`, preserving comments and making a backup."""

    par_path = project_path / "flash.par"
    if not par_path.exists():
        raise FlashProjectError(f"flash.par not found: {par_path}")
    lines = par_path.read_text(encoding="utf-8", errors="replace").splitlines()
    remaining = dict(updates)
    changed = []
    out_lines = []
    for line in lines:
        match = PARAM_RE.match(line)
        if match and not line.lstrip().startswith("#"):
            name = match.group(2)
            if name in remaining:
                formatted = format_value(remaining.pop(name))
                value_part, inline_comment = _strip_inline_comment(match.group(4))
                comment = inline_comment or match.group(5)
                new_line = f"{match.group(1)}{name}{match.group(3)}{formatted}"
                if comment:
                    new_line += f" {comment.strip()}"
                changed.append({"name": name, "old": value_part.strip(), "new": formatted})
                out_lines.append(new_line)
                continue
        out_lines.append(line)
    if remaining and not allow_new:
        missing = ", ".join(sorted(remaining))
        raise FlashProjectError(f"Parameter(s) not found: {missing}. Use --allow-new to append.")
    if remaining:
        out_lines.append("")
        out_lines.append("# Added by cli-anything-flash")
        for name, value in remaining.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise FlashProjectError(f"Invalid parameter name: {name}")
            formatted = format_value(value)
            out_lines.append(f"{name} = {formatted}")
            changed.append({"name": name, "old": None, "new": formatted})
    backup = par_path.with_name(f"flash.par.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(par_path, backup)
    par_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {"project": str(project_path), "backup": str(backup), "changed": changed}


def create_project(root: Path, name: str, template: str, copy_mode: str = "light") -> dict[str, Any]:
    """Create a new project by copying an existing FLASH run directory."""

    if not PROJECT_RE.match(name):
        raise FlashProjectError("Project name must start with alnum and contain only alnum, dot, underscore, dash.")
    source = resolve_project(root, template)
    assert source is not None
    dest = root / name
    if dest.exists():
        raise FlashProjectError(f"Destination already exists: {dest}")
    ignore = None if copy_mode == "full" else HEAVY_IGNORE
    shutil.copytree(source, dest, symlinks=True, ignore=ignore)
    return {"created": str(dest), "template": str(source), "copy_mode": copy_mode}


def list_files(project_path: Path, limit: int = 500) -> list[dict[str, Any]]:
    """List useful files in a FLASH project directory."""

    interesting = []
    for item in project_path.iterdir():
        if not item.is_file():
            continue
        suffix = item.suffix.lower()
        name = item.name.lower()
        if (
            suffix in {".log", ".out", ".err", ".txt", ".dat", ".csv", ".par", ".png", ".jpg", ".jpeg"}
            or "hdf5" in name
            or "plt" in name
            or "chk" in name
            or item.name in {"flash.par", "Makefile"}
        ):
            stat = item.stat()
            interesting.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    interesting.sort(key=lambda row: (row["mtime"], row["name"]), reverse=True)
    return interesting[:limit]


def process_status(project_path: Path) -> dict[str, Any]:
    """Read the harness PID file and report whether the process is alive."""

    pid_file = project_path / ".agent_flash.pid"
    if not pid_file.exists():
        return {"running": False, "pid": None}
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return {"running": False, "pid": None, "error": "invalid pid file"}
    proc_stat = Path("/proc") / str(pid) / "stat"
    if not proc_stat.exists():
        return {"running": False, "pid": pid}
    try:
        state = proc_stat.read_text(encoding="utf-8", errors="replace").split()[2]
    except Exception:
        state = "?"
    return {"running": state != "Z", "pid": pid, "state": state}
