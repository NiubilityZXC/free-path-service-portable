"""Project diagnostics and result interpretation for FLASH runs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from cli_anything.flash.core.export import hdf5_summary, is_hdf5, table_summary
from cli_anything.flash.core.project import list_files, parse_flash_par, process_status


ERROR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\berror\b",
        r"\bfatal\b",
        r"driver_abortflash",
        r"segmentation fault",
        r"traceback",
        r"\bnan\b",
        r"\binf\b",
        r"not converged",
    ]
]


def _tail(path: Path, lines: int = 80) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _classify_files(files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {"logs": [], "hdf5": [], "tables": [], "plots": [], "parameters": []}
    for row in files:
        name = row["name"].lower()
        suffix = Path(name).suffix
        if suffix in {".log", ".out", ".err"}:
            groups["logs"].append(row)
        if "hdf5" in name or "plt" in name or "chk" in name:
            groups["hdf5"].append(row)
        if suffix in {".dat", ".csv", ".txt"}:
            groups["tables"].append(row)
        if suffix in {".png", ".jpg", ".jpeg"}:
            groups["plots"].append(row)
        if name == "flash.par" or suffix == ".par":
            groups["parameters"].append(row)
    return groups


def inspect_project(project_path: Path, include_logs: bool = True) -> dict[str, Any]:
    """Inspect a FLASH project for likely errors and available result artifacts."""

    files = list_files(project_path, limit=1000)
    groups = _classify_files(files)
    issues: list[dict[str, Any]] = []
    log_summaries = []
    if not (project_path / "flash.par").exists():
        issues.append({"severity": "error", "message": "flash.par is missing."})
    if not (project_path / "flash4").exists():
        issues.append({"severity": "warning", "message": "flash4 executable is missing; build may be required."})
    runtime_libraries = inspect_runtime_libraries(project_path)
    for missing in runtime_libraries.get("missing", []):
        issues.append(
            {
                "severity": "error",
                "message": f"Missing runtime shared library for flash4: {missing}",
                "recommendation": "Install the matching system package or rebuild FLASH against available libraries.",
            }
        )

    if include_logs:
        for row in groups["logs"][:6]:
            path = Path(row["path"])
            tail = _tail(path)
            matches = []
            for line in tail.splitlines():
                if any(pattern.search(line) for pattern in ERROR_PATTERNS):
                    matches.append(line.strip())
            if matches:
                issues.append(
                    {
                        "severity": "error",
                        "message": f"Error-like lines found in {path.name}.",
                        "examples": matches[:5],
                    }
                )
            log_summaries.append({"name": path.name, "tail": tail[-4000:], "matches": matches[:10]})

    hdf5_info = []
    for row in groups["hdf5"][:3]:
        path = Path(row["path"])
        try:
            if is_hdf5(path):
                summary = hdf5_summary(path)
                hdf5_info.append(
                    {
                        "name": path.name,
                        "variables": summary.get("variables", [])[:30],
                        "dataset_count": len(summary.get("datasets", [])),
                    }
                )
        except Exception as exc:
            issues.append({"severity": "warning", "message": f"Could not inspect HDF5 file {path.name}: {exc}"})

    table_info = []
    for row in groups["tables"][:5]:
        path = Path(row["path"])
        try:
            summary = table_summary(path)
            table_info.append({"name": path.name, "rows": summary["rows"], "columns": summary["columns"][:8]})
        except Exception:
            continue

    params = []
    if (project_path / "flash.par").exists():
        try:
            params = parse_flash_par(project_path / "flash.par")
        except Exception as exc:
            issues.append({"severity": "error", "message": f"Could not parse flash.par: {exc}"})

    if not groups["hdf5"] and not groups["tables"]:
        issues.append({"severity": "warning", "message": "No obvious HDF5 or numeric table outputs found yet."})

    interpretation = interpret_results(groups, hdf5_info, table_info, issues)
    return {
        "project": project_path.name,
        "path": str(project_path),
        "process": process_status(project_path),
        "parameter_count": len(params),
        "artifact_counts": {key: len(value) for key, value in groups.items()},
        "runtime_libraries": runtime_libraries,
        "newest_files": files[:20],
        "hdf5": hdf5_info,
        "tables": table_info,
        "logs": log_summaries,
        "issues": issues,
        "interpretation": interpretation,
        "next_steps": suggest_next_steps(issues, hdf5_info, table_info),
    }


def interpret_results(
    groups: dict[str, list[dict[str, Any]]],
    hdf5_info: list[dict[str, Any]],
    table_info: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> list[str]:
    """Return concise human-readable interpretation bullets."""

    notes = []
    if any(issue["severity"] == "error" for issue in issues):
        notes.append("The run needs attention before trusting the physical result because error-like messages were detected.")
    elif hdf5_info or table_info:
        notes.append("The run produced inspectable result artifacts; no obvious fatal log pattern was detected in the scanned files.")
    if hdf5_info:
        variables = sorted({var for item in hdf5_info for var in item.get("variables", [])})
        notes.append(f"HDF5 outputs are available. Candidate variables for plots include: {', '.join(variables[:12])}.")
    if table_info:
        names = ", ".join(item["name"] for item in table_info[:5])
        notes.append(f"Numeric diagnostic tables are available: {names}.")
    if groups["plots"]:
        notes.append("Previously rendered plot images are present; use them together with fresh HDF5/table plots for final interpretation.")
    if not notes:
        notes.append("There is not enough output yet to interpret the simulation result.")
    return notes


def suggest_next_steps(
    issues: list[dict[str, Any]],
    hdf5_info: list[dict[str, Any]],
    table_info: list[dict[str, Any]],
) -> list[str]:
    """Suggest follow-up actions after a run."""

    steps = []
    if any(issue["severity"] == "error" for issue in issues):
        steps.append("Inspect the matched log lines, then adjust the relevant flash.par parameter or rebuild before rerunning.")
        steps.append("If auto-modify was not enabled at workflow start, ask the user before changing parameters.")
    if hdf5_info:
        preferred = next((var for item in hdf5_info for var in item.get("variables", []) if var in {"dens", "tele", "tion", "magz"}), None)
        if preferred:
            steps.append(f"Render and compare the `{preferred}` field across the newest plot files.")
        else:
            steps.append("Choose the physical variable to visualize from the HDF5 variable list.")
    if table_info:
        steps.append("Plot the most relevant diagnostic table columns and compare trends with the expected physics.")
    if not steps:
        steps.append("Run or rerun the simulation with a clear stop condition and requested output variables.")
    return steps


def inspect_runtime_libraries(project_path: Path) -> dict[str, Any]:
    """Run ldd on flash4 and report missing shared libraries without executing FLASH."""

    exe = project_path / "flash4"
    if not exe.exists():
        return {"checked": False, "missing": [], "reason": "flash4 missing"}
    try:
        completed = subprocess.run(
            ["ldd", str(exe)],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:
        return {"checked": False, "missing": [], "reason": str(exc)}
    missing = []
    for line in completed.stdout.splitlines():
        if "not found" in line:
            missing.append(line.strip().split()[0])
    return {
        "checked": True,
        "returncode": completed.returncode,
        "missing": missing,
        "stdout_tail": completed.stdout[-4000:],
    }
