"""Simulation mesh resolution helpers for FLASH projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cli_anything.flash.core.project import FlashProjectError, get_param, parse_flash_par, set_params


GRID_KEYS = ("iGridSize", "jGridSize", "kGridSize")
PROC_KEYS = ("iProcs", "jProcs", "kProcs", "meshCopyCount")
DOMAIN_KEYS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
KNOWN_RESOLUTION_KEYS = GRID_KEYS + PROC_KEYS + DOMAIN_KEYS


def _param_map(project_path: Path) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in parse_flash_par(project_path / "flash.par")}


def _value(params: dict[str, dict[str, Any]], key: str, default: Any = None) -> Any:
    return params.get(key, {}).get("value", default)


def _int_value(value: Any, key: str, minimum: int = 1) -> int:
    try:
        number = int(value)
    except Exception as exc:
        raise FlashProjectError(f"{key} must be an integer") from exc
    if number < minimum:
        raise FlashProjectError(f"{key} must be >= {minimum}")
    return number


def _float_value(value: Any, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise FlashProjectError(f"{key} must be a number") from exc


def summarize_resolution(project_path: Path) -> dict[str, Any]:
    """Summarize the simulation mesh/domain resolution from flash.par."""

    params = _param_map(project_path)
    grid = {key: _value(params, key) for key in GRID_KEYS if key in params}
    procs = {key: _value(params, key) for key in PROC_KEYS if key in params}
    domain = {key: _value(params, key) for key in DOMAIN_KEYS if key in params}

    grid_int: dict[str, int] = {}
    for key, value in grid.items():
        if value is not None:
            grid_int[key] = _int_value(value, key)

    proc_int = {}
    for key, value in procs.items():
        if value is not None:
            proc_int[key] = _int_value(value, key)

    i_cells = grid_int.get("iGridSize")
    j_cells = grid_int.get("jGridSize")
    k_cells = grid_int.get("kGridSize")
    active_axes = [value for value in (i_cells, j_cells, k_cells) if value and value > 1]
    dimensions = max(1, len(active_axes)) if grid_int else None
    total_cells = None
    if grid_int:
        total_cells = 1
        for key in GRID_KEYS:
            total_cells *= grid_int.get(key, 1)

    spacings = {}
    if i_cells and "xmin" in domain and "xmax" in domain:
        spacings["dx"] = (_float_value(domain["xmax"], "xmax") - _float_value(domain["xmin"], "xmin")) / i_cells
    if j_cells and "ymin" in domain and "ymax" in domain:
        spacings["dy"] = (_float_value(domain["ymax"], "ymax") - _float_value(domain["ymin"], "ymin")) / j_cells
    if k_cells and "zmin" in domain and "zmax" in domain:
        spacings["dz"] = (_float_value(domain["zmax"], "zmax") - _float_value(domain["zmin"], "zmin")) / k_cells

    mpi_ranks = None
    if {"iProcs", "jProcs", "kProcs"}.issubset(proc_int):
        mpi_ranks = proc_int["iProcs"] * proc_int["jProcs"] * proc_int["kProcs"] * proc_int.get("meshCopyCount", 1)

    warnings = []
    for grid_key, proc_key in [("iGridSize", "iProcs"), ("jGridSize", "jProcs"), ("kGridSize", "kProcs")]:
        grid_value = grid_int.get(grid_key)
        proc_value = proc_int.get(proc_key)
        if grid_value and proc_value and grid_value % proc_value != 0:
            warnings.append(f"{grid_key}={grid_value} is not evenly divisible by {proc_key}={proc_value}.")

    notes = [
        "iGridSize/jGridSize/kGridSize control the simulation mesh resolution.",
        "iProcs*jProcs*kProcs*meshCopyCount should match the MPI process count used to launch FLASH.",
        "Changing resolution can change memory use, runtime, stability, and physical convergence.",
    ]
    if mpi_ranks:
        notes.append(f"Current processor grid expects {mpi_ranks} MPI rank(s).")

    return {
        "project": project_path.name,
        "grid": grid,
        "domain": domain,
        "processors": procs,
        "derived": {
            "dimensions": dimensions,
            "total_cells": total_cells,
            "spacings": spacings,
            "mpi_ranks": mpi_ranks,
            "cells_per_mpi_rank": (total_cells / mpi_ranks) if total_cells and mpi_ranks else None,
            "recommended_run_command": (
                None if mpi_ranks is None else f"mpirun -np {mpi_ranks} ./flash4" if mpi_ranks > 1 else "./flash4"
            ),
        },
        "warnings": warnings,
        "notes": notes,
    }


def update_resolution(project_path: Path, updates: dict[str, Any], allow_new: bool = False) -> dict[str, Any]:
    """Validate and update simulation resolution parameters in flash.par."""

    normalized: dict[str, str] = {}
    for key, value in updates.items():
        if value is None:
            continue
        if key not in KNOWN_RESOLUTION_KEYS:
            raise FlashProjectError(f"Unsupported resolution parameter: {key}")
        if key in GRID_KEYS or key in PROC_KEYS:
            normalized[key] = str(_int_value(value, key))
        else:
            normalized[key] = str(_float_value(value, key))

    if not normalized:
        raise FlashProjectError("No resolution updates were provided.")

    projected = summarize_resolution(project_path)
    domain = dict(projected.get("domain", {}))
    for key in DOMAIN_KEYS:
        if key in normalized:
            domain[key] = float(normalized[key])
    for lo, hi in [("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax")]:
        if lo in domain and hi in domain and float(domain[hi]) <= float(domain[lo]):
            raise FlashProjectError(f"{hi} must be greater than {lo}")

    result = set_params(project_path, normalized, allow_new=allow_new)
    result["resolution"] = summarize_resolution(project_path)
    return result
