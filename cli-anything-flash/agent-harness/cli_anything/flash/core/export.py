"""Inspection and plotting utilities for FLASH outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class FlashExportError(RuntimeError):
    """Raised for unsupported result files or plotting failures."""


def _import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise FlashExportError("h5py is required for HDF5 inspection and plotting.") from exc
    return h5py


def is_hdf5(path: Path) -> bool:
    """Return whether a file can be opened as HDF5."""

    h5py = _import_h5py()
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def hdf5_summary(path: Path) -> dict[str, Any]:
    """Summarize datasets and variables in a FLASH HDF5 file."""

    h5py = _import_h5py()
    datasets = []
    with h5py.File(path, "r") as h5:
        for name, obj in h5.items():
            shape = tuple(int(v) for v in getattr(obj, "shape", ()))
            dtype = str(getattr(obj, "dtype", "group"))
            datasets.append({"name": name, "shape": shape, "dtype": dtype})
    variables = [row["name"] for row in datasets if row["shape"] and len(row["shape"]) >= 2]
    return {"path": str(path), "datasets": datasets, "variables": variables}


def _dataset_to_image(array: np.ndarray, block: int = 0, z_index: int = 0) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 4:
        block = max(0, min(block, data.shape[0] - 1))
        z_index = max(0, min(z_index, data.shape[1] - 1))
        return np.asarray(data[block, z_index, :, :], dtype=float)
    if data.ndim == 3:
        block = max(0, min(block, data.shape[0] - 1))
        return np.asarray(data[block, :, :], dtype=float)
    if data.ndim == 2:
        return np.asarray(data, dtype=float)
    if data.ndim == 1:
        return np.asarray(data, dtype=float).reshape(1, -1)
    raise FlashExportError(f"Cannot render array with ndim={data.ndim}")


def _figure_size(width: float, height: float, dpi: int) -> tuple[tuple[float, float], list[int]]:
    """Validate figure dimensions and return inches plus pixel size."""

    if width <= 0 or height <= 0:
        raise FlashExportError("Figure width and height must be positive.")
    if dpi <= 0:
        raise FlashExportError("DPI must be positive.")
    if width > 40 or height > 40 or dpi > 1200:
        raise FlashExportError("Requested image is too large; keep width/height <= 40 inches and dpi <= 1200.")
    return (float(width), float(height)), [int(round(width * dpi)), int(round(height * dpi))]


def plot_hdf5(
    path: Path,
    variable: str,
    output: Path,
    block: int = 0,
    z_index: int = 0,
    scale: str = "linear",
    width: float = 7.0,
    height: float = 5.0,
    dpi: int = 140,
) -> dict[str, Any]:
    """Render one HDF5 dataset to a PNG image."""

    h5py = _import_h5py()
    with h5py.File(path, "r") as h5:
        if variable not in h5:
            names = ", ".join(sorted(h5.keys())[:40])
            raise FlashExportError(f"Variable not found: {variable}. Available starts with: {names}")
        image = _dataset_to_image(h5[variable][()], block=block, z_index=z_index)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise FlashExportError("No finite values to plot.")
    plot_data = image
    if scale == "log10":
        safe = np.where(image > 0, image, np.nan)
        plot_data = np.log10(safe)
    output.parent.mkdir(parents=True, exist_ok=True)
    figsize, pixel_size = _figure_size(width, height, dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(plot_data, origin="lower", aspect="auto")
    ax.set_title(f"{path.name}: {variable}")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return {
        "output": str(output),
        "variable": variable,
        "shape": list(image.shape),
        "scale": scale,
        "figure": {"width": width, "height": height, "dpi": dpi, "pixels": pixel_size},
        "min": float(np.nanmin(image)),
        "max": float(np.nanmax(image)),
        "mean": float(np.nanmean(image)),
    }


def read_text_tail(path: Path, lines: int = 120) -> dict[str, Any]:
    """Return a text file tail."""

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"path": str(path), "line_count": len(text), "tail": "\n".join(text[-lines:])}


def read_numeric_table(path: Path) -> tuple[list[list[float]], list[str] | None]:
    """Read a whitespace or CSV numeric table."""

    rows: list[list[float]] = []
    header: list[str] | None = None
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip().replace(",", " ").split()
            if candidate:
                header = candidate
            continue
        parts = next(csv.reader([stripped])) if "," in stripped else stripped.split()
        try:
            values = [float(part) for part in parts]
        except ValueError:
            if header is None:
                header = [part.strip() for part in parts]
            continue
        if values:
            rows.append(values)
    if not rows:
        raise FlashExportError(f"No numeric rows found in {path}")
    return rows, header


def table_summary(path: Path) -> dict[str, Any]:
    """Summarize numeric columns in a text table."""

    rows, header = read_numeric_table(path)
    arr = np.asarray(rows, dtype=float)
    columns = []
    width = arr.shape[1]
    for idx in range(width):
        name = header[idx] if header and idx < len(header) else f"col{idx}"
        col = arr[:, idx]
        columns.append(
            {
                "index": idx,
                "name": name,
                "min": float(np.nanmin(col)),
                "max": float(np.nanmax(col)),
                "mean": float(np.nanmean(col)),
            }
        )
    return {"path": str(path), "rows": int(arr.shape[0]), "columns": columns}


def plot_table(
    path: Path,
    output: Path,
    x_column: int = 0,
    y_column: int = 1,
    scale: str = "linear",
    width: float = 7.0,
    height: float = 5.0,
    dpi: int = 140,
) -> dict[str, Any]:
    """Plot two numeric columns from a text table."""

    rows, header = read_numeric_table(path)
    arr = np.asarray(rows, dtype=float)
    if x_column >= arr.shape[1] or y_column >= arr.shape[1]:
        raise FlashExportError(f"Column out of range. Table has {arr.shape[1]} columns.")
    x = arr[:, x_column]
    y = arr[:, y_column]
    y_plot = np.log10(np.where(y > 0, y, np.nan)) if scale == "log10" else y
    output.parent.mkdir(parents=True, exist_ok=True)
    figsize, pixel_size = _figure_size(width, height, dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, y_plot, marker=".", linewidth=1)
    x_name = header[x_column] if header and x_column < len(header) else f"col{x_column}"
    y_name = header[y_column] if header and y_column < len(header) else f"col{y_column}"
    ax.set_xlabel(x_name)
    ax.set_ylabel(f"log10({y_name})" if scale == "log10" else y_name)
    ax.set_title(path.name)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return {
        "output": str(output),
        "x_column": x_column,
        "y_column": y_column,
        "rows": int(arr.shape[0]),
        "scale": scale,
        "figure": {"width": width, "height": height, "dpi": dpi, "pixels": pixel_size},
        "y_min": float(np.nanmin(y)),
        "y_max": float(np.nanmax(y)),
    }


def default_output_path(source: Path, suffix: str = ".png") -> Path:
    """Return a deterministic output path next to a source file."""

    safe = source.name.replace("/", "_")
    return source.parent / f"{safe}.agent_plot{suffix}"


def finite_json(value: Any) -> Any:
    """Convert non-finite floats to None for JSON output."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value
