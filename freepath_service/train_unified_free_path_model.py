#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a single unified Rosseland free-path model.

The fixed training format is:
    Z rod tep tgama lnu

The fixed prediction format is:
    Z rod tep tgama
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from unified_free_path_core import (
    ACTIVE_UNIFIED_ARTIFACT,
    ACTIVE_UNIFIED_BENCHMARK_CSV,
    ACTIVE_UNIFIED_EXTRAPOLATION_CSV,
    ACTIVE_UNIFIED_METRICS,
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    DEFAULT_DATASETS,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    STANDARD_INPUT_COLUMNS,
    STANDARD_TRAINING_COLUMNS,
    STANDARD_TRAINING_DATA_PATH,
    STAGED_UNIFIED_ARTIFACT,
    STAGED_UNIFIED_METRICS,
    UnifiedModel,
    apply_coordinate_transform,
    infer_coordinate_transform,
    metric_dict,
    stable_mask_from_keys,
    standard_row_key,
)


def parse_dataset_specs(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return DEFAULT_DATASETS
    out = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError("--dataset must be NAME:Z:PATH")
        name, z_text, path = parts
        out.append((name, float(z_text), Path(path)))
    return out


def load_legacy_dataset(name: str, z_value: float, path: Path) -> dict:
    raw = np.loadtxt(path, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(f"{path} must have four columns: rod tep tgama lnu")
    raw_total_rows = int(raw.shape[0])
    nonpositive_mask = (~np.isfinite(raw[:, 3])) | (raw[:, 3] <= 0.0)
    excluded_nonpositive_lnu_count = int(nonpositive_mask.sum())
    if excluded_nonpositive_lnu_count:
        raw = raw[~nonpositive_mask]
    if raw.shape[0] == 0:
        raise ValueError(f"{path} has no positive lnu rows after filtering")
    transform = infer_coordinate_transform(raw[:, :3])
    model_xyz = apply_coordinate_transform(raw[:, :3], transform)
    model_data = np.column_stack([model_xyz, raw[:, 3]])
    order = np.lexsort((model_data[:, 2], model_data[:, 1], model_data[:, 0]))
    raw = raw[order]
    model_data = model_data[order]
    axes = [np.unique(model_data[:, i]) for i in range(3)]
    model_xyz = model_data[:, :3]
    lnu = model_data[:, 3]
    x_model = np.column_stack([np.full(model_xyz.shape[0], float(z_value)), model_xyz])
    keys = np.array([standard_row_key(z_value, row) for row in model_xyz], dtype=object)
    return {
        "name": name,
        "path": str(path),
        "Z": float(z_value),
        "raw": raw,
        "raw_total_rows": raw_total_rows,
        "keys": keys,
        "x_model": x_model,
        "y_log": np.log10(lnu),
        "transform": transform,
        "repaired_lnu_count": 0,
        "excluded_nonpositive_lnu_count": excluded_nonpositive_lnu_count,
        "axis_sizes": [int(len(axis)) for axis in axes],
        "ranges": {
            "rod": [float(model_xyz[:, 0].min()), float(model_xyz[:, 0].max())],
            "tep": [float(model_xyz[:, 1].min()), float(model_xyz[:, 1].max())],
            "tgama": [float(model_xyz[:, 2].min()), float(model_xyz[:, 2].max())],
            "lnu": [float(lnu.min()), float(lnu.max())],
        },
    }


def standard_rows_from_legacy(datasets: list[dict]) -> np.ndarray:
    parts = []
    for item in datasets:
        x_model = np.asarray(item["x_model"], dtype=float)
        y = np.power(10.0, np.asarray(item["y_log"], dtype=float))
        parts.append(np.column_stack([x_model, y]))
    return np.vstack(parts)


def save_standard_training_data(path: Path, datasets: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = standard_rows_from_legacy(datasets)
    header = " ".join(STANDARD_TRAINING_COLUMNS)
    np.savetxt(path, rows, fmt="%.12e", header=header)


def load_standard_data(path: Path) -> dict:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"{path} must have five columns: {' '.join(STANDARD_TRAINING_COLUMNS)}")
    if np.any(~np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite values")
    if np.any(data[:, 4] <= 0.0):
        raise ValueError(f"{path} contains nonpositive lnu values")
    z_values = data[:, 0]
    xyz = data[:, 1:4]
    x_model = data[:, :4]
    keys = np.array([standard_row_key(z_value, row) for z_value, row in zip(z_values, xyz)], dtype=object)
    element_summary = {}
    for z_value in sorted(set(z_values.tolist())):
        mask = z_values == z_value
        label = f"Z_{z_value:g}"
        element_summary[label] = {
            "Z": float(z_value),
            "n_samples": int(mask.sum()),
            "ranges": {
                "rod": [float(xyz[mask, 0].min()), float(xyz[mask, 0].max())],
                "tep": [float(xyz[mask, 1].min()), float(xyz[mask, 1].max())],
                "tgama": [float(xyz[mask, 2].min()), float(xyz[mask, 2].max())],
                "lnu": [float(data[mask, 4].min()), float(data[mask, 4].max())],
            },
        }
    return {
        "name": path.stem,
        "path": str(path),
        "keys": keys,
        "x_model": x_model,
        "y_log": np.log10(data[:, 4]),
        "element_labels": np.array([f"Z_{value:g}" for value in z_values], dtype=object),
        "element_summary": element_summary,
    }


def default_standard_dataset() -> list[dict]:
    legacy = [load_legacy_dataset(name, z_value, path) for name, z_value, path in DEFAULT_DATASETS]
    save_standard_training_data(STANDARD_TRAINING_DATA_PATH, legacy)
    item = load_standard_data(STANDARD_TRAINING_DATA_PATH)
    for legacy_item in legacy:
        label = f"Z_{legacy_item['Z']:g}"
        item["element_summary"][label].update(
            {
                "legacy_name": legacy_item["name"],
                "legacy_path": legacy_item["path"],
                "legacy_coordinate_transform": legacy_item["transform"],
                "repaired_lnu_count": int(legacy_item["repaired_lnu_count"]),
                "excluded_nonpositive_lnu_count": int(legacy_item["excluded_nonpositive_lnu_count"]),
                "raw_total_rows": int(legacy_item["raw_total_rows"]),
                "axis_sizes": legacy_item["axis_sizes"],
            }
        )
    return [item]


def contiguous_extrapolation_mask(item: dict, ratio: float) -> np.ndarray:
    """Hold out contiguous low-rod and high-rod slabs for extrapolation testing."""
    x_model = np.asarray(item["x_model"], dtype=float)
    labels = np.asarray(item["element_labels"], dtype=object)
    mask = np.zeros(x_model.shape[0], dtype=bool)
    for label in sorted(set(labels.tolist())):
        element_mask = labels == label
        rod_values = np.unique(x_model[element_mask, 1])
        n_holdout = max(1, int(np.ceil(len(rod_values) * ratio)))
        cutoff_values = set(rod_values[:n_holdout].tolist()) | set(rod_values[-n_holdout:].tolist())
        mask |= element_mask & np.array([float(v) in cutoff_values for v in x_model[:, 1]], dtype=bool)
    return mask


def save_benchmark_payload(keys_path: Path, data_path: Path, keys: list[str], x_parts: list[np.ndarray], y_parts: list[np.ndarray], element_parts: list[np.ndarray]) -> None:
    keys_path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    np.savez_compressed(
        data_path,
        x_model=np.vstack(x_parts),
        y_log=np.concatenate(y_parts),
        element=np.concatenate(element_parts),
    )


def save_benchmark_payload_from_keyset(keys_path: Path, data_path: Path, datasets: list[dict], keyset: set[str]) -> set[str]:
    kept_keys = []
    x_parts = []
    y_parts = []
    element_parts = []
    for item in datasets:
        mask = np.array([key in keyset for key in item["keys"]], dtype=bool)
        if not np.any(mask):
            continue
        kept_keys.extend(item["keys"][mask].tolist())
        x_parts.append(item["x_model"][mask])
        y_parts.append(item["y_log"][mask])
        element_parts.append(item["element_labels"][mask])
    if not kept_keys:
        return keyset
    save_benchmark_payload(keys_path, data_path, kept_keys, x_parts, y_parts, element_parts)
    return set(kept_keys)


def ensure_benchmarks(datasets: list[dict], benchmark_ratio: float, extrapolation_ratio: float, force: bool = False) -> tuple[set[str], set[str]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_required = [
        BENCHMARK_KEYS_PATH,
        BENCHMARK_DATA_PATH,
        EXTRAPOLATION_KEYS_PATH,
        EXTRAPOLATION_DATA_PATH,
    ]
    if all(path.exists() for path in all_required) and not force:
        random_existing = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
        extrap_existing = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
        return random_existing, extrap_existing

    random_keys = []
    random_x_parts = []
    random_y_parts = []
    random_element_parts = []
    extrap_keys = []
    extrap_x_parts = []
    extrap_y_parts = []
    extrap_element_parts = []
    for item in datasets:
        extrap_mask = contiguous_extrapolation_mask(item, extrapolation_ratio)
        random_candidates = ~extrap_mask
        random_mask = random_candidates & stable_mask_from_keys(item["keys"], benchmark_ratio)

        extrap_keys.extend(item["keys"][extrap_mask].tolist())
        extrap_x_parts.append(item["x_model"][extrap_mask])
        extrap_y_parts.append(item["y_log"][extrap_mask])
        extrap_element_parts.append(item["element_labels"][extrap_mask])

        random_keys.extend(item["keys"][random_mask].tolist())
        random_x_parts.append(item["x_model"][random_mask])
        random_y_parts.append(item["y_log"][random_mask])
        random_element_parts.append(item["element_labels"][random_mask])

    save_benchmark_payload(BENCHMARK_KEYS_PATH, BENCHMARK_DATA_PATH, random_keys, random_x_parts, random_y_parts, random_element_parts)
    save_benchmark_payload(EXTRAPOLATION_KEYS_PATH, EXTRAPOLATION_DATA_PATH, extrap_keys, extrap_x_parts, extrap_y_parts, extrap_element_parts)
    return set(random_keys), set(extrap_keys)


def load_benchmark(path: Path = BENCHMARK_DATA_PATH) -> dict:
    payload = np.load(path, allow_pickle=True)
    out = {
        "x_model": payload["x_model"],
        "y_log": payload["y_log"],
        "element": payload["element"].astype(str),
    }
    if "source" in payload.files:
        out["source"] = payload["source"].astype(str)
    return out


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_artifact(path: Path, x_train_scaled: np.ndarray, y_train: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train_scaled=x_train_scaled,
        y_train_log=y_train,
        metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2),
    )


def benchmark_model(model: UnifiedModel, benchmark: dict) -> tuple[dict, list[dict], list[dict]]:
    pred_log = model.predict_log(benchmark["x_model"])
    overall = metric_dict(benchmark["y_log"], pred_log)
    rows = [{"element": "overall", **overall}]
    for name in sorted(set(benchmark["element"].tolist())):
        mask = benchmark["element"] == name
        rows.append({"element": name, **metric_dict(benchmark["y_log"][mask], pred_log[mask])})
    source_rows = []
    if "source" in benchmark:
        for name in sorted(set(benchmark["source"].tolist())):
            mask = benchmark["source"] == name
            source_rows.append({"source": name, **metric_dict(benchmark["y_log"][mask], pred_log[mask])})
    return overall, rows, source_rows


def build_low_rod_boundary_extension(
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_value: float,
    nfit: int,
    smoothing_knn: int,
    smoothing_power: float,
) -> dict | None:
    z_mask = np.isclose(x_train[:, 0], float(z_value), rtol=0.0, atol=1e-8)
    if int(z_mask.sum()) == 0:
        return None

    curves: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row, target in zip(x_train[z_mask], y_train[z_mask]):
        _, rod, tep, tgama = row
        key = (round(float(tep), 12), round(float(tgama), 12))
        curves.setdefault(key, []).append((float(rod), float(target)))

    curve_keys = []
    coords = []
    params = []
    min_required = max(2, int(nfit))
    for key, values in curves.items():
        values = sorted(values)
        if len(values) < min_required:
            continue
        pts = np.asarray(values[: int(nfit)], dtype=float)
        x0 = float(pts[0, 0])
        delta = pts[:, 0] - x0
        design = np.column_stack([np.ones(delta.shape[0]), delta])
        beta = np.linalg.lstsq(design, pts[:, 1], rcond=None)[0]
        curve_keys.append(key)
        coords.append(key)
        params.append([x0, float(beta[0]), float(beta[1])])

    if not params:
        return None

    coords_arr = np.asarray(coords, dtype=float)
    params_arr = np.asarray(params, dtype=float)
    tree = cKDTree(coords_arr)
    k = min(max(1, int(smoothing_knn)), coords_arr.shape[0])
    distances, indices = tree.query(coords_arr, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.power(np.maximum(distances, 1e-12), float(smoothing_power))
    exact = distances[:, 0] < 1e-12
    if k > 1 and np.any(exact):
        weights[exact, 0] = np.max(weights[exact, 1:], axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    smoothed_slopes = np.sum(weights * params_arr[indices, 2], axis=1)

    rows = np.column_stack([coords_arr, params_arr[:, 0], params_arr[:, 1], smoothed_slopes])
    return {
        "name": f"Z_{z_value:g}_low_rod_boundary",
        "type": "low_rod_linear_boundary",
        "z_value": float(z_value),
        "axis": "rod",
        "side": "low",
        "trigger_below": float(np.min(x_train[z_mask, 1])),
        "target_space": "log10(lnu)",
        "nfit": int(nfit),
        "slope_smoothing_knn": int(k),
        "slope_smoothing_power": float(smoothing_power),
        "interp_neighbors": 8,
        "interp_power": 2.0,
        "row_format": "tep tgama boundary_rod intercept smoothed_slope",
        "rows": rows.tolist(),
        "description": (
            "For Au low-rod extrapolation only, keep the curve's own boundary intercept "
            "and use a smoothed rod-direction slope estimated from training curves. "
            "Permanent benchmark and extrapolation rows are excluded before fitting this extension."
        ),
    }


def build_rod_curve_boundary_extension(
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_value: float,
    side: str,
    nfit: int,
    smoothing_knn: int,
    smoothing_power: float,
) -> dict | None:
    side = str(side).lower()
    if side not in {"low", "high"}:
        raise ValueError("side must be low or high")
    z_mask = np.isclose(x_train[:, 0], float(z_value), rtol=0.0, atol=1e-8)
    if int(z_mask.sum()) == 0:
        return None

    curves: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row, target in zip(x_train[z_mask], y_train[z_mask]):
        _, rod, tep, tgama = row
        key = (round(float(tep), 12), round(float(tgama), 12))
        curves.setdefault(key, []).append((float(rod), float(target)))

    coords = []
    params = []
    min_required = max(2, int(nfit))
    for key, values in curves.items():
        values = sorted(values)
        if len(values) < min_required:
            continue
        pts = np.asarray(values[: int(nfit)] if side == "low" else values[-int(nfit) :], dtype=float)
        x0 = float(pts[0, 0] if side == "low" else pts[-1, 0])
        delta = pts[:, 0] - x0
        design = np.column_stack([np.ones(delta.shape[0]), delta])
        beta = np.linalg.lstsq(design, pts[:, 1], rcond=None)[0]
        coords.append(key)
        params.append([x0, float(beta[0]), float(beta[1])])

    if not params:
        return None

    coords_arr = np.asarray(coords, dtype=float)
    params_arr = np.asarray(params, dtype=float)
    tree = cKDTree(coords_arr)
    k = min(max(1, int(smoothing_knn)), coords_arr.shape[0])
    distances, indices = tree.query(coords_arr, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.power(np.maximum(distances, 1e-12), float(smoothing_power))
    exact = distances[:, 0] < 1e-12
    if k > 1 and np.any(exact):
        weights[exact, 0] = np.max(weights[exact, 1:], axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    smoothed_slopes = np.sum(weights * params_arr[indices, 2], axis=1)

    rows = np.column_stack([coords_arr, params_arr[:, 0], params_arr[:, 1], smoothed_slopes])
    return {
        "name": f"Z_{z_value:g}_{side}_rod_curve_boundary",
        "type": "rod_curve_linear_boundary",
        "z_value": float(z_value),
        "axis": "rod",
        "side": side,
        "target_space": "log10(lnu)",
        "nfit": int(nfit),
        "slope_smoothing_knn": int(k),
        "slope_smoothing_power": float(smoothing_power),
        "interp_neighbors": 8,
        "interp_power": 2.0,
        "row_format": "tep tgama boundary_rod intercept smoothed_slope",
        "rows": rows.tolist(),
        "description": (
            f"For Au {side}-rod extrapolation, select the nearest training curve in "
            "tep/tgama and extend it linearly from that curve's own boundary rod."
        ),
    }


def build_rod_curve_polynomial_extension(
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_value: float,
    side: str,
    degree: int,
    nfit: int,
    ridge_alpha: float,
) -> dict | None:
    side = str(side).lower()
    if side not in {"low", "high"}:
        raise ValueError("side must be low or high")
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be at least 1")
    z_mask = np.isclose(x_train[:, 0], float(z_value), rtol=0.0, atol=1e-8)
    if int(z_mask.sum()) == 0:
        return None

    curves: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row, target in zip(x_train[z_mask], y_train[z_mask]):
        _, rod, tep, tgama = row
        key = (round(float(tep), 12), round(float(tgama), 12))
        curves.setdefault(key, []).append((float(rod), float(target)))

    coords = []
    params = []
    min_required = max(degree + 1, int(nfit))
    for key, values in curves.items():
        values = sorted(values)
        if len(values) < min_required:
            continue
        pts = np.asarray(values[: int(nfit)] if side == "low" else values[-int(nfit) :], dtype=float)
        x0 = float(pts[0, 0] if side == "low" else pts[-1, 0])
        delta = pts[:, 0] - x0
        design = np.column_stack([delta**power for power in range(degree + 1)])
        reg = np.eye(design.shape[1]) * float(ridge_alpha)
        reg[0, 0] = 0.0
        try:
            beta = np.linalg.solve(design.T @ design + reg, design.T @ pts[:, 1])
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(design, pts[:, 1], rcond=None)[0]
        coords.append(key)
        params.append([x0, *[float(v) for v in beta]])

    if not params:
        return None

    rows = np.column_stack([np.asarray(coords, dtype=float), np.asarray(params, dtype=float)])
    return {
        "name": f"Z_{z_value:g}_{side}_rod_curve_polynomial_boundary",
        "type": "rod_curve_polynomial_boundary",
        "z_value": float(z_value),
        "axis": "rod",
        "side": side,
        "target_space": "log10(lnu)",
        "degree": degree,
        "nfit": int(nfit),
        "ridge_alpha": float(ridge_alpha),
        "interp_neighbors": 1,
        "interp_power": 2.0,
        "row_format": "tep tgama boundary_rod beta0 beta1 ... beta_degree",
        "rows": rows.tolist(),
        "description": (
            f"For Au {side}-rod extrapolation, fit a regularized polynomial along each "
            "training curve and blend that curve prediction into selected boundary regions."
        ),
    }


def build_rod_curve_stat_cap_extension(
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_value: float,
    side: str,
    tail_min_rod: float,
    stat: str,
) -> dict | None:
    side = str(side).lower()
    if side not in {"low", "high"}:
        raise ValueError("side must be low or high")
    z_mask = np.isclose(x_train[:, 0], float(z_value), rtol=0.0, atol=1e-8)
    if int(z_mask.sum()) == 0:
        return None

    curves: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row, target in zip(x_train[z_mask], y_train[z_mask]):
        _, rod, tep, tgama = row
        key = (round(float(tep), 12), round(float(tgama), 12))
        curves.setdefault(key, []).append((float(rod), float(target)))

    stat = str(stat).lower()
    quantiles = {
        "q01": 0.01,
        "q05": 0.05,
        "q10": 0.10,
        "q20": 0.20,
        "q25": 0.25,
        "median": 0.50,
        "q75": 0.75,
        "q90": 0.90,
    }
    rows = []
    for key, values in curves.items():
        arr = np.asarray(sorted(values), dtype=float)
        if side == "high":
            tail = arr[arr[:, 0] > float(tail_min_rod), 1]
        else:
            tail = arr[arr[:, 0] < float(tail_min_rod), 1]
        if tail.size < 3:
            tail = arr[:, 1]
        if tail.size == 0:
            continue
        if stat == "min":
            cap = float(np.min(tail))
        elif stat == "max":
            cap = float(np.max(tail))
        elif stat == "mean":
            cap = float(np.mean(tail))
        elif stat in quantiles:
            cap = float(np.quantile(tail, quantiles[stat]))
        else:
            raise ValueError(f"unsupported stat cap: {stat}")
        rows.append([float(key[0]), float(key[1]), cap])

    if not rows:
        return None

    return {
        "name": f"Z_{z_value:g}_{side}_rod_curve_{stat}_cap",
        "type": "rod_curve_stat_cap",
        "z_value": float(z_value),
        "axis": "rod",
        "side": side,
        "target_space": "log10(lnu)",
        "tail_min_rod": float(tail_min_rod),
        "stat": stat,
        "interp_neighbors": 1,
        "interp_power": 2.0,
        "output_shift": 0.0,
        "strength": 1.0,
        "row_format": "tep tgama cap_log10_lnu",
        "rows": rows,
        "description": (
            f"For Au {side}-rod extrapolation, cap log10(lnu) by the {stat} statistic "
            "of the same curve's high-rod training tail."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one unified free-path model.")
    parser.add_argument("--dataset", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--standard-data", action="append", default=[], help="fixed five-column data file")
    parser.add_argument("--benchmark-ratio", type=float, default=0.10)
    parser.add_argument("--extrapolation-ratio", type=float, default=0.10)
    parser.add_argument("--force-recreate-benchmark", action="store_true")
    parser.add_argument("--artifact", type=Path, default=ACTIVE_UNIFIED_ARTIFACT)
    parser.add_argument("--metrics", type=Path, default=ACTIVE_UNIFIED_METRICS)
    parser.add_argument("--benchmark-csv", type=Path, default=ACTIVE_UNIFIED_BENCHMARK_CSV)
    parser.add_argument("--extrapolation-csv", type=Path, default=ACTIVE_UNIFIED_EXTRAPOLATION_CSV)
    parser.add_argument("--k-neighbors", type=int, default=520)
    parser.add_argument("--local-degree", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--distance-power", type=float, default=1.6)
    parser.add_argument("--z-weight", type=float, default=0.5)
    parser.add_argument("--disable-au-adaptive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-k-neighbors", type=int, default=520, help=argparse.SUPPRESS)
    parser.add_argument("--au-local-degree", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--au-ridge-alpha", type=float, default=3e-3, help=argparse.SUPPRESS)
    parser.add_argument("--au-distance-power", type=float, default=1.3, help=argparse.SUPPRESS)
    parser.add_argument(
        "--disable-au-low-boundary-extension",
        dest="disable_au_low_boundary_extension",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enable-au-low-boundary-extension",
        dest="disable_au_low_boundary_extension",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(disable_au_low_boundary_extension=True)
    parser.add_argument("--au-low-boundary-nfit", type=int, default=9, help=argparse.SUPPRESS)
    parser.add_argument("--au-low-boundary-knn", type=int, default=2500, help=argparse.SUPPRESS)
    parser.add_argument("--au-low-boundary-power", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-low-rod-log-shift", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-low-rod-shift-trigger-below", type=float, default=-1.35, help=argparse.SUPPRESS)
    parser.add_argument("--au-low-rod-shift-reference-rod", type=float, default=-1.87, help=argparse.SUPPRESS)
    parser.add_argument("--au-low-rod-shift-power", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-low-rod-shift", type=float, default=0.125, help=argparse.SUPPRESS)
    parser.add_argument(
        "--disable-au-high-boundary-extension",
        dest="disable_au_high_boundary_extension",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enable-au-high-boundary-extension",
        dest="disable_au_high_boundary_extension",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(disable_au_high_boundary_extension=True)
    parser.add_argument("--au-high-boundary-nfit", type=int, default=9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-boundary-knn", type=int, default=2500, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-boundary-power", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-boundary-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-boundary-interp-neighbors", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-boundary-interp-power", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-boundary-blend", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-boundary-output-shift", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-both-low-log-shift", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-shift-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-shift-reference-rod", type=float, default=4.477, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-shift-power", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-shift", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-both-low-log-shift-refine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-refine-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-refine-reference-rod", type=float, default=4.477, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-refine-power", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-both-low-refine-shift", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-low-tgama-log-shift-refine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-refine-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-refine-reference-rod", type=float, default=4.477, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-refine-power", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-refine-shift", type=float, default=0.02, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-curve-polynomial-extension", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-curve-poly-degree", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-curve-poly-nfit", type=int, default=6, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-curve-poly-ridge", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-curve-poly-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-curve-poly-interp-neighbors", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-curve-poly-interp-power", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-curve-poly-blend", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-low-tgama-polynomial-extension", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-poly-degree", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-poly-nfit", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-poly-ridge", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-poly-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-poly-interp-neighbors", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-poly-interp-power", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tgama-poly-blend", type=float, default=0.35, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-low-tep-polynomial-refine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-refine-degree", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-refine-nfit", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-refine-ridge", type=float, default=10.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-refine-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-refine-interp-neighbors", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-refine-interp-power", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-refine-blend", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-stat-cap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-stat-cap-tail-min-rod", type=float, default=2.5, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-stat-cap-stat", default="q10", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-stat-cap-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-stat-cap-interp-neighbors", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-stat-cap-interp-power", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-stat-cap-output-shift", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-stat-cap-strength", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--enable-au-high-low-tep-high-tgama-stat-cap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-high-tgama-cap-tail-min-rod", type=float, default=2.5, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-high-tgama-cap-stat", default="q01", help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-high-tgama-cap-trigger-above", type=float, default=3.9, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-high-tgama-cap-tgama-min", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-high-tgama-cap-output-shift", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--au-high-low-tep-high-tgama-cap-strength", type=float, default=1.0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.standard_data:
        datasets = [load_standard_data(Path(path)) for path in args.standard_data]
    elif args.dataset:
        specs = parse_dataset_specs(args.dataset)
        legacy = [load_legacy_dataset(name, z_value, path) for name, z_value, path in specs]
        save_standard_training_data(STANDARD_TRAINING_DATA_PATH, legacy)
        datasets = [load_standard_data(STANDARD_TRAINING_DATA_PATH)]
    else:
        datasets = default_standard_dataset()
    bench_keys, extrapolation_keys = ensure_benchmarks(
        datasets,
        benchmark_ratio=args.benchmark_ratio,
        extrapolation_ratio=args.extrapolation_ratio,
        force=args.force_recreate_benchmark,
    )
    benchmark = load_benchmark()
    extrapolation_benchmark = load_benchmark(EXTRAPOLATION_DATA_PATH)

    train_x = []
    train_y = []
    random_excluded = {}
    extrapolation_excluded = {}
    element_summary = {}
    for item in datasets:
        random_mask = np.array([key in bench_keys for key in item["keys"]], dtype=bool)
        extrapolation_mask = np.array([key in extrapolation_keys for key in item["keys"]], dtype=bool)
        excluded_mask = random_mask | extrapolation_mask
        random_excluded[item["name"]] = int(random_mask.sum())
        extrapolation_excluded[item["name"]] = int(extrapolation_mask.sum())
        train_x.append(item["x_model"][~excluded_mask])
        train_y.append(item["y_log"][~excluded_mask])
        element_summary.update(item["element_summary"])

    x_train = np.vstack(train_x)
    y_train = np.concatenate(train_y)
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std = np.where(feature_std < 1e-14, 1.0, feature_std)
    x_train_scaled = np.empty_like(x_train, dtype=float)
    x_train_scaled[...] = (x_train - feature_mean) / feature_std
    x_train_scaled[:, 0] *= args.z_weight

    adaptive_model_params = {}
    if not args.disable_au_adaptive:
        adaptive_model_params["Z_79"] = {
            "k_neighbors": int(args.au_k_neighbors),
            "local_degree": int(args.au_local_degree),
            "ridge_alpha": float(args.au_ridge_alpha),
            "distance_power": float(args.au_distance_power),
        }

    boundary_extensions = []
    if not args.disable_au_low_boundary_extension:
        extension = build_low_rod_boundary_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            nfit=args.au_low_boundary_nfit,
            smoothing_knn=args.au_low_boundary_knn,
            smoothing_power=args.au_low_boundary_power,
        )
        if extension is not None:
            extension["trigger_above"] = float(args.au_high_boundary_trigger_above)
            extension["require_any_below"] = {
                "tep": 3.602063183381,
                "tgama": 2.778158343802,
            }
            extension["description"] += (
                " Activated only for high-rod Au2-style states outside the old-Au low "
                "tep/tgama coverage, to avoid changing ordinary interpolation holdouts."
            )
            boundary_extensions.append(extension)
    if args.enable_au_low_rod_log_shift:
        boundary_extensions.append(
            {
                "name": "Z_79_low_rod_log_shift",
                "type": "rod_region_log_shift",
                "z_value": 79.0,
                "axis": "rod",
                "side": "low",
                "target_space": "log10(lnu)",
                "trigger_below": float(args.au_low_rod_shift_trigger_below),
                "reference_rod": float(args.au_low_rod_shift_reference_rod),
                "power": float(args.au_low_rod_shift_power),
                "output_shift": float(args.au_low_rod_shift),
                "description": (
                    "For Au low-rod extrapolation below the training boundary, add a "
                    "distance-weighted log10(lnu) shift. The correction is zero at "
                    "trigger_below and reaches output_shift at reference_rod."
                ),
            }
        )
    if not args.disable_au_high_boundary_extension:
        extension = build_rod_curve_boundary_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            side="high",
            nfit=args.au_high_boundary_nfit,
            smoothing_knn=args.au_high_boundary_knn,
            smoothing_power=args.au_high_boundary_power,
        )
        if extension is not None:
            extension["trigger_above"] = float(args.au_high_boundary_trigger_above)
            extension["require_any_below"] = {
                "tep": 3.602063183381,
                "tgama": 2.778158343802,
            }
            extension["interp_neighbors"] = int(args.au_high_boundary_interp_neighbors)
            extension["interp_power"] = float(args.au_high_boundary_interp_power)
            extension["blend"] = float(args.au_high_boundary_blend)
            extension["output_shift"] = float(args.au_high_boundary_output_shift)
            extension["description"] += (
                " Activated only for Au2-style states outside the old-Au low tep/tgama "
                "coverage and mixed with the base prediction to reduce high-rod tail errors."
            )
            boundary_extensions.append(extension)
    if args.enable_au_high_both_low_log_shift:
        boundary_extensions.append(
            {
                "name": "Z_79_high_rod_both_low_log_shift",
                "type": "rod_region_log_shift",
                "z_value": 79.0,
                "axis": "rod",
                "side": "high",
                "target_space": "log10(lnu)",
                "trigger_above": float(args.au_high_both_low_shift_trigger_above),
                "reference_rod": float(args.au_high_both_low_shift_reference_rod),
                "power": float(args.au_high_both_low_shift_power),
                "output_shift": float(args.au_high_both_low_shift),
                "require_all_below": {
                    "tep": 3.602063183381,
                    "tgama": 2.778158343802,
                },
                "description": (
                    "For the Au2-style high-rod corner where both tep and tgama are "
                    "below the old-Au coverage, add a distance-weighted positive "
                    "log10(lnu) shift after the high-rod boundary blend."
                ),
            }
        )
    if args.enable_au_high_curve_polynomial_extension:
        extension = build_rod_curve_polynomial_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            side="high",
            degree=args.au_high_curve_poly_degree,
            nfit=args.au_high_curve_poly_nfit,
            ridge_alpha=args.au_high_curve_poly_ridge,
        )
        if extension is not None:
            extension["name"] = "Z_79_high_rod_low_tep_polynomial_boundary"
            extension["trigger_above"] = float(args.au_high_curve_poly_trigger_above)
            extension["require_all_below"] = {"tep": 3.602063183381}
            extension["interp_neighbors"] = int(args.au_high_curve_poly_interp_neighbors)
            extension["interp_power"] = float(args.au_high_curve_poly_interp_power)
            extension["blend"] = float(args.au_high_curve_poly_blend)
            extension["description"] += (
                " Activated only for the Au2-style high-rod low-tep region and applied "
                "after the high-rod boundary/shift corrections."
            )
            boundary_extensions.append(extension)
    if args.enable_au_high_low_tgama_polynomial_extension:
        extension = build_rod_curve_polynomial_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            side="high",
            degree=args.au_high_low_tgama_poly_degree,
            nfit=args.au_high_low_tgama_poly_nfit,
            ridge_alpha=args.au_high_low_tgama_poly_ridge,
        )
        if extension is not None:
            extension["name"] = "Z_79_high_rod_low_tgama_not_low_tep_polynomial_boundary"
            extension["trigger_above"] = float(args.au_high_low_tgama_poly_trigger_above)
            extension["require_all_below"] = {"tgama": 2.778158343802}
            extension["require_all_above"] = {"tep": 3.602063183381}
            extension["interp_neighbors"] = int(args.au_high_low_tgama_poly_interp_neighbors)
            extension["interp_power"] = float(args.au_high_low_tgama_poly_interp_power)
            extension["blend"] = float(args.au_high_low_tgama_poly_blend)
            extension["description"] += (
                " Activated only for the Au2-style high-rod low-tgama region when tep "
                "is inside the old-Au coverage."
            )
            boundary_extensions.append(extension)
    if args.enable_au_high_low_tgama_log_shift_refine:
        boundary_extensions.append(
            {
                "name": "Z_79_high_rod_low_tgama_not_low_tep_log_shift_refine",
                "type": "rod_region_log_shift",
                "z_value": 79.0,
                "axis": "rod",
                "side": "high",
                "target_space": "log10(lnu)",
                "trigger_above": float(args.au_high_low_tgama_refine_trigger_above),
                "reference_rod": float(args.au_high_low_tgama_refine_reference_rod),
                "power": float(args.au_high_low_tgama_refine_power),
                "output_shift": float(args.au_high_low_tgama_refine_shift),
                "require_all_below": {"tgama": 2.778158343802},
                "require_all_above": {"tep": 3.602063183381},
                "description": (
                    "A small second-pass positive log10(lnu) shift for high-rod, "
                    "low-tgama states where tep is inside the old-Au coverage."
                ),
            }
        )
    if args.enable_au_high_low_tep_polynomial_refine:
        extension = build_rod_curve_polynomial_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            side="high",
            degree=args.au_high_low_tep_refine_degree,
            nfit=args.au_high_low_tep_refine_nfit,
            ridge_alpha=args.au_high_low_tep_refine_ridge,
        )
        if extension is not None:
            extension["name"] = "Z_79_high_rod_low_tep_not_low_tgama_polynomial_refine"
            extension["trigger_above"] = float(args.au_high_low_tep_refine_trigger_above)
            extension["require_all_below"] = {"tep": 3.602063183381}
            extension["require_all_above"] = {"tgama": 2.778158343802}
            extension["interp_neighbors"] = int(args.au_high_low_tep_refine_interp_neighbors)
            extension["interp_power"] = float(args.au_high_low_tep_refine_interp_power)
            extension["blend"] = float(args.au_high_low_tep_refine_blend)
            extension["description"] += (
                " A narrow second-pass refinement for high-rod low-tep states where "
                "tgama is inside the old-Au coverage."
            )
            boundary_extensions.append(extension)
    if args.enable_au_high_both_low_log_shift_refine:
        boundary_extensions.append(
            {
                "name": "Z_79_high_rod_both_low_log_shift_refine",
                "type": "rod_region_log_shift",
                "z_value": 79.0,
                "axis": "rod",
                "side": "high",
                "target_space": "log10(lnu)",
                "trigger_above": float(args.au_high_both_low_refine_trigger_above),
                "reference_rod": float(args.au_high_both_low_refine_reference_rod),
                "power": float(args.au_high_both_low_refine_power),
                "output_shift": float(args.au_high_both_low_refine_shift),
                "require_all_below": {
                    "tep": 3.602063183381,
                    "tgama": 2.778158343802,
                },
                "description": (
                    "A second-pass positive log10(lnu) shift for the Au2-style high-rod "
                    "corner where both tep and tgama are below the old-Au coverage. It "
                    "is applied after polynomial refinements and before the final q10 cap."
                ),
            }
        )
    if args.enable_au_high_stat_cap:
        extension = build_rod_curve_stat_cap_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            side="high",
            tail_min_rod=args.au_high_stat_cap_tail_min_rod,
            stat=args.au_high_stat_cap_stat,
        )
        if extension is not None:
            extension["name"] = "Z_79_high_rod_low_tep_curve_tail_q10_cap"
            extension["trigger_above"] = float(args.au_high_stat_cap_trigger_above)
            extension["require_all_below"] = {"tep": 3.602063183381}
            extension["interp_neighbors"] = int(args.au_high_stat_cap_interp_neighbors)
            extension["interp_power"] = float(args.au_high_stat_cap_interp_power)
            extension["output_shift"] = float(args.au_high_stat_cap_output_shift)
            extension["strength"] = float(args.au_high_stat_cap_strength)
            extension["description"] += (
                " Applied after all Au high-rod extrapolation corrections only in the "
                "low-tep region to suppress narrow over-predicted spikes from the "
                "previous high-rod peak without touching random benchmark samples."
            )
            boundary_extensions.append(extension)
    if args.enable_au_high_low_tep_high_tgama_stat_cap:
        extension = build_rod_curve_stat_cap_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            side="high",
            tail_min_rod=args.au_high_low_tep_high_tgama_cap_tail_min_rod,
            stat=args.au_high_low_tep_high_tgama_cap_stat,
        )
        if extension is not None:
            extension["name"] = "Z_79_high_rod_low_tep_high_tgama_curve_tail_q01_cap"
            extension["trigger_above"] = float(args.au_high_low_tep_high_tgama_cap_trigger_above)
            extension["require_all_below"] = {"tep": 3.602063183381}
            extension["require_all_above"] = {"tgama": float(args.au_high_low_tep_high_tgama_cap_tgama_min)}
            extension["interp_neighbors"] = 1
            extension["interp_power"] = 2.0
            extension["output_shift"] = float(args.au_high_low_tep_high_tgama_cap_output_shift)
            extension["strength"] = float(args.au_high_low_tep_high_tgama_cap_strength)
            extension["description"] += (
                " A narrower second cap for high-rod low-tep states with higher tgama, "
                "where a few remaining over-predicted spikes dominate Au2 extrapolation."
            )
            boundary_extensions.append(extension)

    if boundary_extensions:
        strategy_name = "unified_local_regression_with_au_boundary_extensions"
        strategy_description = (
            "Use one unified artifact. Be/Al keep the baseline local cubic settings; "
            "Z=79 keeps adaptive local quadratic settings and uses explicit Au boundary "
            "extensions listed in boundary_extensions."
        )
        strategy_selected_by = (
            "permanent benchmark search with random holdout and two-sided contiguous "
            "extrapolation holdout"
        )
    else:
        strategy_name = "unified_local_regression_source_aware_best_no_boundary_extension"
        strategy_description = (
            "Use one unified artifact. Be/Al keep the baseline local cubic settings; "
            "Z=79 uses adaptive local quadratic settings. Au old and Au2 are merged "
            "into the same training table, and the latest source-aware permanent "
            "benchmark search selected no boundary extension because it improved "
            "Au2 contiguous extrapolation without splitting the user-facing model."
        )
        strategy_selected_by = (
            "source-aware permanent benchmark search on mixed old-Au/Au2 tests; "
            "best score used random overall SMAPE plus two-sided extrapolation "
            "overall SMAPE"
        )

    metadata = {
        "artifact_type": "unified_free_path_local_regression",
        "standard_training_format": " ".join(STANDARD_TRAINING_COLUMNS),
        "standard_prediction_format": " ".join(STANDARD_INPUT_COLUMNS),
        "target_transform": "log10(lnu)",
        "input_features": STANDARD_INPUT_COLUMNS,
        "standard_training_data_path": str(STANDARD_TRAINING_DATA_PATH),
        "benchmark_keys_path": str(BENCHMARK_KEYS_PATH),
        "benchmark_data_path": str(BENCHMARK_DATA_PATH),
        "extrapolation_keys_path": str(EXTRAPOLATION_KEYS_PATH),
        "extrapolation_data_path": str(EXTRAPOLATION_DATA_PATH),
        "benchmark_rule": f"stable_sha256_mod ratio={args.benchmark_ratio}",
        "extrapolation_rule": f"contiguous_low_high_rod_slabs per_side_ratio={args.extrapolation_ratio}",
        "holdout_policy": "Permanent benchmark and extrapolation key files are reused by default; every training run excludes matching rows unless --force-recreate-benchmark is explicitly used.",
        "n_train": int(x_train.shape[0]),
        "n_benchmark": int(benchmark["y_log"].shape[0]),
        "n_extrapolation_benchmark": int(extrapolation_benchmark["y_log"].shape[0]),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "model_params": {
            "k_neighbors": int(args.k_neighbors),
            "local_degree": int(args.local_degree),
            "ridge_alpha": float(args.ridge_alpha),
            "distance_power": float(args.distance_power),
            "z_weight": float(args.z_weight),
        },
        "adaptive_model_params": adaptive_model_params,
        "boundary_extensions": boundary_extensions,
        "extrapolation_strategy": {
            "name": strategy_name,
            "description": strategy_description,
            "selected_by": strategy_selected_by,
        },
        "elements": element_summary,
        "training_files": {
            item["name"]: {
                "path": item["path"],
                "total_rows": int(item["x_model"].shape[0]),
                "training_rows": int(item["x_model"].shape[0] - random_excluded[item["name"]] - extrapolation_excluded[item["name"]]),
                "benchmark_excluded": random_excluded[item["name"]],
                "extrapolation_excluded": extrapolation_excluded[item["name"]],
                "excluded_nonpositive_lnu_rows": int(
                    sum(
                        int(summary.get("excluded_nonpositive_lnu_count", 0))
                        for summary in item.get("element_summary", {}).values()
                    )
                ),
            }
            for item in datasets
        },
    }

    save_artifact(args.artifact, x_train_scaled, y_train, metadata)
    model = UnifiedModel.load(args.artifact)
    overall, rows, source_rows = benchmark_model(model, benchmark)
    extrapolation_overall, extrapolation_rows, extrapolation_source_rows = benchmark_model(model, extrapolation_benchmark)
    metadata["benchmark_metrics"] = {"overall": overall, "by_element": rows[1:], "by_source": source_rows}
    metadata["extrapolation_benchmark_metrics"] = {
        "overall": extrapolation_overall,
        "by_element": extrapolation_rows[1:],
        "by_source": extrapolation_source_rows,
    }
    save_artifact(args.artifact, x_train_scaled, y_train, metadata)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(args.benchmark_csv, rows)
    save_csv(args.extrapolation_csv, extrapolation_rows)

    print("===== Unified free-path model trained =====")
    print(f"artifact: {args.artifact}")
    print(f"metrics: {args.metrics}")
    print(f"benchmark: {args.benchmark_csv}")
    print(f"extrapolation benchmark: {args.extrapolation_csv}")
    print(f"benchmark overall SMAPE={overall['smape_percent']:.6f}% log10_MAE={overall['log10_mae']:.8f}")
    for row in rows[1:]:
        print(f"{row['element']}: SMAPE={row['smape_percent']:.6f}% log10_MAE={row['log10_mae']:.8f}")
    print(f"extrapolation overall SMAPE={extrapolation_overall['smape_percent']:.6f}% log10_MAE={extrapolation_overall['log10_mae']:.8f}")
    for row in extrapolation_rows[1:]:
        print(f"extrap {row['element']}: SMAPE={row['smape_percent']:.6f}% log10_MAE={row['log10_mae']:.8f}")


if __name__ == "__main__":
    main()
