#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities for the unified Rosseland free-path model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "free_path_model_outputs"
ACTIVE_UNIFIED_ARTIFACT = OUTPUT_DIR / "unified_free_path_model.npz"
ACTIVE_UNIFIED_METRICS = OUTPUT_DIR / "unified_free_path_metrics.json"
ACTIVE_UNIFIED_BENCHMARK_CSV = OUTPUT_DIR / "unified_free_path_benchmark_metrics.csv"
ACTIVE_UNIFIED_EXTRAPOLATION_CSV = OUTPUT_DIR / "unified_free_path_extrapolation_metrics.csv"
STAGED_UNIFIED_ARTIFACT = OUTPUT_DIR / "staged_unified_free_path_model.npz"
STAGED_UNIFIED_METRICS = OUTPUT_DIR / "staged_unified_free_path_metrics.json"
BENCHMARK_KEYS_PATH = OUTPUT_DIR / "unified_permanent_benchmark_keys.txt"
BENCHMARK_DATA_PATH = OUTPUT_DIR / "unified_permanent_benchmark_data.npz"
EXTRAPOLATION_KEYS_PATH = OUTPUT_DIR / "unified_permanent_extrapolation_keys.txt"
EXTRAPOLATION_DATA_PATH = OUTPUT_DIR / "unified_permanent_extrapolation_data.npz"
STANDARD_TRAINING_DATA_PATH = OUTPUT_DIR / "unified_standard_training_data.txt"
STANDARD_INPUT_COLUMNS = ["Z", "rod", "tep", "tgama"]
STANDARD_TRAINING_COLUMNS = ["Z", "rod", "tep", "tgama", "lnu"]


DEFAULT_DATASETS = [
    ("data_Be", 4.0, ROOT / "data_Be.txt"),
    ("data_Al", 13.0, ROOT / "data_Al.txt"),
    ("data", 79.0, ROOT / "data.txt"),
]


def metric_dict(log_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(log_true) & np.isfinite(log_pred)
    log_true = log_true[valid]
    log_pred = log_pred[valid]
    true = np.power(10.0, log_true)
    pred = np.power(10.0, log_pred)
    log_error = log_pred - log_true
    factor_error = np.power(10.0, np.abs(log_error))
    smape = 2.0 * np.abs(pred - true) / np.maximum(np.abs(pred) + np.abs(true), 1e-300)
    mape = np.abs(pred - true) / np.maximum(np.abs(true), 1e-300)
    return {
        "n": int(log_true.size),
        "log10_mae": float(np.mean(np.abs(log_error))),
        "log10_rmse": float(np.sqrt(np.mean(log_error**2))),
        "mape_percent": float(np.mean(mape) * 100.0),
        "smape_percent": float(np.mean(smape) * 100.0),
        "median_factor_error": float(np.median(factor_error)),
        "p90_factor_error": float(np.percentile(factor_error, 90.0)),
        "p99_factor_error": float(np.percentile(factor_error, 99.0)),
        "max_factor_error": float(np.max(factor_error)),
    }


def infer_coordinate_transform(raw_inputs: np.ndarray) -> str:
    if np.all(raw_inputs > 0.0):
        ratios = raw_inputs.max(axis=0) / raw_inputs.min(axis=0)
        if np.all(ratios > 20.0):
            return "log10"
    return "identity"


def apply_coordinate_transform(raw_inputs: np.ndarray, transform: str) -> np.ndarray:
    arr = np.asarray(raw_inputs, dtype=float)
    if transform == "identity":
        return arr.copy()
    if transform == "log10":
        if np.any(arr <= 0.0):
            raise ValueError("log10 coordinate transform requires positive inputs")
        return np.log10(arr)
    raise ValueError(f"unknown coordinate transform: {transform}")


def repair_nonpositive_grid(values: np.ndarray) -> tuple[np.ndarray, int]:
    repaired = values.astype(float, copy=True)
    bad = (~np.isfinite(repaired)) | (repaired <= 0.0)
    bad_count = int(np.sum(bad))
    if bad_count == 0:
        return repaired, 0
    shape = repaired.shape
    for radius in range(1, 8):
        for index in np.argwhere(bad):
            window = tuple(
                slice(max(0, int(i) - radius), min(shape[d], int(i) + radius + 1))
                for d, i in enumerate(index)
            )
            vals = repaired[window]
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size:
                repaired[tuple(index)] = float(np.median(vals))
        bad = (~np.isfinite(repaired)) | (repaired <= 0.0)
        if not np.any(bad):
            break
    if np.any(bad):
        vals = repaired[np.isfinite(repaired) & (repaired > 0.0)]
        repaired[bad] = float(np.median(vals))
    return repaired, bad_count


def row_key(dataset_name: str, raw_xyz: np.ndarray) -> str:
    return dataset_name + "|" + "|".join(f"{float(v):.12g}" for v in raw_xyz)


def standard_row_key(z_value: float, xyz: np.ndarray) -> str:
    return f"{float(z_value):.12g}|" + "|".join(f"{float(v):.12g}" for v in xyz)


def stable_benchmark_mask(dataset_name: str, raw_xyz: np.ndarray, ratio: float) -> np.ndarray:
    threshold = int(ratio * 10000)
    out = np.zeros(raw_xyz.shape[0], dtype=bool)
    for i, row in enumerate(raw_xyz):
        digest = hashlib.sha256(row_key(dataset_name, row).encode("utf-8")).hexdigest()
        out[i] = (int(digest[:16], 16) % 10000) < threshold
    return out


def stable_mask_from_keys(keys: np.ndarray, ratio: float) -> np.ndarray:
    threshold = int(ratio * 10000)
    out = np.zeros(len(keys), dtype=bool)
    for i, key in enumerate(keys):
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        out[i] = (int(digest[:16], 16) % 10000) < threshold
    return out


def design_matrix(delta: np.ndarray, degree: int) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    n, d = delta.shape
    cols = [np.ones(n)]
    cols.extend(delta[:, i] for i in range(d))
    if degree >= 2:
        cols.extend(delta[:, i] * delta[:, i] for i in range(d))
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(delta[:, i] * delta[:, j])
    if degree >= 3:
        cols.extend(delta[:, i] * delta[:, i] * delta[:, i] for i in range(d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    cols.append(delta[:, i] * delta[:, i] * delta[:, j])
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    cols.append(delta[:, i] * delta[:, j] * delta[:, k])
    return np.column_stack(cols)


@dataclass
class UnifiedModel:
    artifact_path: Path
    metadata: dict
    x_train_scaled: np.ndarray
    y_train_log: np.ndarray
    tree: cKDTree
    boundary_extensions: list[dict]

    @classmethod
    def load(cls, artifact_path: Path) -> "UnifiedModel":
        payload = np.load(artifact_path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata_json"]))
        x_train_scaled = payload["x_train_scaled"]
        y_train_log = payload["y_train_log"]
        tree = cKDTree(x_train_scaled)
        boundary_extensions = cls._prepare_boundary_extensions(metadata)
        return cls(artifact_path, metadata, x_train_scaled, y_train_log, tree, boundary_extensions)

    @staticmethod
    def _prepare_boundary_extensions(metadata: dict) -> list[dict]:
        prepared = []
        for spec in metadata.get("boundary_extensions", []):
            spec_type = spec.get("type")
            if spec_type not in {
                "low_rod_linear_boundary",
                "rod_curve_linear_boundary",
                "rod_curve_polynomial_boundary",
                "rod_curve_stat_cap",
                "rod_region_log_shift",
            }:
                continue
            if spec_type == "rod_region_log_shift":
                prepared.append(dict(spec))
                continue
            rows = np.asarray(spec.get("rows", []), dtype=float)
            if spec_type == "rod_curve_polynomial_boundary":
                if rows.ndim != 2 or rows.shape[1] < 5 or rows.shape[0] == 0:
                    continue
                item = dict(spec)
                item["rows_array"] = rows
                item["tree"] = cKDTree(rows[:, :2])
                prepared.append(item)
                continue
            if spec_type == "rod_curve_stat_cap":
                if rows.ndim != 2 or rows.shape[1] != 3 or rows.shape[0] == 0:
                    continue
                item = dict(spec)
                item["rows_array"] = rows
                item["tree"] = cKDTree(rows[:, :2])
                prepared.append(item)
                continue
            if rows.ndim != 2 or rows.shape[1] != 5 or rows.shape[0] == 0:
                continue
            item = dict(spec)
            item["rows_array"] = rows
            item["tree"] = cKDTree(rows[:, :2])
            prepared.append(item)
        return prepared

    def standard_point(self, z_value: float, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,):
            raise ValueError("standard prediction input must be: Z rod tep tgama")
        return np.array([float(z_value), xyz[0], xyz[1], xyz[2]], dtype=float)

    def scale_features(self, x: np.ndarray) -> np.ndarray:
        mu = np.asarray(self.metadata["feature_mean"], dtype=float)
        sigma = np.asarray(self.metadata["feature_std"], dtype=float)
        z_weight = float(self.metadata["model_params"].get("z_weight", self.metadata["model_params"].get("source_weight", 1.0)))
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x, dtype=float)
        out[...] = (x - mu) / sigma
        out[..., 0] *= z_weight
        return out

    def params_for_point(self, x_model: np.ndarray) -> dict:
        params = dict(self.metadata["model_params"])
        adaptive = self.metadata.get("adaptive_model_params") or {}
        z_value = float(np.asarray(x_model, dtype=float)[0])
        rounded = int(round(z_value))
        keys = [f"Z_{z_value:g}", f"Z_{rounded:g}", f"{z_value:g}", f"{rounded:g}"]
        for key in keys:
            if key in adaptive:
                params.update(adaptive[key])
                break
        return params

    def _predict_one_scaled(self, x_scaled: np.ndarray, params: dict) -> float:
        return float(self._predict_scaled_batch(np.asarray(x_scaled, dtype=float).reshape(1, -1), params)[0])

    def _predict_scaled_batch(self, x_scaled: np.ndarray, params: dict) -> np.ndarray:
        x_scaled = np.asarray(x_scaled, dtype=float)
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)
        k = int(params["k_neighbors"])
        degree = int(params["local_degree"])
        alpha = float(params["ridge_alpha"])
        power = float(params["distance_power"])
        distances, indices = self.tree.query(x_scaled, k=k, workers=-1)
        if k == 1:
            return np.asarray(self.y_train_log[indices], dtype=float).reshape(-1)
        pred = np.empty(x_scaled.shape[0], dtype=float)
        for i in range(x_scaled.shape[0]):
            x_neighbors = self.x_train_scaled[indices[i]]
            y_neighbors = self.y_train_log[indices[i]]
            delta = x_neighbors - x_scaled[i]
            phi = design_matrix(delta, degree)
            dist = np.maximum(distances[i], 1e-10)
            weights = 1.0 / np.power(dist, power)
            sw = np.sqrt(weights)
            a = phi * sw[:, None]
            b = y_neighbors * sw
            reg = np.eye(a.shape[1]) * alpha
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(a.T @ a + reg, a.T @ b)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(a, b, rcond=None)[0]
            pred[i] = beta[0]
        return pred

    @staticmethod
    def _params_group_key(params: dict) -> tuple:
        return (
            int(params["k_neighbors"]),
            int(params["local_degree"]),
            float(params["ridge_alpha"]),
            float(params["distance_power"]),
        )

    def _apply_boundary_extensions(self, x_model: np.ndarray, pred: np.ndarray) -> np.ndarray:
        if not self.boundary_extensions:
            return pred
        out = np.asarray(pred, dtype=float).copy()
        x_model = np.asarray(x_model, dtype=float)
        for spec in self.boundary_extensions:
            z_value = float(spec["z_value"])
            z_match = np.isclose(x_model[:, 0], z_value, rtol=0.0, atol=1e-8)
            if spec.get("type") == "rod_region_log_shift":
                side = str(spec.get("side", "low")).lower()
                if side == "low":
                    trigger = float(spec["trigger_below"])
                    active = z_match & (x_model[:, 1] < trigger)
                    distance = trigger - x_model[active, 1]
                    reference = float(spec.get("reference_rod", trigger - 1.0))
                    width = max(trigger - reference, 1e-12)
                elif side == "high":
                    trigger = float(spec["trigger_above"])
                    active = z_match & (x_model[:, 1] > trigger)
                    distance = x_model[active, 1] - trigger
                    reference = float(spec.get("reference_rod", trigger + 1.0))
                    width = max(reference - trigger, 1e-12)
                else:
                    continue
                if not np.any(active):
                    continue
                require_any_below = spec.get("require_any_below") or {}
                if require_any_below:
                    offsets = {"rod": 1, "tep": 2, "tgama": 3}
                    condition = np.zeros(int(active.sum()), dtype=bool)
                    active_rows = x_model[active]
                    for name, limit in require_any_below.items():
                        condition |= active_rows[:, offsets[name]] < float(limit)
                    active_indices = np.flatnonzero(active)
                    active[active_indices[~condition]] = False
                    distance = distance[condition]
                require_all_below = spec.get("require_all_below") or {}
                if require_all_below:
                    offsets = {"rod": 1, "tep": 2, "tgama": 3}
                    condition = np.ones(int(active.sum()), dtype=bool)
                    active_rows = x_model[active]
                    for name, limit in require_all_below.items():
                        condition &= active_rows[:, offsets[name]] < float(limit)
                    active_indices = np.flatnonzero(active)
                    active[active_indices[~condition]] = False
                    distance = distance[condition]
                if not np.any(active):
                    continue
                power = float(spec.get("power", 1.0))
                factor = np.clip(distance / width, 0.0, 1.0)
                if power != 1.0:
                    factor = np.power(factor, power)
                out[active] += float(spec.get("output_shift", 0.0)) * factor
                continue
            rows = spec["rows_array"]
            tree = spec["tree"]
            k = min(int(spec.get("interp_neighbors", 8)), rows.shape[0])
            power = float(spec.get("interp_power", 2.0))
            if spec.get("type") == "low_rod_linear_boundary":
                trigger_below = float(spec["trigger_below"])
                active = z_match & (x_model[:, 1] < trigger_below)
                if not np.any(active):
                    continue
                candidate_indices = np.flatnonzero(active)
            else:
                candidate_indices = np.flatnonzero(z_match)
                if candidate_indices.size == 0:
                    continue
            side = str(spec.get("side", "low")).lower()
            for i in candidate_indices:
                if spec.get("type") in {
                    "rod_curve_linear_boundary",
                    "rod_curve_polynomial_boundary",
                    "rod_curve_stat_cap",
                }:
                    trigger_above = spec.get("trigger_above")
                    if side == "high" and trigger_above is not None and x_model[i, 1] <= float(trigger_above):
                        continue
                    trigger_below = spec.get("trigger_below")
                    if side == "low" and trigger_below is not None and x_model[i, 1] >= float(trigger_below):
                        continue
                    require_any_below = spec.get("require_any_below") or {}
                    if require_any_below:
                        offsets = {"rod": 1, "tep": 2, "tgama": 3}
                        if not any(x_model[i, offsets[name]] < float(limit) for name, limit in require_any_below.items()):
                            continue
                    require_all_below = spec.get("require_all_below") or {}
                    if require_all_below:
                        offsets = {"rod": 1, "tep": 2, "tgama": 3}
                        if not all(x_model[i, offsets[name]] < float(limit) for name, limit in require_all_below.items()):
                            continue
                    require_all_above = spec.get("require_all_above") or {}
                    if require_all_above:
                        offsets = {"rod": 1, "tep": 2, "tgama": 3}
                        if not all(x_model[i, offsets[name]] >= float(limit) for name, limit in require_all_above.items()):
                            continue
                tep_tgama = x_model[i, 2:4]
                distances, indices = tree.query(tep_tgama, k=k)
                distances = np.atleast_1d(distances).astype(float)
                indices = np.atleast_1d(indices).astype(int)
                selected = rows[indices]
                if spec.get("type") == "rod_curve_stat_cap":
                    if distances[0] < 1e-10:
                        cap = float(selected[0, 2])
                    else:
                        weights = 1.0 / np.power(np.maximum(distances, 1e-10), power)
                        weights /= weights.sum()
                        cap = float(np.sum(weights * selected[:, 2]))
                    cap += float(spec.get("output_shift", 0.0))
                    if out[i] > cap:
                        strength = float(spec.get("strength", 1.0))
                        strength = min(1.0, max(0.0, strength))
                        out[i] = float((1.0 - strength) * out[i] + strength * cap)
                    continue
                if spec.get("type") == "rod_curve_polynomial_boundary":
                    if distances[0] < 1e-10:
                        curves = selected[:1]
                        weights = np.ones(1, dtype=float)
                    else:
                        weights = 1.0 / np.power(np.maximum(distances, 1e-10), power)
                        weights /= weights.sum()
                        curves = selected
                    curve_pred = []
                    for row in curves:
                        x0 = float(row[2])
                        if side == "low" and x_model[i, 1] >= x0:
                            continue
                        if side == "high" and x_model[i, 1] <= x0:
                            continue
                        delta = float(x_model[i, 1] - x0)
                        betas = row[3:]
                        curve_pred.append(float(sum(float(beta) * (delta**degree) for degree, beta in enumerate(betas))))
                    if not curve_pred:
                        continue
                    if len(curve_pred) != len(weights):
                        weights = weights[: len(curve_pred)]
                        weights /= weights.sum()
                    candidate = float(np.sum(weights * np.asarray(curve_pred, dtype=float)) + float(spec.get("output_shift", 0.0)))
                    blend = float(spec.get("blend", 1.0))
                    blend = min(1.0, max(0.0, blend))
                    out[i] = float((1.0 - blend) * out[i] + blend * candidate)
                    continue
                if distances[0] < 1e-10:
                    x0, intercept, slope = selected[0, 2:5]
                else:
                    weights = 1.0 / np.power(np.maximum(distances, 1e-10), power)
                    weights /= weights.sum()
                    x0 = float(np.sum(weights * selected[:, 2]))
                    intercept = float(np.sum(weights * selected[:, 3]))
                    slope = float(np.sum(weights * selected[:, 4]))
                if spec.get("type") == "rod_curve_linear_boundary":
                    if side == "low" and x_model[i, 1] >= x0:
                        continue
                    if side == "high" and x_model[i, 1] <= x0:
                        continue
                candidate = float(intercept + slope * (x_model[i, 1] - x0) + float(spec.get("output_shift", 0.0)))
                blend = float(spec.get("blend", 1.0))
                blend = min(1.0, max(0.0, blend))
                out[i] = float((1.0 - blend) * out[i] + blend * candidate)
        return out

    def predict_log(self, x_model: np.ndarray) -> np.ndarray:
        x_model = np.asarray(x_model, dtype=float)
        one = x_model.ndim == 1
        if one:
            x_model = x_model.reshape(1, -1)
        x_scaled = self.scale_features(x_model)
        if self.metadata.get("adaptive_model_params"):
            pred = np.empty(x_scaled.shape[0], dtype=float)
            groups: dict[tuple, list[int]] = {}
            group_params = {}
            for i in range(x_scaled.shape[0]):
                params = self.params_for_point(x_model[i])
                key = self._params_group_key(params)
                groups.setdefault(key, []).append(i)
                group_params[key] = params
            for key, rows in groups.items():
                row_indices = np.asarray(rows, dtype=int)
                pred[row_indices] = self._predict_scaled_batch(x_scaled[row_indices], group_params[key])
            pred = self._apply_boundary_extensions(x_model, pred)
            return pred[0] if one else pred
        params = self.metadata["model_params"]
        pred = self._predict_scaled_batch(x_scaled, params)
        pred = self._apply_boundary_extensions(x_model, pred)
        return pred[0] if one else pred

    def predict(self, z_value: float, xyz: np.ndarray) -> dict:
        x_model = self.standard_point(z_value, xyz)
        log_lnu = float(self.predict_log(x_model))
        lnu = float(10.0 ** log_lnu)
        return {
            "artifact": str(self.artifact_path),
            "Z": float(z_value),
            "mode": "unified_local_regression",
            "input": {
                "Z": float(z_value),
                "rod": float(xyz[0]),
                "tep": float(xyz[1]),
                "tgama": float(xyz[2]),
            },
            "model_coordinates": {
                "Z": float(x_model[0]),
                "rod": float(x_model[1]),
                "tep": float(x_model[2]),
                "tgama": float(x_model[3]),
            },
            "lnu": lnu,
            "log10_lnu": log_lnu,
        }
