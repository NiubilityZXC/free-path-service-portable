#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Specialized high-Z free-path model wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unified_free_path_core import OUTPUT_DIR


HIGH_Z_MODEL_PATH = OUTPUT_DIR / "high_z_au_catboost_model.cbm"
HIGH_Z_METADATA_PATH = OUTPUT_DIR / "high_z_au_catboost_metrics.json"


@dataclass
class HighZModel:
    model_path: Path
    metadata_path: Path
    metadata: dict
    model: object
    boundary_model: object | None = None

    @classmethod
    def load(cls, model_path: Path = HIGH_Z_MODEL_PATH, metadata_path: Path = HIGH_Z_METADATA_PATH) -> "HighZModel":
        from catboost import CatBoostRegressor

        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        def resolve_path(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else metadata_path.parent / path

        if metadata.get("model_specs"):
            import joblib

            models = {}
            for spec in metadata["model_specs"]:
                path = resolve_path(spec["path"])
                if spec["type"] == "catboost":
                    item = CatBoostRegressor()
                    item.load_model(str(path))
                elif spec["type"] == "joblib":
                    item = joblib.load(path)
                else:
                    raise ValueError(f"unsupported high-Z model type: {spec['type']}")
                models[spec["name"]] = item
            model = models
        else:
            model_paths = metadata.get("model_paths") or [str(model_path)]
            models = []
            for item in model_paths:
                path = resolve_path(item)
                model = CatBoostRegressor()
                model.load_model(str(path))
                models.append(model)
            model = models if len(models) > 1 else models[0]

        boundary_model = None
        if metadata.get("boundary_model_path"):
            from unified_free_path_core import UnifiedModel

            boundary_path = resolve_path(str(metadata["boundary_model_path"]))
            if boundary_path.exists():
                boundary_model = UnifiedModel.load(boundary_path)
        return cls(
            model_path=model_path,
            metadata_path=metadata_path,
            metadata=metadata,
            model=model,
            boundary_model=boundary_model,
        )

    @classmethod
    def load_if_available(cls) -> "HighZModel | None":
        if not HIGH_Z_METADATA_PATH.exists():
            return None
        try:
            metadata = json.loads(HIGH_Z_METADATA_PATH.read_text(encoding="utf-8"))
            # A disabled route is metadata-only. Do not import CatBoost/joblib
            # or load the multi-gigabyte historical ensemble.
            if metadata.get("route_enabled") is False:
                return None
            if not HIGH_Z_MODEL_PATH.exists() and not metadata.get("model_specs") and not metadata.get("model_paths"):
                return None
            return cls.load()
        except Exception:
            return None

    @staticmethod
    def _data_source_values(data_source: str | list | np.ndarray | None, n_rows: int) -> list[str]:
        if data_source is None:
            return ["auto"] * n_rows
        if isinstance(data_source, str):
            return [data_source.strip().lower() or "auto"] * n_rows
        values = list(np.asarray(data_source, dtype=object).reshape(-1))
        if len(values) == 1:
            return [str(values[0]).strip().lower() or "auto"] * n_rows
        if len(values) != n_rows:
            raise ValueError("data_source length must match x_model rows")
        return [str(value).strip().lower() or "auto" for value in values]

    @staticmethod
    def _range_mask(x: np.ndarray, ranges: dict) -> np.ndarray:
        mask = np.ones(x.shape[0], dtype=bool)
        for offset, name in enumerate(("rod", "tep", "tgama"), start=1):
            if name not in ranges:
                continue
            low, high = [float(v) for v in ranges[name]]
            mask &= (x[:, offset] >= low) & (x[:, offset] <= high)
        return mask

    def route_mask(self, x_model: np.ndarray, data_source: str | list | np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x_model, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.metadata.get("route_enabled") is False:
            return np.zeros(x.shape[0], dtype=bool)
        z_values = {float(value) for value in self.metadata.get("z_values", [79.0])}
        mask = np.array([float(row[0]) in z_values for row in x], dtype=bool)
        policy = str(self.metadata.get("route_policy", "")).strip().lower()
        if policy == "data_source_aware_old_au":
            data_sources = self._data_source_values(data_source, x.shape[0])
            allowed = {
                str(value).strip().lower()
                for value in self.metadata.get("direct_data_sources", ["auto", "old_au"])
            }
            mask &= np.array([source in allowed for source in data_sources], dtype=bool)
            ranges = self.metadata.get("old_au_midrod_ranges") or self.metadata.get("train_ranges", {})
            mask &= self._range_mask(x, ranges)
            return mask
        ranges = self.metadata.get("train_ranges", {})
        mask &= self._range_mask(x, ranges)
        return mask

    def boundary_mask(self, x_model: np.ndarray, data_source: str | list | np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x_model, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.boundary_model is None or self.metadata.get("route_enabled") is False:
            return np.zeros(x.shape[0], dtype=bool)
        policy = str(self.metadata.get("route_policy", "")).strip().lower()
        if policy != "data_source_aware_old_au":
            return np.zeros(x.shape[0], dtype=bool)
        z_values = {float(value) for value in self.metadata.get("z_values", [79.0])}
        mask = np.array([float(row[0]) in z_values for row in x], dtype=bool)
        data_sources = self._data_source_values(data_source, x.shape[0])
        allowed = {
            str(value).strip().lower()
            for value in self.metadata.get("boundary_data_sources", ["auto", "old_au"])
        }
        mask &= np.array([source in allowed for source in data_sources], dtype=bool)
        full_ranges = self.metadata.get("old_au_full_ranges") or {}
        if full_ranges:
            mask &= self._range_mask(x, full_ranges)
        direct_ranges = self.metadata.get("old_au_midrod_ranges") or self.metadata.get("train_ranges", {})
        if direct_ranges:
            mask &= ~self._range_mask(x, direct_ranges)
        return mask

    def _predict_direct(self, features: np.ndarray) -> np.ndarray:
        if isinstance(self.model, dict):
            predictions = {
                name: np.asarray(model.predict(features), dtype=float).reshape(-1)
                for name, model in self.model.items()
            }
            ensemble = self.metadata.get("ensemble", {})
            if ensemble.get("mode") == "weighted_components":
                cat_sources = ensemble.get("cat_median_sources") or [
                    spec["name"]
                    for spec in self.metadata.get("model_specs", [])
                    if spec.get("type") == "catboost"
                ]
                if cat_sources:
                    cat_stack = np.vstack([predictions[name] for name in cat_sources])
                    predictions["cat_median"] = np.median(cat_stack, axis=0)
                    predictions["cat_mean"] = np.mean(cat_stack, axis=0)
                out = np.zeros(features.shape[0], dtype=float)
                for component in ensemble.get("components", []):
                    out += float(component["weight"]) * predictions[component["source"]]
                return out
            stack = np.vstack(list(predictions.values()))
            return np.mean(stack, axis=0)
        if isinstance(self.model, list):
            stack = np.vstack([np.asarray(model.predict(features), dtype=float).reshape(-1) for model in self.model])
            ensemble = self.metadata.get("ensemble", {})
            mode = ensemble.get("mode", "mean")
            if mode == "blend_best_median":
                best_index = int(ensemble.get("best_model_index", 0))
                median_indices = [int(i) for i in ensemble.get("median_model_indices", list(range(stack.shape[0])))]
                best_weight = float(ensemble.get("best_weight", 0.0))
                median_pred = np.median(stack[median_indices], axis=0)
                return best_weight * stack[best_index] + (1.0 - best_weight) * median_pred
            if mode == "median":
                return np.median(stack, axis=0)
            return np.mean(stack, axis=0)
        return np.asarray(self.model.predict(features), dtype=float).reshape(-1)

    def predict_log(
        self,
        x_model: np.ndarray,
        fallback_log: np.ndarray | None = None,
        data_source: str | list | np.ndarray | None = None,
        return_details: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
        x = np.asarray(x_model, dtype=float)
        one = x.ndim == 1
        if one:
            x = x.reshape(1, -1)
        mask = self.route_mask(x, data_source=data_source)
        boundary_mask = self.boundary_mask(x, data_source=data_source)
        if fallback_log is None:
            pred = np.full(x.shape[0], np.nan, dtype=float)
        else:
            pred = np.asarray(fallback_log, dtype=float).reshape(-1).copy()
        if np.any(boundary_mask):
            pred[boundary_mask] = np.asarray(self.boundary_model.predict_log(x[boundary_mask]), dtype=float).reshape(-1)
        if np.any(mask):
            direct = self._predict_direct(x[mask, 1:4])
            alpha = float(self.metadata.get("alpha_base", 0.0))
            if fallback_log is not None and alpha > 0.0:
                pred[mask] = alpha * np.asarray(fallback_log, dtype=float).reshape(-1)[mask] + (1.0 - alpha) * direct
            else:
                pred[mask] = direct
        combined_mask = mask | boundary_mask
        details = {
            "direct_rows": int(np.sum(mask)),
            "boundary_rows": int(np.sum(boundary_mask)),
            "combined_rows": int(np.sum(combined_mask)),
            "data_source": data_source if data_source is not None else "auto",
            "route_policy": self.metadata.get("route_policy"),
            "boundary_model_available": self.boundary_model is not None,
        }
        pred_out = pred[0] if one else pred
        mask_out = combined_mask[0] if one else combined_mask
        if return_details:
            return pred_out, mask_out, details
        return pred_out, mask_out
