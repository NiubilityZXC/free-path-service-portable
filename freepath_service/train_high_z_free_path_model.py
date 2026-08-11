#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a specialized high-Z model for Au/Z=79."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from high_z_free_path_model import HIGH_Z_METADATA_PATH, HIGH_Z_MODEL_PATH, HighZModel
from train_unified_free_path_model import load_benchmark, load_standard_data
from unified_free_path_core import (
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    STANDARD_TRAINING_DATA_PATH,
    STAGED_UNIFIED_ARTIFACT,
    UnifiedModel,
    metric_dict,
)


DEFAULT_ENSEMBLE_MEMBERS = [
    {
        "name": "d11_lr016_it6500_l205_seed20260629",
        "iterations": 6500,
        "learning_rate": 0.016,
        "depth": 11,
        "l2_leaf_reg": 0.5,
        "random_seed": 20260629,
    },
    {
        "name": "d11_lr016_it6500_l205_seed20260638",
        "iterations": 6500,
        "learning_rate": 0.016,
        "depth": 11,
        "l2_leaf_reg": 0.5,
        "random_seed": 20260638,
    },
    {
        "name": "d11_lr012_it8500_l21_seed20260637",
        "iterations": 8500,
        "learning_rate": 0.012,
        "depth": 11,
        "l2_leaf_reg": 1.0,
        "random_seed": 20260637,
    },
    {
        "name": "d11_lr012_it8500_l21_seed20260629",
        "iterations": 8500,
        "learning_rate": 0.012,
        "depth": 11,
        "l2_leaf_reg": 1.0,
        "random_seed": 20260629,
    },
]


DEFAULT_AUXILIARY_MEMBERS = [
    {
        "name": "extra_leaf2_500",
        "type": "extra_trees",
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "random_seed": 20260630,
    },
    {
        "name": "extra_leaf3_700",
        "type": "extra_trees",
        "n_estimators": 700,
        "min_samples_leaf": 3,
        "max_features": 1.0,
        "random_seed": 20260631,
    },
    {
        "name": "hgb_025_leaf255_2200",
        "type": "hist_gradient_boosting",
        "max_iter": 2200,
        "learning_rate": 0.025,
        "max_leaf_nodes": 255,
        "l2_regularization": 0.0,
        "random_seed": 20260630,
    },
]


DEFAULT_MIXED_COMPONENTS = [
    {"source": "d11_lr016_it6500_l205_seed20260629", "weight": 0.2174097563545284},
    {"source": "d11_lr016_it6500_l205_seed20260638", "weight": 0.16526096510973204},
    {"source": "d11_lr012_it8500_l21_seed20260637", "weight": 0.10592382331619384},
    {"source": "cat_median", "weight": 0.19068866723347133},
    {"source": "extra_leaf2_500", "weight": 0.18801220873151475},
    {"source": "extra_leaf3_700", "weight": 0.074653589509278},
    {"source": "hgb_025_leaf255_2200", "weight": 0.05805098974528155},
]


def element_metrics(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "overall": metric_dict(y_true, pred),
        "by_element": [
            {"element": label, **metric_dict(y_true[labels == label], pred[labels == label])}
            for label in sorted(set(labels.tolist()))
        ],
    }


def ensemble_member_path(model_path: Path, index: int) -> Path:
    if index == 0:
        return model_path
    return model_path.with_name(f"{model_path.stem}_member{index + 1:02d}{model_path.suffix}")


def auxiliary_member_path(model_path: Path, name: str) -> Path:
    return model_path.with_name(f"{model_path.stem}_{name}.joblib")


def predict_direct(
    cat_models: dict[str, CatBoostRegressor],
    aux_models: dict[str, object],
    x_features: np.ndarray,
    best_weight: float,
    mixed_components: list[dict] | None = None,
) -> np.ndarray:
    if mixed_components:
        predictions = {
            name: np.asarray(model.predict(x_features), dtype=float).reshape(-1)
            for name, model in cat_models.items()
        }
        predictions.update(
            {
                name: np.asarray(model.predict(x_features), dtype=float).reshape(-1)
                for name, model in aux_models.items()
            }
        )
        cat_stack = np.vstack([predictions[name] for name in cat_models])
        predictions["cat_median"] = np.median(cat_stack, axis=0)
        predictions["cat_mean"] = np.mean(cat_stack, axis=0)
        out = np.zeros(x_features.shape[0], dtype=float)
        for component in mixed_components:
            out += float(component["weight"]) * predictions[component["source"]]
        return out
    models = list(cat_models.values())
    if len(models) == 1:
        return np.asarray(models[0].predict(x_features), dtype=float).reshape(-1)
    stack = np.vstack([np.asarray(model.predict(x_features), dtype=float).reshape(-1) for model in models])
    return best_weight * stack[0] + (1.0 - best_weight) * np.median(stack, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a specialized high-Z free-path model.")
    parser.add_argument("--z", type=float, default=79.0)
    parser.add_argument("--model-path", type=Path, default=HIGH_Z_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=HIGH_Z_METADATA_PATH)
    parser.add_argument("--base-model", type=Path, default=STAGED_UNIFIED_ARTIFACT)
    parser.add_argument("--standard-data", type=Path, default=STANDARD_TRAINING_DATA_PATH)
    parser.add_argument("--iterations", type=int, default=6500)
    parser.add_argument("--learning-rate", type=float, default=0.016)
    parser.add_argument("--depth", type=int, default=11)
    parser.add_argument("--l2-leaf-reg", type=float, default=0.5)
    parser.add_argument("--alpha-base", type=float, default=0.0)
    parser.add_argument("--ensemble-best-weight", type=float, default=0.2)
    parser.add_argument("--single-model", action="store_true", help="train one CatBoost model instead of the default four-member ensemble")
    parser.add_argument("--catboost-only", action="store_true", help="skip the ExtraTrees/HGB auxiliary models")
    args = parser.parse_args()

    item = load_standard_data(args.standard_data)
    benchmark = load_benchmark(BENCHMARK_DATA_PATH)
    extrapolation = load_benchmark(EXTRAPOLATION_DATA_PATH)
    base = UnifiedModel.load(args.base_model)

    benchmark_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extrapolation_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    train_mask = np.array([(key not in benchmark_keys) and (key not in extrapolation_keys) for key in item["keys"]], dtype=bool)
    z_mask = np.isclose(item["x_model"][:, 0], args.z, rtol=0.0, atol=1e-8)
    x_train = item["x_model"][z_mask & train_mask]
    y_train = item["y_log"][z_mask & train_mask]
    if x_train.shape[0] == 0:
        raise ValueError(f"no training rows for Z={args.z:g}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.single_model:
        member_specs = [
            {
                "name": "single",
                "iterations": int(args.iterations),
                "learning_rate": float(args.learning_rate),
                "depth": int(args.depth),
                "l2_leaf_reg": float(args.l2_leaf_reg),
                "random_seed": 20260629,
            }
        ]
        best_weight = 1.0
    else:
        member_specs = [dict(item) for item in DEFAULT_ENSEMBLE_MEMBERS]
        best_weight = float(args.ensemble_best_weight)

    cat_models = {}
    aux_models = {}
    model_paths = []
    member_results = []
    train_seconds = 0.0
    for index, spec in enumerate(member_specs):
        model = CatBoostRegressor(
            iterations=int(spec["iterations"]),
            learning_rate=float(spec["learning_rate"]),
            depth=int(spec["depth"]),
            l2_leaf_reg=float(spec["l2_leaf_reg"]),
            loss_function="RMSE",
            random_seed=int(spec["random_seed"]),
            verbose=False,
            thread_count=-1,
        )
        t0 = time.perf_counter()
        model.fit(x_train[:, 1:4], y_train)
        elapsed = time.perf_counter() - t0
        train_seconds += elapsed
        path = ensemble_member_path(args.model_path, index)
        model.save_model(str(path))
        cat_models[spec["name"]] = model
        path = path.resolve()
        model_paths.append(path)
        member_results.append({**spec, "type": "catboost", "model_path": str(path), "train_seconds": float(elapsed)})

    auxiliary_specs = [] if args.single_model or args.catboost_only else [dict(item) for item in DEFAULT_AUXILIARY_MEMBERS]
    auxiliary_results = []
    for spec in auxiliary_specs:
        if spec["type"] == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=int(spec["n_estimators"]),
                min_samples_leaf=int(spec["min_samples_leaf"]),
                max_features=float(spec["max_features"]),
                random_state=int(spec["random_seed"]),
                n_jobs=-1,
                bootstrap=False,
            )
        elif spec["type"] == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=int(spec["max_iter"]),
                learning_rate=float(spec["learning_rate"]),
                max_leaf_nodes=int(spec["max_leaf_nodes"]),
                l2_regularization=float(spec["l2_regularization"]),
                random_state=int(spec["random_seed"]),
                early_stopping=False,
            )
        else:
            raise ValueError(f"unknown auxiliary model type: {spec['type']}")
        t0 = time.perf_counter()
        model.fit(x_train[:, 1:4], y_train)
        elapsed = time.perf_counter() - t0
        train_seconds += elapsed
        path = auxiliary_member_path(args.model_path, spec["name"])
        joblib.dump(model, path)
        path = path.resolve()
        aux_models[spec["name"]] = model
        auxiliary_results.append({**spec, "model_path": str(path), "train_seconds": float(elapsed)})

    mixed_components = None if args.single_model or args.catboost_only else [dict(item) for item in DEFAULT_MIXED_COMPONENTS]

    base_bench_pred = base.predict_log(benchmark["x_model"])
    base_extra_pred = base.predict_log(extrapolation["x_model"])
    direct_bench_pred = base_bench_pred.copy()
    direct_extra_pred = base_extra_pred.copy()
    bench_z_mask = np.isclose(benchmark["x_model"][:, 0], args.z, rtol=0.0, atol=1e-8)
    direct_bench = predict_direct(cat_models, aux_models, benchmark["x_model"][bench_z_mask, 1:4], best_weight, mixed_components)
    direct_bench_pred[bench_z_mask] = args.alpha_base * base_bench_pred[bench_z_mask] + (1.0 - args.alpha_base) * direct_bench

    train_ranges = {
        name: [float(x_train[:, offset].min()), float(x_train[:, offset].max())]
        for offset, name in enumerate(("rod", "tep", "tgama"), start=1)
    }
    # Extrapolation points outside the high-Z training box stay on the base model.
    extra_z_mask = np.isclose(extrapolation["x_model"][:, 0], args.z, rtol=0.0, atol=1e-8)
    extra_route = extra_z_mask.copy()
    for offset, name in enumerate(("rod", "tep", "tgama"), start=1):
        low, high = train_ranges[name]
        extra_route &= (extrapolation["x_model"][:, offset] >= low) & (extrapolation["x_model"][:, offset] <= high)
    if np.any(extra_route):
        direct_extra = predict_direct(cat_models, aux_models, extrapolation["x_model"][extra_route, 1:4], best_weight, mixed_components)
        direct_extra_pred[extra_route] = args.alpha_base * base_extra_pred[extra_route] + (1.0 - args.alpha_base) * direct_extra

    base_benchmark_metrics = element_metrics(benchmark["y_log"], base_bench_pred, benchmark["element"].astype(str))
    base_extrapolation_metrics = element_metrics(extrapolation["y_log"], base_extra_pred, extrapolation["element"].astype(str))
    candidate_routed_benchmark_metrics = element_metrics(benchmark["y_log"], direct_bench_pred, benchmark["element"].astype(str))
    candidate_routed_extrapolation_metrics = element_metrics(extrapolation["y_log"], direct_extra_pred, extrapolation["element"].astype(str))
    candidate_high_z_benchmark_metrics = metric_dict(benchmark["y_log"][bench_z_mask], direct_bench_pred[bench_z_mask])
    candidate_high_z_extrapolation_metrics = metric_dict(extrapolation["y_log"][extra_z_mask], direct_extra_pred[extra_z_mask])
    base_high_z_benchmark_metrics = metric_dict(benchmark["y_log"][bench_z_mask], base_bench_pred[bench_z_mask])
    route_enabled = candidate_high_z_benchmark_metrics["smape_percent"] <= base_high_z_benchmark_metrics["smape_percent"]
    route_disabled_reason = ""
    if not route_enabled:
        route_disabled_reason = (
            "High-Z candidate is disabled because it is worse than the unified base model "
            "on the frozen Au benchmark."
        )
        direct_bench_pred = base_bench_pred.copy()
        direct_extra_pred = base_extra_pred.copy()
        extra_route = np.zeros_like(extra_route, dtype=bool)

    model_specs = [
        {"name": result["name"], "type": "catboost", "path": result["model_path"]}
        for result in member_results
    ] + [
        {"name": result["name"], "type": "joblib", "path": result["model_path"]}
        for result in auxiliary_results
    ]

    metadata = {
        "artifact_type": "high_z_specialized_mixed_ensemble"
        if mixed_components
        else ("high_z_specialized_catboost_ensemble" if len(cat_models) > 1 else "high_z_specialized_catboost"),
        "description": "Dedicated high-Z model for Z=79/Au. The web app routes Z=79 points inside the high-Z training box to this CatBoost model and falls back to the unified model outside that box.",
        "model_path": str(args.model_path),
        "model_paths": [str(path) for path in model_paths],
        "model_specs": model_specs,
        "base_model_path": str(args.base_model),
        "z_values": [float(args.z)],
        "input_features": ["rod", "tep", "tgama"],
        "target_transform": "log10(lnu)",
        "alpha_base": float(args.alpha_base),
        "route_enabled": bool(route_enabled),
        "route_disabled_reason": route_disabled_reason,
        "route_selection_rule": "enable high-Z route only when candidate Au SMAPE is no worse than unified base on the frozen benchmark",
        "train_rows": int(x_train.shape[0]),
        "train_seconds": float(train_seconds),
        "train_ranges": train_ranges,
        "ensemble": {
            "mode": "weighted_components" if mixed_components else ("blend_best_median" if len(cat_models) > 1 else "single"),
            "best_model_index": 0,
            "median_model_indices": list(range(len(cat_models))),
            "best_weight": float(best_weight),
            "cat_median_sources": list(cat_models.keys()),
            "components": mixed_components or [],
        },
        "params": {
            "ensemble_size": int(len(cat_models) + len(aux_models)),
            "single_model": bool(args.single_model),
            "catboost_only": bool(args.catboost_only),
            "members": member_results,
            "auxiliary_members": auxiliary_results,
        },
        "base_benchmark_metrics": base_benchmark_metrics,
        "base_extrapolation_metrics": base_extrapolation_metrics,
        "routed_benchmark_metrics": element_metrics(benchmark["y_log"], direct_bench_pred, benchmark["element"].astype(str)),
        "routed_extrapolation_metrics": element_metrics(extrapolation["y_log"], direct_extra_pred, extrapolation["element"].astype(str)),
        "high_z_benchmark_metrics": candidate_high_z_benchmark_metrics,
        "high_z_extrapolation_metrics": candidate_high_z_extrapolation_metrics,
        "candidate_routed_benchmark_metrics": candidate_routed_benchmark_metrics,
        "candidate_routed_extrapolation_metrics": candidate_routed_extrapolation_metrics,
        "extrapolation_routed_rows": int(extra_route.sum()),
    }
    args.metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = HighZModel.load(args.model_path, args.metrics_path)
    check_pred, check_mask = loaded.predict_log(benchmark["x_model"][bench_z_mask], fallback_log=base_bench_pred[bench_z_mask])
    check_metrics = metric_dict(benchmark["y_log"][bench_z_mask], check_pred)
    print("===== High-Z model trained =====")
    print(f"model: {args.model_path}")
    print(f"metrics: {args.metrics_path}")
    print(f"train rows: {x_train.shape[0]} seconds: {train_seconds:.3f} ensemble_size: {len(cat_models) + len(aux_models)}")
    print(f"route enabled: {route_enabled}")
    if route_disabled_reason:
        print(f"route disabled reason: {route_disabled_reason}")
    print(f"route rows check: {int(np.sum(check_mask))}")
    print(f"Au benchmark SMAPE={check_metrics['smape_percent']:.6f}% log10_MAE={check_metrics['log10_mae']:.8f}")
    print(f"Au extrapolation SMAPE={metadata['high_z_extrapolation_metrics']['smape_percent']:.6f}% log10_MAE={metadata['high_z_extrapolation_metrics']['log10_mae']:.8f}")


if __name__ == "__main__":
    main()
