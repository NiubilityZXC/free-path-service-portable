#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Offline MoE / physical-state partition experiments for Au free-path.

This script does not publish or modify the web model. It compares practical
variants of physical-state partitioning on the frozen benchmark payloads.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree

from train_unified_free_path_model import load_standard_data
from unified_free_path_core import (
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    UnifiedModel,
    metric_dict,
)


ROOT = Path(__file__).resolve().parent
COMBINED_DATA = OUTPUT_DIR / "experiment_standard_with_old_au_and_au2.txt"
CURRENT_DATA = OUTPUT_DIR / "unified_standard_training_data.txt"
BASE_MODEL_PATH = OUTPUT_DIR / "experiment2_au_k100_model.npz"
BASE_METRICS_PATH = OUTPUT_DIR / "experiment2_au_k100_metrics.json"
OUT_JSON = OUTPUT_DIR / "moe_partition_experiment_results.json"
OUT_MD = ROOT / "moe_partition_experiment_report.md"


def load_benchmark(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)
    return {
        "x_model": payload["x_model"],
        "y_log": payload["y_log"],
        "element": payload["element"].astype(str),
    }


def predict_batches(model: UnifiedModel, x: np.ndarray, *, batch_size: int = 2048, label: str = "") -> np.ndarray:
    out = np.empty(x.shape[0], dtype=float)
    start = time.time()
    for i in range(0, x.shape[0], batch_size):
        j = min(i + batch_size, x.shape[0])
        out[i:j] = model.predict_log(x[i:j])
        if label and (i == 0 or j == x.shape[0] or (j // batch_size) % 10 == 0):
            print(f"{label}: {j}/{x.shape[0]} rows, {time.time() - start:.1f}s", flush=True)
    return out


def smape_per_point(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    true = np.power(10.0, y_true_log)
    pred = np.power(10.0, y_pred_log)
    return 2.0 * np.abs(pred - true) / np.maximum(np.abs(pred) + np.abs(true), 1e-300) * 100.0


def load_reference_element_metrics(split: str) -> dict[str, dict]:
    metrics = json.loads(BASE_METRICS_PATH.read_text(encoding="utf-8"))
    key = "benchmark_metrics" if split == "benchmark" else "extrapolation_benchmark_metrics"
    rows = metrics[key]["by_element"]
    return {row["element"]: row for row in rows}


def weighted_overall(split: str, au_metrics: dict) -> dict:
    refs = load_reference_element_metrics(split)
    rows = [refs["Z_13"], refs["Z_4"], au_metrics]
    n = sum(int(row["n"]) for row in rows)
    return {
        "n": n,
        "smape_percent": sum(row["smape_percent"] * row["n"] for row in rows) / n,
        "log10_mae": sum(row["log10_mae"] * row["n"] for row in rows) / n,
    }


def route_metrics(
    method_name: str,
    split: str,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    expert_pred: np.ndarray,
    route_mask: np.ndarray,
) -> dict:
    pred = np.asarray(base_pred, dtype=float).copy()
    pred[route_mask] = np.asarray(expert_pred, dtype=float)[route_mask]
    au = metric_dict(y_true, pred)
    overall = weighted_overall(split, au)
    return {
        "method": method_name,
        "split": split,
        "routed_rows": int(np.sum(route_mask)),
        "au": au,
        "overall": overall,
    }


@dataclass
class ExpertPrediction:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]


class ClusterExperts:
    def __init__(self, scaler_mean: np.ndarray, scaler_std: np.ndarray, centers: np.ndarray, models: list, *, soft_power: float = 2.0):
        self.mean = scaler_mean
        self.std = scaler_std
        self.centers = centers
        self.models = models
        self.soft_power = float(soft_power)

    def _scaled(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def predict_hard(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        d2 = ((xs[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        out = np.empty(xs.shape[0], dtype=float)
        for k, model in enumerate(self.models):
            mask = labels == k
            if np.any(mask):
                out[mask] = np.asarray(model.predict(xs[mask]), dtype=float)
        return out

    def predict_soft(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        d = np.sqrt(((xs[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2))
        weights = 1.0 / np.power(np.maximum(d, 1e-8), self.soft_power)
        weights /= weights.sum(axis=1, keepdims=True)
        stack = np.column_stack([np.asarray(model.predict(xs), dtype=float) for model in self.models])
        return np.sum(weights * stack, axis=1)


def fit_hgb() -> object:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.035,
        max_iter=900,
        max_leaf_nodes=63,
        l2_regularization=0.02,
        min_samples_leaf=18,
        random_state=20260703,
    )


def fit_cluster_experts(x_train: np.ndarray, y_train: np.ndarray, *, n_clusters: int, io_target: np.ndarray | None = None) -> ClusterExperts:
    from sklearn.cluster import MiniBatchKMeans

    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
    xs = (x_train - mean) / std
    if io_target is None:
        cluster_input = xs
    else:
        yz = (io_target - io_target.mean()) / max(float(io_target.std()), 1e-12)
        cluster_input = np.column_stack([xs, yz])
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        n_init=8,
        max_iter=300,
        random_state=20260703,
    )
    labels = km.fit_predict(cluster_input)
    centers = km.cluster_centers_[:, : xs.shape[1]]
    models = []
    global_model = fit_hgb()
    global_model.fit(xs, y_train)
    for k in range(n_clusters):
        mask = labels == k
        if int(mask.sum()) < 600:
            models.append(global_model)
            continue
        model = fit_hgb()
        model.fit(xs[mask], y_train[mask])
        models.append(model)
    return ClusterExperts(mean, std, centers, models)


class ClassifierExperts:
    def __init__(self, mean: np.ndarray, std: np.ndarray, classifier: object, models: list):
        self.mean = mean
        self.std = std
        self.classifier = classifier
        self.models = models

    def _scaled(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def predict_soft(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        prob = np.asarray(self.classifier.predict_proba(xs), dtype=float)
        labels = list(self.classifier.classes_)
        stack = np.column_stack([np.asarray(self.models[int(label)].predict(xs), dtype=float) for label in labels])
        return np.sum(prob * stack, axis=1)


def fit_io_classifier_experts(x_train: np.ndarray, y_train: np.ndarray, residual_like: np.ndarray, *, n_clusters: int) -> ClassifierExperts:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.ensemble import ExtraTreesClassifier

    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
    xs = (x_train - mean) / std
    yz = (residual_like - residual_like.mean()) / max(float(residual_like.std()), 1e-12)
    cluster_input = np.column_stack([xs, yz])
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        n_init=8,
        max_iter=300,
        random_state=20260703,
    )
    labels = km.fit_predict(cluster_input)
    clf = ExtraTreesClassifier(
        n_estimators=240,
        max_features=2,
        min_samples_leaf=12,
        random_state=20260703,
        n_jobs=-1,
    )
    clf.fit(xs, labels)
    models = []
    global_model = fit_hgb()
    global_model.fit(xs, y_train)
    for k in range(n_clusters):
        mask = labels == k
        if int(mask.sum()) < 600:
            models.append(global_model)
            continue
        model = fit_hgb()
        model.fit(xs[mask], y_train[mask])
        models.append(model)
    return ClassifierExperts(mean, std, clf, models)


class BgmmExperts:
    def __init__(self, mean: np.ndarray, std: np.ndarray, bgmm: object, models: list):
        self.mean = mean
        self.std = std
        self.bgmm = bgmm
        self.models = models

    def _scaled(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def predict_soft(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        prob = np.asarray(self.bgmm.predict_proba(xs), dtype=float)
        stack = np.column_stack([np.asarray(model.predict(xs), dtype=float) for model in self.models])
        return np.sum(prob * stack, axis=1)


def fit_bgmm_experts(x_train: np.ndarray, y_train: np.ndarray, *, n_components: int) -> BgmmExperts:
    from sklearn.mixture import BayesianGaussianMixture

    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
    xs = (x_train - mean) / std
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        max_iter=250,
        random_state=20260703,
    )
    labels = bgmm.fit_predict(xs)
    models = []
    global_model = fit_hgb()
    global_model.fit(xs, y_train)
    for k in range(n_components):
        mask = labels == k
        if int(mask.sum()) < 600:
            models.append(global_model)
            continue
        model = fit_hgb()
        model.fit(xs[mask], y_train[mask])
        models.append(model)
    return BgmmExperts(mean, std, bgmm, models)


class ResidualModel:
    def __init__(self, mean: np.ndarray, std: np.ndarray, model: object):
        self.mean = mean
        self.std = std
        self.model = model

    def predict_residual(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=float) - self.mean) / self.std
        return np.asarray(self.model.predict(xs), dtype=float)


def fit_residual_hgb(x_train: np.ndarray, residual: np.ndarray) -> ResidualModel:
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
    xs = (x_train - mean) / std
    model = fit_hgb()
    model.fit(xs, residual)
    return ResidualModel(mean, std, model)


def fit_catboost_direct(x_train: np.ndarray, y_train: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=2600,
        depth=10,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        random_seed=20260703,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)
    return lambda x: np.asarray(model.predict(x), dtype=float).reshape(-1)


def make_distance_route(train_x: np.ndarray, val_x: np.ndarray, q: float = 0.95) -> Callable[[np.ndarray], np.ndarray]:
    mean = train_x.mean(axis=0)
    std = np.where(train_x.std(axis=0) < 1e-12, 1.0, train_x.std(axis=0))
    tree = cKDTree((train_x - mean) / std)
    val_d, _ = tree.query((val_x - mean) / std, k=1)
    threshold = float(np.quantile(val_d, q))

    def route(x: np.ndarray) -> np.ndarray:
        d, _ = tree.query((np.asarray(x, dtype=float) - mean) / std, k=1)
        return d <= threshold

    route.threshold = threshold  # type: ignore[attr-defined]
    return route


def read_stored_method(path: Path, label: str) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for split_name, key in [("benchmark", "routed_benchmark_metrics"), ("extrapolation", "routed_extrapolation_metrics")]:
        metrics = data.get(key)
        high_z = data.get("high_z_benchmark_metrics" if split_name == "benchmark" else "high_z_extrapolation_metrics")
        if isinstance(metrics, dict) and "overall" in metrics:
            metrics = metrics["overall"]
        if isinstance(high_z, dict) and "overall" in high_z:
            high_z = high_z["overall"]
        if metrics:
            out.append(
                {
                    "method": label,
                    "split": split_name,
                    "routed_rows": int(data.get("benchmark_routed_rows_midrod" if split_name == "benchmark" else "extrapolation_routed_rows_midrod", -1)),
                    "overall": {
                        "n": metrics["n"],
                        "smape_percent": metrics["smape_percent"],
                        "log10_mae": metrics["log10_mae"],
                    },
                    "au": high_z or {},
                    "source": str(path),
                }
            )
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(20260703)
    data_path = COMBINED_DATA if COMBINED_DATA.exists() else CURRENT_DATA
    print(f"data: {data_path}")
    item = load_standard_data(data_path)
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    key_ok = np.array([key not in bench_keys and key not in extra_keys for key in item["keys"]], dtype=bool)
    au_train_mask = (item["element_labels"] == "Z_79") & key_ok
    x_au = item["x_model"][au_train_mask, 1:4]
    y_au = item["y_log"][au_train_mask]
    print(f"Au train candidates after fixed holdout exclusion: {x_au.shape[0]}")

    order = rng.permutation(x_au.shape[0])
    train_n = min(70000, order.shape[0] - 25000)
    val_n = min(30000, order.shape[0] - train_n)
    train_idx = order[:train_n]
    val_idx = order[train_n : train_n + val_n]
    x_fit, y_fit = x_au[train_idx], y_au[train_idx]
    x_val, y_val = x_au[val_idx], y_au[val_idx]
    print(f"fit rows={x_fit.shape[0]}, validation rows={x_val.shape[0]}")

    benchmark = load_benchmark(BENCHMARK_DATA_PATH)
    extrap = load_benchmark(EXTRAPOLATION_DATA_PATH)
    bench_au_mask = benchmark["element"] == "Z_79"
    extra_au_mask = extrap["element"] == "Z_79"
    x_bench_au = benchmark["x_model"][bench_au_mask, 1:4]
    y_bench_au = benchmark["y_log"][bench_au_mask]
    x_extra_au = extrap["x_model"][extra_au_mask, 1:4]
    y_extra_au = extrap["y_log"][extra_au_mask]

    base_model = UnifiedModel.load(BASE_MODEL_PATH)
    base_bench_au = predict_batches(base_model, benchmark["x_model"][bench_au_mask], label="base bench Au")
    base_extra_au = predict_batches(base_model, extrap["x_model"][extra_au_mask], label="base extrap Au")
    base_val = predict_batches(base_model, np.column_stack([np.full(x_val.shape[0], 79.0), x_val]), label="base validation Au")
    base_fit_small_n = min(30000, x_fit.shape[0])
    base_fit_small = predict_batches(
        base_model,
        np.column_stack([np.full(base_fit_small_n, 79.0), x_fit[:base_fit_small_n]]),
        label="base residual-fit Au",
    )

    results: list[dict] = []
    base_bench_au_metrics = metric_dict(y_bench_au, base_bench_au)
    base_extra_au_metrics = metric_dict(y_extra_au, base_extra_au)
    results.extend(
        [
            {
                "method": "baseline_unified_local_k100",
                "split": "benchmark",
                "routed_rows": 0,
                "au": base_bench_au_metrics,
                "overall": weighted_overall("benchmark", base_bench_au_metrics),
                "source": str(BASE_MODEL_PATH),
            },
            {
                "method": "baseline_unified_local_k100",
                "split": "extrapolation",
                "routed_rows": 0,
                "au": base_extra_au_metrics,
                "overall": weighted_overall("extrapolation", base_extra_au_metrics),
                "source": str(BASE_MODEL_PATH),
            },
        ]
    )

    results.extend(read_stored_method(OUTPUT_DIR / "experiment_highz_combined_single_midrod_metrics.json", "stored_catboost_single_midrod"))
    results.extend(read_stored_method(OUTPUT_DIR / "experiment_highz_combined_ensemble_midrod_metrics.json", "stored_heterogeneous_full_ensemble_midrod"))
    results.extend(read_stored_method(OUTPUT_DIR / "experiment_highz_combined_ensemble_metrics.json", "stored_heterogeneous_full_ensemble_all_range"))

    routes: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "all_au": lambda x: np.ones(x.shape[0], dtype=bool),
        "inner_rod_p10_p95": lambda x: (x[:, 0] >= np.percentile(x_fit[:, 0], 10.0)) & (x[:, 0] <= np.percentile(x_fit[:, 0], 95.0)),
    }
    distance_route = make_distance_route(x_fit, x_val, q=0.95)
    routes[f"ood_distance_q95_{distance_route.threshold:.4g}"] = distance_route

    expert_predictions: list[ExpertPrediction] = []

    print("fit global HGB direct", flush=True)
    hgb_global = fit_hgb()
    hgb_global.fit(x_fit, y_fit)
    expert_predictions.append(ExpertPrediction("hgb_global_direct", lambda x, m=hgb_global: np.asarray(m.predict(x), dtype=float)))

    print("fit feature KMeans experts", flush=True)
    feature_cluster = fit_cluster_experts(x_fit, y_fit, n_clusters=8)
    expert_predictions.append(ExpertPrediction("feature_kmeans_hard_hgb", feature_cluster.predict_hard))
    expert_predictions.append(ExpertPrediction("feature_kmeans_soft_hgb", feature_cluster.predict_soft))

    print("fit input-output KMeans classifier experts", flush=True)
    io_cluster = fit_io_classifier_experts(x_fit, y_fit, y_fit, n_clusters=10)
    expert_predictions.append(ExpertPrediction("input_output_kmeans_soft_classifier_hgb", io_cluster.predict_soft))

    print("fit DPMM/BGMM soft experts", flush=True)
    bgmm_sample_n = min(35000, x_fit.shape[0])
    bgmm_experts = fit_bgmm_experts(x_fit[:bgmm_sample_n], y_fit[:bgmm_sample_n], n_components=10)
    expert_predictions.append(ExpertPrediction("dpmm_bgmm_soft_hgb", bgmm_experts.predict_soft))

    print("fit residual HGB", flush=True)
    residual_model = fit_residual_hgb(x_fit[:base_fit_small_n], y_fit[:base_fit_small_n] - base_fit_small)
    expert_predictions.append(
        ExpertPrediction(
            "residual_hgb_on_unified",
            lambda x, m=residual_model: np.full(x.shape[0], np.nan, dtype=float),
        )
    )

    print("fit CatBoost direct", flush=True)
    cat_predict = fit_catboost_direct(x_fit, y_fit)
    expert_predictions.append(ExpertPrediction("catboost_direct_sampled", cat_predict))

    bench_preds: dict[str, np.ndarray] = {}
    extra_preds: dict[str, np.ndarray] = {}
    val_preds: dict[str, np.ndarray] = {}
    for expert in expert_predictions:
        print(f"predict {expert.name}", flush=True)
        if expert.name == "residual_hgb_on_unified":
            bench_preds[expert.name] = base_bench_au + residual_model.predict_residual(x_bench_au)
            extra_preds[expert.name] = base_extra_au + residual_model.predict_residual(x_extra_au)
            val_preds[expert.name] = base_val + residual_model.predict_residual(x_val)
        else:
            bench_preds[expert.name] = expert.predict(x_bench_au)
            extra_preds[expert.name] = expert.predict(x_extra_au)
            val_preds[expert.name] = expert.predict(x_val)

    for name in list(bench_preds):
        for route_name, route in routes.items():
            rb = route(x_bench_au)
            re = route(x_extra_au)
            results.append(route_metrics(f"{name}+{route_name}", "benchmark", y_bench_au, base_bench_au, bench_preds[name], rb))
            results.append(route_metrics(f"{name}+{route_name}", "extrapolation", y_extra_au, base_extra_au, extra_preds[name], re))

    print("fit validation nonnegative stack", flush=True)
    stack_names = [
        "hgb_global_direct",
        "feature_kmeans_soft_hgb",
        "input_output_kmeans_soft_classifier_hgb",
        "dpmm_bgmm_soft_hgb",
        "residual_hgb_on_unified",
        "catboost_direct_sampled",
    ]
    stack_val = np.column_stack([val_preds[name] for name in stack_names])
    weights, _ = nnls(stack_val, y_val)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(stack_names), dtype=float) / len(stack_names)
    else:
        weights = weights / weights.sum()
    stack_bench = np.sum(np.column_stack([bench_preds[name] for name in stack_names]) * weights[None, :], axis=1)
    stack_extra = np.sum(np.column_stack([extra_preds[name] for name in stack_names]) * weights[None, :], axis=1)
    for route_name, route in routes.items():
        rb = route(x_bench_au)
        re = route(x_extra_au)
        results.append(route_metrics(f"validation_nnls_soft_stack+{route_name}", "benchmark", y_bench_au, base_bench_au, stack_bench, rb))
        results.append(route_metrics(f"validation_nnls_soft_stack+{route_name}", "extrapolation", y_extra_au, base_extra_au, stack_extra, re))

    results_sorted = sorted(
        results,
        key=lambda r: (
            r["split"],
            r["overall"].get("smape_percent", 999.0),
            r.get("au", {}).get("smape_percent", 999.0),
        ),
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(data_path),
        "fixed_benchmark": str(BENCHMARK_DATA_PATH),
        "fixed_extrapolation": str(EXTRAPOLATION_DATA_PATH),
        "base_model": str(BASE_MODEL_PATH),
        "fit_rows": int(x_fit.shape[0]),
        "validation_rows": int(x_val.shape[0]),
        "routes": {
            "all_au": "replace every Au prediction with the expert",
            "inner_rod_p10_p95": "expert only between the 10th and 95th percentiles of Au training rod; fallback to unified model outside",
            "ood_distance_q95": "expert only when nearest-neighbor distance is inside the 95th percentile of validation distances",
        },
        "stack_weights": {name: float(w) for name, w in zip(stack_names, weights)},
        "results": results_sorted,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    bench_rows = [r for r in results_sorted if r["split"] == "benchmark"]
    extra_rows = [r for r in results_sorted if r["split"] == "extrapolation"]
    lines = [
        "# MoE / 物理状态分区离线实验报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 训练数据：`{data_path}`",
        f"- 固定随机 benchmark：`{BENCHMARK_DATA_PATH}`",
        f"- 固定连续外推 benchmark：`{EXTRAPOLATION_DATA_PATH}`",
        "- 本实验没有发布模型，也没有修改网页正在使用的权重。",
        "",
        "## 随机 benchmark 排名",
        "",
        "| 排名 | 方法 | overall SMAPE | Au SMAPE | Au log10 MAE | Au P99倍数 | 路由行数 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(bench_rows[:20], 1):
        au = row.get("au", {})
        lines.append(
            f"| {i} | `{row['method']}` | {row['overall']['smape_percent']:.6f}% | "
            f"{au.get('smape_percent', float('nan')):.6f}% | {au.get('log10_mae', float('nan')):.8f} | "
            f"{au.get('p99_factor_error', float('nan')):.6f} | {row.get('routed_rows', -1)} |"
        )
    lines.extend(
        [
            "",
            "## 连续外推 benchmark 排名",
            "",
            "| 排名 | 方法 | overall SMAPE | Au SMAPE | Au log10 MAE | Au P99倍数 | 路由行数 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for i, row in enumerate(extra_rows[:20], 1):
        au = row.get("au", {})
        lines.append(
            f"| {i} | `{row['method']}` | {row['overall']['smape_percent']:.6f}% | "
            f"{au.get('smape_percent', float('nan')):.6f}% | {au.get('log10_mae', float('nan')):.8f} | "
            f"{au.get('p99_factor_error', float('nan')):.6f} | {row.get('routed_rows', -1)} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `all_au`：所有 Au 点都走专家模型。",
            "- `inner_rod_p10_p95`：只在 Au 训练 rod 的 10%-95% 内启用专家，边界区域回退统一模型。",
            "- `ood_distance_q95`：离训练样本太远就回退统一模型。",
            "- `stored_*` 是前面已经训练出的高 Z CatBoost/异构集成候选，本次直接纳入同一张表对比。",
            "- overall 里 Be/Al 使用同一个基础统一模型，因此对比主要反映 Au 专家和路由策略的差异。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
