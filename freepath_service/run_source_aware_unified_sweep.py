#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sweep unified-model parameters on the source-aware permanent benchmarks."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from train_unified_free_path_model import (
    benchmark_model,
    build_low_rod_boundary_extension,
    load_benchmark,
    load_standard_data,
    save_artifact,
)
from unified_free_path_core import (
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    STANDARD_INPUT_COLUMNS,
    STANDARD_TRAINING_COLUMNS,
    UnifiedModel,
)


ROOT = Path(__file__).resolve().parent
STANDARD_DATA = OUTPUT_DIR / "unified_standard_training_data.txt"
TEMP_MODEL = OUTPUT_DIR / "tmp_source_aware_sweep_model.npz"
OUT_JSON = OUTPUT_DIR / "source_aware_unified_sweep_results.json"
OUT_MD = ROOT / "source_aware_unified_sweep_report.md"


def prepare_train():
    item = load_standard_data(STANDARD_DATA)
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    excluded = np.array([key in bench_keys or key in extra_keys for key in item["keys"]], dtype=bool)
    return item, item["x_model"][~excluded], item["y_log"][~excluded]


def source_smape(metrics: dict, source: str) -> float:
    for row in metrics.get("by_source", []):
        if row.get("source") == source:
            return float(row["smape_percent"])
    return float("nan")


def make_model(item: dict, x_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> UnifiedModel:
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std = np.where(feature_std < 1e-14, 1.0, feature_std)
    x_train_scaled = (x_train - feature_mean) / feature_std
    x_train_scaled[:, 0] *= float(cfg["z_weight"])
    adaptive_model_params = {
        "Z_79": {
            "k_neighbors": int(cfg["au_k"]),
            "local_degree": int(cfg["au_degree"]),
            "ridge_alpha": float(cfg["au_ridge"]),
            "distance_power": float(cfg["au_power"]),
        }
    }
    boundary_extensions = []
    if cfg.get("low_boundary", True):
        extension = build_low_rod_boundary_extension(
            x_train=x_train,
            y_train=y_train,
            z_value=79.0,
            nfit=int(cfg.get("low_nfit", 9)),
            smoothing_knn=int(cfg.get("low_knn", 2500)),
            smoothing_power=float(cfg.get("low_power", 1.5)),
        )
        if extension is not None:
            boundary_extensions.append(extension)
    metadata = {
        "artifact_type": "unified_free_path_local_regression_source_aware_sweep",
        "standard_training_format": " ".join(STANDARD_TRAINING_COLUMNS),
        "standard_prediction_format": " ".join(STANDARD_INPUT_COLUMNS),
        "target_transform": "log10(lnu)",
        "input_features": STANDARD_INPUT_COLUMNS,
        "standard_training_data_path": str(STANDARD_DATA),
        "n_train": int(x_train.shape[0]),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "model_params": {
            "k_neighbors": int(cfg["k"]),
            "local_degree": int(cfg["degree"]),
            "ridge_alpha": float(cfg["ridge"]),
            "distance_power": float(cfg["power"]),
            "z_weight": float(cfg["z_weight"]),
        },
        "adaptive_model_params": adaptive_model_params,
        "boundary_extensions": boundary_extensions,
        "elements": item["element_summary"],
        "training_files": {
            item["name"]: {
                "path": item["path"],
                "total_rows": int(item["x_model"].shape[0]),
                "training_rows": int(x_train.shape[0]),
            }
        },
    }
    save_artifact(TEMP_MODEL, x_train_scaled, y_train, metadata)
    return UnifiedModel.load(TEMP_MODEL)


def evaluate_cfg(item: dict, x_train: np.ndarray, y_train: np.ndarray, cfg: dict, benchmark: dict, extrap: dict) -> dict:
    start = time.time()
    model = make_model(item, x_train, y_train, cfg)
    bench_overall, bench_rows, bench_sources = benchmark_model(model, benchmark)
    extra_overall, extra_rows, extra_sources = benchmark_model(model, extrap)
    bench_metrics = {"overall": bench_overall, "by_element": bench_rows[1:], "by_source": bench_sources}
    extra_metrics = {"overall": extra_overall, "by_element": extra_rows[1:], "by_source": extra_sources}
    return {
        "config": cfg,
        "benchmark_metrics": bench_metrics,
        "extrapolation_benchmark_metrics": extra_metrics,
        "score": float(bench_overall["smape_percent"] + extra_overall["smape_percent"]),
        "au2_extrap_smape": source_smape(extra_metrics, "Au2_data_Au2.txt"),
        "old_au_extrap_smape": source_smape(extra_metrics, "Au_old_data.txt"),
        "seconds": time.time() - start,
    }


def main() -> None:
    t0 = time.time()
    item, x_train, y_train = prepare_train()
    benchmark = load_benchmark()
    extrap = load_benchmark(EXTRAPOLATION_DATA_PATH)
    configs = []
    base = {
        "k": 260,
        "degree": 3,
        "ridge": 1e-3,
        "power": 1.6,
        "z_weight": 2.0,
        "au_degree": 2,
        "au_ridge": 3e-3,
        "low_boundary": True,
        "low_nfit": 9,
        "low_knn": 2500,
        "low_power": 1.5,
    }
    for au_k, au_power in itertools.product([60, 100, 160, 260, 420, 700], [1.0, 1.2, 1.3, 1.5, 1.8]):
        cfg = dict(base, au_k=au_k, au_power=au_power)
        configs.append(cfg)
    for z_weight, k in itertools.product([0.5, 1.0, 2.0, 4.0], [180, 260, 360]):
        cfg = dict(base, z_weight=z_weight, k=k, au_k=160, au_power=1.2)
        configs.append(cfg)
    for low_boundary in [False, True]:
        for au_degree in [1, 2, 3]:
            cfg = dict(base, au_k=160, au_power=1.2, au_degree=au_degree, low_boundary=low_boundary)
            configs.append(cfg)

    # Stable order and remove exact duplicates.
    unique = []
    seen = set()
    for cfg in configs:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            unique.append(cfg)
            seen.add(key)

    print(f"train={x_train.shape[0]}, configs={len(unique)}", flush=True)
    rows = []
    for idx, cfg in enumerate(unique, 1):
        row = evaluate_cfg(item, x_train, y_train, cfg, benchmark, extrap)
        rows.append(row)
        print(
            f"{idx:03d}/{len(unique)} score={row['score']:.6f} "
            f"bench={row['benchmark_metrics']['overall']['smape_percent']:.6f}% "
            f"extra={row['extrapolation_benchmark_metrics']['overall']['smape_percent']:.6f}% "
            f"au2_extra={row['au2_extrap_smape']:.6f}% cfg={cfg}",
            flush=True,
        )
    rows.sort(key=lambda row: (row["score"], row["extrapolation_benchmark_metrics"]["overall"]["smape_percent"]))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(STANDARD_DATA),
        "n_train": int(x_train.shape[0]),
        "n_configs": len(unique),
        "top": rows[:25],
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Source-aware 统一模型参数扫描",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 训练行数：{payload['n_train']}",
        f"- 搜索组合：{payload['n_configs']}",
        "",
        "| 排名 | score | 随机 overall | 外推 overall | Au2 外推 | 旧 Au 外推 | au_k | au_power | z_weight | k | au_degree | low_boundary |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(rows[:20], 1):
        cfg = row["config"]
        lines.append(
            f"| {i} | {row['score']:.6f} | {row['benchmark_metrics']['overall']['smape_percent']:.6f}% | "
            f"{row['extrapolation_benchmark_metrics']['overall']['smape_percent']:.6f}% | "
            f"{row['au2_extrap_smape']:.6f}% | {row['old_au_extrap_smape']:.6f}% | "
            f"{cfg['au_k']} | {cfg['au_power']} | {cfg['z_weight']} | {cfg['k']} | {cfg['au_degree']} | {cfg['low_boundary']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"best: {rows[0]}")


if __name__ == "__main__":
    main()
