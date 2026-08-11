#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Au-only local regression sweep on frozen benchmark/extrapolation sets.

This is an offline experiment. It only writes report files and does not update
the active web model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from train_unified_free_path_model import load_standard_data
from unified_free_path_core import (
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    design_matrix,
    metric_dict,
)


ROOT = Path(__file__).resolve().parent
COMBINED_DATA = OUTPUT_DIR / "experiment_standard_with_old_au_and_au2.txt"
OUT_JSON = OUTPUT_DIR / "au_local_sweep_results.json"
OUT_MD = ROOT / "au_local_sweep_report.md"


def load_benchmark(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return payload["x_model"], payload["y_log"], payload["element"].astype(str)


def weighted_average_prediction(distances: np.ndarray, indices: np.ndarray, y: np.ndarray, power: float) -> np.ndarray:
    w = 1.0 / np.power(np.maximum(distances, 1e-12), power)
    exact = distances[:, 0] < 1e-12
    out = np.sum(w * y[indices], axis=1) / np.sum(w, axis=1)
    if np.any(exact):
        out[exact] = y[indices[exact, 0]]
    return out


def local_poly_prediction(
    x_scaled: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    x_train_scaled: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    power: float,
    ridge: float,
) -> np.ndarray:
    if degree == 0:
        return weighted_average_prediction(distances, indices, y, power)
    out = np.empty(x_scaled.shape[0], dtype=float)
    for i in range(x_scaled.shape[0]):
        idx = indices[i]
        delta = x_train_scaled[idx] - x_scaled[i]
        phi = design_matrix(delta, degree)
        dist = np.maximum(distances[i], 1e-12)
        weights = 1.0 / np.power(dist, power)
        if dist[0] < 1e-12:
            out[i] = y[idx[0]]
            continue
        sw = np.sqrt(weights)
        a = phi * sw[:, None]
        b = y[idx] * sw
        reg = np.eye(a.shape[1]) * ridge
        reg[0, 0] = 0.0
        try:
            beta = np.linalg.solve(a.T @ a + reg, a.T @ b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(a, b, rcond=None)[0]
        out[i] = beta[0]
    return out


def main() -> None:
    t0 = time.time()
    item = load_standard_data(COMBINED_DATA)
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    holdout_ok = np.array([key not in bench_keys and key not in extra_keys for key in item["keys"]], dtype=bool)
    au_mask = (item["element_labels"] == "Z_79") & holdout_ok
    x_train = item["x_model"][au_mask, 1:4]
    y_train = item["y_log"][au_mask]

    bench_x_all, bench_y_all, bench_el = load_benchmark(BENCHMARK_DATA_PATH)
    extra_x_all, extra_y_all, extra_el = load_benchmark(EXTRAPOLATION_DATA_PATH)
    bench_mask = bench_el == "Z_79"
    extra_mask = extra_el == "Z_79"
    bench_x = bench_x_all[bench_mask, 1:4]
    bench_y = bench_y_all[bench_mask]
    extra_x = extra_x_all[extra_mask, 1:4]
    extra_y = extra_y_all[extra_mask]

    scale_variants = {
        "std": np.array([1.0, 1.0, 1.0]),
        "rod075": np.array([0.75, 1.0, 1.0]),
        "rod125": np.array([1.25, 1.0, 1.0]),
        "rod150": np.array([1.5, 1.0, 1.0]),
        "tep125": np.array([1.0, 1.25, 1.0]),
        "tgama125": np.array([1.0, 1.0, 1.25]),
        "tep_tgama125": np.array([1.0, 1.25, 1.25]),
        "rod075_tgama125": np.array([0.75, 1.0, 1.25]),
    }
    k_values = [12, 20, 32, 48, 64, 80, 100, 128, 180, 260, 360]
    degrees = [0, 1, 2]
    powers = [1.1, 1.4, 1.7, 2.1]
    ridge_by_degree = {0: [0.0], 1: [1e-5, 1e-4, 1e-3], 2: [1e-4, 1e-3, 1e-2]}
    max_k = max(k_values)

    rows = []
    print(f"Au train rows={x_train.shape[0]}, bench={bench_x.shape[0]}, extrap={extra_x.shape[0]}")
    for scale_name, extra_weight in scale_variants.items():
        mean = x_train.mean(axis=0)
        std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
        x_train_scaled = ((x_train - mean) / std) * extra_weight
        bench_scaled = ((bench_x - mean) / std) * extra_weight
        tree = cKDTree(x_train_scaled)
        bench_dist_all, bench_idx_all = tree.query(bench_scaled, k=max_k, workers=-1)
        print(f"scale={scale_name} queried benchmark", flush=True)
        for k in k_values:
            dist = bench_dist_all[:, :k]
            idx = bench_idx_all[:, :k]
            for degree in degrees:
                if degree > 0 and k < 20:
                    continue
                for power in powers:
                    for ridge in ridge_by_degree[degree]:
                        pred = local_poly_prediction(
                            bench_scaled,
                            dist,
                            idx,
                            x_train_scaled,
                            y_train,
                            degree=degree,
                            power=power,
                            ridge=ridge,
                        )
                        metrics = metric_dict(bench_y, pred)
                        rows.append(
                            {
                                "scale": scale_name,
                                "extra_weight": extra_weight.tolist(),
                                "k": int(k),
                                "degree": int(degree),
                                "power": float(power),
                                "ridge": float(ridge),
                                "benchmark": metrics,
                            }
                        )

    rows.sort(key=lambda r: (r["benchmark"]["smape_percent"], r["benchmark"]["log10_mae"]))
    top = rows[:30]
    for i, row in enumerate(top, 1):
        extra_weight = np.asarray(row["extra_weight"], dtype=float)
        mean = x_train.mean(axis=0)
        std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
        x_train_scaled = ((x_train - mean) / std) * extra_weight
        extra_scaled = ((extra_x - mean) / std) * extra_weight
        tree = cKDTree(x_train_scaled)
        dist, idx = tree.query(extra_scaled, k=int(row["k"]), workers=-1)
        pred = local_poly_prediction(
            extra_scaled,
            dist,
            idx,
            x_train_scaled,
            y_train,
            degree=int(row["degree"]),
            power=float(row["power"]),
            ridge=float(row["ridge"]),
        )
        row["extrapolation"] = metric_dict(extra_y, pred)
        print(
            f"top {i}: bench={row['benchmark']['smape_percent']:.6f}% "
            f"extra={row['extrapolation']['smape_percent']:.6f}% {row['scale']} k={row['k']} d={row['degree']}",
            flush=True,
        )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(COMBINED_DATA),
        "fixed_benchmark": str(BENCHMARK_DATA_PATH),
        "fixed_extrapolation": str(EXTRAPOLATION_DATA_PATH),
        "au_train_rows": int(x_train.shape[0]),
        "searched_rows": len(rows),
        "top": top,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Au 专用局部回归大扫描报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- Au 训练行数：{payload['au_train_rows']}",
        f"- 搜索组合数：{payload['searched_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | scale | k | degree | power | ridge | Au benchmark SMAPE | Au extrap SMAPE | Au benchmark P99 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(top[:20], 1):
        extra = row.get("extrapolation", {})
        lines.append(
            f"| {i} | `{row['scale']}` | {row['k']} | {row['degree']} | {row['power']} | {row['ridge']:.1e} | "
            f"{row['benchmark']['smape_percent']:.6f}% | {extra.get('smape_percent', float('nan')):.6f}% | "
            f"{row['benchmark']['p99_factor_error']:.6f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
