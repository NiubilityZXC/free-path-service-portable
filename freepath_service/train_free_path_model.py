#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train and evaluate a surrogate model for Rosseland free path lnu.

The data files in this directory use four columns:

    rod  tep  tgama  lnu

The first three columns form a complete 3D grid and the last column is the
free path.  Because lnu is positive and spans many orders of magnitude, the
model is trained on log10(lnu) and converted back during prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data_Al.txt"
DEFAULT_OUTPUT = ROOT / "free_path_model_outputs"
COLUMNS = ["rod", "tep", "tgama", "lnu"]


@dataclass
class GridData:
    path: Path
    data: np.ndarray
    axes: list[np.ndarray]
    lnu_grid: np.ndarray
    log_lnu_grid: np.ndarray


def load_grid_data(path: Path) -> GridData:
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"{path} must have exactly 4 numeric columns: {COLUMNS}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path} contains NaN or inf values")
    if np.any(data[:, 3] <= 0):
        n_bad = int(np.sum(data[:, 3] <= 0))
        raise ValueError(
            f"{path} contains {n_bad} non-positive lnu values; "
            "log10(lnu) model requires positive targets."
        )

    unique_rows = np.unique(data[:, :3], axis=0).shape[0]
    if unique_rows != data.shape[0]:
        raise ValueError(f"{path} has duplicated input parameter rows")

    axes = [np.unique(data[:, i]) for i in range(3)]
    expected_n = math.prod(len(axis) for axis in axes)
    if expected_n != data.shape[0]:
        raise ValueError(
            f"{path} is not a complete rectangular grid: "
            f"rows={data.shape[0]}, expected={expected_n}"
        )

    order = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
    sorted_data = data[order]
    mesh = np.meshgrid(*axes, indexing="ij")
    expected_xyz = np.column_stack([m.ravel() for m in mesh])
    if not np.allclose(sorted_data[:, :3], expected_xyz, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path} grid ordering cannot be reconstructed safely")

    shape = tuple(len(axis) for axis in axes)
    lnu_grid = sorted_data[:, 3].reshape(shape)
    return GridData(
        path=path,
        data=sorted_data,
        axes=axes,
        lnu_grid=lnu_grid,
        log_lnu_grid=np.log10(lnu_grid),
    )


def make_interpolator(
    axes: list[np.ndarray],
    log_lnu_grid: np.ndarray,
    method: str,
    allow_extrapolate: bool = False,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        axes,
        log_lnu_grid,
        method=method,
        bounds_error=not allow_extrapolate,
        fill_value=None if allow_extrapolate else np.nan,
    )


def metric_dict(log_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    log_true = np.asarray(log_true, dtype=float)
    log_pred = np.asarray(log_pred, dtype=float)
    valid = np.isfinite(log_pred) & np.isfinite(log_true)
    if not valid.all():
        log_true = log_true[valid]
        log_pred = log_pred[valid]
    true = np.power(10.0, log_true)
    pred = np.power(10.0, log_pred)
    log_error = log_pred - log_true
    factor_error = np.power(10.0, np.abs(log_error))
    rel = np.abs(pred - true) / np.maximum(np.abs(true), 1e-300)
    smape = 2.0 * np.abs(pred - true) / np.maximum(np.abs(pred) + np.abs(true), 1e-300)
    return {
        "n": int(log_true.size),
        "log10_mae": float(np.mean(np.abs(log_error))),
        "log10_rmse": float(np.sqrt(np.mean(log_error**2))),
        "mape_percent": float(np.mean(rel) * 100.0),
        "smape_percent": float(np.mean(smape) * 100.0),
        "median_factor_error": float(np.median(factor_error)),
        "p90_factor_error": float(np.percentile(factor_error, 90.0)),
        "p99_factor_error": float(np.percentile(factor_error, 99.0)),
        "max_factor_error": float(np.max(factor_error)),
    }


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones(x.shape[0], dtype=float)]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(x.shape[1]), deg):
            col = np.ones(x.shape[0], dtype=float)
            for idx in combo:
                col *= x[:, idx]
            cols.append(col)
    return np.column_stack(cols)


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    degree: int = 6,
    alpha: float = 1e-4,
) -> np.ndarray:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma < 1e-14, 1.0, sigma)
    x_train_s = (x_train - mu) / sigma
    x_test_s = (x_test - mu) / sigma
    phi_train = polynomial_features(x_train_s, degree)
    reg = np.eye(phi_train.shape[1]) * alpha
    reg[0, 0] = 0.0
    weights = np.linalg.solve(phi_train.T @ phi_train + reg, phi_train.T @ y_train)
    return polynomial_features(x_test_s, degree) @ weights


def coarse_holdout_points(grid: GridData) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    train_indices = [np.arange(0, len(axis), 2, dtype=int) for axis in grid.axes]
    max_inside = [idx[-1] for idx in train_indices]
    full_index_axes = [np.arange(0, max_i + 1, dtype=int) for max_i in max_inside]
    index_mesh = np.meshgrid(*full_index_axes, indexing="ij")
    is_train_node = np.ones(index_mesh[0].shape, dtype=bool)
    for mesh_axis in index_mesh:
        is_train_node &= (mesh_axis % 2) == 0
    test_mask = ~is_train_node
    test_index = tuple(mesh_axis[test_mask] for mesh_axis in index_mesh)
    test_points = np.column_stack(
        [grid.axes[dim][test_index[dim]] for dim in range(3)]
    )
    test_log_true = grid.log_lnu_grid[test_index]
    return train_indices, test_points, test_log_true, test_index


def benchmark_models(grid: GridData) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    train_indices, test_points, test_log_true, _ = coarse_holdout_points(grid)
    train_axes = [axis[idx] for axis, idx in zip(grid.axes, train_indices)]
    train_log_grid = grid.log_lnu_grid[np.ix_(*train_indices)]

    rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    for method in ["nearest", "linear", "cubic"]:
        name = f"grid_{method}_log10"
        t0 = perf_counter()
        interpolator = make_interpolator(train_axes, train_log_grid, method, allow_extrapolate=False)
        pred_log = interpolator(test_points)
        elapsed = perf_counter() - t0
        row = {"model": name, "seconds": elapsed}
        row.update(metric_dict(test_log_true, pred_log))
        rows.append(row)
        predictions[name] = pred_log

    mesh = np.meshgrid(*train_axes, indexing="ij")
    x_train = np.column_stack([m.ravel() for m in mesh])
    y_train = train_log_grid.ravel()
    t0 = perf_counter()
    pred_log = ridge_predict(x_train, y_train, test_points, degree=6, alpha=1e-4)
    elapsed = perf_counter() - t0
    row = {"model": "poly6_ridge_log10", "seconds": elapsed}
    row.update(metric_dict(test_log_true, pred_log))
    rows.append(row)
    predictions["poly6_ridge_log10"] = pred_log

    rows.sort(key=lambda item: (float(item["smape_percent"]), float(item["log10_mae"])))
    return rows, {"test_points": test_points, "test_log_true": test_log_true, **predictions}


def save_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_artifact(path: Path, grid: GridData, method: str, metadata: dict) -> None:
    np.savez_compressed(
        path,
        axis_rod=grid.axes[0],
        axis_tep=grid.axes[1],
        axis_tgama=grid.axes[2],
        log_lnu_grid=grid.log_lnu_grid,
        metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2),
    )


def plot_diagnostics(output_dir: Path, grid: GridData, bench_cache: dict[str, np.ndarray], best_model: str) -> None:
    lnu = grid.lnu_grid.ravel()
    plt.figure(figsize=(8, 4.8))
    plt.hist(np.log10(lnu), bins=80, color="#386cb0", alpha=0.85)
    plt.xlabel("log10(lnu)")
    plt.ylabel("count")
    plt.title("Free Path Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_lnu_log_distribution.png", dpi=160)
    plt.close()

    true = bench_cache["test_log_true"]
    pred = bench_cache[best_model]
    sample_n = min(25000, true.size)
    rng = np.random.default_rng(20260617)
    idx = rng.choice(true.size, size=sample_n, replace=False)
    lo = min(float(true[idx].min()), float(pred[idx].min()))
    hi = max(float(true[idx].max()), float(pred[idx].max()))
    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(true[idx], pred[idx], s=3, alpha=0.25, color="#386cb0", linewidths=0)
    plt.plot([lo, hi], [lo, hi], color="#d95f02", linewidth=1.2)
    plt.xlabel("true log10(lnu)")
    plt.ylabel("predicted log10(lnu)")
    plt.title(f"Coarse-grid Holdout: {best_model}")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_holdout_true_vs_pred.png", dpi=170)
    plt.close()


def write_report(path: Path, summary: dict, benchmark_rows: list[dict[str, float | str]]) -> None:
    grid = summary["grid"]
    best = summary["selected_model"]
    best_metrics = next(row for row in benchmark_rows if row["model"] == best)

    lines = [
        "# 自由程代理模型方法报告",
        "",
        "## 1. 数据与目标",
        "",
        f"- 数据文件：`{summary['data_path']}`",
        "- 列含义：`rod tep tgama lnu`，其中 `lnu` 是 Fortran 程序中 `lnu = integral / rod` 得到的自由程。",
        f"- 样本数：{grid['n_samples']}，规则网格：{grid['axis_sizes'][0]} × {grid['axis_sizes'][1]} × {grid['axis_sizes'][2]}。",
        f"- 输入范围：rod [{grid['ranges']['rod'][0]:.6g}, {grid['ranges']['rod'][1]:.6g}]，"
        f"tep [{grid['ranges']['tep'][0]:.6g}, {grid['ranges']['tep'][1]:.6g}]，"
        f"tgama [{grid['ranges']['tgama'][0]:.6g}, {grid['ranges']['tgama'][1]:.6g}]。",
        f"- 自由程范围：lnu [{grid['ranges']['lnu'][0]:.6e}, {grid['ranges']['lnu'][1]:.6e}]。",
        "",
        "## 2. 选择的方法",
        "",
        "最适合当前任务的方法是 **规则三维网格上的 log10(lnu) 三次插值**：",
        "",
        "1. 数据本身是完整的 50×50×50 规则网格，插值比神经网络或随机森林更直接。",
        "2. 自由程始终为正，并且跨很多数量级，所以先学习 `log10(lnu)`，预测后再还原为 `lnu`。",
        "3. 插值模型在已有网格点上可精确复现原始计算结果；在网格内部新点上速度快、无训练随机性。",
        "4. 三次插值在粗网格留出测试中优于线性插值和全局多项式 Ridge。",
        "",
        "## 3. 测试设计",
        "",
        "- 完整网格用于最终模型保存。",
        "- 方法比较时，只取每隔一个网格点作为训练网格，即 25×25×25。",
        "- 其余内部网格点作为测试集，共 102024 个点。",
        "- 评价指标包括 log10 空间误差、原始自由程 MAPE/SMAPE，以及倍数误差。",
        "",
        "## 4. 测试结果",
        "",
        "| 方法 | log10 MAE | log10 RMSE | MAPE | SMAPE | 中位倍数误差 | P90 倍数误差 | 用时(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in benchmark_rows:
        lines.append(
            f"| {row['model']} | {float(row['log10_mae']):.6f} | "
            f"{float(row['log10_rmse']):.6f} | {float(row['mape_percent']):.4f}% | "
            f"{float(row['smape_percent']):.4f}% | {float(row['median_factor_error']):.6f} | "
            f"{float(row['p90_factor_error']):.6f} | {float(row['seconds']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 结论",
            "",
            f"- 选定模型：`{best}`。",
            f"- 粗网格留出测试：log10 MAE = {float(best_metrics['log10_mae']):.6f}，"
            f"SMAPE = {float(best_metrics['smape_percent']):.4f}%，"
            f"P90 倍数误差 = {float(best_metrics['p90_factor_error']):.6f}。",
            "- 该模型适合在训练数据范围内做快速插值预测；不建议把它用于范围外强外推。",
            "",
            "## 6. 使用方法",
            "",
            "训练并重新生成报告：",
            "",
            "```bash",
            f"cd {ROOT}",
            "python train_free_path_model.py",
            "```",
            "",
            "输入参数预测自由程：",
            "",
            "```bash",
            "python predict_free_path.py -1.870 1.771 1.755",
            "```",
            "",
            "如果输入的是物理量而模型文件使用的是对数坐标，可加：",
            "",
            "```bash",
            "python predict_free_path.py 0.013489628825916533 59.02010801718442 56.88529308438413 --log10-input",
            "```",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(grid: GridData, selected_model: str, method: str, benchmark_rows: list[dict[str, float | str]]) -> dict:
    return {
        "data_path": str(grid.path),
        "selected_model": selected_model,
        "interpolation_method": method,
        "target_transform": "log10(lnu)",
        "input_columns": COLUMNS[:3],
        "target_column": COLUMNS[3],
        "grid": {
            "n_samples": int(grid.data.shape[0]),
            "axis_sizes": [int(len(axis)) for axis in grid.axes],
            "ranges": {
                "rod": [float(grid.axes[0].min()), float(grid.axes[0].max())],
                "tep": [float(grid.axes[1].min()), float(grid.axes[1].max())],
                "tgama": [float(grid.axes[2].min()), float(grid.axes[2].max())],
                "lnu": [float(grid.lnu_grid.min()), float(grid.lnu_grid.max())],
            },
        },
        "benchmark": benchmark_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test a Rosseland free-path surrogate model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="4-column data file: rod tep tgama lnu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="directory for model and reports")
    parser.add_argument(
        "--method",
        choices=["auto", "linear", "cubic"],
        default="auto",
        help="final interpolation method; auto selects the best grid interpolator from the benchmark",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid_data(args.data)
    benchmark_rows, bench_cache = benchmark_models(grid)

    grid_rows = [row for row in benchmark_rows if str(row["model"]).startswith("grid_")]
    selected_row = min(grid_rows, key=lambda row: (float(row["smape_percent"]), float(row["log10_mae"])))
    if args.method == "auto":
        method = str(selected_row["model"]).split("_")[1]
    else:
        method = args.method
        selected_row = next(row for row in grid_rows if row["model"] == f"grid_{method}_log10")
    selected_model = f"grid_{method}_log10"

    summary = build_summary(grid, selected_model, method, benchmark_rows)
    metadata = {
        "model_type": "scipy.interpolate.RegularGridInterpolator",
        "method": method,
        "target_transform": "log10",
        "columns": COLUMNS,
        "data_path": str(args.data),
        "selected_benchmark": selected_row,
        "reliability_note": "Use inside the training coordinate ranges. Extrapolation is disabled by default.",
    }

    stem = args.data.stem
    artifact_path = output_dir / f"free_path_model_{stem}_{method}.npz"
    summary_path = output_dir / f"free_path_metrics_{stem}.json"
    benchmark_path = output_dir / f"free_path_benchmark_{stem}.csv"
    report_path = ROOT / "free_path_method_report.md"

    save_artifact(artifact_path, grid, method, metadata)
    summary["artifact_path"] = str(artifact_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(benchmark_path, benchmark_rows)
    plot_diagnostics(output_dir, grid, bench_cache, selected_model)
    write_report(report_path, summary, benchmark_rows)

    print("===== Free-path model training complete =====")
    print(f"data: {args.data}")
    print(f"selected model: {selected_model}")
    print(f"artifact: {artifact_path}")
    print(f"metrics: {summary_path}")
    print(f"benchmark: {benchmark_path}")
    print(f"report: {report_path}")
    print(
        "best coarse-holdout: "
        f"log10_mae={float(selected_row['log10_mae']):.6f}, "
        f"smape={float(selected_row['smape_percent']):.4f}%, "
        f"p90_factor={float(selected_row['p90_factor_error']):.6f}"
    )


if __name__ == "__main__":
    main()
