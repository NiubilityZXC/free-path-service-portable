#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a two-dataset hybrid surrogate for Rosseland free path.

Each input file has four columns:

    rod  tep  tgama  lnu

The two files describe different data sources/material settings, so this
script trains one sub-model per file and stores them in one artifact.  Inside
the training domain it uses the best grid interpolator selected by a coarse
holdout test.  Outside the domain it uses a separate smoothed linear model
selected by a boundary holdout test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "free_path_model_outputs"
DEFAULT_DATASETS = [
    ("data_Al", ROOT / "data_Al.txt"),
    ("data", ROOT / "data.txt"),
]
COLUMNS = ["rod", "tep", "tgama", "lnu"]
INTERP_METHODS = ["linear", "cubic"]
INTERP_SIGMAS = [0.0, 0.15, 0.3, 0.5]
ENSEMBLE_WEIGHTS = np.linspace(-0.1, 1.1, 241)
EXTRAP_SIGMAS = [0.0, 0.3, 0.5, 0.75, 1.0]
EXTRAP_CLIP_MARGINS = [None, 0.0, 0.25, 0.5, 1.0, 2.0]


@dataclass
class DatasetModel:
    name: str
    path: Path
    raw_data: np.ndarray
    model_data: np.ndarray
    axes: list[np.ndarray]
    lnu_grid: np.ndarray
    log_lnu_grid: np.ndarray
    coordinate_transform: str
    repaired_lnu_count: int


def metric_dict(log_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(log_true) & np.isfinite(log_pred)
    log_true = log_true[valid]
    log_pred = log_pred[valid]
    true = np.power(10.0, log_true)
    pred = np.power(10.0, log_pred)
    log_error = log_pred - log_true
    with np.errstate(over="ignore", invalid="ignore"):
        factor_error = np.power(10.0, np.abs(log_error))
    smape = 2.0 * np.abs(pred - true) / np.maximum(np.abs(pred) + np.abs(true), 1e-300)
    mape = np.abs(pred - true) / np.maximum(np.abs(true), 1e-300)
    finite_factor = factor_error[np.isfinite(factor_error)]
    max_factor = float(np.max(finite_factor)) if finite_factor.size else float("inf")
    return {
        "n": int(log_true.size),
        "log10_mae": float(np.mean(np.abs(log_error))),
        "log10_rmse": float(np.sqrt(np.mean(log_error**2))),
        "mape_percent": float(np.mean(mape) * 100.0),
        "smape_percent": float(np.mean(smape) * 100.0),
        "median_factor_error": float(np.median(factor_error)),
        "p90_factor_error": float(np.percentile(factor_error, 90.0)),
        "p99_factor_error": float(np.percentile(factor_error, 99.0)),
        "max_finite_factor_error": max_factor,
    }


def infer_coordinate_transform(raw_inputs: np.ndarray) -> str:
    """Use log10 coordinates for wide positive physical grids."""
    if np.all(raw_inputs > 0.0):
        ratios = raw_inputs.max(axis=0) / raw_inputs.min(axis=0)
        if np.all(ratios > 20.0):
            return "log10"
    return "identity"


def apply_coordinate_transform(raw_inputs: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return raw_inputs.astype(float, copy=True)
    if transform == "log10":
        if np.any(raw_inputs <= 0.0):
            raise ValueError("log10 coordinate transform requires positive inputs")
        return np.log10(raw_inputs)
    raise ValueError(f"unknown coordinate transform: {transform}")


def repair_nonpositive_grid(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Repair rare non-positive Monte Carlo artifacts using local positive medians."""
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
        if not vals.size:
            raise ValueError("target column has no positive values")
        repaired[bad] = float(np.median(vals))

    return repaired, bad_count


def load_dataset(name: str, path: Path, coordinate_transform: str = "auto") -> DatasetModel:
    raw = np.loadtxt(path, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(f"{path} must contain exactly four columns: {COLUMNS}")
    if not np.isfinite(raw[:, :3]).all():
        raise ValueError(f"{path} contains non-finite input coordinates")

    transform = infer_coordinate_transform(raw[:, :3]) if coordinate_transform == "auto" else coordinate_transform
    model_inputs = apply_coordinate_transform(raw[:, :3], transform)
    model_data = np.column_stack([model_inputs, raw[:, 3]])

    order = np.lexsort((model_data[:, 2], model_data[:, 1], model_data[:, 0]))
    model_data = model_data[order]
    raw_sorted = raw[order]
    axes = [np.unique(model_data[:, i]) for i in range(3)]
    expected_n = math.prod(len(axis) for axis in axes)
    if expected_n != raw.shape[0]:
        raise ValueError(f"{path} is not a complete rectangular grid")
    if np.unique(model_data[:, :3], axis=0).shape[0] != raw.shape[0]:
        raise ValueError(f"{path} has duplicated transformed coordinates")

    mesh = np.meshgrid(*axes, indexing="ij")
    expected = np.column_stack([m.ravel() for m in mesh])
    if not np.allclose(model_data[:, :3], expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path} cannot be reshaped as a regular grid")

    shape = tuple(len(axis) for axis in axes)
    lnu_grid, repaired_count = repair_nonpositive_grid(model_data[:, 3].reshape(shape))
    log_lnu_grid = np.log10(lnu_grid)
    return DatasetModel(
        name=name,
        path=path,
        raw_data=raw_sorted,
        model_data=model_data,
        axes=axes,
        lnu_grid=lnu_grid,
        log_lnu_grid=log_lnu_grid,
        coordinate_transform=transform,
        repaired_lnu_count=repaired_count,
    )


def make_interpolator(
    axes: list[np.ndarray],
    log_lnu_grid: np.ndarray,
    method: str,
    allow_extrapolate: bool,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        axes,
        log_lnu_grid,
        method=method,
        bounds_error=not allow_extrapolate,
        fill_value=None if allow_extrapolate else np.nan,
    )


def maybe_smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return grid
    return gaussian_filter(grid, sigma=sigma, mode="nearest")


def maybe_clip(pred: np.ndarray, train_grid: np.ndarray, clip_margin: float | None) -> np.ndarray:
    if clip_margin is None:
        return pred
    lo = float(np.min(train_grid)) - clip_margin
    hi = float(np.max(train_grid)) + clip_margin
    return np.clip(pred, lo, hi)


def coarse_interpolation_split(model: DatasetModel) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    train_indices = [np.arange(0, len(axis), 2, dtype=int) for axis in model.axes]
    train_axes = [axis[index] for axis, index in zip(model.axes, train_indices)]
    train_grid = model.log_lnu_grid[np.ix_(*train_indices)]
    index_axes = [np.arange(0, index[-1] + 1, dtype=int) for index in train_indices]
    mesh = np.meshgrid(*index_axes, indexing="ij")
    is_train = np.ones(mesh[0].shape, dtype=bool)
    for item in mesh:
        is_train &= (item % 2) == 0
    test_mask = ~is_train
    test_index = tuple(item[test_mask] for item in mesh)
    test_points = np.column_stack([model.axes[d][test_index[d]] for d in range(3)])
    test_true = model.log_lnu_grid[test_index]
    return train_axes, train_grid, test_points, test_true


def boundary_extrapolation_split(
    model: DatasetModel,
    margin: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    train_indices = [np.arange(margin, len(axis) - margin, dtype=int) for axis in model.axes]
    train_axes = [axis[index] for axis, index in zip(model.axes, train_indices)]
    train_grid = model.log_lnu_grid[np.ix_(*train_indices)]
    index_axes = [np.arange(len(axis), dtype=int) for axis in model.axes]
    mesh = np.meshgrid(*index_axes, indexing="ij")
    outside = np.zeros(mesh[0].shape, dtype=bool)
    for d, item in enumerate(mesh):
        outside |= (item < margin) | (item >= len(model.axes[d]) - margin)
    test_index = tuple(item[outside] for item in mesh)
    test_points = np.column_stack([model.axes[d][test_index[d]] for d in range(3)])
    test_true = model.log_lnu_grid[test_index]
    return train_axes, train_grid, test_points, test_true


def select_interpolation_model(model: DatasetModel) -> tuple[dict, list[dict]]:
    train_axes, train_grid, test_points, test_true = coarse_interpolation_split(model)
    rows = []
    base_predictions = {}
    for sigma in INTERP_SIGMAS:
        candidate_grid = maybe_smooth(train_grid, sigma)
        for method in INTERP_METHODS:
            if method == "cubic" and sigma != 0.0:
                continue
            start = perf_counter()
            interp = make_interpolator(train_axes, candidate_grid, method=method, allow_extrapolate=False)
            pred = interp(test_points)
            elapsed = perf_counter() - start
            if sigma == 0.0:
                base_predictions[method] = pred
            row = {
                "dataset": model.name,
                "task": "interpolation",
                "model": f"{method}_log10",
                "method": method,
                "smoothing_sigma": sigma,
                "clip_margin_log10": "",
                "ensemble_weight_cubic": "",
                "seconds": elapsed,
            }
            row.update(metric_dict(test_true, pred))
            rows.append(row)

    if "linear" in base_predictions and "cubic" in base_predictions:
        for weight_cubic in ENSEMBLE_WEIGHTS:
            start = perf_counter()
            pred = weight_cubic * base_predictions["cubic"] + (1.0 - weight_cubic) * base_predictions["linear"]
            elapsed = perf_counter() - start
            row = {
                "dataset": model.name,
                "task": "interpolation",
                "model": "ensemble_linear_cubic_log10",
                "method": "ensemble_linear_cubic",
                "smoothing_sigma": 0.0,
                "clip_margin_log10": "",
                "ensemble_weight_cubic": float(weight_cubic),
                "seconds": elapsed,
            }
            row.update(metric_dict(test_true, pred))
            rows.append(row)
    rows.sort(key=lambda row: (row["smape_percent"], row["log10_mae"]))
    return rows[0], rows


def select_extrapolation_model(model: DatasetModel, margin: int) -> tuple[dict, list[dict]]:
    train_axes, train_grid, test_points, test_true = boundary_extrapolation_split(model, margin=margin)
    rows = []
    for sigma in EXTRAP_SIGMAS:
        smoothed = maybe_smooth(train_grid, sigma)
        interp = make_interpolator(train_axes, smoothed, method="linear", allow_extrapolate=True)
        raw_pred = interp(test_points)
        for clip_margin in EXTRAP_CLIP_MARGINS:
            start = perf_counter()
            pred = maybe_clip(raw_pred, smoothed, clip_margin)
            elapsed = perf_counter() - start
            row = {
                "dataset": model.name,
                "task": "boundary_extrapolation",
                "model": "linear_log10_extrap",
                "method": "linear",
                "smoothing_sigma": sigma,
                "clip_margin_log10": "" if clip_margin is None else clip_margin,
                "ensemble_weight_cubic": "",
                "seconds": elapsed,
            }
            row.update(metric_dict(test_true, pred))
            rows.append(row)
    rows.sort(key=lambda row: (row["smape_percent"], row["log10_mae"], row["p99_factor_error"]))
    return rows[0], rows


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_artifact(path: Path, models: list[DatasetModel], metadata: dict) -> None:
    payload = {"metadata_json": json.dumps(metadata, ensure_ascii=False, indent=2)}
    for model in models:
        prefix = model.name
        payload[f"{prefix}_axis_rod"] = model.axes[0]
        payload[f"{prefix}_axis_tep"] = model.axes[1]
        payload[f"{prefix}_axis_tgama"] = model.axes[2]
        payload[f"{prefix}_log_lnu_grid"] = model.log_lnu_grid
    np.savez_compressed(path, **payload)


def plot_dataset_diagnostics(output_dir: Path, model: DatasetModel, selected: dict) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.hist(model.log_lnu_grid.ravel(), bins=80, color="#386cb0", alpha=0.85)
    plt.xlabel("log10(lnu)")
    plt.ylabel("count")
    plt.title(f"{model.name}: Free Path Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_{model.name}_log_lnu_distribution.png", dpi=160)
    plt.close()

    train_axes, train_grid, test_points, test_true = coarse_interpolation_split(model)
    interp_config = selected["interpolation"]
    sigma = float(interp_config.get("smoothing_sigma", 0.0) or 0.0)
    plot_grid = maybe_smooth(train_grid, sigma)
    if interp_config["method"] == "ensemble_linear_cubic":
        weight_cubic = float(interp_config["ensemble_weight_cubic"])
        linear = make_interpolator(train_axes, plot_grid, method="linear", allow_extrapolate=False)
        cubic = make_interpolator(train_axes, plot_grid, method="cubic", allow_extrapolate=False)
        pred = weight_cubic * cubic(test_points) + (1.0 - weight_cubic) * linear(test_points)
    else:
        interp = make_interpolator(
            train_axes,
            plot_grid,
            method=interp_config["method"],
            allow_extrapolate=False,
        )
        pred = interp(test_points)
    rng = np.random.default_rng(20260617)
    sample_n = min(25000, len(test_true))
    idx = rng.choice(len(test_true), size=sample_n, replace=False)
    lo = float(min(test_true[idx].min(), pred[idx].min()))
    hi = float(max(test_true[idx].max(), pred[idx].max()))
    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(test_true[idx], pred[idx], s=3, alpha=0.25, color="#386cb0", linewidths=0)
    plt.plot([lo, hi], [lo, hi], color="#d95f02", linewidth=1.2)
    plt.xlabel("true log10(lnu)")
    plt.ylabel("predicted log10(lnu)")
    plt.title(f"{model.name}: Interpolation Holdout")
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_{model.name}_interp_true_vs_pred.png", dpi=170)
    plt.close()


def write_report(path: Path, metadata: dict, benchmark_rows: list[dict]) -> None:
    lines = [
        "# 双数据集自由程代理模型方法报告",
        "",
        "## 1. 总体方法",
        "",
        "两份训练数据不直接硬合并为一个三输入函数，而是训练成一个统一工件里的两个子模型。",
        "原因是两份数据对应的坐标体系和物理对象不同；如果不加入数据源/材料标识，直接合并会把同一类坐标映射到互相冲突的自由程。",
        "",
        "最终采用 **自适应混合网格模型**：",
        "",
        "1. 目标统一在 `log10(lnu)` 空间建模，保证自由程预测为正并适应跨数量级变化。",
        "2. 网格范围内用粗网格留出测试选择最佳内插器。",
        "3. 网格范围外用边界留出测试选择外推器；三次外推容易发散，因此外推候选限定为 log 空间线性斜率外推，并允许轻微平滑和输出保护。",
        "4. `data.txt` 中少量非正自由程点被视为数值积分噪声，用邻域正值中位数修复后再取 log。",
        "",
        "## 2. 数据集",
        "",
    ]

    for name, info in metadata["datasets"].items():
        ranges = info["ranges"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 文件：`{info['path']}`",
                f"- 样本数：{info['n_samples']}，网格：{info['axis_sizes'][0]} × {info['axis_sizes'][1]} × {info['axis_sizes'][2]}",
                f"- 坐标变换：`{info['coordinate_transform']}`",
                f"- 修复非正自由程点数：{info['repaired_lnu_count']}",
                f"- 模型坐标范围：rod [{ranges['rod'][0]:.6g}, {ranges['rod'][1]:.6g}]，"
                f"tep [{ranges['tep'][0]:.6g}, {ranges['tep'][1]:.6g}]，"
                f"tgama [{ranges['tgama'][0]:.6g}, {ranges['tgama'][1]:.6g}]",
                f"- 自由程范围：lnu [{ranges['lnu'][0]:.6e}, {ranges['lnu'][1]:.6e}]",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 选定模型",
            "",
            "| 数据集 | 内插方法 | 内插 SMAPE | 内插 log10 MAE | 外推方法 | 外推平滑 sigma | 外推裁剪 margin | 外推 SMAPE | 外推 log10 MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, selected in metadata["selected_models"].items():
        interp = selected["interpolation"]
        extrap = selected["extrapolation"]
        clip = extrap["clip_margin_log10"]
        clip_text = "None" if clip == "" else str(clip)
        interp_method = str(interp["method"])
        if interp_method == "ensemble_linear_cubic":
            interp_method = f"ensemble(w_cubic={float(interp['ensemble_weight_cubic']):.3f})"
        lines.append(
            f"| {name} | {interp_method} | {interp['smape_percent']:.4f}% | "
            f"{interp['log10_mae']:.6f} | {extrap['method']} | "
            f"{extrap['smoothing_sigma']} | {clip_text} | "
            f"{extrap['smape_percent']:.4f}% | {extrap['log10_mae']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 测试方式",
            "",
            "- 内插测试：每隔一个网格点作为训练网格，其余内部点作为测试点。",
            f"- 外推测试：只用内部网格训练，留出外层 {metadata['extrapolation_margin']} 层边界点测试。",
            "- 外推指标只能代表边界附近有限外推；离训练范围很远的强外推仍需新增计算点验证。",
            "",
            "## 5. 使用方法",
            "",
            "训练两份数据：",
            "",
            "```bash",
            f"cd {ROOT}",
            "python train_two_dataset_free_path_model.py",
            "```",
            "",
            "预测 `data_Al` 模型坐标中的自由程：",
            "",
            "```bash",
            "python predict_two_dataset_free_path.py --dataset data_Al -1.870 1.771 1.755",
            "```",
            "",
            "`data.txt` 会自动把原始物理坐标转成 log10 模型坐标：",
            "",
            "```bash",
            "python predict_two_dataset_free_path.py --dataset data 600.000098 4000.0294 600.0098",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_dataset_args(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return DEFAULT_DATASETS
    datasets = []
    for value in values:
        if ":" not in value:
            raise ValueError("--dataset entries must be NAME:PATH")
        name, path = value.split(":", 1)
        datasets.append((name, Path(path)))
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid free-path model for two or more datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="dataset spec NAME:PATH; default uses data_Al.txt and data.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extrapolation-margin", type=int, default=4)
    parser.add_argument("--artifact-name", default="free_path_two_dataset_hybrid_model.npz")
    parser.add_argument("--metrics-name", default="free_path_two_dataset_metrics.json")
    parser.add_argument("--benchmark-name", default="free_path_two_dataset_benchmark.csv")
    parser.add_argument("--report-name", default="two_dataset_free_path_method_report.md")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_specs = parse_dataset_args(args.dataset)

    models = []
    all_rows = []
    metadata = {
        "artifact_type": "two_dataset_hybrid_free_path_model",
        "target_transform": "log10(lnu)",
        "input_columns": COLUMNS[:3],
        "target_column": COLUMNS[3],
        "extrapolation_margin": args.extrapolation_margin,
        "datasets": {},
        "selected_models": {},
    }

    for name, path in dataset_specs:
        model = load_dataset(name, path)
        models.append(model)
        interp_selected, interp_rows = select_interpolation_model(model)
        extrap_selected, extrap_rows = select_extrapolation_model(model, margin=args.extrapolation_margin)
        all_rows.extend(interp_rows)
        all_rows.extend(extrap_rows)
        metadata["datasets"][name] = {
            "path": str(path),
            "n_samples": int(model.raw_data.shape[0]),
            "axis_sizes": [int(len(axis)) for axis in model.axes],
            "coordinate_transform": model.coordinate_transform,
            "repaired_lnu_count": int(model.repaired_lnu_count),
            "ranges": {
                "rod": [float(model.axes[0].min()), float(model.axes[0].max())],
                "tep": [float(model.axes[1].min()), float(model.axes[1].max())],
                "tgama": [float(model.axes[2].min()), float(model.axes[2].max())],
                "lnu": [float(model.lnu_grid.min()), float(model.lnu_grid.max())],
            },
        }
        metadata["selected_models"][name] = {
            "interpolation": interp_selected,
            "extrapolation": extrap_selected,
        }
        plot_dataset_diagnostics(args.output_dir, model, metadata["selected_models"][name])

    artifact_path = args.output_dir / args.artifact_name
    metrics_path = args.output_dir / args.metrics_name
    benchmark_path = args.output_dir / args.benchmark_name
    report_path = ROOT / args.report_name

    metadata["artifact_path"] = str(artifact_path)
    save_artifact(artifact_path, models, metadata)
    metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(benchmark_path, all_rows)
    write_report(report_path, metadata, all_rows)

    print("===== Two-dataset free-path model complete =====")
    print(f"artifact: {artifact_path}")
    print(f"metrics: {metrics_path}")
    print(f"benchmark: {benchmark_path}")
    print(f"report: {report_path}")
    for name, selected in metadata["selected_models"].items():
        interp = selected["interpolation"]
        extrap = selected["extrapolation"]
        interp_detail = interp["method"]
        if interp["method"] == "ensemble_linear_cubic":
            interp_detail = f"ensemble_linear_cubic(w_cubic={float(interp['ensemble_weight_cubic']):.3f})"
        print(
            f"{name}: interp={interp_detail} "
            f"SMAPE={interp['smape_percent']:.4f}%, "
            f"extrap={extrap['method']} sigma={extrap['smoothing_sigma']} "
            f"clip={extrap['clip_margin_log10'] or 'None'} "
            f"SMAPE={extrap['smape_percent']:.4f}%"
        )


if __name__ == "__main__":
    main()
