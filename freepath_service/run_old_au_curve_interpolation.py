#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Old-Au rod-curve interpolation/extrapolation experiment.

For each fixed (tep, tgama) curve, fit a 1D interpolator along rod using only
non-holdout old-Au grid points. This exploits the original 50x50x50 table
structure and does not use frozen benchmark/extrapolation target values.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, UnivariateSpline, interp1d

from unified_free_path_core import (
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    metric_dict,
    standard_row_key,
)


ROOT = Path(__file__).resolve().parent
OLD_STANDARD = OUTPUT_DIR / "unified_standard_training_data.bak_20260702_170959.txt"
OUT_JSON = OUTPUT_DIR / "old_au_curve_interpolation_results.json"
OUT_MD = ROOT / "old_au_curve_interpolation_report.md"


def curve_key(row: np.ndarray) -> tuple[float, float]:
    return (round(float(row[2]), 12), round(float(row[3]), 12))


def load_old_curves() -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    data = np.loadtxt(OLD_STANDARD)
    data = data[data[:, 0] == 79.0]
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    curves: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row in data:
        key = standard_row_key(79.0, row[1:4])
        if key in bench_keys or key in extra_keys:
            continue
        curves.setdefault(curve_key(row), []).append((float(row[1]), float(np.log10(row[4]))))
    out = {}
    for key, values in curves.items():
        arr = np.asarray(sorted(values), dtype=float)
        # Duplicate rod values are not expected, but average defensively.
        rods = np.unique(arr[:, 0])
        ys = np.array([arr[arr[:, 0] == r, 1].mean() for r in rods], dtype=float)
        out[key] = (rods, ys)
    return out


def load_au(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    mask = payload["element"].astype(str) == "Z_79"
    return payload["x_model"][mask], payload["y_log"][mask]


def linear_extrap(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    out = np.interp(query, x, y)
    low = query < x[0]
    high = query > x[-1]
    if x.size >= 2:
        left_slope = (y[1] - y[0]) / max(x[1] - x[0], 1e-12)
        right_slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-12)
        out[low] = y[0] + left_slope * (query[low] - x[0])
        out[high] = y[-1] + right_slope * (query[high] - x[-1])
    return out


def fit_predict_curve(x: np.ndarray, y: np.ndarray, query: np.ndarray, method: str) -> np.ndarray:
    if x.size < 4 and method not in {"linear", "nearest"}:
        method = "linear"
    if method == "nearest":
        idx = np.argmin(np.abs(query[:, None] - x[None, :]), axis=1)
        return y[idx]
    if method == "linear":
        return linear_extrap(x, y, query)
    if method == "interp_cubic":
        return interp1d(x, y, kind="cubic", fill_value="extrapolate", assume_sorted=True)(query)
    if method == "pchip":
        return PchipInterpolator(x, y, extrapolate=True)(query)
    if method == "akima":
        return Akima1DInterpolator(x, y, extrapolate=True)(query)
    if method == "cubic_notaknot":
        return CubicSpline(x, y, bc_type="not-a-knot", extrapolate=True)(query)
    if method == "cubic_natural":
        return CubicSpline(x, y, bc_type="natural", extrapolate=True)(query)
    if method.startswith("spline_s"):
        s = float(method[len("spline_s") :])
        return UnivariateSpline(x, y, k=3, s=s, ext=0)(query)
    raise ValueError(method)


def predict_rows(curves: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]], rows: np.ndarray, method: str) -> np.ndarray:
    pred = np.empty(rows.shape[0], dtype=float)
    grouped: dict[tuple[float, float], list[int]] = {}
    for i, row in enumerate(rows):
        grouped.setdefault(curve_key(row), []).append(i)
    missing = 0
    for key, indices in grouped.items():
        idx = np.asarray(indices, dtype=int)
        if key not in curves:
            missing += idx.size
            pred[idx] = np.nan
            continue
        x, y = curves[key]
        pred[idx] = fit_predict_curve(x, y, rows[idx, 1], method)
    if missing:
        raise RuntimeError(f"{missing} rows have no matching curve")
    return pred


def main() -> None:
    t0 = time.time()
    curves = load_old_curves()
    bench_x, bench_y = load_au(BENCHMARK_DATA_PATH)
    extra_x, extra_y = load_au(EXTRAPOLATION_DATA_PATH)
    methods = [
        "nearest",
        "linear",
        "interp_cubic",
        "pchip",
        "akima",
        "cubic_notaknot",
        "cubic_natural",
        "spline_s0",
        "spline_s0.0001",
        "spline_s0.001",
        "spline_s0.01",
    ]
    results = []
    print(f"curves={len(curves)}, bench={bench_x.shape[0]}, extra={extra_x.shape[0]}")
    for method in methods:
        start = time.time()
        pred_b = predict_rows(curves, bench_x, method)
        pred_e = predict_rows(curves, extra_x, method)
        row = {
            "method": method,
            "benchmark": metric_dict(bench_y, pred_b),
            "extrapolation": metric_dict(extra_y, pred_e),
            "seconds": time.time() - start,
        }
        results.append(row)
        print(
            f"{method}: bench={row['benchmark']['smape_percent']:.6f}% "
            f"extra={row['extrapolation']['smape_percent']:.6f}%",
            flush=True,
        )
    results.sort(key=lambda r: (r["benchmark"]["smape_percent"], r["extrapolation"]["smape_percent"]))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(OLD_STANDARD),
        "n_curves": len(curves),
        "results": results,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 旧 Au rod 曲线插值实验报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 曲线数量：{payload['n_curves']}",
        "- 每条曲线固定 `(tep,tgama)`，只用非测试 old-Au 点沿 rod 方向插值/外推。",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | 方法 | Au benchmark SMAPE | Au extrap SMAPE | benchmark log10 MAE | extrap log10 MAE | P99 倍数 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(results, 1):
        lines.append(
            f"| {i} | `{row['method']}` | {row['benchmark']['smape_percent']:.6f}% | "
            f"{row['extrapolation']['smape_percent']:.6f}% | {row['benchmark']['log10_mae']:.8f} | "
            f"{row['extrapolation']['log10_mae']:.8f} | {row['benchmark']['p99_factor_error']:.6f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
