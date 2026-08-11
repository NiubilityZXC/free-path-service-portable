#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Old-Au CatBoost ensemble and blend experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

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
OUT_JSON = OUTPUT_DIR / "old_au_catboost_ensemble_results.json"
OUT_MD = ROOT / "old_au_catboost_ensemble_report.md"


def load_old_train() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(OLD_STANDARD)
    data = data[data[:, 0] == 79.0]
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    keep = np.array(
        [standard_row_key(79.0, row[1:4]) not in bench_keys and standard_row_key(79.0, row[1:4]) not in extra_keys for row in data],
        dtype=bool,
    )
    return data[keep, 1:4], np.log10(data[keep, 4])


def load_au_benchmark(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    mask = payload["element"].astype(str) == "Z_79"
    return payload["x_model"][mask, 1:4], payload["y_log"][mask]


def augment(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if mode == "raw":
        return x
    r, t, g = x[:, 0], x[:, 1], x[:, 2]
    cols = [
        r,
        t,
        g,
        r * r,
        t * t,
        g * g,
        r * t,
        r * g,
        t * g,
        r * t * g,
        np.sin(r),
        np.cos(r),
        np.sin(t),
        np.cos(t),
        np.sin(g),
        np.cos(g),
    ]
    return np.column_stack(cols)


def smape_mean(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    true = 10.0**y_true_log
    pred = 10.0**y_pred_log
    return float(np.mean(2.0 * np.abs(pred - true) / np.maximum(np.abs(pred) + np.abs(true), 1e-300)) * 100.0)


def main() -> None:
    from catboost import CatBoostRegressor

    t0 = time.time()
    x_all, y_all = load_old_train()
    rng = np.random.default_rng(20260703)
    order = rng.permutation(x_all.shape[0])
    val_n = 12000
    val_idx = order[:val_n]
    fit_idx = order[val_n:]
    x_fit, y_fit = x_all[fit_idx], y_all[fit_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_bench, y_bench = load_au_benchmark(BENCHMARK_DATA_PATH)
    x_extra, y_extra = load_au_benchmark(EXTRAPOLATION_DATA_PATH)
    specs = [
        ("d11_mae_lr025_l2_10_seed03", "raw", "MAE", 11, 0.025, 10000, 10.0, 20260703),
        ("d11_mae_lr022_l2_14_seed04", "raw", "MAE", 11, 0.022, 12000, 14.0, 20260704),
        ("d10_mae_lr030_l2_08_seed05", "raw", "MAE", 10, 0.030, 10000, 8.0, 20260705),
        ("d12_mae_lr018_l2_16_seed06", "raw", "MAE", 12, 0.018, 12000, 16.0, 20260706),
        ("d10_rmse_lr035_l2_08_seed07", "raw", "RMSE", 10, 0.035, 9000, 8.0, 20260707),
        ("aug_d10_mae_lr030_l2_10_seed08", "aug", "MAE", 10, 0.030, 9000, 10.0, 20260708),
        ("aug_d11_mae_lr022_l2_14_seed09", "aug", "MAE", 11, 0.022, 10000, 14.0, 20260709),
    ]
    pred_val = []
    pred_bench = []
    pred_extra = []
    rows = []
    for name, mode, loss, depth, lr, iterations, l2, seed in specs:
        print(f"fit {name}", flush=True)
        start = time.time()
        model = CatBoostRegressor(
            loss_function=loss,
            depth=depth,
            learning_rate=lr,
            iterations=iterations,
            l2_leaf_reg=l2,
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(augment(x_fit, mode), y_fit)
        pv = np.asarray(model.predict(augment(x_val, mode)), dtype=float).reshape(-1)
        pb = np.asarray(model.predict(augment(x_bench, mode)), dtype=float).reshape(-1)
        pe = np.asarray(model.predict(augment(x_extra, mode)), dtype=float).reshape(-1)
        pred_val.append(pv)
        pred_bench.append(pb)
        pred_extra.append(pe)
        row = {
            "method": name,
            "mode": mode,
            "validation": metric_dict(y_val, pv),
            "benchmark": metric_dict(y_bench, pb),
            "extrapolation": metric_dict(y_extra, pe),
            "train_seconds": time.time() - start,
        }
        rows.append(row)
        print(
            f"{name}: val={row['validation']['smape_percent']:.6f}% "
            f"bench={row['benchmark']['smape_percent']:.6f}% extra={row['extrapolation']['smape_percent']:.6f}%",
            flush=True,
        )

    val_stack = np.column_stack(pred_val)
    bench_stack = np.column_stack(pred_bench)
    extra_stack = np.column_stack(pred_extra)
    mean_b = bench_stack.mean(axis=1)
    mean_e = extra_stack.mean(axis=1)
    median_b = np.median(bench_stack, axis=1)
    median_e = np.median(extra_stack, axis=1)
    weights, _ = nnls(val_stack, y_val)
    if weights.sum() <= 0:
        weights = np.ones(len(specs)) / len(specs)
    else:
        weights = weights / weights.sum()
    nnls_b = bench_stack @ weights
    nnls_e = extra_stack @ weights
    ensemble_rows = [
        {
            "method": "old_au_catboost_mean",
            "benchmark": metric_dict(y_bench, mean_b),
            "extrapolation": metric_dict(y_extra, mean_e),
            "weights": [1.0 / len(specs)] * len(specs),
        },
        {
            "method": "old_au_catboost_median",
            "benchmark": metric_dict(y_bench, median_b),
            "extrapolation": metric_dict(y_extra, median_e),
            "weights": None,
        },
        {
            "method": "old_au_catboost_nnls_validation_blend",
            "benchmark": metric_dict(y_bench, nnls_b),
            "extrapolation": metric_dict(y_extra, nnls_e),
            "weights": weights.tolist(),
        },
    ]
    all_rows = rows + ensemble_rows
    all_rows.sort(key=lambda r: (r["benchmark"]["smape_percent"], r["extrapolation"]["smape_percent"]))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(OLD_STANDARD),
        "fit_rows": int(x_fit.shape[0]),
        "validation_rows": int(x_val.shape[0]),
        "specs": [s[0] for s in specs],
        "results": all_rows,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 旧 Au CatBoost 集成实验报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 训练行数：{payload['fit_rows']}",
        f"- 内部验证行数：{payload['validation_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | 方法 | Au benchmark SMAPE | Au extrap SMAPE | benchmark log10 MAE | P99 倍数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(all_rows, 1):
        lines.append(
            f"| {i} | `{row['method']}` | {row['benchmark']['smape_percent']:.6f}% | "
            f"{row['extrapolation']['smape_percent']:.6f}% | {row['benchmark']['log10_mae']:.8f} | "
            f"{row['benchmark']['p99_factor_error']:.6f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
