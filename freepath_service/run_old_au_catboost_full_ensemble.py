#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Full-data old-Au CatBoost ensemble experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

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
OUT_JSON = OUTPUT_DIR / "old_au_catboost_full_ensemble_results.json"
OUT_MD = ROOT / "old_au_catboost_full_ensemble_report.md"


def load_old_train() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(OLD_STANDARD)
    data = data[data[:, 0] == 79.0]
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    keep = np.array(
        [standard_row_key(79.0, row[1:4]) not in bench_keys and standard_row_key(79.0, row[1:4]) not in extra_keys for row in data],
        dtype=bool,
    )
    return data[keep, 1:4], np.log10(data[keep, 4]), data[keep, 4]


def load_au(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    mask = payload["element"].astype(str) == "Z_79"
    return payload["x_model"][mask, 1:4], payload["y_log"][mask]


def augment(x: np.ndarray) -> np.ndarray:
    r, t, g = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack(
        [
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
        ]
    )


def main() -> None:
    from catboost import CatBoostRegressor

    t0 = time.time()
    x_train, y_log_train, y_raw_train = load_old_train()
    x_bench, y_bench = load_au(BENCHMARK_DATA_PATH)
    x_extra, y_extra = load_au(EXTRAPOLATION_DATA_PATH)
    specs = [
        ("full_d11_mae_lr025_l2_10_seed11", "log", "raw", "MAE", 11, 0.025, 11000, 10.0, 20260711),
        ("full_d11_mae_lr022_l2_14_seed12", "log", "raw", "MAE", 11, 0.022, 13000, 14.0, 20260712),
        ("full_d12_mae_lr018_l2_16_seed13", "log", "raw", "MAE", 12, 0.018, 13000, 16.0, 20260713),
        ("full_d10_mae_lr030_l2_08_seed14", "log", "raw", "MAE", 10, 0.030, 10000, 8.0, 20260714),
        ("full_d10_rmse_lr035_l2_08_seed15", "log", "raw", "RMSE", 10, 0.035, 9000, 8.0, 20260715),
        ("full_aug_d11_mae_lr022_l2_14_seed16", "log", "aug", "MAE", 11, 0.022, 11000, 14.0, 20260716),
        ("full_rawtarget_d10_mape_lr030_seed17", "raw_target", "raw", "MAPE", 10, 0.030, 7000, 8.0, 20260717),
    ]
    bench_preds = []
    extra_preds = []
    rows = []
    for name, target_mode, feature_mode, loss, depth, lr, iterations, l2, seed in specs:
        print(f"fit {name}", flush=True)
        start = time.time()
        x_fit = augment(x_train) if feature_mode == "aug" else x_train
        xb = augment(x_bench) if feature_mode == "aug" else x_bench
        xe = augment(x_extra) if feature_mode == "aug" else x_extra
        y_fit = y_raw_train if target_mode == "raw_target" else y_log_train
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
        model.fit(x_fit, y_fit)
        if target_mode == "raw_target":
            pb_raw = np.maximum(np.asarray(model.predict(xb), dtype=float).reshape(-1), 1e-300)
            pe_raw = np.maximum(np.asarray(model.predict(xe), dtype=float).reshape(-1), 1e-300)
            pb = np.log10(pb_raw)
            pe = np.log10(pe_raw)
        else:
            pb = np.asarray(model.predict(xb), dtype=float).reshape(-1)
            pe = np.asarray(model.predict(xe), dtype=float).reshape(-1)
        bench_preds.append(pb)
        extra_preds.append(pe)
        row = {
            "method": name,
            "target_mode": target_mode,
            "feature_mode": feature_mode,
            "benchmark": metric_dict(y_bench, pb),
            "extrapolation": metric_dict(y_extra, pe),
            "train_seconds": time.time() - start,
        }
        rows.append(row)
        print(
            f"{name}: bench={row['benchmark']['smape_percent']:.6f}% "
            f"extra={row['extrapolation']['smape_percent']:.6f}%",
            flush=True,
        )
    stack_b = np.column_stack(bench_preds)
    stack_e = np.column_stack(extra_preds)
    for name, pb, pe in [
        ("full_catboost_mean_all", stack_b.mean(axis=1), stack_e.mean(axis=1)),
        ("full_catboost_median_all", np.median(stack_b, axis=1), np.median(stack_e, axis=1)),
        ("full_catboost_mean_log_only", stack_b[:, :-1].mean(axis=1), stack_e[:, :-1].mean(axis=1)),
        ("full_catboost_median_log_only", np.median(stack_b[:, :-1], axis=1), np.median(stack_e[:, :-1], axis=1)),
    ]:
        rows.append(
            {
                "method": name,
                "benchmark": metric_dict(y_bench, pb),
                "extrapolation": metric_dict(y_extra, pe),
                "train_seconds": 0.0,
            }
        )
    rows.sort(key=lambda r: (r["benchmark"]["smape_percent"], r["extrapolation"]["smape_percent"]))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(OLD_STANDARD),
        "train_rows": int(x_train.shape[0]),
        "results": rows,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 旧 Au 全量 CatBoost 集成实验报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 训练行数：{payload['train_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | 方法 | Au benchmark SMAPE | Au extrap SMAPE | benchmark log10 MAE | P99 倍数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, 1):
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
