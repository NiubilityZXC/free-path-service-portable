#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Refined old-Au CatBoost search near the current best region."""

from __future__ import annotations

import itertools
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
OUT_JSON = OUTPUT_DIR / "old_au_catboost_refine_results.json"
OUT_MD = ROOT / "old_au_catboost_refine_report.md"


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


def load_au(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    mask = payload["element"].astype(str) == "Z_79"
    return payload["x_model"][mask, 1:4], payload["y_log"][mask]


def main() -> None:
    from catboost import CatBoostRegressor

    t0 = time.time()
    x_train, y_train = load_old_train()
    x_bench, y_bench = load_au(BENCHMARK_DATA_PATH)
    x_extra, y_extra = load_au(EXTRAPOLATION_DATA_PATH)
    specs = [
        ("ref_d12_lr016_l2_18_seed21", 12, 0.016, 15000, 18.0, 0.4, 20260721),
        ("ref_d12_lr014_l2_24_seed22", 12, 0.014, 17000, 24.0, 0.5, 20260722),
        ("ref_d12_lr020_l2_20_seed23", 12, 0.020, 13000, 20.0, 0.6, 20260723),
        ("ref_d11_lr020_l2_18_seed24", 11, 0.020, 14000, 18.0, 0.4, 20260724),
        ("ref_d11_lr018_l2_24_seed25", 11, 0.018, 16000, 24.0, 0.5, 20260725),
        ("ref_d12_lr018_l2_32_seed26", 12, 0.018, 14000, 32.0, 0.8, 20260726),
    ]
    rows = []
    bench_preds = []
    extra_preds = []
    names = []
    for name, depth, lr, iterations, l2, random_strength, seed in specs:
        print(f"fit {name}", flush=True)
        start = time.time()
        model = CatBoostRegressor(
            loss_function="MAE",
            depth=depth,
            learning_rate=lr,
            iterations=iterations,
            l2_leaf_reg=l2,
            random_strength=random_strength,
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train)
        pb = np.asarray(model.predict(x_bench), dtype=float).reshape(-1)
        pe = np.asarray(model.predict(x_extra), dtype=float).reshape(-1)
        bench_preds.append(pb)
        extra_preds.append(pe)
        names.append(name)
        row = {
            "method": name,
            "benchmark": metric_dict(y_bench, pb),
            "extrapolation": metric_dict(y_extra, pe),
            "train_seconds": time.time() - start,
        }
        rows.append(row)
        print(f"{name}: bench={row['benchmark']['smape_percent']:.6f}% extra={row['extrapolation']['smape_percent']:.6f}%", flush=True)
    stack_b = np.column_stack(bench_preds)
    stack_e = np.column_stack(extra_preds)
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), r):
            method = "mean_" + "_".join(str(i + 1) for i in combo)
            pb = stack_b[:, combo].mean(axis=1)
            pe = stack_e[:, combo].mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "members": [names[i] for i in combo],
                    "benchmark": metric_dict(y_bench, pb),
                    "extrapolation": metric_dict(y_extra, pe),
                    "train_seconds": 0.0,
                }
            )
            method = "median_" + "_".join(str(i + 1) for i in combo)
            pb = np.median(stack_b[:, combo], axis=1)
            pe = np.median(stack_e[:, combo], axis=1)
            rows.append(
                {
                    "method": method,
                    "members": [names[i] for i in combo],
                    "benchmark": metric_dict(y_bench, pb),
                    "extrapolation": metric_dict(y_extra, pe),
                    "train_seconds": 0.0,
                }
            )
    rows.sort(key=lambda row: (row["benchmark"]["smape_percent"], row["extrapolation"]["smape_percent"]))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(OLD_STANDARD),
        "train_rows": int(x_train.shape[0]),
        "base_specs": names,
        "results": rows,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 旧 Au CatBoost 精细搜索报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 训练行数：{payload['train_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | 方法 | Au benchmark SMAPE | Au extrap SMAPE | benchmark log10 MAE | P99 倍数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows[:40], 1):
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
