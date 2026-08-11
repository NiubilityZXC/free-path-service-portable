#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Old-Au-only tree model experiments on frozen benchmark/extrapolation."""

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
OUT_JSON = OUTPUT_DIR / "old_au_tree_model_results.json"
OUT_MD = ROOT / "old_au_tree_model_report.md"


def load_au_benchmark(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    mask = payload["element"].astype(str) == "Z_79"
    return payload["x_model"][mask, 1:4], payload["y_log"][mask]


def load_old_train() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(OLD_STANDARD)
    data = data[data[:, 0] == 79.0]
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    keep = []
    for row in data:
        key = standard_row_key(79.0, row[1:4])
        keep.append(key not in bench_keys and key not in extra_keys)
    keep = np.asarray(keep, dtype=bool)
    return data[keep, 1:4], np.log10(data[keep, 4])


def eval_model(name: str, predict_fn, x_bench, y_bench, x_extra, y_extra) -> dict:
    pred_b = np.asarray(predict_fn(x_bench), dtype=float).reshape(-1)
    pred_e = np.asarray(predict_fn(x_extra), dtype=float).reshape(-1)
    return {
        "method": name,
        "benchmark": metric_dict(y_bench, pred_b),
        "extrapolation": metric_dict(y_extra, pred_e),
    }


def main() -> None:
    t0 = time.time()
    x_train, y_train = load_old_train()
    x_bench, y_bench = load_au_benchmark(BENCHMARK_DATA_PATH)
    x_extra, y_extra = load_au_benchmark(EXTRAPOLATION_DATA_PATH)
    print(f"old Au train={x_train.shape[0]}, bench={x_bench.shape[0]}, extra={x_extra.shape[0]}")
    results = []

    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sklearn_models = [
        (
            "old_au_extra_trees_l2_600",
            ExtraTreesRegressor(
                n_estimators=600,
                criterion="squared_error",
                max_features=1.0,
                min_samples_leaf=1,
                random_state=20260704,
                n_jobs=-1,
            ),
        ),
        (
            "old_au_hgb_abs_leaf31",
            make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.035,
                    max_iter=2500,
                    max_leaf_nodes=31,
                    l2_regularization=0.005,
                    min_samples_leaf=8,
                    random_state=20260703,
                ),
            ),
        ),
        (
            "old_au_hgb_abs_leaf127",
            make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.025,
                    max_iter=3000,
                    max_leaf_nodes=127,
                    l2_regularization=0.01,
                    min_samples_leaf=5,
                    random_state=20260704,
                ),
            ),
        ),
    ]
    for name, model in sklearn_models:
        print(f"fit {name}", flush=True)
        start = time.time()
        model.fit(x_train, y_train)
        row = eval_model(name, model.predict, x_bench, y_bench, x_extra, y_extra)
        row["train_seconds"] = time.time() - start
        results.append(row)
        print(
            f"{name}: bench={row['benchmark']['smape_percent']:.6f}% "
            f"extra={row['extrapolation']['smape_percent']:.6f}% time={row['train_seconds']:.1f}s",
            flush=True,
        )

    try:
        from catboost import CatBoostRegressor

        cat_specs = [
            ("old_au_catboost_mae_d10", "MAE", 10, 0.03, 9000, 8.0, 20260703),
            ("old_au_catboost_mae_d11", "MAE", 11, 0.025, 10000, 10.0, 20260704),
            ("old_au_catboost_rmse_d10", "RMSE", 10, 0.035, 8000, 8.0, 20260705),
        ]
        for name, loss, depth, lr, iterations, l2, seed in cat_specs:
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
            model.fit(x_train, y_train)
            row = eval_model(name, model.predict, x_bench, y_bench, x_extra, y_extra)
            row["train_seconds"] = time.time() - start
            results.append(row)
            print(
                f"{name}: bench={row['benchmark']['smape_percent']:.6f}% "
                f"extra={row['extrapolation']['smape_percent']:.6f}% time={row['train_seconds']:.1f}s",
                flush=True,
            )
    except Exception as exc:
        results.append({"method": "catboost_failed", "error": repr(exc)})

    try:
        from lightgbm import LGBMRegressor

        lgbm_specs = [
            ("old_au_lgbm_l1_leaf255", "l1", 255, 0.025, 6500),
            ("old_au_lgbm_l2_leaf255", "l2", 255, 0.025, 6500),
            ("old_au_lgbm_l1_leaf511", "l1", 511, 0.018, 9000),
        ]
        for name, objective, leaves, lr, n_estimators in lgbm_specs:
            print(f"fit {name}", flush=True)
            start = time.time()
            model = LGBMRegressor(
                objective=objective,
                n_estimators=n_estimators,
                learning_rate=lr,
                num_leaves=leaves,
                min_child_samples=5,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=0.01,
                random_state=20260703,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(x_train, y_train)
            row = eval_model(name, model.predict, x_bench, y_bench, x_extra, y_extra)
            row["train_seconds"] = time.time() - start
            results.append(row)
            print(
                f"{name}: bench={row['benchmark']['smape_percent']:.6f}% "
                f"extra={row['extrapolation']['smape_percent']:.6f}% time={row['train_seconds']:.1f}s",
                flush=True,
            )
    except Exception as exc:
        results.append({"method": "lightgbm_failed", "error": repr(exc)})

    good = [r for r in results if "benchmark" in r]
    good.sort(key=lambda r: (r["benchmark"]["smape_percent"], r["extrapolation"]["smape_percent"]))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(OLD_STANDARD),
        "train_rows": int(x_train.shape[0]),
        "results": good + [r for r in results if "benchmark" not in r],
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 旧 Au 专用树模型实验报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 训练行数：{payload['train_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | 方法 | Au benchmark SMAPE | Au extrap SMAPE | benchmark log10 MAE | extrap log10 MAE | 训练秒 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(good, 1):
        lines.append(
            f"| {i} | `{row['method']}` | {row['benchmark']['smape_percent']:.6f}% | "
            f"{row['extrapolation']['smape_percent']:.6f}% | {row['benchmark']['log10_mae']:.8f} | "
            f"{row['extrapolation']['log10_mae']:.8f} | {row.get('train_seconds', 0.0):.1f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
