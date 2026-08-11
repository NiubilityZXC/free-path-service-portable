#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Focused Au boundary/extrapolation local-regression sweep."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from run_au_local_sweep import local_poly_prediction
from train_unified_free_path_model import load_standard_data
from unified_free_path_core import (
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    metric_dict,
)


ROOT = Path(__file__).resolve().parent
COMBINED_DATA = OUTPUT_DIR / "experiment_standard_with_old_au_and_au2.txt"
OUT_JSON = OUTPUT_DIR / "au_extrap_local_sweep_results.json"
OUT_MD = ROOT / "au_extrap_local_sweep_report.md"


def main() -> None:
    t0 = time.time()
    item = load_standard_data(COMBINED_DATA)
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    train_mask = (item["element_labels"] == "Z_79") & np.array(
        [key not in bench_keys and key not in extra_keys for key in item["keys"]], dtype=bool
    )
    x_train = item["x_model"][train_mask, 1:4]
    y_train = item["y_log"][train_mask]
    payload = np.load(EXTRAPOLATION_DATA_PATH, allow_pickle=True)
    au_mask = payload["element"].astype(str) == "Z_79"
    x_eval = payload["x_model"][au_mask, 1:4]
    y_eval = payload["y_log"][au_mask]

    # Narrow scan around the best region found by run_au_local_sweep.py.
    scale_variants = {
        "rod150": np.array([1.5, 1.0, 1.0]),
        "rod175": np.array([1.75, 1.0, 1.0]),
    }
    k_values = [260, 320, 360, 440, 520, 700]
    degrees = [2]
    powers = [1.2, 1.4, 1.6, 1.8]
    ridges = [1e-3, 1e-2, 5e-2]
    max_k = max(k_values)

    rows = []
    print(f"Au train={x_train.shape[0]}, Au extrap={x_eval.shape[0]}")
    for scale_name, extra_weight in scale_variants.items():
        mean = x_train.mean(axis=0)
        std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
        x_train_scaled = ((x_train - mean) / std) * extra_weight
        x_eval_scaled = ((x_eval - mean) / std) * extra_weight
        tree = cKDTree(x_train_scaled)
        dist_all, idx_all = tree.query(x_eval_scaled, k=max_k, workers=-1)
        print(f"scale={scale_name}", flush=True)
        for k in k_values:
            dist = dist_all[:, :k]
            idx = idx_all[:, :k]
            for degree in degrees:
                for power in powers:
                    for ridge in ridges:
                        pred = local_poly_prediction(
                            x_eval_scaled,
                            dist,
                            idx,
                            x_train_scaled,
                            y_train,
                            degree=degree,
                            power=power,
                            ridge=ridge,
                        )
                        metrics = metric_dict(y_eval, pred)
                        rows.append(
                            {
                                "scale": scale_name,
                                "extra_weight": extra_weight.tolist(),
                                "k": int(k),
                                "degree": int(degree),
                                "power": float(power),
                                "ridge": float(ridge),
                                "extrapolation": metrics,
                            }
                        )
    rows.sort(key=lambda r: (r["extrapolation"]["smape_percent"], r["extrapolation"]["log10_mae"]))
    payload_out = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(COMBINED_DATA),
        "fixed_extrapolation": str(EXTRAPOLATION_DATA_PATH),
        "au_train_rows": int(x_train.shape[0]),
        "searched_rows": len(rows),
        "top": rows[:40],
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Au 外推边界局部模型扫描报告",
        "",
        f"- 生成时间：{payload_out['created_at']}",
        f"- Au 训练行数：{payload_out['au_train_rows']}",
        f"- 搜索组合数：{payload_out['searched_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | scale | k | degree | power | ridge | Au extrap SMAPE | Au log10 MAE | Au P99倍数 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows[:20], 1):
        m = row["extrapolation"]
        lines.append(
            f"| {i} | `{row['scale']}` | {row['k']} | {row['degree']} | {row['power']} | {row['ridge']:.1e} | "
            f"{m['smape_percent']:.6f}% | {m['log10_mae']:.8f} | {m['p99_factor_error']:.6f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"best: {rows[0]}")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={payload_out['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
