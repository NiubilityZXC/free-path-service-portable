#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Old-Au local RBF interpolation experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

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
OUT_JSON = OUTPUT_DIR / "old_au_rbf_results.json"
OUT_MD = ROOT / "old_au_rbf_report.md"


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
    t0 = time.time()
    x_train, y_train = load_old_train()
    x_bench, y_bench = load_au(BENCHMARK_DATA_PATH)
    x_extra, y_extra = load_au(EXTRAPOLATION_DATA_PATH)
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
    scale_variants = {
        "std": np.array([1.0, 1.0, 1.0]),
        "rod125": np.array([1.25, 1.0, 1.0]),
        "rod150": np.array([1.5, 1.0, 1.0]),
        "tgama125": np.array([1.0, 1.0, 1.25]),
    }
    specs = []
    for scale in scale_variants:
        for kernel in ["linear", "thin_plate_spline", "cubic", "quintic"]:
            for neighbors in [24, 40, 64, 96, 144, 220]:
                for smoothing in [0.0, 1e-8, 1e-6]:
                    specs.append((scale, kernel, neighbors, smoothing))
    results = []
    for scale, kernel, neighbors, smoothing in specs:
        start = time.time()
        weight = scale_variants[scale]
        xt = ((x_train - mean) / std) * weight
        xb = ((x_bench - mean) / std) * weight
        xe = ((x_extra - mean) / std) * weight
        try:
            model = RBFInterpolator(
                xt,
                y_train,
                neighbors=neighbors,
                kernel=kernel,
                smoothing=smoothing,
                degree=1 if kernel in {"thin_plate_spline", "cubic", "quintic"} else 0,
            )
            pred_b = model(xb)
            b = metric_dict(y_bench, pred_b)
            # Only evaluate extrap for promising benchmark candidates.
            if b["smape_percent"] <= 6.5:
                pred_e = model(xe)
                e = metric_dict(y_extra, pred_e)
            else:
                e = None
            row = {
                "scale": scale,
                "kernel": kernel,
                "neighbors": int(neighbors),
                "smoothing": float(smoothing),
                "benchmark": b,
                "extrapolation": e,
                "seconds": time.time() - start,
            }
            results.append(row)
            print(
                f"{scale} {kernel} n={neighbors} sm={smoothing:g}: bench={b['smape_percent']:.6f}%"
                + (f" extra={e['smape_percent']:.6f}%" if e else ""),
                flush=True,
            )
        except Exception as exc:
            results.append(
                {
                    "scale": scale,
                    "kernel": kernel,
                    "neighbors": int(neighbors),
                    "smoothing": float(smoothing),
                    "error": repr(exc),
                    "seconds": time.time() - start,
                }
            )
    good = [r for r in results if "benchmark" in r]
    good.sort(key=lambda r: (r["benchmark"]["smape_percent"], r.get("extrapolation")["smape_percent"] if r.get("extrapolation") else 999.0))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(OLD_STANDARD),
        "train_rows": int(x_train.shape[0]),
        "results": good + [r for r in results if "benchmark" not in r],
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 旧 Au 局部 RBF 实验报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 训练行数：{payload['train_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | scale | kernel | neighbors | smoothing | Au benchmark SMAPE | Au extrap SMAPE | P99 倍数 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(good[:30], 1):
        extra = row.get("extrapolation") or {}
        lines.append(
            f"| {i} | `{row['scale']}` | `{row['kernel']}` | {row['neighbors']} | {row['smoothing']:.0e} | "
            f"{row['benchmark']['smape_percent']:.6f}% | "
            f"{extra.get('smape_percent', float('nan')):.6f}% | "
            f"{row['benchmark']['p99_factor_error']:.6f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
