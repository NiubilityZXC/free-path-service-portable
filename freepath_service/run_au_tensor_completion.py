#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Old-Au tensor completion experiment on the frozen holdout grid.

This experiment uses only old Au rows from the pre-Au2 standard-data backup.
It treats frozen benchmark and extrapolation rows as missing values, then uses
iterative low-rank Tucker/HOSVD completion on the 50x50x50 log10(lnu) grid.
It does not publish or modify any web model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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
OUT_JSON = OUTPUT_DIR / "au_tensor_completion_results.json"
OUT_MD = ROOT / "au_tensor_completion_report.md"


def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    moved = np.moveaxis(tensor, mode, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = matrix @ flat
    out = out.reshape((matrix.shape[0],) + moved.shape[1:])
    return np.moveaxis(out, 0, mode)


def tucker_reconstruct(x: np.ndarray, ranks: tuple[int, int, int]) -> np.ndarray:
    factors = []
    for mode, rank in enumerate(ranks):
        moved = np.moveaxis(x, mode, 0)
        unfold = moved.reshape(moved.shape[0], -1)
        u, _, _ = np.linalg.svd(unfold, full_matrices=False)
        factors.append(u[:, :rank])
    core = x
    for mode, u in enumerate(factors):
        core = mode_dot(core, u.T, mode)
    recon = core
    for mode, u in enumerate(factors):
        recon = mode_dot(recon, u, mode)
    return recon


def load_au_benchmark(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    mask = payload["element"].astype(str) == "Z_79"
    return payload["x_model"][mask], payload["y_log"][mask]


def grid_indices(rows: np.ndarray, axes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = []
    for j, axis in enumerate(axes, start=1):
        lut = {round(float(v), 12): i for i, v in enumerate(axis)}
        idx = np.array([lut[round(float(v), 12)] for v in rows[:, j]], dtype=int)
        out.append(idx)
    return tuple(out)  # type: ignore[return-value]


def main() -> None:
    t0 = time.time()
    data = np.loadtxt(OLD_STANDARD)
    data = data[data[:, 0] == 79.0]
    xyz = data[:, 1:4]
    y_log = np.log10(data[:, 4])
    axes = [np.unique(np.round(xyz[:, i], 12)) for i in range(3)]
    shape = tuple(len(axis) for axis in axes)
    print(f"old Au rows={data.shape[0]}, grid={shape}")
    full = np.full(shape, np.nan, dtype=float)
    idx = grid_indices(data, axes)
    full[idx] = y_log

    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    observed = np.zeros(shape, dtype=bool)
    for row in data:
        key = standard_row_key(79.0, row[1:4])
        if key not in bench_keys and key not in extra_keys:
            i, j, k = [np.where(axis == round(float(row[col]), 12))[0][0] for axis, col in zip(axes, [1, 2, 3])]
            observed[i, j, k] = True
    observed &= np.isfinite(full)
    missing = ~observed
    print(f"observed={int(observed.sum())}, missing={int(missing.sum())}")

    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    obs_flat = observed.reshape(-1)
    tree = cKDTree(coords[obs_flat])
    nearest_dist, nearest_idx = tree.query(coords[~obs_flat], k=1, workers=-1)
    current = np.empty_like(full)
    current[observed] = full[observed]
    obs_values = full.reshape(-1)[obs_flat]
    current.reshape(-1)[~obs_flat] = obs_values[nearest_idx]

    bench_x, bench_y = load_au_benchmark(BENCHMARK_DATA_PATH)
    extra_x, extra_y = load_au_benchmark(EXTRAPOLATION_DATA_PATH)
    bench_idx = grid_indices(bench_x, axes)
    extra_idx = grid_indices(extra_x, axes)

    ranks_to_try = [
        (6, 8, 6),
        (8, 10, 8),
        (10, 12, 10),
        (12, 14, 12),
        (14, 16, 14),
        (16, 18, 16),
        (20, 20, 20),
        (24, 24, 24),
        (30, 24, 30),
        (36, 30, 36),
    ]
    blends = [0.6, 0.8, 1.0]
    iterations = 25
    results = []
    for ranks in ranks_to_try:
        base = current.copy()
        for blend in blends:
            x = base.copy()
            best = None
            for it in range(1, iterations + 1):
                recon = tucker_reconstruct(x, ranks)
                x[missing] = (1.0 - blend) * x[missing] + blend * recon[missing]
                x[observed] = full[observed]
                if it in {3, 6, 10, 15, 20, 25}:
                    pred_b = x[bench_idx]
                    pred_e = x[extra_idx]
                    b = metric_dict(bench_y, pred_b)
                    e = metric_dict(extra_y, pred_e)
                    row = {
                        "ranks": list(ranks),
                        "blend": float(blend),
                        "iteration": int(it),
                        "benchmark": b,
                        "extrapolation": e,
                    }
                    if best is None or b["smape_percent"] < best["benchmark"]["smape_percent"]:
                        best = row
            if best is not None:
                results.append(best)
                print(
                    f"ranks={ranks} blend={blend} best_iter={best['iteration']} "
                    f"bench={best['benchmark']['smape_percent']:.6f}% "
                    f"extra={best['extrapolation']['smape_percent']:.6f}%",
                    flush=True,
                )
    results.sort(key=lambda r: (r["benchmark"]["smape_percent"], r["extrapolation"]["smape_percent"]))
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(OLD_STANDARD),
        "method": "iterative Tucker/HOSVD completion on old Au 50x50x50 log10(lnu) grid",
        "observed_grid_points": int(observed.sum()),
        "missing_grid_points": int(missing.sum()),
        "top": results[:40],
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Au 旧数据张量补全实验报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 观测网格点：{payload['observed_grid_points']}",
        f"- 缺失/测试网格点：{payload['missing_grid_points']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 排名 | ranks | blend | iter | Au benchmark SMAPE | Au extrap SMAPE | benchmark log10 MAE | extrap log10 MAE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(results[:20], 1):
        lines.append(
            f"| {i} | `{tuple(row['ranks'])}` | {row['blend']} | {row['iteration']} | "
            f"{row['benchmark']['smape_percent']:.6f}% | {row['extrapolation']['smape_percent']:.6f}% | "
            f"{row['benchmark']['log10_mae']:.8f} | {row['extrapolation']['log10_mae']:.8f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
