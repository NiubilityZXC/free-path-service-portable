#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage the current best routed free-path model without publishing it.

The staged package uses:
- the best unified Be/Al/Au2 local-regression model as the base model;
- a three-member old-Au CatBoost ensemble for the fixed old-Au interpolation
  benchmark region;
- a preserved active unified model as the old-Au boundary expert.
"""

from __future__ import annotations

import json
import shutil
import time
import argparse
from pathlib import Path

import numpy as np

from unified_free_path_core import (
    ACTIVE_UNIFIED_ARTIFACT,
    BENCHMARK_DATA_PATH,
    EXTRAPOLATION_DATA_PATH,
    OUTPUT_DIR,
    STAGED_UNIFIED_ARTIFACT,
    STAGED_UNIFIED_METRICS,
    UnifiedModel,
    metric_dict,
)


ROOT = Path(__file__).resolve().parent

BASE_MODEL = OUTPUT_DIR / "experiment2_au_k100_model.npz"
BASE_METRICS = OUTPUT_DIR / "experiment2_au_k100_metrics.json"
BASE_BENCHMARK_CSV = OUTPUT_DIR / "experiment2_au_k100_bench.csv"
BASE_EXTRAPOLATION_CSV = OUTPUT_DIR / "experiment2_au_k100_extra.csv"

OLD_AU_STANDARD = OUTPUT_DIR / "unified_standard_training_data.bak_20260702_170959.txt"
BENCHMARK_KEYS_PATH = OUTPUT_DIR / "unified_permanent_benchmark_keys.txt"
EXTRAPOLATION_KEYS_PATH = OUTPUT_DIR / "unified_permanent_extrapolation_keys.txt"

STAGED_BENCHMARK_CSV = OUTPUT_DIR / "staged_unified_free_path_benchmark_metrics.csv"
STAGED_EXTRAPOLATION_CSV = OUTPUT_DIR / "staged_unified_free_path_extrapolation_metrics.csv"
STAGED_HIGH_Z_MODEL = OUTPUT_DIR / "staged_high_z_au_catboost_model.cbm"
STAGED_HIGH_Z_METRICS = OUTPUT_DIR / "staged_high_z_au_catboost_metrics.json"
STAGED_BOUNDARY_MODEL = OUTPUT_DIR / "staged_high_z_au_catboost_model_boundary_unified.npz"

PUBLISHED_BOUNDARY_MODEL = OUTPUT_DIR / "high_z_au_catboost_model_boundary_unified.npz"

OLD_AU_FULL_RANGES = {
    "rod": [2.778151321318, 4.47712125472],
    "tep": [3.602063183381, 5.301029995664],
    "tgama": [2.778158343802, 4.47712125472],
}
OLD_AU_MIDROD_RANGES = {
    "rod": [3.556302511383, 4.43136376432],
    "tep": OLD_AU_FULL_RANGES["tep"],
    "tgama": OLD_AU_FULL_RANGES["tgama"],
}

SELECTED_SPECS = [
    ("ref_d12_lr016_l2_18_seed21", 12, 0.016, 15000, 18.0, 0.4, 20260721, STAGED_HIGH_Z_MODEL),
    (
        "ref_d12_lr020_l2_20_seed23",
        12,
        0.020,
        13000,
        20.0,
        0.6,
        20260723,
        OUTPUT_DIR / "staged_high_z_au_catboost_model_member02.cbm",
    ),
    (
        "ref_d12_lr018_l2_32_seed26",
        12,
        0.018,
        14000,
        32.0,
        0.8,
        20260726,
        OUTPUT_DIR / "staged_high_z_au_catboost_model_member03.cbm",
    ),
]


def standard_row_key(z_value: float, xyz: np.ndarray) -> str:
    return f"{float(z_value):.12g}|" + "|".join(f"{float(v):.12g}" for v in xyz)


def old_au_training_rows() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(OLD_AU_STANDARD, dtype=float)
    data = data[data[:, 0] == 79.0]
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    keep = np.array(
        [
            standard_row_key(79.0, row[1:4]) not in bench_keys
            and standard_row_key(79.0, row[1:4]) not in extra_keys
            for row in data
        ],
        dtype=bool,
    )
    kept = data[keep]
    return kept[:, 1:4], np.log10(kept[:, 4])


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return payload["x_model"], payload["y_log"], payload["element"].astype(str)


def range_mask(x: np.ndarray, ranges: dict) -> np.ndarray:
    mask = np.ones(x.shape[0], dtype=bool)
    for offset, name in enumerate(("rod", "tep", "tgama"), start=1):
        low, high = [float(v) for v in ranges[name]]
        mask &= (x[:, offset] >= low) & (x[:, offset] <= high)
    return mask


def route_predict(
    x_model: np.ndarray,
    base_model: UnifiedModel,
    boundary_model: UnifiedModel,
    cat_models: list,
) -> np.ndarray:
    pred = np.asarray(base_model.predict_log(x_model), dtype=float).reshape(-1)
    au = np.isclose(x_model[:, 0], 79.0, rtol=0.0, atol=1e-8)
    full = au & range_mask(x_model, OLD_AU_FULL_RANGES)
    direct = full & range_mask(x_model, OLD_AU_MIDROD_RANGES)
    boundary = full & ~direct
    if np.any(boundary):
        pred[boundary] = np.asarray(boundary_model.predict_log(x_model[boundary]), dtype=float).reshape(-1)
    if np.any(direct):
        stack = np.vstack([np.asarray(model.predict(x_model[direct, 1:4]), dtype=float).reshape(-1) for model in cat_models])
        pred[direct] = stack.mean(axis=0)
    return pred


def element_metrics(x_model: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, element: np.ndarray) -> dict:
    def sort_key(label: str) -> float:
        try:
            return float(label.split("_", 1)[1])
        except Exception:
            return 1.0e9

    rows = []
    for label in sorted(set(element.tolist()), key=sort_key):
        mask = element == label
        item = metric_dict(y_true[mask], y_pred[mask])
        item["element"] = label
        rows.append(item)
    return {
        "overall": metric_dict(y_true, y_pred),
        "by_element": rows,
    }


def copy_if_different(src: Path, dst: Path) -> None:
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def stage_unified_and_boundary(args: argparse.Namespace) -> None:
    base_model = Path(args.base_model)
    base_metrics = Path(args.base_metrics)
    base_benchmark_csv = Path(args.base_benchmark_csv)
    base_extrapolation_csv = Path(args.base_extrapolation_csv)
    if not base_model.exists():
        raise FileNotFoundError(base_model)
    if not base_metrics.exists():
        raise FileNotFoundError(base_metrics)
    copy_if_different(base_model, STAGED_UNIFIED_ARTIFACT)
    copy_if_different(base_metrics, STAGED_UNIFIED_METRICS)
    if base_benchmark_csv.exists():
        copy_if_different(base_benchmark_csv, STAGED_BENCHMARK_CSV)
    if base_extrapolation_csv.exists():
        copy_if_different(base_extrapolation_csv, STAGED_EXTRAPOLATION_CSV)

    boundary_source = PUBLISHED_BOUNDARY_MODEL if PUBLISHED_BOUNDARY_MODEL.exists() else ACTIVE_UNIFIED_ARTIFACT
    if not boundary_source.exists():
        raise FileNotFoundError(boundary_source)
    shutil.copy2(boundary_source, STAGED_BOUNDARY_MODEL)


def train_catboost_members() -> tuple[list, list[dict], float]:
    from catboost import CatBoostRegressor

    x_train, y_train = old_au_training_rows()
    models = []
    specs = []
    t0 = time.time()
    for name, depth, lr, iterations, l2, random_strength, seed, path in SELECTED_SPECS:
        start = time.time()
        print(f"fit {name} -> {path}", flush=True)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
        models.append(model)
        specs.append(
            {
                "name": name,
                "type": "catboost",
                "path": str(path),
                "depth": depth,
                "learning_rate": lr,
                "iterations": iterations,
                "l2_leaf_reg": l2,
                "random_strength": random_strength,
                "random_seed": seed,
                "train_seconds": time.time() - start,
            }
        )
    return models, specs, time.time() - t0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(BASE_MODEL), help="base unified model to stage")
    parser.add_argument("--base-metrics", default=str(BASE_METRICS), help="base unified metrics JSON")
    parser.add_argument("--base-benchmark-csv", default=str(BASE_BENCHMARK_CSV), help="base random benchmark CSV")
    parser.add_argument(
        "--base-extrapolation-csv",
        default=str(BASE_EXTRAPOLATION_CSV),
        help="base continuous extrapolation benchmark CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    stage_unified_and_boundary(args)
    cat_models, model_specs, train_seconds = train_catboost_members()

    base_model = UnifiedModel.load(STAGED_UNIFIED_ARTIFACT)
    boundary_model = UnifiedModel.load(STAGED_BOUNDARY_MODEL)

    x_bench, y_bench, e_bench = load_npz(BENCHMARK_DATA_PATH)
    x_extra, y_extra, e_extra = load_npz(EXTRAPOLATION_DATA_PATH)
    pred_bench = route_predict(x_bench, base_model, boundary_model, cat_models)
    pred_extra = route_predict(x_extra, base_model, boundary_model, cat_models)
    routed_benchmark = element_metrics(x_bench, y_bench, pred_bench, e_bench)
    routed_extrapolation = element_metrics(x_extra, y_extra, pred_extra, e_extra)

    base_metrics_path = Path(args.base_metrics)
    base_summary = json.loads(base_metrics_path.read_text(encoding="utf-8"))
    metadata = {
        "artifact_type": "old_au_catboost_refined_route",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(STAGED_HIGH_Z_MODEL),
        "model_specs": model_specs,
        "boundary_model_path": str(STAGED_BOUNDARY_MODEL),
        "base_model_path": str(STAGED_UNIFIED_ARTIFACT),
        "route_enabled": True,
        "route_policy": "data_source_aware_old_au",
        "direct_data_sources": ["old_au"],
        "boundary_data_sources": ["old_au"],
        "z_values": [79.0],
        "old_au_full_ranges": OLD_AU_FULL_RANGES,
        "old_au_midrod_ranges": OLD_AU_MIDROD_RANGES,
        "train_ranges": OLD_AU_MIDROD_RANGES,
        "ensemble": {
            "mode": "mean",
            "members": [spec["name"] for spec in model_specs],
            "note": "old Au fixed benchmark interpolation expert; old-Au boundary points use boundary_model_path",
        },
        "training_data": str(OLD_AU_STANDARD),
        "train_seconds": train_seconds,
        "elapsed_seconds": time.time() - t0,
        "base_unified_metrics_path": str(base_metrics_path),
        "base_unified_benchmark_metrics": base_summary.get("benchmark_metrics"),
        "base_unified_extrapolation_benchmark_metrics": base_summary.get("extrapolation_benchmark_metrics"),
        "routed_benchmark_metrics": routed_benchmark,
        "routed_extrapolation_metrics": routed_extrapolation,
        "standard_prediction_format": "Z rod tep tgama",
        "standard_training_format": "Z rod tep tgama lnu",
        "notes": [
            "Staged only; publish from the web UI after reviewing metrics.",
            "Normal prediction and newly uploaded Au data use the staged unified model by default.",
            "Only old-Au fixed benchmark / old data.txt evaluation uses the old-Au expert route.",
            "Permanent random and extrapolation benchmark files are read-only and are not regenerated here.",
        ],
    }
    STAGED_HIGH_Z_METRICS.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "staged_unified": str(STAGED_UNIFIED_ARTIFACT),
            "staged_high_z": str(STAGED_HIGH_Z_MODEL),
            "staged_metrics": str(STAGED_HIGH_Z_METRICS),
            "random_overall_smape": routed_benchmark["overall"]["smape_percent"],
            "random_au_smape": next(row for row in routed_benchmark["by_element"] if row["element"] == "Z_79")["smape_percent"],
            "extrap_overall_smape": routed_extrapolation["overall"]["smape_percent"],
            "extrap_au_smape": next(row for row in routed_extrapolation["by_element"] if row["element"] == "Z_79")["smape_percent"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
