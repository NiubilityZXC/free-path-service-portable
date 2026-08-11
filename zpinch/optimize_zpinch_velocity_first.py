#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from train_zpinch_surrogate import (
    DOC_DATA_PATH,
    EnergyToVelocityOutputModel,
    V_FORMULA_COEF,
    build_dataset,
    calc_v_from_E_m,
    extract_rows_from_doc,
    mae,
    make_features,
    mape,
    rmse,
    train_val_test_split,
    zpinch_exp_inverse,
    zpinch_log_positive,
)


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("ZPINCH_OUTPUT_DIR", ROOT_DIR / "ai_training_outputs")).expanduser().resolve()
ARTIFACT_PATH = OUTPUT_DIR / "best_velocity_first_optimized_model_artifact.pkl"
SUMMARY_PATH = OUTPUT_DIR / "velocity_first_optimized_summary.json"
BASELINE_SUMMARY_PATH = OUTPUT_DIR / "summary_metrics.json"


def metrics_from_velocity(y_true_E, y_true_v, pred_v, m_input):
    pred_E = (np.asarray(pred_v, dtype=float) ** 2) * m_input / V_FORMULA_COEF
    return {
        "E_MAE": mae(y_true_E, pred_E),
        "E_RMSE": rmse(y_true_E, pred_E),
        "E_MAPE_percent": mape(y_true_E, pred_E),
        "v_MAE": mae(y_true_v, pred_v),
        "v_RMSE": rmse(y_true_v, pred_v),
        "v_MAPE_percent": mape(y_true_v, pred_v),
    }


def metrics_from_energy(y_true_E, y_true_v, pred_E, m_input):
    pred_E = np.maximum(np.asarray(pred_E, dtype=float), 1e-12)
    pred_v = calc_v_from_E_m(pred_E, m_input)
    return metrics_from_velocity(y_true_E, y_true_v, pred_v, m_input)


def build_optimized_velocity_model():
    energy_regressor = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=4, include_bias=False),
            Ridge(alpha=1e-4),
        ),
        func=zpinch_log_positive,
        inverse_func=zpinch_exp_inverse,
    )
    return EnergyToVelocityOutputModel(energy_regressor)


def training_ranges(X_base, y_E, y_v):
    base_names = [
        "I_MA",
        "tr_ns",
        "liner_r_cm",
        "foam_r_cm",
        "m_mg_per_cm",
        "Z_atomic_number",
    ]
    ranges = {
        name: {"min": float(X_base[:, idx].min()), "max": float(X_base[:, idx].max())}
        for idx, name in enumerate(base_names)
    }
    target_ranges = {
        "E_MJ_per_cm": {"min": float(y_E.min()), "max": float(y_E.max())},
        "v_cm_per_s": {"min": float(y_v.min()), "max": float(y_v.max())},
    }
    return ranges, target_ranges


def mean_metrics(records):
    return {key: float(np.mean([item[key] for item in records])) for key in records[0]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = extract_rows_from_doc(DOC_DATA_PATH)
    X_base, y_E, y_v, base_feature_names = build_dataset(raw_rows)
    X, feature_names = make_features(X_base)

    seeds = [42, 2024, 2026, 7, 13, 99, 123]
    optimized_records = []
    for seed in seeds:
        tr_idx, va_idx, te_idx = train_val_test_split(len(X), seed=seed)
        train_idx = np.concatenate([tr_idx, va_idx])
        model = build_optimized_velocity_model()
        model.fit(X[train_idx], y_E[train_idx])
        pred_v = model.predict(X[te_idx])
        optimized_records.append(metrics_from_velocity(y_E[te_idx], y_v[te_idx], pred_v, X[te_idx, 4]))

    tr_idx, va_idx, te_idx = train_val_test_split(len(X), seed=2026)
    train_idx = np.concatenate([tr_idx, va_idx])
    final_model = build_optimized_velocity_model()
    final_model.fit(X[train_idx], y_E[train_idx])
    final_pred_v = final_model.predict(X[te_idx])
    final_metrics = metrics_from_velocity(y_E[te_idx], y_v[te_idx], final_pred_v, X[te_idx, 4])

    baseline_metrics = {}
    if BASELINE_SUMMARY_PATH.exists():
        with BASELINE_SUMMARY_PATH.open("r", encoding="utf-8") as f:
            baseline_summary = json.load(f)
        baseline_metrics = baseline_summary.get("final_test_metrics", {})

    ranges, target_ranges = training_ranges(X_base, y_E, y_v)
    artifact_payload = {
        "model_name": "VelocityFirstLogEPoly4RidgeOptimized",
        "model": final_model,
        "doc_path": DOC_DATA_PATH,
        "feature_names": feature_names,
        "base_feature_names": base_feature_names,
        "final_seed": 2026,
        "train_idx": train_idx,
        "test_idx": te_idx,
        "final_test_metrics": final_metrics,
        "multi_seed_test_metrics": mean_metrics(optimized_records),
        "multi_seed_test_seeds": seeds,
        "prediction_target": "v_cm_per_s",
        "pipeline": "velocity_first_physics_constrained",
        "formula": "model.predict returns v_cm_per_s; web computes E_MJ_per_cm = v^2*m/(2e16)",
        "trained_on": "log(E_MJ_per_cm) with degree-4 polynomial Ridge, wrapped to output velocity",
        "n_samples": int(len(X)),
        "training_ranges": ranges,
        "target_ranges": target_ranges,
        "baseline_existing_best_final_metrics": baseline_metrics,
    }
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(artifact_payload, f)

    summary = {
        "model_family": "zpinch_velocity_first_optimized",
        "root_dir": str(ROOT_DIR),
        "artifact_path": str(ARTIFACT_PATH),
        "doc_path": DOC_DATA_PATH,
        "prediction_target": "v_cm_per_s",
        "pipeline": "网页先获得速度 v，再由 E=v^2*m/(2e16) 反算动能；模型内部使用 log(E) 物理约束拟合后输出 v。",
        "model_name": artifact_payload["model_name"],
        "n_samples": int(len(X)),
        "test_seed": 2026,
        "test_size": int(len(te_idx)),
        "final_test_metrics": final_metrics,
        "multi_seed_test_metrics": mean_metrics(optimized_records),
        "multi_seed_test_seeds": seeds,
        "baseline_existing_best_final_metrics": baseline_metrics,
        "improvement_vs_existing_best_final": {
            key: (
                float(baseline_metrics[key] / final_metrics[key])
                if key in baseline_metrics and key in final_metrics and final_metrics[key] != 0
                else None
            )
            for key in ["E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]
        },
        "feature_names": feature_names,
        "base_feature_names": base_feature_names,
        "training_ranges": ranges,
        "target_ranges": target_ranges,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
