#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from train_missing_models_on_xlsx import (
    OUTPUT_DIR,
    SOURCE_SHEET,
    TARGET_SHEET,
    XLSX_PATH,
    read_sheet_as_base_dataset,
    repair_velocity_values,
    run_random_split_benchmark,
    save_csv,
    save_json,
)
from train_zpinch_surrogate import ensure_output_dir, evaluate_model, make_features, quality_report


def build_models():
    return {
        "XGBoost": XGBRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            random_seed=42,
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            verbose=False,
        ),
    }


def load_existing_summary(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_pickle(path: str, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    ensure_output_dir(OUTPUT_DIR)

    _, Xs_base, yEs, yvs_raw = read_sheet_as_base_dataset(XLSX_PATH, SOURCE_SHEET)
    _, Xt_base, yEt, yvt_raw = read_sheet_as_base_dataset(XLSX_PATH, TARGET_SHEET)
    yvs, repaired_source = repair_velocity_values(Xs_base, yEs, yvs_raw)
    yvt, repaired_target = repair_velocity_values(Xt_base, yEt, yvt_raw)
    Xs, feat_names = make_features(Xs_base)
    Xt, _ = make_features(Xt_base)

    models = build_models()
    source_random_summary = run_random_split_benchmark(Xs, yEs, yvs, seeds=[42, 2024, 2026], models=models)

    external_summary = {}
    for name, model in models.items():
        print(f"[external-fit] {name}", flush=True)
        model.fit(Xs, yEs)
        pred = np.maximum(model.predict(Xt), 1e-12)
        score = evaluate_model(yEt, pred, Xt[:, 4], yvt)
        external_summary[name] = score
        save_model_pickle(
            os.path.join(OUTPUT_DIR, f"{name}_artifact.pkl"),
            {
                "model_name": name,
                "model": model,
                "source_sheet": SOURCE_SHEET,
                "target_sheet": TARGET_SHEET,
                "xlsx_path": XLSX_PATH,
                "feature_names": feat_names,
                "target_fullset_metrics": score,
            },
        )

    summary_path = os.path.join(OUTPUT_DIR, "summary_metrics.json")
    summary = load_existing_summary(summary_path)

    summary.setdefault("xlsx_path", XLSX_PATH)
    summary.setdefault("source_sheet", SOURCE_SHEET)
    summary.setdefault("target_sheet", TARGET_SHEET)
    summary["source_n_samples"] = int(len(Xs))
    summary["target_n_samples"] = int(len(Xt))
    summary["repaired_source_velocity_rows"] = repaired_source
    summary["repaired_target_velocity_rows"] = repaired_target
    summary["source_quality"] = quality_report(Xs_base, yEs, yvs)
    summary["target_quality"] = quality_report(Xt_base, yEt, yvt)
    summary.setdefault("source_random_split_summary", {})
    summary.setdefault("target_external_test_summary", {})
    summary["source_random_split_summary"].update(source_random_summary)
    summary["target_external_test_summary"].update(external_summary)

    available_models = sorted(summary["target_external_test_summary"].keys())
    summary["available_models"] = available_models
    summary["feature_names"] = feat_names
    summary["best_external_model"] = min(
        summary["target_external_test_summary"].items(),
        key=lambda kv: kv[1]["E_MAE"],
    )[0]
    save_json(summary_path, summary)

    source_rows = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, s in summary["source_random_split_summary"].items():
        source_rows.append([name, s["E_MAE"], s["E_RMSE"], s["E_MAPE_percent"], s["v_MAE"], s["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "source_random_split_metrics.csv"), source_rows)

    target_rows = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, s in summary["target_external_test_summary"].items():
        target_rows.append([name, s["E_MAE"], s["E_RMSE"], s["E_MAPE_percent"], s["v_MAE"], s["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "target_external_test_metrics.csv"), target_rows)

    print(json.dumps({
        "source_random_split_summary": source_random_summary,
        "target_external_test_summary": external_summary,
        "best_external_model": summary["best_external_model"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
