#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from train_zpinch_surrogate import (
    calc_v_from_E_m,
    ensure_output_dir,
    evaluate_model,
    make_features,
    mape,
    plot_basic_distribution,
    plot_residual,
    plot_true_pred,
    quality_report,
)

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


XLSX_PATH = "/home/user/zpinch_package_20260306/零维表格转换 (2).xlsx"
SOURCE_SHEET = "对应优化质量"
TARGET_SHEET = "丝阵质量20_30_40mgcm"
OUTPUT_DIR = "/home/user/ai_xlsx_cross_test_outputs"


def read_sheet_as_base_dataset(xlsx_path: str, sheet_name: str):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    rename_map = {
        "电流(MA)": "I_MA",
        "上升时间(ns)": "tr_ns",
        "套筒半径(cm)": "liner_r_cm",
        "泡沫半径(cm)": "foam_r_cm",
        "套筒质量(mg/cm)": "m_mg_per_cm",
        "动能(MJ/cm)": "E_MJ_per_cm",
        "速度(cm/s)": "v_cm_per_s",
    }
    df = df.rename(columns=rename_map)
    needed = [
        "I_MA",
        "tr_ns",
        "liner_r_cm",
        "foam_r_cm",
        "m_mg_per_cm",
        "E_MJ_per_cm",
        "v_cm_per_s",
    ]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{sheet_name} 缺少列: {missing_cols}")

    # 保持和原始代码一致，Z 先固定常数。
    z_col = np.full(len(df), 13.0, dtype=float)
    X_base = np.column_stack(
        [
            df["I_MA"].to_numpy(dtype=float),
            df["tr_ns"].to_numpy(dtype=float),
            df["liner_r_cm"].to_numpy(dtype=float),
            df["foam_r_cm"].to_numpy(dtype=float),
            df["m_mg_per_cm"].to_numpy(dtype=float),
            z_col,
        ]
    )
    yE = df["E_MJ_per_cm"].to_numpy(dtype=float)
    yv = df["v_cm_per_s"].to_numpy(dtype=float)
    return df, X_base, yE, yv


def repair_velocity_values(X_base: np.ndarray, yE: np.ndarray, yv: np.ndarray):
    v_back = calc_v_from_E_m(yE, X_base[:, 4])
    denom = np.maximum(np.abs(yv), 1e-12)
    rel_percent = np.abs((yv - v_back) / denom) * 100.0
    bad_mask = (np.abs(yv) < 1.0e5) | (rel_percent > 5.0)
    repaired = yv.copy()
    repaired[bad_mask] = v_back[bad_mask]
    return repaired, int(bad_mask.sum())


def train_val_test_split(n: int, seed: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def build_missing_models():
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        ),
    }
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            verbose=-1,
        )
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
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
        )
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            random_seed=42,
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            verbose=False,
        )
    return models


def run_random_split_benchmark(X: np.ndarray, yE: np.ndarray, yv: np.ndarray, seeds, models=None):
    records = {}
    bench_models = build_missing_models() if models is None else models
    for seed in seeds:
        tr_idx, va_idx, te_idx = train_val_test_split(len(X), seed)
        train_idx = np.concatenate([tr_idx, va_idx])
        Xtr, Xte = X[train_idx], X[te_idx]
        ytr, yte = yE[train_idx], yE[te_idx]
        vte = yv[te_idx]
        mte = Xte[:, 4]

        for name, model in bench_models.items():
            model.fit(Xtr, ytr)
            pred = np.maximum(model.predict(Xte), 1e-12)
            score = evaluate_model(yte, pred, mte, vte)
            records.setdefault(name, []).append(score)

    summary = {}
    for name, values in records.items():
        keys = values[0].keys()
        summary[name] = {k: float(np.mean([item[k] for item in values])) for k in keys}
    return summary


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows):
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path, index=False, encoding="utf-8-sig")


def save_model_pickle(path: str, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def plot_external_comparison(external_summary, out_path: str):
    import matplotlib.pyplot as plt

    names = list(external_summary.keys())
    maes = [external_summary[n]["E_MAE"] for n in names]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, maes, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plt.title("External Test on 20/30/40 mg/cm - E MAE")
    plt.ylabel("E MAE (MJ/cm)")
    for bar, value in zip(bars, maes):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ensure_output_dir(OUTPUT_DIR)
    available_models = list(build_missing_models().keys())
    if not available_models:
        raise RuntimeError("当前环境没有可用的补充模型。")

    _, Xs_base, yEs, yvs_raw = read_sheet_as_base_dataset(XLSX_PATH, SOURCE_SHEET)
    _, Xt_base, yEt, yvt_raw = read_sheet_as_base_dataset(XLSX_PATH, TARGET_SHEET)

    yvs, repaired_source = repair_velocity_values(Xs_base, yEs, yvs_raw)
    yvt, repaired_target = repair_velocity_values(Xt_base, yEt, yvt_raw)

    source_quality = quality_report(Xs_base, yEs, yvs)
    target_quality = quality_report(Xt_base, yEt, yvt)

    Xs, feat_names = make_features(Xs_base)
    Xt, _ = make_features(Xt_base)

    source_random_summary = run_random_split_benchmark(Xs, yEs, yvs, seeds=[42, 2024, 2026])

    external_summary = {}
    best_external_model_name = None
    best_external_mae = None

    for name, model in build_missing_models().items():
        model.fit(Xs, yEs)
        predE_target = np.maximum(model.predict(Xt), 1e-12)
        score_target = evaluate_model(yEt, predE_target, Xt[:, 4], yvt)
        external_summary[name] = score_target

        save_model_pickle(
            os.path.join(OUTPUT_DIR, f"{name}_artifact.pkl"),
            {
                "model_name": name,
                "model": model,
                "source_sheet": SOURCE_SHEET,
                "target_sheet": TARGET_SHEET,
                "xlsx_path": XLSX_PATH,
                "feature_names": feat_names,
                "target_fullset_metrics": score_target,
            },
        )

        if best_external_mae is None or score_target["E_MAE"] < best_external_mae:
            best_external_mae = score_target["E_MAE"]
            best_external_model_name = name

    best_model = build_missing_models()[best_external_model_name]
    best_model.fit(Xs, yEs)
    best_predE_target = np.maximum(best_model.predict(Xt), 1e-12)

    plot_true_pred(yEt, best_predE_target, os.path.join(OUTPUT_DIR, "fig_best_external_true_vs_pred.png"))
    plot_residual(yEt, best_predE_target, os.path.join(OUTPUT_DIR, "fig_best_external_residual.png"))
    plot_external_comparison(external_summary, os.path.join(OUTPUT_DIR, "fig_external_model_comparison.png"))
    plot_basic_distribution(yEt, os.path.join(OUTPUT_DIR, "fig_target_E_distribution.png"))

    source_rows = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, s in source_random_summary.items():
        source_rows.append([name, s["E_MAE"], s["E_RMSE"], s["E_MAPE_percent"], s["v_MAE"], s["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "source_random_split_metrics.csv"), source_rows)

    target_rows = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, s in external_summary.items():
        target_rows.append([name, s["E_MAE"], s["E_RMSE"], s["E_MAPE_percent"], s["v_MAE"], s["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "target_external_test_metrics.csv"), target_rows)

    summary = {
        "xlsx_path": XLSX_PATH,
        "source_sheet": SOURCE_SHEET,
        "target_sheet": TARGET_SHEET,
        "source_n_samples": int(len(Xs)),
        "target_n_samples": int(len(Xt)),
        "repaired_source_velocity_rows": repaired_source,
        "repaired_target_velocity_rows": repaired_target,
        "source_quality": source_quality,
        "target_quality": target_quality,
        "source_random_split_summary": source_random_summary,
        "target_external_test_summary": external_summary,
        "best_external_model": best_external_model_name,
        "available_models": available_models,
        "feature_names": feat_names,
    }
    save_json(os.path.join(OUTPUT_DIR, "summary_metrics.json"), summary)

    print("xlsx_path=", XLSX_PATH)
    print("source_sheet=", SOURCE_SHEET)
    print("target_sheet=", TARGET_SHEET)
    print("source_n_samples=", len(Xs))
    print("target_n_samples=", len(Xt))
    print("available_models=", available_models)
    print("best_external_model=", best_external_model_name)
    print("source_random_split_summary=")
    print(json.dumps(source_random_summary, ensure_ascii=False, indent=2))
    print("target_external_test_summary=")
    print(json.dumps(external_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
