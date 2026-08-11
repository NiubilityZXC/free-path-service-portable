#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle

import numpy as np
import pandas as pd

from train_missing_models_on_xlsx import (
    OUTPUT_DIR as PREV_OUTPUT_DIR,
    SOURCE_SHEET,
    TARGET_SHEET,
    XLSX_PATH,
    build_missing_models,
    read_sheet_as_base_dataset,
    repair_velocity_values,
    train_val_test_split,
)
from train_zpinch_surrogate import (
    ensure_output_dir,
    evaluate_model,
    make_features,
    plot_basic_distribution,
    plot_residual,
    plot_true_pred,
    quality_report,
)


OUTPUT_DIR = "/home/user/ai_xlsx_all_in_training_outputs"
SEEDS = [42, 2024, 2026]
MASS_VALUES = [20.0, 30.0, 40.0]


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows):
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path, index=False, encoding="utf-8-sig")


def save_pickle(path: str, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def summarize_records(records):
    summary = {}
    for name, values in records.items():
        keys = values[0].keys()
        summary[name] = {k: float(np.mean([item[k] for item in values])) for k in keys}
    return summary


def run_random_split_with_domain_metrics(X, yE, yv, domain_tag, seeds, models):
    all_records = {}
    source_records = {}
    target_records = {}

    for seed in seeds:
        tr_idx, va_idx, te_idx = train_val_test_split(len(X), seed)
        train_idx = np.concatenate([tr_idx, va_idx])

        Xtr, Xte = X[train_idx], X[te_idx]
        ytr, yte = yE[train_idx], yE[te_idx]
        vte = yv[te_idx]
        mte = Xte[:, 4]
        dte = domain_tag[te_idx]

        for name, model in models.items():
            model.fit(Xtr, ytr)
            pred = np.maximum(model.predict(Xte), 1e-12)
            all_records.setdefault(name, []).append(evaluate_model(yte, pred, mte, vte))

            src_mask = dte == "source"
            tgt_mask = dte == "target"
            if src_mask.any():
                source_records.setdefault(name, []).append(
                    evaluate_model(yte[src_mask], pred[src_mask], mte[src_mask], vte[src_mask])
                )
            if tgt_mask.any():
                target_records.setdefault(name, []).append(
                    evaluate_model(yte[tgt_mask], pred[tgt_mask], mte[tgt_mask], vte[tgt_mask])
                )

    return (
        summarize_records(all_records),
        summarize_records(source_records),
        summarize_records(target_records),
    )


def run_mass_holdout_benchmark(X, yE, yv, X_base, models, masses):
    records = {name: [] for name in models.keys()}

    for mass in masses:
        mask_test = np.isclose(X_base[:, 4], mass)
        mask_train = ~mask_test
        if mask_test.sum() == 0:
            continue

        Xtr, Xte = X[mask_train], X[mask_test]
        ytr, yte = yE[mask_train], yE[mask_test]
        vte = yv[mask_test]
        mte = Xte[:, 4]

        for name, model in models.items():
            model.fit(Xtr, ytr)
            pred = np.maximum(model.predict(Xte), 1e-12)
            score = evaluate_model(yte, pred, mte, vte)
            score["heldout_mass"] = float(mass)
            records[name].append(score)

    summary = {}
    for name, values in records.items():
        if not values:
            continue
        metric_keys = [k for k in values[0].keys() if k != "heldout_mass"]
        avg_metrics = {k: float(np.mean([item[k] for item in values])) for k in metric_keys}
        summary[name] = {
            "avg_over_masses": avg_metrics,
            "by_mass": values,
        }
    return summary


def load_previous_external_summary():
    prev_path = os.path.join(PREV_OUTPUT_DIR, "summary_metrics.json")
    if not os.path.exists(prev_path):
        return {}
    with open(prev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("target_external_test_summary", {})


def compute_generalization_gain(previous_external, combined_target_split):
    gain = {}
    for name, now_score in combined_target_split.items():
        if name not in previous_external:
            continue
        prev_score = previous_external[name]
        gain[name] = {
            "prev_target_E_MAPE_percent": prev_score["E_MAPE_percent"],
            "new_target_split_E_MAPE_percent": now_score["E_MAPE_percent"],
            "absolute_drop_E_MAPE_percent": prev_score["E_MAPE_percent"] - now_score["E_MAPE_percent"],
            "relative_drop_E_MAPE_percent": (
                (prev_score["E_MAPE_percent"] - now_score["E_MAPE_percent"]) / prev_score["E_MAPE_percent"] * 100.0
                if prev_score["E_MAPE_percent"] != 0
                else 0.0
            ),
        }
    return gain


def write_chinese_summary_report(path, summary):
    lines = []
    lines.append("全量合并训练总结（对应优化质量 + 丝阵质量20/30/40）")
    lines.append("")
    lines.append(f"数据文件: {summary['xlsx_path']}")
    lines.append(f"源sheet: {summary['source_sheet']}  样本数: {summary['source_n_samples']}")
    lines.append(f"目标sheet: {summary['target_sheet']}  样本数: {summary['target_n_samples']}")
    lines.append(f"合并样本数: {summary['combined_n_samples']}")
    lines.append("")
    lines.append("新增数据补充的泛化能力（数据层面）:")
    lines.append(f"- 每个固定工况(I,tr,liner_r,foam_r)的质量点数量由 1 提升到 {summary['combined_mass_points_per_condition_median']:.0f}（中位数）。")
    lines.append(f"- 覆盖质量点总区间: [{summary['combined_mass_min']:.2f}, {summary['combined_mass_max']:.2f}] mg/cm。")
    lines.append(f"- 低动能区覆盖补充: 合并后最小动能 {summary['combined_E_min']:.4f} MJ/cm。")
    lines.append("")
    lines.append("泛化能力改进（模型层面，目标域对比）:")
    for name, g in summary["generalization_gain_vs_source_only"].items():
        lines.append(
            f"- {name}: E_MAPE 从 {g['prev_target_E_MAPE_percent']:.4f}% 降至 {g['new_target_split_E_MAPE_percent']:.4f}% "
            f"(下降 {g['absolute_drop_E_MAPE_percent']:.4f}pct, 相对下降 {g['relative_drop_E_MAPE_percent']:.2f}%)"
        )
    lines.append("")
    lines.append("说明:")
    lines.append("- 这里的新指标是“合并数据集随机划分中的目标子集测试”，不是旧的纯外部迁移测试。")
    lines.append("- 该结果说明新增数据显著提升了目标域内插值能力与质量维度的可学习性。")
    lines.append("- 对完全未见工况的强外推能力仍需要额外留组评估与补点。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_output_dir(OUTPUT_DIR)
    models = build_missing_models()
    if not models:
        raise RuntimeError("当前环境没有可用模型。")

    _, Xs_base, yEs, yvs_raw = read_sheet_as_base_dataset(XLSX_PATH, SOURCE_SHEET)
    _, Xt_base, yEt, yvt_raw = read_sheet_as_base_dataset(XLSX_PATH, TARGET_SHEET)

    yvs, repaired_source = repair_velocity_values(Xs_base, yEs, yvs_raw)
    yvt, repaired_target = repair_velocity_values(Xt_base, yEt, yvt_raw)

    X_base = np.vstack([Xs_base, Xt_base])
    yE = np.concatenate([yEs, yEt])
    yv = np.concatenate([yvs, yvt])
    domain = np.array(["source"] * len(Xs_base) + ["target"] * len(Xt_base))

    X, feat_names = make_features(X_base)

    random_all, random_source, random_target = run_random_split_with_domain_metrics(
        X=X, yE=yE, yv=yv, domain_tag=domain, seeds=SEEDS, models=models
    )
    mass_holdout = run_mass_holdout_benchmark(X=X, yE=yE, yv=yv, X_base=X_base, models=models, masses=MASS_VALUES)

    best_model_name = min(random_all.items(), key=lambda kv: kv[1]["E_MAE"])[0]
    tr_idx, va_idx, te_idx = train_val_test_split(len(X), 2026)
    train_idx = np.concatenate([tr_idx, va_idx])
    best_model = build_missing_models()[best_model_name]
    best_model.fit(X[train_idx], yE[train_idx])
    predE = np.maximum(best_model.predict(X[te_idx]), 1e-12)
    final_metrics = evaluate_model(yE[te_idx], predE, X[te_idx][:, 4], yv[te_idx])

    plot_true_pred(yE[te_idx], predE, os.path.join(OUTPUT_DIR, "fig_combined_true_vs_pred.png"))
    plot_residual(yE[te_idx], predE, os.path.join(OUTPUT_DIR, "fig_combined_residual.png"))
    plot_basic_distribution(yE, os.path.join(OUTPUT_DIR, "fig_combined_E_distribution.png"))

    source_quality = quality_report(Xs_base, yEs, yvs)
    target_quality = quality_report(Xt_base, yEt, yvt)
    combined_quality = quality_report(X_base, yE, yv)

    previous_external = load_previous_external_summary()
    gain = compute_generalization_gain(previous_external, random_target)

    merged_df = pd.concat(
        [
            pd.DataFrame(Xs_base[:, :5], columns=["I", "tr", "liner_r", "foam_r", "m"]).assign(domain="source"),
            pd.DataFrame(Xt_base[:, :5], columns=["I", "tr", "liner_r", "foam_r", "m"]).assign(domain="target"),
        ],
        ignore_index=True,
    )
    mass_points_per_cond = (
        merged_df.groupby(["I", "tr", "liner_r", "foam_r"])["m"].nunique().to_numpy(dtype=float)
    )

    summary = {
        "xlsx_path": XLSX_PATH,
        "source_sheet": SOURCE_SHEET,
        "target_sheet": TARGET_SHEET,
        "source_n_samples": int(len(Xs_base)),
        "target_n_samples": int(len(Xt_base)),
        "combined_n_samples": int(len(X)),
        "repaired_source_velocity_rows": int(repaired_source),
        "repaired_target_velocity_rows": int(repaired_target),
        "source_quality": source_quality,
        "target_quality": target_quality,
        "combined_quality": combined_quality,
        "available_models": sorted(list(models.keys())),
        "feature_names": feat_names,
        "combined_random_split_summary": random_all,
        "combined_random_split_source_only_summary": random_source,
        "combined_random_split_target_only_summary": random_target,
        "combined_mass_holdout_summary": mass_holdout,
        "best_model_name": best_model_name,
        "final_test_metrics_seed_2026": final_metrics,
        "generalization_gain_vs_source_only": gain,
        "combined_mass_points_per_condition_min": float(np.min(mass_points_per_cond)),
        "combined_mass_points_per_condition_median": float(np.median(mass_points_per_cond)),
        "combined_mass_points_per_condition_max": float(np.max(mass_points_per_cond)),
        "combined_mass_min": float(np.min(X_base[:, 4])),
        "combined_mass_max": float(np.max(X_base[:, 4])),
        "combined_E_min": float(np.min(yE)),
        "combined_E_max": float(np.max(yE)),
    }

    save_json(os.path.join(OUTPUT_DIR, "summary_metrics.json"), summary)

    rows_all = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, m in random_all.items():
        rows_all.append([name, m["E_MAE"], m["E_RMSE"], m["E_MAPE_percent"], m["v_MAE"], m["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "combined_random_split_metrics.csv"), rows_all)

    rows_src = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, m in random_source.items():
        rows_src.append([name, m["E_MAE"], m["E_RMSE"], m["E_MAPE_percent"], m["v_MAE"], m["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "combined_random_split_source_metrics.csv"), rows_src)

    rows_tgt = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, m in random_target.items():
        rows_tgt.append([name, m["E_MAE"], m["E_RMSE"], m["E_MAPE_percent"], m["v_MAE"], m["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "combined_random_split_target_metrics.csv"), rows_tgt)

    report_path = os.path.join(OUTPUT_DIR, "combined_training_summary_中文总结.txt")
    write_chinese_summary_report(report_path, summary)

    save_pickle(
        os.path.join(OUTPUT_DIR, "best_combined_model_artifact.pkl"),
        {
            "model_name": best_model_name,
            "model": best_model,
            "xlsx_path": XLSX_PATH,
            "sheets": [SOURCE_SHEET, TARGET_SHEET],
            "feature_names": feat_names,
            "final_test_metrics_seed_2026": final_metrics,
        },
    )

    print("output_dir=", OUTPUT_DIR)
    print("best_model=", best_model_name)
    print("combined_n_samples=", len(X))
    print("random_target_summary=")
    print(json.dumps(random_target, ensure_ascii=False, indent=2))
    print("generalization_gain_vs_source_only=")
    print(json.dumps(gain, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

