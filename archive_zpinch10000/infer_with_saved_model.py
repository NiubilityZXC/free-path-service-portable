#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np

from train_zpinch_surrogate import (
    BEST_MODEL_ARTIFACT_PATH,
    build_dataset,
    calc_v_from_E_m,
    evaluate_model,
    extract_rows_from_doc,
    load_model_artifact,
    make_features,
)


def find_latest_artifact(root_dir: str = "/home/user") -> Path:
    """自动选择最近修改的模型权重文件。"""
    root = Path(root_dir)
    candidates = sorted(root.rglob("*artifact*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = load_model_artifact(str(candidate))
        except Exception:
            continue
        if isinstance(payload, dict) and "model" in payload and "model_name" in payload:
            return candidate
    raise FileNotFoundError("未找到可用的模型权重文件。")


def repair_velocity_values(X_base, yE, yv):
    """修复明显损坏的速度值，避免评估时被坏数据主导。"""
    v_back = calc_v_from_E_m(yE, X_base[:, 4])
    denom = np.maximum(np.abs(yv), 1e-12)
    rel_percent = np.abs((yv - v_back) / denom) * 100.0
    bad_mask = (np.abs(yv) < 1.0e5) | (rel_percent > 5.0)
    repaired = yv.copy()
    repaired[bad_mask] = v_back[bad_mask]
    return repaired, int(bad_mask.sum())


def print_preview(X_base, yE_true, yE_pred, yv_true, yv_pred, limit: int):
    print("\n预测结果预览:")
    print("idx,I_MA,tr_ns,liner_r_cm,foam_r_cm,m_mg_per_cm,E_true,E_pred,v_true,v_pred")
    preview_n = min(limit, len(X_base))
    for i in range(preview_n):
        row = X_base[i]
        print(
            f"{i},"
            f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f},"
            f"{yE_true[i]:.6f},{yE_pred[i]:.6f},{yv_true[i]:.6e},{yv_pred[i]:.6e}"
        )


def main():
    parser = argparse.ArgumentParser(description="使用已保存权重文件对其他 .doc 数据做推理")
    parser.add_argument("target_doc", help="待推理的目标数据 .doc 文件路径")
    parser.add_argument(
        "--artifact-path",
        default=None,
        help=f"模型权重路径；不传时自动选择最新权重，默认候选包含 {BEST_MODEL_ARTIFACT_PATH}",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="终端预览输出多少条样本预测结果",
    )
    args = parser.parse_args()

    target_doc = Path(args.target_doc)
    if not target_doc.exists():
        raise FileNotFoundError(f"找不到目标数据文件: {target_doc}")

    artifact_path = Path(args.artifact_path) if args.artifact_path else find_latest_artifact()
    payload = load_model_artifact(str(artifact_path))
    model = payload["model"]
    model_name = payload["model_name"]
    source_doc = payload.get("doc_path", "unknown")

    raw_rows = extract_rows_from_doc(str(target_doc))
    X_base, yE_true, yv_raw, _ = build_dataset(raw_rows)
    yv_true, repaired_velocity_rows = repair_velocity_values(X_base, yE_true, yv_raw)
    X_feat, _ = make_features(X_base)

    yE_pred = np.maximum(model.predict(X_feat), 1e-12)
    yv_pred = calc_v_from_E_m(yE_pred, X_base[:, 4])
    metrics = evaluate_model(yE_true, yE_pred, X_base[:, 4], yv_true)

    print("===== 推理配置 =====")
    print(f"选择的权重文件: {artifact_path}")
    print(f"权重对应模型: {model_name}")
    print(f"权重来源数据: {source_doc}")
    print(f"推理目标数据: {target_doc}")
    print(f"目标样本数: {len(X_base)}")
    print(f"自动修复的速度异常行数: {repaired_velocity_rows}")

    print("\n===== 推理评估结果 =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print_preview(X_base, yE_true, yE_pred, yv_true, yv_pred, args.preview_rows)


if __name__ == "__main__":
    main()
