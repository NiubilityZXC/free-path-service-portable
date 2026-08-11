#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
手动输入参数进行推理：
1) 自动从原始 .doc 重新抽取训练数据；
2) 使用最佳模型 Poly2Ridge 在全量数据上训练；
3) 对手动输入参数输出 E 和由物理公式反算的 v。
"""

import argparse
from pathlib import Path

import numpy as np

from train_zpinch_surrogate import (
    BEST_MODEL_ARTIFACT_PATH,
    DOC_DATA_PATH,
    CONST_Z,
    build_dataset,
    make_features,
    Poly2RidgeModel,
    calc_v_from_E_m,
    extract_rows_from_doc,
    load_model_artifact,
)


def main():
    parser = argparse.ArgumentParser(description="手动输入参数，推理 E 和 v")
    parser.add_argument("--I", type=float, help="电流 I (MA)")
    parser.add_argument("--tr", type=float, help="上升时间 tr (ns)")
    parser.add_argument("--liner_r", type=float, help="套筒半径 liner_r (cm)")
    parser.add_argument("--foam_r", type=float, help="泡沫半径 foam_r (cm)")
    parser.add_argument("--m", type=float, help="套筒质量 m (mg/cm)")
    parser.add_argument("--Z", type=float, default=CONST_Z, help="原子序数 Z（默认常数）")
    args = parser.parse_args()

    # 如果命令行没有给参数，则进入交互输入。
    if args.I is None:
        args.I = float(input("请输入 I (MA): ").strip())
        args.tr = float(input("请输入 tr (ns): ").strip())
        args.liner_r = float(input("请输入 liner_r (cm): ").strip())
        args.foam_r = float(input("请输入 foam_r (cm): ").strip())
        args.m = float(input("请输入 m (mg/cm): ").strip())
        z_in = input(f"请输入 Z (默认 {CONST_Z}): ").strip()
        args.Z = float(z_in) if z_in else CONST_Z

    # 优先加载已保存的最佳模型权重；如果不存在，则退回到即时重训。
    if Path(BEST_MODEL_ARTIFACT_PATH).exists():
        payload = load_model_artifact(BEST_MODEL_ARTIFACT_PATH)
        model = payload["model"]
        model_name = payload["model_name"]
    else:
        raw = extract_rows_from_doc(DOC_DATA_PATH)
        X_base, yE, _, _ = build_dataset(raw)
        X_train, _ = make_features(X_base)
        model = Poly2RidgeModel(alpha=1e-2)
        model.fit(X_train, yE)
        model_name = "Poly2Ridge(fallback_retrain)"

    # 组织手动输入为基础特征，再做同样特征工程。
    x_base = np.array([[args.I, args.tr, args.liner_r, args.foam_r, args.m, args.Z]], dtype=float)
    x_feat, _ = make_features(x_base)

    # 推理 E，再由公式反算 v。
    E_pred = float(max(model.predict(x_feat)[0], 1e-12))
    v_pred = float(calc_v_from_E_m(np.array([E_pred]), np.array([args.m]))[0])

    print("\n===== 推理结果 =====")
    print(f"使用模型: {model_name}")
    print(f"输入: I={args.I:.6f} MA, tr={args.tr:.6f} ns, liner_r={args.liner_r:.6f} cm, foam_r={args.foam_r:.6f} cm, m={args.m:.6f} mg/cm, Z={args.Z:.6f}")
    print(f"预测动能 E: {E_pred:.6f} MJ/cm")
    print(f"反算速度 v: {v_pred:.6e} cm/s")


if __name__ == "__main__":
    main()
