#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
零维 Z 箍缩代理模型训练脚本（最小依赖版本）
说明：
1) 从 .doc 文件中抽取 7 列数值数据；
2) 按方案执行数据校验、特征工程与多模型训练；
3) 输出图表、指标和报告到本地目录；
4) 只预测 E（动能），再按物理公式反算 v（速度）。
"""

import csv
import json
import math
import os
import pickle
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# 0) 基础路径和常量配置
# -----------------------------
DOC_DATA_PATH = "/home/user/零维计算结果（对应优化质量）_40MA-60MA_150ns-600ns.doc"
OUTPUT_DIR = "/home/user/ai_training_outputs"
REPORT_DOC_PATH = "/home/user/零维计算结果（对应优化质量）_40MA-60MA_150ns-600ns_训练报告.doc"
NOTEBOOK_PATH = "/home/user/zpinch_surrogate_training.ipynb"
BEST_MODEL_ARTIFACT_PATH = os.path.join(OUTPUT_DIR, "best_model_artifact.pkl")

# 当前数据没有材料变化，这里固定一个原子序数字段，保留接口一致性。
CONST_Z = 13.0

# 物理换算常数：v^2 = 2e16 * E / m  (E: MJ/cm, m: mg/cm, v: cm/s)
V_FORMULA_COEF = 2.0e16


# -----------------------------
# 1) 数据抽取与结构化
# -----------------------------
def extract_rows_from_doc(doc_path: str) -> np.ndarray:
    """从 .doc 中通过 strings -el 抽取数值行，并解析为二维数组。"""
    # 读取 UTF-16LE 可见字符串，文档表格数字在该输出中可见。
    proc = subprocess.run(
        ["strings", "-el", doc_path],
        check=True,
        capture_output=True,
        text=True,
    )

    # 行模式：7 列数值（最后一列可能是科学计数法）。
    row_pattern = re.compile(
        r"^\s*"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
    )

    rows: List[List[float]] = []
    for line in proc.stdout.splitlines():
        m = row_pattern.match(line)
        if m:
            rows.append([float(x) for x in m.groups()])

    if not rows:
        raise RuntimeError("未从 .doc 中提取到可训练数据行。")

    return np.array(rows, dtype=float)


def build_dataset(raw_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """构造 X, y_E, y_v 和列名。"""
    # 原始列顺序（按文档）：I, tr, liner_r, foam_r, m, E, v
    I = raw_rows[:, 0]
    tr = raw_rows[:, 1]
    liner_r = raw_rows[:, 2]
    foam_r = raw_rows[:, 3]
    m = raw_rows[:, 4]
    E = raw_rows[:, 5]
    v = raw_rows[:, 6]

    # 将 Z 常数列并入输入，以保持后续多材料扩展兼容。
    Z = np.full_like(I, CONST_Z)

    # 拼接基础输入矩阵。
    X_base = np.column_stack([I, tr, liner_r, foam_r, m, Z])
    base_names = [
        "I_MA",
        "tr_ns",
        "liner_r_cm",
        "foam_r_cm",
        "m_mg_per_cm",
        "Z_atomic_number",
    ]

    return X_base, E, v, base_names


# -----------------------------
# 2) 质量检查与物理一致性
# -----------------------------
def calc_v_from_E_m(E: np.ndarray, m: np.ndarray) -> np.ndarray:
    """由 E 和 m 反算速度，方向按内爆取负号。"""
    safe = np.maximum(m, 1e-12)
    return -np.sqrt(V_FORMULA_COEF * np.maximum(E, 0.0) / safe)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """计算 MAPE（百分比）。"""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 MAE。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def quality_report(X_base: np.ndarray, y_E: np.ndarray, y_v: np.ndarray) -> Dict[str, float]:
    """输出数据质量报告字典。"""
    # 缺失检查：numpy 数组中 NaN 即视为缺失。
    missing_count = int(np.isnan(X_base).sum() + np.isnan(y_E).sum() + np.isnan(y_v).sum())

    # 重复检查：按全部基础输入+输出拼接后看去重数量。
    full = np.column_stack([X_base, y_E, y_v])
    unique_count = int(np.unique(full, axis=0).shape[0])
    duplicate_count = int(full.shape[0] - unique_count)

    # 物理一致性：用 E,m 反算 v 并与文档 v 比较。
    m_col = X_base[:, 4]
    v_back = calc_v_from_E_m(y_E, m_col)
    v_mae = mae(y_v, v_back)
    v_mape = mape(y_v, v_back)

    return {
        "n_samples": int(full.shape[0]),
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "physics_v_backcalc_mae": v_mae,
        "physics_v_backcalc_mape_percent": v_mape,
    }


# -----------------------------
# 3) 特征工程（基础 + 物理启发）
# -----------------------------
def make_features(X_base: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """构造用于学习 E 的扩展特征。"""
    I = X_base[:, 0]
    tr = X_base[:, 1]
    liner_r = X_base[:, 2]
    foam_r = X_base[:, 3]
    m = X_base[:, 4]
    Z = X_base[:, 5]

    # 物理启发特征：几何比、间隙、面积差、驱动强度代理和交叉项。
    ratio_rf = foam_r / np.maximum(liner_r, 1e-12)
    gap_r = liner_r - foam_r
    area_diff = math.pi * (np.maximum(liner_r, 0.0) ** 2 - np.maximum(foam_r, 0.0) ** 2)
    drive_proxy = I / np.maximum(tr, 1e-12)

    cross_I_liner = I * liner_r
    cross_I_foam = I * foam_r
    cross_tr_liner = tr * liner_r
    cross_tr_foam = tr * foam_r

    # 拼接总特征。
    X = np.column_stack(
        [
            I,
            tr,
            liner_r,
            foam_r,
            m,
            Z,
            ratio_rf,
            gap_r,
            area_diff,
            drive_proxy,
            cross_I_liner,
            cross_I_foam,
            cross_tr_liner,
            cross_tr_foam,
        ]
    )

    names = [
        "I_MA",
        "tr_ns",
        "liner_r_cm",
        "foam_r_cm",
        "m_mg_per_cm",
        "Z_atomic_number",
        "ratio_foam_liner",
        "gap_r_cm",
        "area_diff_cm2",
        "drive_proxy_I_over_tr",
        "I_mul_liner",
        "I_mul_foam",
        "tr_mul_liner",
        "tr_mul_foam",
    ]
    return X, names


# -----------------------------
# 4) 模型实现（不依赖外部 ML 框架）
# -----------------------------
@dataclass
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        m = X.mean(axis=0)
        s = X.std(axis=0)
        s = np.where(s < 1e-12, 1.0, s)
        return cls(mean_=m, std_=s)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """闭式解 Ridge。"""
    xtx = X.T @ X
    reg = np.eye(X.shape[1]) * alpha
    reg[0, 0] = 0.0
    w = np.linalg.pinv(xtx + reg) @ X.T @ y
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """线性预测。"""
    return X @ w


def add_bias(X: np.ndarray) -> np.ndarray:
    """加截距列。"""
    return np.column_stack([np.ones(X.shape[0]), X])


def poly2_expand(X: np.ndarray) -> np.ndarray:
    """二阶多项式展开（含一次项、平方项、两两交叉项）。"""
    n, d = X.shape
    cols = [X]

    # 平方项。
    cols.append(X * X)

    # 交叉项。
    cross_list = []
    for i in range(d):
        for j in range(i + 1, d):
            cross_list.append((X[:, i] * X[:, j]).reshape(n, 1))

    if cross_list:
        cols.append(np.hstack(cross_list))

    return np.hstack(cols)


class PowerLawModel:
    """幂律回归：log(E)=b0+sum(bi*log(xi_shift))."""

    def __init__(self):
        self.w = None
        self.shift = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 对每一列做正平移，避免 log(<=0)。
        min_col = X.min(axis=0)
        self.shift = np.where(min_col <= 0.0, 1.0 - min_col, 0.0)
        X_pos = X + self.shift

        # 对目标也做下限裁剪，避免 log(0)。
        y_pos = np.maximum(y, 1e-12)

        # 在 log 空间做线性回归。
        X_log = np.log(np.maximum(X_pos, 1e-12))
        y_log = np.log(y_pos)
        X_design = add_bias(X_log)
        self.w = ridge_fit(X_design, y_log, alpha=1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_pos = X + self.shift
        X_log = np.log(np.maximum(X_pos, 1e-12))
        y_log = ridge_predict(add_bias(X_log), self.w)
        return np.exp(y_log)


class Poly2RidgeModel:
    """二阶多项式 + Ridge。"""

    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self.std = None
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.std = Standardizer.fit(X)
        Xs = self.std.transform(X)
        Xp = poly2_expand(Xs)
        self.w = ridge_fit(add_bias(Xp), y, alpha=self.alpha)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.std.transform(X)
        Xp = poly2_expand(Xs)
        return ridge_predict(add_bias(Xp), self.w)


class KNNRegressorModel:
    """KNN 回归（numpy 实现）。"""

    def __init__(self, k: int = 8):
        self.k = k
        self.std = None
        self.Xtr = None
        self.ytr = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.std = Standardizer.fit(X)
        self.Xtr = self.std.transform(X)
        self.ytr = y.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xq = self.std.transform(X)
        preds = np.zeros(Xq.shape[0], dtype=float)
        for i in range(Xq.shape[0]):
            # 逐样本算欧式距离并取前 k 个。
            d2 = np.sum((self.Xtr - Xq[i]) ** 2, axis=1)
            idx = np.argpartition(d2, self.k)[: self.k]
            # 用距离倒数做加权平均，近邻更有影响力。
            w = 1.0 / np.maximum(np.sqrt(d2[idx]), 1e-12)
            preds[i] = float(np.sum(w * self.ytr[idx]) / np.sum(w))
        return preds


class MLP2LayerModel:
    """两层 MLP（numpy 手写，ReLU + Adam）。"""

    def __init__(self, h1: int = 64, h2: int = 32, lr: float = 1e-3, epochs: int = 250, seed: int = 42):
        self.h1 = h1
        self.h2 = h2
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.std_x = None
        self.std_y = None
        self.params = {}
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.loss_history_ = []

        # 标准化输入和目标，提高收敛稳定性。
        self.std_x = Standardizer.fit(X)
        Xn = self.std_x.transform(X)
        y_mean = y.mean()
        y_std = y.std() if y.std() > 1e-12 else 1.0
        self.std_y = (y_mean, y_std)
        yn = ((y - y_mean) / y_std).reshape(-1, 1)

        n, d = Xn.shape
        h1, h2 = self.h1, self.h2

        # Xavier 风格初始化。
        W1 = rng.normal(0.0, np.sqrt(2.0 / (d + h1)), size=(d, h1))
        b1 = np.zeros((1, h1))
        W2 = rng.normal(0.0, np.sqrt(2.0 / (h1 + h2)), size=(h1, h2))
        b2 = np.zeros((1, h2))
        W3 = rng.normal(0.0, np.sqrt(2.0 / (h2 + 1)), size=(h2, 1))
        b3 = np.zeros((1, 1))

        # Adam 一阶/二阶矩。
        m = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(W3), np.zeros_like(b3)]
        v = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(W3), np.zeros_like(b3)]

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.epochs + 1):
            # 前向传播。
            z1 = Xn @ W1 + b1
            a1 = np.maximum(z1, 0.0)
            z2 = a1 @ W2 + b2
            a2 = np.maximum(z2, 0.0)
            yhat = a2 @ W3 + b3

            # 均方误差梯度。
            diff = (yhat - yn)
            self.loss_history_.append(float(np.mean(diff ** 2)))
            dy = (2.0 / n) * diff

            # 反向传播。
            dW3 = a2.T @ dy
            db3 = np.sum(dy, axis=0, keepdims=True)

            da2 = dy @ W3.T
            dz2 = da2 * (z2 > 0.0)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0.0)
            dW1 = Xn.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            grads = [dW1, db1, dW2, db2, dW3, db3]
            params = [W1, b1, W2, b2, W3, b3]

            # Adam 更新。
            for i in range(len(params)):
                m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i]
                v[i] = beta2 * v[i] + (1.0 - beta2) * (grads[i] ** 2)
                m_hat = m[i] / (1.0 - beta1 ** t)
                v_hat = v[i] / (1.0 - beta2 ** t)
                params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self.params = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = self.std_x.transform(X)
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        z1 = Xn @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ W2 + b2
        a2 = np.maximum(z2, 0.0)
        yhat_n = a2 @ W3 + b3

        y_mean, y_std = self.std_y
        yhat = yhat_n.reshape(-1) * y_std + y_mean
        return yhat


# -----------------------------
# 5) 评估流程（随机划分 + 分组留出）
# -----------------------------
def train_val_test_split(n: int, seed: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def evaluate_model(y_true_E: np.ndarray, y_pred_E: np.ndarray, m_input: np.ndarray, y_true_v: np.ndarray) -> Dict[str, float]:
    """统一计算 E 与派生 v 的评价指标。"""
    pred_E = np.maximum(y_pred_E, 1e-12)
    pred_v = calc_v_from_E_m(pred_E, m_input)

    return {
        "E_MAE": mae(y_true_E, pred_E),
        "E_RMSE": rmse(y_true_E, pred_E),
        "E_MAPE_percent": mape(y_true_E, pred_E),
        "v_MAE": mae(y_true_v, pred_v),
        "v_MAPE_percent": mape(y_true_v, pred_v),
    }


def build_models() -> Dict[str, object]:
    """返回模型字典。"""
    return {
        "PowerLaw": PowerLawModel(),
        "Poly2Ridge": Poly2RidgeModel(alpha=1e-2),
        "KNN": KNNRegressorModel(k=8),
        "MLP": MLP2LayerModel(h1=64, h2=32, lr=1e-3, epochs=220, seed=42),
    }


def run_random_split_benchmark(X: np.ndarray, yE: np.ndarray, yv: np.ndarray, seeds: List[int]) -> Dict[str, Dict[str, float]]:
    """随机划分重复实验，并返回各模型平均指标。"""
    records: Dict[str, List[Dict[str, float]]] = {}

    for seed in seeds:
        tr_idx, va_idx, te_idx = train_val_test_split(len(X), seed)
        train_idx = np.concatenate([tr_idx, va_idx])

        Xtr, Xte = X[train_idx], X[te_idx]
        ytr, yte = yE[train_idx], yE[te_idx]
        vte = yv[te_idx]
        mte = Xte[:, 4]

        for name, model in build_models().items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            score = evaluate_model(yte, pred, mte, vte)
            records.setdefault(name, []).append(score)

    # 求各指标均值。
    summary: Dict[str, Dict[str, float]] = {}
    for name, lst in records.items():
        keys = lst[0].keys()
        summary[name] = {k: float(np.mean([item[k] for item in lst])) for k in keys}
    return summary


def run_group_holdout_benchmark(X: np.ndarray, yE: np.ndarray, yv: np.ndarray, group_col: int) -> Dict[str, Dict[str, float]]:
    """按某列做分组留出（整组外推）。"""
    groups = np.unique(X[:, group_col])
    records: Dict[str, List[Dict[str, float]]] = {}

    for g in groups:
        test_mask = X[:, group_col] == g
        train_mask = ~test_mask

        Xtr, Xte = X[train_mask], X[test_mask]
        ytr, yte = yE[train_mask], yE[test_mask]
        vte = yv[test_mask]
        mte = Xte[:, 4]

        for name, model in build_models().items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            score = evaluate_model(yte, pred, mte, vte)
            records.setdefault(name, []).append(score)

    summary: Dict[str, Dict[str, float]] = {}
    for name, lst in records.items():
        keys = lst[0].keys()
        summary[name] = {k: float(np.mean([item[k] for item in lst])) for k in keys}
    return summary


# -----------------------------
# 6) 结果可视化与导出
# -----------------------------
def ensure_output_dir(path: str):
    """创建输出目录。"""
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, rows: List[List[object]]):
    """写 CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)


def save_model_artifact(
    artifact_path: str,
    model_name: str,
    model: object,
    doc_path: str,
    feature_names: List[str],
    base_feature_names: List[str],
    final_seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    final_metrics: Dict[str, float],
):
    """保存最终最佳模型及其上下文，便于后续直接加载推理。"""
    payload = {
        "model_name": model_name,
        "model": model,
        "doc_path": doc_path,
        "feature_names": feature_names,
        "base_feature_names": base_feature_names,
        "final_seed": final_seed,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "final_test_metrics": final_metrics,
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(payload, f)


def load_model_artifact(artifact_path: str = BEST_MODEL_ARTIFACT_PATH):
    """加载已保存的最佳模型权重。"""
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def plot_basic_distribution(yE: np.ndarray, out_path: str):
    """动能分布图。"""
    plt.figure(figsize=(8, 5))
    plt.hist(yE, bins=40, color="#4C78A8", edgecolor="white")
    plt.title("Distribution of Kinetic Energy E")
    plt.xlabel("E (MJ/cm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(random_summary: Dict[str, Dict[str, float]], out_path: str):
    """模型 E_MAE 对比柱状图。"""
    names = list(random_summary.keys())
    maes = [random_summary[n]["E_MAE"] for n in names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, maes, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plt.title("Model Comparison (Random Split) - E MAE")
    plt.ylabel("MAE (MJ/cm)")
    for b, v in zip(bars, maes):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_true_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """真实值-预测值散点图。"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=14, alpha=0.65, color="#4C78A8")
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    plt.title("True vs Predicted E")
    plt.xlabel("True E (MJ/cm)")
    plt.ylabel("Predicted E (MJ/cm)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """残差图。"""
    res = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, res, s=14, alpha=0.65, color="#F58518")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Residual Plot")
    plt.xlabel("Predicted E (MJ/cm)")
    plt.ylabel("Residual (Pred - True)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# 7) Jupyter Notebook 生成
# -----------------------------
def write_notebook_from_script(script_path: str, notebook_path: str):
    """将脚本包成一个可直接打开的 ipynb。"""
    with open(script_path, "r", encoding="utf-8") as f:
        code_text = f.read()

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Z箍缩0D代理模型训练笔记本\n",
                    "\n",
                    "本笔记本按方案执行：数据提取、质量校验、特征工程、模型训练、图表输出与报告导出。\n",
                    "仅预测动能E，再通过物理公式反算速度v。\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_text.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)


# -----------------------------
# 8) 主流程
# -----------------------------
def main():
    ensure_output_dir(OUTPUT_DIR)

    # 读取并构造数据。
    raw_rows = extract_rows_from_doc(DOC_DATA_PATH)
    X_base, yE, yv, base_names = build_dataset(raw_rows)

    # 质量检查。
    q = quality_report(X_base, yE, yv)

    # 特征工程。
    X, feat_names = make_features(X_base)

    # 随机划分评估。
    random_summary = run_random_split_benchmark(X, yE, yv, seeds=[42, 2024, 2026])

    # 分组留出评估：按电流（col0）和上升时间（col1）。
    group_I_summary = run_group_holdout_benchmark(X, yE, yv, group_col=0)
    group_tr_summary = run_group_holdout_benchmark(X, yE, yv, group_col=1)

    # 选主模型：随机划分 E_MAE 最低者。
    best_model_name = min(random_summary.keys(), key=lambda n: random_summary[n]["E_MAE"])

    # 用固定随机划分做最终可视化测试。
    tr_idx, va_idx, te_idx = train_val_test_split(len(X), seed=2026)
    train_idx = np.concatenate([tr_idx, va_idx])
    Xtr, Xte = X[train_idx], X[te_idx]
    ytr, yte = yE[train_idx], yE[te_idx]
    vte = yv[te_idx]
    mte = Xte[:, 4]

    model = build_models()[best_model_name]
    model.fit(Xtr, ytr)
    predE = np.maximum(model.predict(Xte), 1e-12)
    final_metrics = evaluate_model(yte, predE, mte, vte)

    save_model_artifact(
        BEST_MODEL_ARTIFACT_PATH,
        best_model_name,
        model,
        DOC_DATA_PATH,
        feat_names,
        base_names,
        2026,
        train_idx,
        te_idx,
        final_metrics,
    )

    # 画图输出。
    fig_dist = os.path.join(OUTPUT_DIR, "fig_01_E_distribution.png")
    fig_cmp = os.path.join(OUTPUT_DIR, "fig_02_model_mae_comparison.png")
    fig_tp = os.path.join(OUTPUT_DIR, "fig_03_true_vs_pred.png")
    fig_res = os.path.join(OUTPUT_DIR, "fig_04_residual.png")

    plot_basic_distribution(yE, fig_dist)
    plot_model_comparison(random_summary, fig_cmp)
    plot_true_pred(yte, predE, fig_tp)
    plot_residual(yte, predE, fig_res)

    # 导出随机划分对比表。
    random_csv_rows = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, s in random_summary.items():
        random_csv_rows.append([name, s["E_MAE"], s["E_RMSE"], s["E_MAPE_percent"], s["v_MAE"], s["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "random_split_metrics.csv"), random_csv_rows)

    # 导出分组留出对比表。
    group_i_rows = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, s in group_I_summary.items():
        group_i_rows.append([name, s["E_MAE"], s["E_RMSE"], s["E_MAPE_percent"], s["v_MAE"], s["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "group_holdout_I_metrics.csv"), group_i_rows)

    group_tr_rows = [["model", "E_MAE", "E_RMSE", "E_MAPE_percent", "v_MAE", "v_MAPE_percent"]]
    for name, s in group_tr_summary.items():
        group_tr_rows.append([name, s["E_MAE"], s["E_RMSE"], s["E_MAPE_percent"], s["v_MAE"], s["v_MAPE_percent"]])
    save_csv(os.path.join(OUTPUT_DIR, "group_holdout_tr_metrics.csv"), group_tr_rows)

    # 导出 JSON 汇总。
    summary_json = {
        "quality": q,
        "best_model": best_model_name,
        "final_test_metrics": final_metrics,
        "random_split_summary": random_summary,
        "group_holdout_I_summary": group_I_summary,
        "group_holdout_tr_summary": group_tr_summary,
        "feature_names": feat_names,
        "base_feature_names": base_names,
        "best_model_artifact_path": BEST_MODEL_ARTIFACT_PATH,
        "n_samples": int(len(X)),
    }
    with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # 写最终报告到 .doc（文本形式，便于直接打开查看）。
    with open(REPORT_DOC_PATH, "w", encoding="utf-8") as f:
        f.write("零维Z箍缩代理模型训练报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"数据文件: {DOC_DATA_PATH}\n")
        f.write(f"样本数: {len(X)}\n")
        f.write("\n[数据质量检查]\n")
        for k, v in q.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n[随机划分平均指标]\n")
        for name, s in random_summary.items():
            f.write(
                f"- {name}: E_MAE={s['E_MAE']:.6f}, E_RMSE={s['E_RMSE']:.6f}, "
                f"E_MAPE={s['E_MAPE_percent']:.4f}%, v_MAE={s['v_MAE']:.6e}, v_MAPE={s['v_MAPE_percent']:.4f}%\n"
            )

        f.write("\n[分组留出-按电流]\n")
        for name, s in group_I_summary.items():
            f.write(
                f"- {name}: E_MAE={s['E_MAE']:.6f}, E_RMSE={s['E_RMSE']:.6f}, "
                f"E_MAPE={s['E_MAPE_percent']:.4f}%, v_MAE={s['v_MAE']:.6e}, v_MAPE={s['v_MAPE_percent']:.4f}%\n"
            )

        f.write("\n[分组留出-按上升时间]\n")
        for name, s in group_tr_summary.items():
            f.write(
                f"- {name}: E_MAE={s['E_MAE']:.6f}, E_RMSE={s['E_RMSE']:.6f}, "
                f"E_MAPE={s['E_MAPE_percent']:.4f}%, v_MAE={s['v_MAE']:.6e}, v_MAPE={s['v_MAPE_percent']:.4f}%\n"
            )

        f.write("\n[最终模型]\n")
        f.write(f"- 最佳模型: {best_model_name}\n")
        f.write(f"- 权重文件: {BEST_MODEL_ARTIFACT_PATH}\n")
        f.write(
            f"- 固定测试集指标: E_MAE={final_metrics['E_MAE']:.6f}, "
            f"E_RMSE={final_metrics['E_RMSE']:.6f}, E_MAPE={final_metrics['E_MAPE_percent']:.4f}%, "
            f"v_MAE={final_metrics['v_MAE']:.6e}, v_MAPE={final_metrics['v_MAPE_percent']:.4f}%\n"
        )

        f.write("\n[图表文件]\n")
        f.write(f"- {fig_dist}\n")
        f.write(f"- {fig_cmp}\n")
        f.write(f"- {fig_tp}\n")
        f.write(f"- {fig_res}\n")

    # 生成 notebook。
    write_notebook_from_script("/home/user/train_zpinch_surrogate.py", NOTEBOOK_PATH)

    print("训练完成。")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"报告文件: {REPORT_DOC_PATH}")
    print(f"Notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
