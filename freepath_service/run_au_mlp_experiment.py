#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Au-only deep MLP experiment on fixed benchmark/extrapolation sets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_unified_free_path_model import load_standard_data
from unified_free_path_core import (
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    metric_dict,
)


ROOT = Path(__file__).resolve().parent
COMBINED_DATA = OUTPUT_DIR / "experiment_standard_with_old_au_and_au2.txt"
OUT_JSON = OUTPUT_DIR / "au_mlp_experiment_results.json"
OUT_MD = ROOT / "au_mlp_experiment_report.md"


class FourierMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    cols = [z]
    for freq in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        cols.append(np.sin(freq * z))
        cols.append(np.cos(freq * z))
    return np.concatenate(cols, axis=1).astype(np.float32)


def load_au_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    mask = payload["element"].astype(str) == "Z_79"
    return payload["x_model"][mask, 1:4], payload["y_log"][mask]


@torch.inference_mode()
def predict(model: nn.Module, x_feat: np.ndarray, device: torch.device, y_mean: float, y_std: float) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, x_feat.shape[0], 16384):
        xb = torch.from_numpy(x_feat[i : i + 16384]).to(device)
        pred = model(xb).detach().cpu().numpy()
        outs.append(pred)
    return np.concatenate(outs) * y_std + y_mean


def main() -> None:
    t0 = time.time()
    torch.manual_seed(20260703)
    np.random.seed(20260703)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item = load_standard_data(COMBINED_DATA)
    bench_keys = set(BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    extra_keys = set(EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines())
    train_mask = (item["element_labels"] == "Z_79") & np.array(
        [key not in bench_keys and key not in extra_keys for key in item["keys"]], dtype=bool
    )
    x = item["x_model"][train_mask, 1:4]
    y = item["y_log"][train_mask]
    rng = np.random.default_rng(20260703)
    order = rng.permutation(x.shape[0])
    val_n = 25000
    val_idx = order[:val_n]
    fit_idx = order[val_n:]
    x_fit, y_fit = x[fit_idx], y[fit_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    mean = x_fit.mean(axis=0)
    std = np.where(x_fit.std(axis=0) < 1e-12, 1.0, x_fit.std(axis=0))
    y_mean = float(y_fit.mean())
    y_std = float(y_fit.std())
    y_std = y_std if y_std > 1e-12 else 1.0
    x_fit_f = make_features(x_fit, mean, std)
    x_val_f = make_features(x_val, mean, std)
    y_fit_s = ((y_fit - y_mean) / y_std).astype(np.float32)
    y_val_s = ((y_val - y_mean) / y_std).astype(np.float32)

    model = FourierMLP(x_fit_f.shape[1], hidden=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=2e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.05)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit_f), torch.from_numpy(y_fit_s)),
        batch_size=8192,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    best_state = None
    best_val = float("inf")
    patience = 0
    history = []
    for epoch in range(1, 81):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        sched.step()
        pred_val = predict(model, x_val_f, device, y_mean, y_std)
        val_metric = metric_dict(y_val, pred_val)
        history.append({"epoch": epoch, "train_loss": total / count, "val_smape": val_metric["smape_percent"], "val_log10_mae": val_metric["log10_mae"]})
        print(f"epoch={epoch:03d} train={total/count:.6f} val_smape={val_metric['smape_percent']:.6f}% val_mae={val_metric['log10_mae']:.7f}", flush=True)
        if val_metric["log10_mae"] < best_val:
            best_val = val_metric["log10_mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if epoch >= 30 and patience >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    bench_x, bench_y = load_au_payload(BENCHMARK_DATA_PATH)
    extra_x, extra_y = load_au_payload(EXTRAPOLATION_DATA_PATH)
    pred_bench = predict(model, make_features(bench_x, mean, std), device, y_mean, y_std)
    pred_extra = predict(model, make_features(extra_x, mean, std), device, y_mean, y_std)
    bench_metrics = metric_dict(bench_y, pred_bench)
    extra_metrics = metric_dict(extra_y, pred_extra)
    out = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(COMBINED_DATA),
        "device": str(device),
        "fit_rows": int(x_fit.shape[0]),
        "validation_rows": int(x_val.shape[0]),
        "features": "standardized rod/tep/tgama plus Fourier sin/cos frequencies 0.5..16",
        "benchmark": bench_metrics,
        "extrapolation": extra_metrics,
        "history": history,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Au 专用 Fourier MLP 实验报告",
        "",
        f"- 生成时间：{out['created_at']}",
        f"- 设备：{out['device']}",
        f"- 训练行数：{out['fit_rows']}",
        "- 本实验只离线评测，不更新网页模型。",
        "",
        "| 测试集 | Au SMAPE | log10 MAE | P99 倍数 |",
        "|---|---:|---:|---:|",
        f"| 固定随机 benchmark | {bench_metrics['smape_percent']:.6f}% | {bench_metrics['log10_mae']:.8f} | {bench_metrics['p99_factor_error']:.6f} |",
        f"| 固定连续外推 | {extra_metrics['smape_percent']:.6f}% | {extra_metrics['log10_mae']:.8f} | {extra_metrics['p99_factor_error']:.6f} |",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print(f"elapsed={out['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
