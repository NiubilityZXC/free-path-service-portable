from __future__ import annotations

import copy
import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from app import MODEL_META as CLASSICAL_MODEL_META
from app import EvaluationError, forecast_with_method

STRICT_DATASET = "ceshishuju_patent_features.csv"
LIFE_FORECAST_STEPS = 5000
LIFE_VALIDATION_MIN_ACCURACY_PERCENT = 85.0
LIFE_HORIZON_CANDIDATES = [1, 5, 10, 20, 40, 80, 120, 160, 200, 240, 300, 400, 500]
LOOKBACK_CANDIDATES = [8, 16, 24, 32, 48, 64, 80, 120]
QUANTIZED_WINDOW_CANDIDATES = [8, 16, 24, 32, 48, 64, 80, 120]
CLASSICAL_METHODS = ["gm11", "linear_trend", "holt_damped", "autoregression"]
STRICT_FEATURES = [
    "VoltageMaxRaw",
    "VoltageFirstZeroTimeUs",
    "VoltageReversePeakCoefficient",
    "VoltageMinAbsRaw",
    "DischargePeriodSec",
]

FEATURE_LABELS = {
    "VoltageMaxRaw": "CH1 最大幅值（电流换算采样）",
    "VoltageFirstZeroTimeUs": "CH1 首次过零时间（微秒）",
    "VoltageReversePeakCoefficient": "CH1 反向峰系数",
    "VoltageMinAbsRaw": "CH1 反向最小幅值绝对值（电流换算采样）",
    "DischargePeriodSec": "放电周期（秒）",
}

@dataclass
class PreparedData:
    raw: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    value_range: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    validation_targets: np.ndarray
    test_targets: np.ndarray
    train_boundary: int
    validation_boundary: int

class RecurrentForecaster(nn.Module):
    def __init__(self, cell: str, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        recurrent = nn.GRU if cell == "gru" else nn.LSTM
        self.recurrent = recurrent(feature_count, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, feature_count),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(inputs)
        return inputs[:, -1] + self.head(output[:, -1])

class CausalConv1d(nn.Conv1d):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        return output[..., : inputs.shape[-1]]

class TemporalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.network = nn.Sequential(
            CausalConv1d(
                input_channels,
                output_channels,
                kernel_size=3,
                dilation=dilation,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            CausalConv1d(
                output_channels,
                output_channels,
                kernel_size=3,
                dilation=dilation,
                padding=padding,
            ),
            nn.GELU(),
        )
        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.normalization = nn.LayerNorm(output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.network(inputs) + self.residual(inputs)
        return self.normalization(output.transpose(1, 2)).transpose(1, 2)

class TCNForecaster(nn.Module):
    def __init__(self, feature_count: int, channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            TemporalBlock(feature_count, channels, dilation=1),
            TemporalBlock(channels, channels, dilation=2),
            TemporalBlock(channels, channels, dilation=4),
        )
        self.head = nn.Linear(channels, feature_count)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.network(inputs.transpose(1, 2))
        return inputs[:, -1] + self.head(output[:, :, -1])

class TransformerForecaster(nn.Module):
    def __init__(self, feature_count: int, lookback: int, model_size: int = 32) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_count, model_size)
        self.position = nn.Parameter(torch.zeros(1, lookback, model_size))
        layer = nn.TransformerEncoderLayer(
            d_model=model_size,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(model_size),
            nn.Linear(model_size, feature_count),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(inputs) + self.position[:, : inputs.shape[1]]
        encoded = self.encoder(embedded)
        return inputs[:, -1] + self.head(encoded[:, -1])

def load_strict_data(csv_path: Path) -> np.ndarray:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("strict waveform feature dataset is empty")
    data = np.asarray(
        [[float(row[feature]) for feature in STRICT_FEATURES] for row in rows],
        dtype=np.float32,
    )
    if not np.isfinite(data).all():
        raise ValueError("strict waveform feature dataset contains non-finite values")
    return data

def prepare_data(raw: np.ndarray, lookback: int, horizon: int) -> PreparedData:
    row_count = len(raw)
    train_boundary = int(row_count * 0.70)
    validation_boundary = int(row_count * 0.85)
    train_rows = raw[:train_boundary]
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    value_range = train_rows.max(axis=0) - train_rows.min(axis=0)
    value_range = np.where(value_range < 1e-8, std, value_range)
    normalized = (raw - mean) / std

    inputs = []
    targets = []
    target_indices = []
    stop = row_count - lookback - horizon + 1
    for start in range(stop):
        target_index = start + lookback + horizon - 1
        inputs.append(normalized[start : start + lookback])
        targets.append(normalized[target_index])
        target_indices.append(target_index)
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    indices = np.asarray(target_indices, dtype=np.int64)

    train_mask = indices < train_boundary
    validation_mask = (indices >= train_boundary) & (indices < validation_boundary)
    test_mask = indices >= validation_boundary
    if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) < 10:
        raise ValueError("not enough chronological samples for train/validation/test split")
    return PreparedData(
        raw=raw,
        mean=mean,
        std=std,
        value_range=value_range,
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_validation=x[validation_mask],
        y_validation=y[validation_mask],
        x_test=x[test_mask],
        y_test=y[test_mask],
        validation_targets=indices[validation_mask],
        test_targets=indices[test_mask],
        train_boundary=train_boundary,
        validation_boundary=validation_boundary,
    )

def denormalize(values: np.ndarray, prepared: PreparedData) -> np.ndarray:
    return values * prepared.std + prepared.mean

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    value_range: np.ndarray,
) -> dict[str, Any]:
    error = predicted - actual
    absolute = np.abs(error)
    mae = absolute.mean(axis=0)
    rmse = np.sqrt(np.square(error).mean(axis=0))
    nmae = mae / value_range * 100.0
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(2.0 * absolute / np.maximum(denominator, 1e-8), axis=0) * 100.0
    mape = np.mean(absolute / np.maximum(np.abs(actual), 1e-8), axis=0) * 100.0
    per_feature = []
    for index, feature in enumerate(STRICT_FEATURES):
        per_feature.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "mae": float(mae[index]),
                "rmse": float(rmse[index]),
                "nmaePercent": float(nmae[index]),
                "smapePercent": float(smape[index]),
                "mapePercent": float(mape[index]),
            }
        )
    mean_nmae = float(nmae.mean())
    return {
        "samples": int(len(actual)),
        "meanNmaePercent": mean_nmae,
        "normalizedAccuracyPercent": max(0.0, 100.0 - mean_nmae),
        "meanSmapePercent": float(smape.mean()),
        "meanMapePercent": float(mape.mean()),
        "perFeature": per_feature,
    }

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)

def predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(tensor(values, device)).detach().cpu().numpy()

def recursive_deep_forecast(
    model: nn.Module,
    prepared: PreparedData,
    device: torch.device,
    direct_horizon: int,
    future_steps: int,
) -> np.ndarray:
    normalized = (prepared.raw - prepared.mean) / prepared.std
    history = tensor(normalized, device)
    lower_bound = tensor(-prepared.mean / prepared.std, device)
    blocks = []
    generated = 0
    model.eval()
    with torch.inference_mode():
        while generated < future_steps:
            endpoint = model(history[-prepared.x_train.shape[1] :].unsqueeze(0))[0]
            endpoint = torch.maximum(torch.clamp(endpoint, max=8.0), lower_bound)
            block_size = min(direct_horizon, future_steps - generated)
            ratios = (
                torch.arange(1, block_size + 1, device=device, dtype=torch.float32)
                / direct_horizon
            ).unsqueeze(1)
            block = history[-1] + ratios * (endpoint - history[-1])
            history = torch.cat([history, block], dim=0)
            blocks.append(block.detach().cpu().numpy())
            generated += block_size
    return np.concatenate(blocks, axis=0)

def train_deep_architecture(
    name: str,
    model_factory: Callable[[], nn.Module],
    prepared: PreparedData,
    device: torch.device,
    seeds: list[int],
    max_epochs: int,
    patience: int,
    forecast_horizon: int,
    future_steps: int,
) -> dict[str, Any]:
    validation_predictions = []
    test_predictions = []
    future_predictions = []
    epochs_by_seed = []
    started = time.perf_counter()
    x_train = tensor(prepared.x_train, device)
    y_train = tensor(prepared.y_train, device)
    x_validation = tensor(prepared.x_validation, device)
    y_validation = tensor(prepared.y_validation, device)

    for seed in seeds:
        set_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_function = nn.MSELoss()
        best_loss = math.inf
        best_state = None
        stale_epochs = 0
        best_epoch = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_prediction = model(x_train)
            train_loss = loss_function(train_prediction, y_train)
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                validation_loss = loss_function(model(x_validation), y_validation).item()
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
                best_epoch = epoch
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break
        if best_state is None:
            raise RuntimeError(f"{name} did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        validation_predictions.append(predict(model, prepared.x_validation, device))
        test_predictions.append(predict(model, prepared.x_test, device))
        future_predictions.append(
            recursive_deep_forecast(
                model,
                prepared,
                device,
                forecast_horizon,
                future_steps,
            )
        )
        epochs_by_seed.append(best_epoch)
        del model

    validation_normalized = np.mean(validation_predictions, axis=0)
    test_normalized = np.mean(test_predictions, axis=0)
    future_normalized = np.mean(future_predictions, axis=0)
    validation_actual = denormalize(prepared.y_validation, prepared)
    test_actual = denormalize(prepared.y_test, prepared)
    validation_prediction = denormalize(validation_normalized, prepared)
    test_prediction = denormalize(test_normalized, prepared)
    return {
        "id": name.lower(),
        "name": name,
        "modelType": "deep_learning",
        "validationMetrics": calculate_metrics(
            validation_actual, validation_prediction, prepared.value_range
        ),
        "testMetrics": calculate_metrics(test_actual, test_prediction, prepared.value_range),
        "epochsBySeed": epochs_by_seed,
        "seeds": seeds,
        "trainingSeconds": time.perf_counter() - started,
        "validationPrediction": validation_prediction,
        "testPrediction": test_prediction,
        "futurePrediction": np.maximum(
            denormalize(future_normalized, prepared),
            0.0,
        ),
    }

def recursive_ridge_forecast(
    prepared: PreparedData,
    weights: np.ndarray,
    direct_horizon: int,
    future_steps: int,
) -> np.ndarray:
    history = ((prepared.raw - prepared.mean) / prepared.std).astype(np.float64)
    lower_bound = -prepared.mean.astype(np.float64) / prepared.std.astype(np.float64)
    blocks = []
    generated = 0
    lookback = prepared.x_train.shape[1]
    while generated < future_steps:
        model_input = np.concatenate([history[-lookback:].reshape(-1), [1.0]])
        endpoint = history[-1] + model_input @ weights
        endpoint = np.maximum(np.minimum(endpoint, 8.0), lower_bound)
        block_size = min(direct_horizon, future_steps - generated)
        ratios = np.arange(1, block_size + 1, dtype=np.float64)[:, None] / direct_horizon
        block = history[-1] + ratios * (endpoint - history[-1])
        history = np.concatenate([history, block], axis=0)
        blocks.append(block)
        generated += block_size
    normalized = np.concatenate(blocks, axis=0).astype(np.float32)
    return np.maximum(denormalize(normalized, prepared), 0.0)

def classical_target_predictions(
    prepared: PreparedData,
    target_indices: np.ndarray,
    method: str,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    predictions = np.empty((len(target_indices), len(STRICT_FEATURES)), dtype=np.float32)
    for row_index, target_index in enumerate(target_indices):
        start = int(target_index) - lookback - horizon + 1
        window = prepared.raw[start : start + lookback]
        for feature_index in range(len(STRICT_FEATURES)):
            values = [float(value) for value in window[:, feature_index]]
            model = forecast_with_method(method, values, horizon)
            predicted = model.forecast[lookback + horizon - 1]
            if not math.isfinite(predicted):
                raise EvaluationError(f"{method} produced a non-finite prediction")
            safety_bound = max(
                1.0,
                abs(float(prepared.mean[feature_index]))
                + 1000.0 * float(prepared.std[feature_index]),
            )
            predictions[row_index, feature_index] = max(
                -safety_bound,
                min(safety_bound, predicted),
            )
    return predictions

def classical_future_forecast(
    prepared: PreparedData,
    method: str,
    lookback: int,
    direct_horizon: int,
    future_steps: int,
) -> np.ndarray:
    upper_bound = prepared.mean + 8.0 * prepared.std
    feature_forecasts = []
    for feature_index in range(len(STRICT_FEATURES)):
        history = [float(value) for value in prepared.raw[:, feature_index]]
        generated: list[float] = []
        try:
            model = forecast_with_method(method, history[-lookback:], future_steps)
            generated = [float(value) for value in model.forecast[lookback:]]
            if len(generated) != future_steps or not all(
                math.isfinite(value) for value in generated
            ):
                raise EvaluationError(f"{method} long forecast is not finite")
        except (EvaluationError, OverflowError, ValueError):
            generated = []
            while len(generated) < future_steps:
                block_size = min(direct_horizon, future_steps - len(generated))
                try:
                    model = forecast_with_method(method, history[-lookback:], block_size)
                    block = [float(value) for value in model.forecast[lookback:]]
                    if len(block) != block_size or not all(math.isfinite(value) for value in block):
                        raise EvaluationError(f"{method} recursive forecast is not finite")
                except (EvaluationError, OverflowError, ValueError):
                    block = [history[-1]] * block_size
                block = [
                    max(0.0, min(float(upper_bound[feature_index]), value))
                    for value in block
                ]
                history.extend(block)
                generated.extend(block)
        feature_forecasts.append(
            np.clip(
                np.asarray(generated, dtype=np.float32),
                0.0,
                float(upper_bound[feature_index]),
            )
        )
    return np.column_stack(feature_forecasts)

def classical_results(
    prepared: PreparedData,
    lookback: int,
    direct_horizon: int,
    future_steps: int,
) -> list[dict[str, Any]]:
    validation_actual = denormalize(prepared.y_validation, prepared)
    test_actual = denormalize(prepared.y_test, prepared)
    results = []
    for method in CLASSICAL_METHODS:
        started = time.perf_counter()
        validation_prediction = classical_target_predictions(
            prepared,
            prepared.validation_targets,
            method,
            lookback,
            direct_horizon,
        )
        test_prediction = classical_target_predictions(
            prepared,
            prepared.test_targets,
            method,
            lookback,
            direct_horizon,
        )
        future_prediction = classical_future_forecast(
            prepared,
            method,
            lookback,
            direct_horizon,
            future_steps,
        )
        results.append(
            {
                "id": method,
                "name": CLASSICAL_MODEL_META[method]["name"],
                "modelType": "patent" if method == "gm11" else "statistical",
                "description": CLASSICAL_MODEL_META[method]["description"],
                "validationMetrics": calculate_metrics(
                    validation_actual,
                    validation_prediction,
                    prepared.value_range,
                ),
                "testMetrics": calculate_metrics(
                    test_actual,
                    test_prediction,
                    prepared.value_range,
                ),
                "trainingSeconds": time.perf_counter() - started,
                "validationPrediction": validation_prediction,
                "testPrediction": test_prediction,
                "futurePrediction": future_prediction,
                "futureMethod": "在最后建模窗口上按该方法直接外推；数值异常时按 20 步分块递推",
            }
        )
    return results

def baseline_results(
    prepared: PreparedData,
    direct_horizon: int,
    future_steps: int,
) -> list[dict[str, Any]]:
    validation_actual = denormalize(prepared.y_validation, prepared)
    test_actual = denormalize(prepared.y_test, prepared)
    persistence_validation = denormalize(prepared.x_validation[:, -1], prepared)
    persistence_test = denormalize(prepared.x_test[:, -1], prepared)
    results = [
        {
            "id": "persistence",
            "name": "持续值基线",
            "modelType": "baseline",
            "validationMetrics": calculate_metrics(
                validation_actual, persistence_validation, prepared.value_range
            ),
            "testMetrics": calculate_metrics(
                test_actual, persistence_test, prepared.value_range
            ),
            "trainingSeconds": 0.0,
            "validationPrediction": persistence_validation,
            "testPrediction": persistence_test,
            "futurePrediction": np.repeat(
                prepared.raw[-1][None, :], future_steps, axis=0
            ),
            "futureMethod": "保持最后一行实测值不变",
        }
    ]

    train_flat = prepared.x_train.reshape(len(prepared.x_train), -1).astype(np.float64)
    validation_flat = prepared.x_validation.reshape(len(prepared.x_validation), -1).astype(
        np.float64
    )
    test_flat = prepared.x_test.reshape(len(prepared.x_test), -1).astype(np.float64)
    train_with_bias = np.column_stack([train_flat, np.ones(len(train_flat))])
    validation_with_bias = np.column_stack(
        [validation_flat, np.ones(len(validation_flat))]
    )
    test_with_bias = np.column_stack([test_flat, np.ones(len(test_flat))])
    started = time.perf_counter()
    best = None
    for penalty in (0.01, 0.1, 1.0, 10.0):
        regularizer = np.eye(train_with_bias.shape[1]) * penalty
        regularizer[-1, -1] = 0.0
        train_delta = prepared.y_train.astype(np.float64) - prepared.x_train[:, -1].astype(
            np.float64
        )
        weights = np.linalg.solve(
            train_with_bias.T @ train_with_bias + regularizer,
            train_with_bias.T @ train_delta,
        )
        validation_prediction = denormalize(
            (
                prepared.x_validation[:, -1]
                + (validation_with_bias @ weights).astype(np.float32)
            ),
            prepared,
        )
        metrics = calculate_metrics(
            validation_actual, validation_prediction, prepared.value_range
        )
        if best is None or metrics["meanNmaePercent"] < best["metrics"]["meanNmaePercent"]:
            best = {
                "penalty": penalty,
                "weights": weights,
                "metrics": metrics,
                "validationPrediction": validation_prediction,
            }
    if best is None:
        raise RuntimeError("ridge baseline fitting failed")
    ridge_test = denormalize(
        prepared.x_test[:, -1] + (test_with_bias @ best["weights"]).astype(np.float32),
        prepared,
    )
    results.append(
        {
            "id": "ridge",
            "name": "Ridge 线性基线",
            "modelType": "baseline",
            "validationMetrics": best["metrics"],
            "testMetrics": calculate_metrics(test_actual, ridge_test, prepared.value_range),
            "parameters": {"penalty": best["penalty"]},
            "trainingSeconds": time.perf_counter() - started,
            "validationPrediction": best["validationPrediction"],
            "testPrediction": ridge_test,
            "futurePrediction": recursive_ridge_forecast(
                prepared,
                best["weights"],
                direct_horizon,
                future_steps,
            ),
            "futureMethod": "按 20 步直接预测端点递推；块内线性插值",
        }
    )
    return results

def stable_mode(values: np.ndarray) -> float:
    unique, counts = np.unique(values, return_counts=True)
    winners = set(unique[counts == counts.max()].tolist())
    for value in values[::-1]:
        if float(value) in winners:
            return float(value)
    raise RuntimeError("cannot determine a stable mode from an empty sequence")

def quantization_aware_point(
    history: np.ndarray,
    window: int,
    period_reference: np.ndarray | None = None,
) -> np.ndarray:
    if len(history) == 0:
        raise ValueError("quantization-aware forecast requires observed history")
    recent = history[-min(window, len(history)) :]
    maximum = float(np.max(recent[:, 0]))
    first_zero = float(np.median(recent[:, 1]))
    reverse_minimum = stable_mode(recent[:, 3])
    valid_periods = (
        period_reference if period_reference is not None else recent[:, 4]
    )
    valid_periods = valid_periods[valid_periods > 1e-3]
    discharge_period = float(
        np.median(valid_periods) if len(valid_periods) else np.median(recent[:, 4])
    )
    reverse_coefficient = (
        1e-6
        if reverse_minimum <= 1.0
        else reverse_minimum / max(maximum, 1e-8)
    )
    return np.asarray(
        [
            maximum,
            first_zero,
            reverse_coefficient,
            reverse_minimum,
            discharge_period,
        ],
        dtype=np.float32,
    )

def quantization_aware_target_predictions(
    prepared: PreparedData,
    target_indices: np.ndarray,
    direct_horizon: int,
    window: int,
) -> np.ndarray:
    predictions = []
    for target_index in target_indices:
        history_end = int(target_index) - direct_horizon + 1
        predictions.append(
            quantization_aware_point(
                prepared.raw[:history_end],
                window,
                prepared.raw[: prepared.train_boundary, 4],
            )
        )
    return np.asarray(predictions, dtype=np.float32)

def quantization_aware_results(
    prepared: PreparedData,
    direct_horizon: int,
    future_steps: int,
) -> list[dict[str, Any]]:
    validation_actual = denormalize(prepared.y_validation, prepared)
    test_actual = denormalize(prepared.y_test, prepared)
    candidates = []
    started = time.perf_counter()
    for window in QUANTIZED_WINDOW_CANDIDATES:
        validation_prediction = quantization_aware_target_predictions(
            prepared,
            prepared.validation_targets,
            direct_horizon,
            window,
        )
        test_prediction = quantization_aware_target_predictions(
            prepared,
            prepared.test_targets,
            direct_horizon,
            window,
        )
        candidates.append(
            {
                "lookback": window,
                "validationPrediction": validation_prediction,
                "testPrediction": test_prediction,
                "validationMetrics": calculate_metrics(
                    validation_actual,
                    validation_prediction,
                    prepared.value_range,
                ),
                "testMetrics": calculate_metrics(
                    test_actual,
                    test_prediction,
                    prepared.value_range,
                ),
            }
        )

    minimum_validation = min(
        candidate["validationMetrics"]["meanNmaePercent"]
        for candidate in candidates
    )
    tied = [
        candidate
        for candidate in candidates
        if abs(
            candidate["validationMetrics"]["meanNmaePercent"]
            - minimum_validation
        )
        <= 1e-6
    ]
    tied.sort(key=lambda candidate: candidate["lookback"])
    selected = tied[len(tied) // 2]
    selected_window = int(selected["lookback"])
    future_point = quantization_aware_point(
        prepared.raw,
        selected_window,
        prepared.raw[: prepared.train_boundary, 4],
    )
    future_prediction = np.repeat(future_point[None, :], future_steps, axis=0)

    return [
        {
            "id": "quantized_hybrid",
            "name": "量化感知混合模型",
            "modelType": "hybrid",
            "description": "离散特征采用上包络、中位数和众数，并重建反向峰系数。",
            "lookback": selected_window,
            "validationMetrics": selected["validationMetrics"],
            "testMetrics": selected["testMetrics"],
            "trainingSeconds": time.perf_counter() - started,
            "validationPrediction": selected["validationPrediction"],
            "testPrediction": selected["testPrediction"],
            "futurePrediction": future_prediction,
            "futureMethod": "保持验证集选定的稳健量化档位",
            "featureStrategies": {
                "VoltageMaxRaw": "最近窗口上包络",
                "VoltageFirstZeroTimeUs": "最近窗口中位数",
                "VoltageReversePeakCoefficient": "由预测反向幅值/预测主幅值重建",
                "VoltageMinAbsRaw": "最近窗口众数档位",
                "DischargePeriodSec": "训练段有效周期中位数",
            },
            "windowSelectionRule": "验证集 NMAE 最低；并列时取中间窗口",
            "windowSearch": [
                {
                    "lookback": int(candidate["lookback"]),
                    "validationNmaePercent": candidate["validationMetrics"][
                        "meanNmaePercent"
                    ],
                    "testNmaePercent": candidate["testMetrics"][
                        "meanNmaePercent"
                    ],
                }
                for candidate in candidates
            ],
        }
    ]

def evaluate_life_horizons(prepared: PreparedData, model: dict[str, Any], direct_horizon: int) -> dict[str, Any]:
    minimum = LIFE_VALIDATION_MIN_ACCURACY_PERCENT
    results = []
    if model["id"] == "quantized_hybrid":
        actual = denormalize(prepared.y_test, prepared)
        for horizon in LIFE_HORIZON_CANDIDATES:
            mask = prepared.test_targets - horizon + 1 >= model["lookback"]
            targets = prepared.test_targets[mask]
            if len(targets) < 30:
                continue
            predicted = quantization_aware_target_predictions(prepared, targets, horizon, model["lookback"])
            metrics = calculate_metrics(actual[mask], predicted, prepared.value_range)
            results.append({"horizon": horizon, "sampleCount": int(len(targets)),
                "nmaePercent": metrics["meanNmaePercent"],
                "accuracyPercent": metrics["normalizedAccuracyPercent"]})
    else:
        metrics = model["testMetrics"]
        results.append({"horizon": direct_horizon,
            "sampleCount": int(len(prepared.test_targets)),
            "nmaePercent": metrics["meanNmaePercent"],
            "accuracyPercent": metrics["normalizedAccuracyPercent"]})
    qualified = [item["horizon"] for item in results if item["accuracyPercent"] >= minimum]
    return {"minimumAccuracyPercent": minimum,
        "validatedHorizonSteps": max(qualified, default=0),
        "selectionRule": "独立测试综合精度不低于 85%，且至少 30 个测试点",
        "results": results}

def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model.items()
        if key not in {"validationPrediction", "testPrediction", "futurePrediction"}
    }

def build_test_series(
    model: dict[str, Any],
    prepared: PreparedData,
) -> list[dict[str, Any]]:
    test_actual = denormalize(prepared.y_test, prepared)
    test_prediction = model["testPrediction"]
    series = []
    for feature_index, feature in enumerate(STRICT_FEATURES):
        series.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "points": [
                    {
                        "targetIndex": int(target) + 1,
                        "actual": float(test_actual[row_index, feature_index]),
                        "predicted": float(test_prediction[row_index, feature_index]),
                        "normalizedAbsErrorPercent": float(
                            abs(
                                test_actual[row_index, feature_index]
                                - test_prediction[row_index, feature_index]
                            )
                            / prepared.value_range[feature_index]
                            * 100.0
                        ),
                    }
                    for row_index, target in enumerate(prepared.test_targets)
                ],
            }
        )
    return series

def train_and_evaluate(
    csv_path: Path,
    lookback: int = 32,
    horizon: int = 20,
    seeds: list[int] | None = None,
    max_epochs: int = 250,
    patience: int = 25,
    future_steps: int = LIFE_FORECAST_STEPS,
    life_model_id: str | None = None,
) -> dict[str, Any]:
    if lookback < 8:
        raise ValueError("lookback must be at least 8")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if future_steps < horizon:
        raise ValueError("future_steps must be at least the direct forecast horizon")
    seeds = seeds or [11, 23, 37]
    raw = load_strict_data(csv_path)
    prepared = prepare_data(raw, lookback, horizon)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    feature_count = len(STRICT_FEATURES)
    factories: list[tuple[str, Callable[[], nn.Module]]] = [
        ("GRU", lambda: RecurrentForecaster("gru", feature_count)),
        ("LSTM", lambda: RecurrentForecaster("lstm", feature_count)),
        ("TCN", lambda: TCNForecaster(feature_count)),
        (
            "Transformer",
            lambda: TransformerForecaster(feature_count, lookback=lookback),
        ),
    ]

    started = time.perf_counter()
    models = classical_results(prepared, lookback, horizon, future_steps)
    models.extend(baseline_results(prepared, horizon, future_steps))
    models.extend(quantization_aware_results(prepared, horizon, future_steps))
    for name, factory in factories:
        models.append(
            train_deep_architecture(
                name,
                factory,
                prepared,
                device,
                seeds,
                max_epochs,
                patience,
                horizon,
                future_steps,
            )
        )
    for model in models:
        model.setdefault("lookback", lookback)
        if model["modelType"] == "deep_learning":
            model["futureMethod"] = "按 20 步直接预测端点递推；块内线性插值"

    selected_model = min(
        models,
        key=lambda model: model["validationMetrics"]["meanNmaePercent"],
    )
    deep_models = [model for model in models if model["modelType"] == "deep_learning"]
    baselines = [model for model in models if model["modelType"] == "baseline"]
    selected_deep = min(
        deep_models,
        key=lambda model: model["validationMetrics"]["meanNmaePercent"],
    )
    selected_baseline = min(
        baselines,
        key=lambda model: model["validationMetrics"]["meanNmaePercent"],
    )
    best_test_model = min(
        models,
        key=lambda model: model["testMetrics"]["meanNmaePercent"],
    )
    model_test_series = [
        {
            "modelId": model["id"],
            "modelName": model["name"],
            "modelType": model["modelType"],
            "lookback": model["lookback"],
            "series": build_test_series(model, prepared),
        }
        for model in models
    ]
    selected_deep_series = next(
        item["series"]
        for item in model_test_series
        if item["modelId"] == selected_deep["id"]
    )
    best_test_series = next(
        item["series"]
        for item in model_test_series
        if item["modelId"] == best_test_model["id"]
    )
    life_model = next(
        (model for model in models if model["id"] == life_model_id),
        selected_model,
    )
    life_prediction = life_model["futurePrediction"]
    horizon_validation = evaluate_life_horizons(prepared, life_model, horizon)
    healthy_rows = 1
    healthy_baseline = raw[0].copy()
    baseline_rows = {feature: 1 for feature in STRICT_FEATURES}
    period_index = STRICT_FEATURES.index("DischargePeriodSec")
    valid_period_rows = np.flatnonzero(raw[:, period_index] > 1e-3)
    if len(valid_period_rows):
        first_period_row = int(valid_period_rows[0])
        healthy_baseline[period_index] = raw[first_period_row, period_index]
        baseline_rows["DischargePeriodSec"] = first_period_row + 1
    robust_current = life_prediction[0]
    recent_rows = min(120, len(raw))
    recent_x = np.arange(recent_rows, dtype=np.float64)
    life_metrics = {
        item["feature"]: item
        for item in life_model["testMetrics"]["perFeature"]
    }
    life_diagnostics = []
    for index, feature in enumerate(STRICT_FEATURES):
        trend_windows = []
        for width in (32, 64, 120, 200):
            values = raw[-min(width, len(raw)) :, index].astype(np.float64)
            axis = np.arange(len(values), dtype=np.float64)
            slope, intercept = np.polyfit(axis, values, 1)
            residual = values - (slope * axis + intercept)
            mad = float(np.median(np.abs(residual)))
            change = float(slope * max(len(values) - 1, 1))
            trend_windows.append({"window": width, "slopePerStep": float(slope),
                "trendDetectable": abs(change) > max(3.0 * mad, 1e-6)})
        recent = raw[-recent_rows:, index].astype(np.float64)
        slope = trend_windows[2]["slopePerStep"]
        residual_mad = float(np.median(np.abs(recent - np.median(recent))))
        trend_change = slope * max(recent_rows - 1, 1)
        detectable = trend_windows[2]["trendDetectable"]
        unique_count = int(len(np.unique(recent)))
        nmae = float(life_metrics[feature]["nmaePercent"])
        accuracy_weight = max(0.0, min(1.0, 1.0 - nmae / 100.0))
        resolution_weight = min(1.0, unique_count / max(12.0, recent_rows * 0.2))
        reliability = accuracy_weight * (0.35 + 0.65 * resolution_weight)
        life_diagnostics.append(
            {
                "feature": feature,
                "healthyBaseline": float(healthy_baseline[index]),
                "robustCurrent": float(robust_current[index]),
                "recentSlopePerStep": slope,
                "recentTrendChange": trend_change,
                "residualMad": residual_mad,
                "trendDetectable": bool(detectable),
                "recentUniqueLevels": unique_count,
                "testNmaePercent": nmae,
                "reliabilityWeight": float(reliability),
                "trendWindows": trend_windows,
            }
        )

    return {
        "dataset": STRICT_DATASET,
        "features": [
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "trainingRange": float(prepared.value_range[index]),
            }
            for index, feature in enumerate(STRICT_FEATURES)
        ],
        "rowCount": int(len(raw)),
        "lookback": lookback,
        "horizon": horizon,
        "split": {
            "trainEndRow": prepared.train_boundary,
            "validationStartRow": prepared.train_boundary + 1,
            "validationEndRow": prepared.validation_boundary,
            "testStartRow": prepared.validation_boundary + 1,
            "testEndRow": int(len(raw)),
            "trainSamples": int(len(prepared.x_train)),
            "validationSamples": int(len(prepared.x_validation)),
            "testSamples": int(len(prepared.x_test)),
        },
        "selectionMetric": "验证集五参数平均归一化 MAE",
        "selectedModel": selected_model["id"],
        "selectedModelName": selected_model["name"],
        "selectedDeepModel": selected_deep["id"],
        "selectedDeepModelName": selected_deep["name"],
        "selectedBaseline": selected_baseline["id"],
        "selectedBaselineName": selected_baseline["name"],
        "bestTestModel": best_test_model["id"],
        "bestTestModelName": best_test_model["name"],
        "summary": {
            "validationNmaePercent": selected_model["validationMetrics"][
                "meanNmaePercent"
            ],
            "testNmaePercent": selected_model["testMetrics"]["meanNmaePercent"],
            "testNormalizedAccuracyPercent": selected_model["testMetrics"][
                "normalizedAccuracyPercent"
            ],
            "testSmapePercent": selected_model["testMetrics"]["meanSmapePercent"],
            "baselineTestNmaePercent": selected_baseline["testMetrics"][
                "meanNmaePercent"
            ],
            "deepVsBaselineNmaePoints": selected_baseline["testMetrics"][
                "meanNmaePercent"
            ]
            - selected_deep["testMetrics"]["meanNmaePercent"],
            "deepBeatsBaseline": selected_deep["testMetrics"]["meanNmaePercent"]
            < selected_baseline["testMetrics"]["meanNmaePercent"],
            "bestTestNmaePercent": best_test_model["testMetrics"][
                "meanNmaePercent"
            ],
            "bestTestNormalizedAccuracyPercent": best_test_model["testMetrics"][
                "normalizedAccuracyPercent"
            ],
            "bestVsBaselineNmaePoints": selected_baseline["testMetrics"][
                "meanNmaePercent"
            ]
            - best_test_model["testMetrics"]["meanNmaePercent"],
        },
        "models": [serializable_model(model) for model in models],
        "selectedDeepTestSeries": selected_deep_series,
        "bestTestSeries": best_test_series,
        "modelTestSeries": model_test_series,
        "lifeForecast": {
            "modelId": life_model["id"],
            "modelName": life_model["name"],
            "startRow": int(len(raw)) + 1,
            "currentRow": int(len(raw)),
            "maxFutureSteps": future_steps,
            "directHorizon": horizon,
            "horizonValidation": horizon_validation,
            "healthyBaselineRows": healthy_rows,
            "healthyBaselineValues": {
                feature: float(healthy_baseline[index])
                for index, feature in enumerate(STRICT_FEATURES)
            },
            "failureBaselineValues": {
                feature: float(healthy_baseline[index])
                for index, feature in enumerate(STRICT_FEATURES)
            },
            "baselineRowsByFeature": baseline_rows,
            "baselineDefinition": (
                "原始数据第1次放电；放电周期取第1次至第2次之间的首个有效间隔"
            ),
            "robustCurrentValues": {
                feature: float(robust_current[index])
                for index, feature in enumerate(STRICT_FEATURES)
            },
            "medianPeriodSeconds": float(
                np.median(
                    raw[
                        : min(len(raw), 424),
                        STRICT_FEATURES.index("DischargePeriodSec"),
                    ]
                )
            ),
            "diagnostics": life_diagnostics,
            "currentValues": {
                feature: float(raw[-1, index])
                for index, feature in enumerate(STRICT_FEATURES)
            },
            "series": [
                {
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "observedValues": [
                        float(value)
                        for value in raw[-lookback:, index]
                    ],
                    "values": [float(value) for value in life_prediction[:, index]],
                }
                for index, feature in enumerate(STRICT_FEATURES)
            ],
            "method": life_model.get("futureMethod", "使用当前模型外推"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "deviceName": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "seeds": seeds,
            "maxEpochs": max_epochs,
            "earlyStoppingPatience": patience,
            "trainingSeconds": time.perf_counter() - started,
        },
        "notes": ["五项 CH1 特征按时间隔离；无一致失效趋势时仅报告经回测验证的寿命下限。"],
    }

def search_lookbacks_and_evaluate(
    csv_path: Path,
    lookbacks: list[int] | None = None,
    horizon: int = 20,
    seeds: list[int] | None = None,
    max_epochs: int = 250,
    patience: int = 25,
    future_steps: int = LIFE_FORECAST_STEPS,
) -> dict[str, Any]:
    candidates = sorted(set(lookbacks or LOOKBACK_CANDIDATES))
    if not candidates or any(value < 8 for value in candidates):
        raise ValueError("all lookback candidates must be at least 8")

    search_started = time.perf_counter()
    candidate_reports = []
    for lookback in candidates:
        print(f"lookback search: training {lookback} steps", flush=True)
        candidate_reports.append(
            train_and_evaluate(
                csv_path,
                lookback=lookback,
                horizon=horizon,
                seeds=seeds,
                max_epochs=max_epochs,
                patience=patience,
                future_steps=horizon,
            )
        )
        winner = min(
            candidate_reports[-1]["models"],
            key=lambda model: model["validationMetrics"]["meanNmaePercent"],
        )
        print(
            f"lookback search: {lookback} steps -> {winner['name']} "
            f"validation NMAE {winner['validationMetrics']['meanNmaePercent']:.4f}%",
            flush=True,
        )

    model_ids = [model["id"] for model in candidate_reports[0]["models"]]
    tuned_models = []
    tuned_series = []
    per_model_search = []
    for model_id in model_ids:
        choices = []
        for report in candidate_reports:
            model = next(item for item in report["models"] if item["id"] == model_id)
            choices.append((report, model))
        source_report, source_model = min(
            choices,
            key=lambda item: (
                item[1]["validationMetrics"]["meanNmaePercent"],
                item[0]["lookback"],
            ),
        )
        tuned_model = copy.deepcopy(source_model)
        tuned_model["lookback"] = source_model.get(
            "lookback", source_report["lookback"]
        )
        tuned_models.append(tuned_model)
        source_series = next(
            item
            for item in source_report["modelTestSeries"]
            if item["modelId"] == model_id
        )
        tuned_group = copy.deepcopy(source_series)
        tuned_group["lookback"] = tuned_model["lookback"]
        tuned_series.append(tuned_group)
        per_model_search.append(
            {
                "modelId": model_id,
                "modelName": source_model["name"],
                "modelType": source_model["modelType"],
                "bestLookback": tuned_model["lookback"],
                "validationNmaePercent": source_model["validationMetrics"][
                    "meanNmaePercent"
                ],
                "testNmaePercent": source_model["testMetrics"]["meanNmaePercent"],
            }
        )

    selected_model = min(
        tuned_models,
        key=lambda model: model["validationMetrics"]["meanNmaePercent"],
    )
    selected_deep = min(
        (model for model in tuned_models if model["modelType"] == "deep_learning"),
        key=lambda model: model["validationMetrics"]["meanNmaePercent"],
    )
    selected_baseline = min(
        (model for model in tuned_models if model["modelType"] == "baseline"),
        key=lambda model: model["validationMetrics"]["meanNmaePercent"],
    )
    best_test_model = min(
        tuned_models,
        key=lambda model: model["testMetrics"]["meanNmaePercent"],
    )
    deployment_lookback = int(selected_model["lookback"])

    final_report = train_and_evaluate(
        csv_path,
        lookback=deployment_lookback,
        horizon=horizon,
        seeds=seeds,
        max_epochs=max_epochs,
        patience=patience,
        future_steps=future_steps,
        life_model_id=str(selected_model["id"]),
    )
    final_report["models"] = tuned_models
    final_report["modelTestSeries"] = tuned_series
    final_report["selectedModel"] = selected_model["id"]
    final_report["selectedModelName"] = selected_model["name"]
    final_report["selectedDeepModel"] = selected_deep["id"]
    final_report["selectedDeepModelName"] = selected_deep["name"]
    final_report["selectedBaseline"] = selected_baseline["id"]
    final_report["selectedBaselineName"] = selected_baseline["name"]
    final_report["bestTestModel"] = best_test_model["id"]
    final_report["bestTestModelName"] = best_test_model["name"]
    final_report["bestTestSeries"] = next(
        item["series"] for item in tuned_series if item["modelId"] == best_test_model["id"]
    )
    final_report["selectedDeepTestSeries"] = next(
        item["series"] for item in tuned_series if item["modelId"] == selected_deep["id"]
    )
    final_report["selectionMetric"] = (
        "每个模型先按验证集五参数平均 NMAE 选择建模步数，再在独立测试集统一比较"
    )
    final_report["lookback"] = deployment_lookback
    if selected_model.get("windowSearch"):
        lookback_candidates = [
            {
                **candidate,
                "winnerModelId": selected_model["id"],
                "winnerModelName": selected_model["name"],
                "focusModelId": selected_model["id"],
                "focusModelName": selected_model["name"],
            }
            for candidate in selected_model["windowSearch"]
        ]
    else:
        lookback_candidates = [
            {
                "lookback": report["lookback"],
                "winnerModelId": winner["id"],
                "winnerModelName": winner["name"],
                "focusModelId": focus_model["id"],
                "focusModelName": focus_model["name"],
                "validationNmaePercent": focus_model["validationMetrics"][
                    "meanNmaePercent"
                ],
                "testNmaePercent": focus_model["testMetrics"]["meanNmaePercent"],
            }
            for report in candidate_reports
            for winner in [
                min(
                    report["models"],
                    key=lambda model: model["validationMetrics"]["meanNmaePercent"],
                )
            ]
            for focus_model in [
                next(
                    model
                    for model in report["models"]
                    if model["id"] == selected_model["id"]
                )
            ]
        ]
    final_report["lookbackSearch"] = {
        "candidates": lookback_candidates,
        "perModel": per_model_search,
        "bestModelId": selected_model["id"],
        "bestModelName": selected_model["name"],
        "bestLookback": deployment_lookback,
        "selectionRule": (
            "模型和窗口只由验证集选择；完全并列时取中间窗口，"
            "测试集仅用于最终统一审计"
        ),
    }

    best_test_nmae = best_test_model["testMetrics"]["meanNmaePercent"]
    selected_test_nmae = selected_model["testMetrics"]["meanNmaePercent"]
    baseline_test_nmae = selected_baseline["testMetrics"]["meanNmaePercent"]
    final_report["summary"] = {
        "validationNmaePercent": selected_model["validationMetrics"][
            "meanNmaePercent"
        ],
        "testNmaePercent": selected_test_nmae,
        "testNormalizedAccuracyPercent": selected_model["testMetrics"][
            "normalizedAccuracyPercent"
        ],
        "testSmapePercent": selected_model["testMetrics"]["meanSmapePercent"],
        "baselineTestNmaePercent": baseline_test_nmae,
        "deepVsBaselineNmaePoints": baseline_test_nmae
        - selected_deep["testMetrics"]["meanNmaePercent"],
        "deepBeatsBaseline": selected_deep["testMetrics"]["meanNmaePercent"]
        < baseline_test_nmae,
        "bestTestNmaePercent": best_test_nmae,
        "bestTestNormalizedAccuracyPercent": best_test_model["testMetrics"][
            "normalizedAccuracyPercent"
        ],
        "bestVsBaselineNmaePoints": baseline_test_nmae - best_test_nmae,
    }
    final_report["environment"]["trainingSeconds"] = sum(
        report["environment"]["trainingSeconds"] for report in candidate_reports
    ) + final_report["environment"]["trainingSeconds"]
    final_report["environment"]["lookbackSearchSeconds"] = (
        time.perf_counter() - search_started
    )
    final_report["environment"]["searchRuns"] = len(candidates)

    return final_report

def save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
