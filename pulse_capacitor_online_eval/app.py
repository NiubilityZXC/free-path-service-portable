#!/usr/bin/env python3
"""Pulse capacitor online state evaluation web service.

The implementation follows the GM(1,1) grey model described in the supplied
patent draft and exposes it through a small standard-library HTTP service.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw"
STATIC = ROOT / "static"
STRICT_DEEP_REPORT = ROOT / "data" / "models" / "strict_deep_report.json"
STRICT_FAILURE_ADDITIONAL_STEPS = 1500
STRICT_CALIBRATED_RULES = {
    "VoltageMaxRaw": {"direction": "increase", "percent": 6.98},
    "VoltageFirstZeroTimeUs": {"direction": "decrease", "percent": 23.32},
}


class EvaluationError(ValueError):
    """Raised when an input sequence cannot be evaluated."""


@dataclass
class GMResult:
    a: float
    b: float
    x0: list[float]
    fitted: list[float]
    forecast: list[float]
    residuals: list[float]
    relative_errors: list[float]
    accuracy_percent: float
    level_ratios: list[float]
    level_ratio_bounds: tuple[float, float]
    level_ratio_valid: bool
    model_precision_band: str


@dataclass
class ForecastResult:
    fitted: list[float]
    forecast: list[float]
    parameters: dict[str, float]


MODEL_META = {
    "gm11": {
        "name": "GM(1,1) 灰色模型",
        "description": "申请文件方法，适合小样本并以指数响应外推趋势。",
    },
    "persistence": {
        "name": "持续值预测",
        "description": "把窗口最后一个实测值保持到未来，适合短期变化很小的参数。",
    },
    "linear_trend": {
        "name": "线性趋势回归",
        "description": "用最小二乘直线拟合整个建模窗口并向未来延伸。",
    },
    "holt_damped": {
        "name": "阻尼 Holt 趋势",
        "description": "同时估计当前水平和变化速度，并让长期趋势逐渐衰减。",
    },
    "autoregression": {
        "name": "AR(1) 自回归",
        "description": "利用相邻两次记录的关系预测下一次，适合具有连续相关性的参数。",
    },
}


def load_dataset(name: str) -> list[dict[str, str]]:
    safe_name = Path(unquote(name)).name
    path = RAW_DATA / safe_name
    if not path.exists() or path.suffix.lower() != ".csv":
        raise EvaluationError(f"dataset not found: {name}")

    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    columns: list[str] = []
    for column in rows[0]:
        ok = True
        seen = False
        for row in rows[:30]:
            value = row.get(column, "").strip()
            if not value:
                continue
            try:
                float(value)
                seen = True
            except ValueError:
                ok = False
                break
        if ok and seen:
            columns.append(column)
    return columns


def to_float_series(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def failure_baseline(rows: list[dict[str, str]], column: str) -> tuple[float, int]:
    """Return the original first-discharge value and its one-based source row."""
    if not rows:
        raise EvaluationError("dataset is empty")
    try:
        value = float(rows[0].get(column, ""))
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"first discharge has no numeric value for {column}") from exc
    if not math.isfinite(value):
        raise EvaluationError(f"first discharge has no finite value for {column}")
    if column == "DischargePeriodSec" and value <= 1e-3:
        for row_number, row in enumerate(rows[1:], start=2):
            try:
                interval = float(row.get(column, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(interval) and interval > 1e-3:
                return interval, row_number
        raise EvaluationError("dataset has no measurable discharge interval")
    return value, 1


def smooth_sequence(values: list[float]) -> list[float]:
    if not values:
        return values
    smoothed = [values[0]]
    for idx, value in enumerate(values[1:], start=2):
        alpha = 2.0 / (idx + 1.0)
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def level_ratio_check(values: list[float]) -> tuple[list[float], tuple[float, float], bool]:
    n = len(values)
    lower = math.exp(-2.0 / (n + 1.0))
    upper = math.exp(2.0 / (n + 1.0))
    ratios = [values[i - 1] / values[i] for i in range(1, n)]
    valid = all(lower < ratio < upper for ratio in ratios)
    return ratios, (lower, upper), valid


def gm11(values: list[float], horizon: int = 800) -> GMResult:
    if len(values) < 4:
        raise EvaluationError("GM(1,1) requires at least 4 positive data points")
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise EvaluationError("GM(1,1) requires all values to be positive and finite")

    n = len(values)
    horizon = max(1, min(int(horizon), 10000))
    ratios, bounds, ratios_valid = level_ratio_check(values)

    x1: list[float] = []
    acc = 0.0
    for value in values:
        acc += value
        x1.append(acc)

    z = [0.5 * (x1[i] + x1[i - 1]) for i in range(1, n)]
    y = values[1:]
    m = len(y)
    sum_z = sum(z)
    sum_z2 = sum(item * item for item in z)
    sum_y = sum(y)
    sum_zy = sum(zi * yi for zi, yi in zip(z, y))
    denom = m * sum_z2 - sum_z * sum_z
    if abs(denom) < 1e-12:
        raise EvaluationError("least-squares matrix is singular for this sequence")

    a = (sum_z * sum_y - m * sum_zy) / denom
    b = (sum_y + a * sum_z) / m

    def cumulative_hat(k_one_based: int) -> float:
        if abs(a) < 1e-10:
            return values[0] + b * (k_one_based - 1)
        return (values[0] - b / a) * math.exp(-a * (k_one_based - 1)) + b / a

    cumulative = [cumulative_hat(k) for k in range(1, n + horizon + 1)]
    forecast = [values[0]]
    for idx in range(1, len(cumulative)):
        forecast.append(cumulative[idx] - cumulative[idx - 1])
    fitted = forecast[:n]

    residuals = [actual - fit for actual, fit in zip(values, fitted)]
    relative_errors = [
        abs(error) / abs(actual) if abs(actual) > 1e-12 else 0.0
        for actual, error in zip(values, residuals)
    ]
    mean_relative_error = sum(relative_errors) / len(relative_errors)
    accuracy_percent = max(0.0, (1.0 - mean_relative_error) * 100.0)

    if -0.3 <= a < 2:
        band = "中长期预测可信"
    elif a < -0.3:
        band = "退化较快，仅建议短期参考"
    else:
        band = "波动过大，建议重新建模"

    return GMResult(
        a=a,
        b=b,
        x0=values,
        fitted=fitted,
        forecast=forecast,
        residuals=residuals,
        relative_errors=relative_errors,
        accuracy_percent=accuracy_percent,
        level_ratios=ratios,
        level_ratio_bounds=bounds,
        level_ratio_valid=ratios_valid,
        model_precision_band=band,
    )


def persistence_forecast(values: list[float], horizon: int) -> ForecastResult:
    if not values:
        raise EvaluationError("persistence forecast requires at least one value")
    fitted = [values[0], *values[:-1]]
    forecast = [*fitted, *([values[-1]] * horizon)]
    return ForecastResult(fitted=fitted, forecast=forecast, parameters={})


def linear_trend_forecast(values: list[float], horizon: int) -> ForecastResult:
    if len(values) < 2:
        raise EvaluationError("linear trend requires at least two values")
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    denominator = sum((idx - mean_x) ** 2 for idx in range(n))
    slope = (
        sum((idx - mean_x) * (value - mean_y) for idx, value in enumerate(values))
        / denominator
        if denominator > 1e-12
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    forecast = [intercept + slope * idx for idx in range(n + horizon)]
    return ForecastResult(
        fitted=forecast[:n],
        forecast=forecast,
        parameters={"slope": slope, "intercept": intercept},
    )


def _holt_damped_run(
    values: list[float],
    horizon: int,
    alpha: float,
    beta: float,
    phi: float,
) -> ForecastResult:
    level = values[0]
    trend = values[1] - values[0]
    fitted = [values[0]]
    for value in values[1:]:
        one_step = level + phi * trend
        fitted.append(one_step)
        previous_level = level
        level = alpha * value + (1.0 - alpha) * one_step
        trend = beta * (level - previous_level) + (1.0 - beta) * phi * trend

    forecast = list(fitted)
    damped_sum = 0.0
    for step in range(1, horizon + 1):
        damped_sum += phi**step
        forecast.append(level + damped_sum * trend)
    return ForecastResult(
        fitted=fitted,
        forecast=forecast,
        parameters={"alpha": alpha, "beta": beta, "phi": phi},
    )


def holt_damped_forecast(values: list[float], horizon: int) -> ForecastResult:
    if len(values) < 2:
        raise EvaluationError("Holt trend requires at least two values")
    best_result = None
    best_error = math.inf
    for alpha in (0.2, 0.5, 0.8):
        for beta in (0.1, 0.3, 0.6):
            for phi in (0.8, 0.9, 0.98):
                result = _holt_damped_run(values, horizon, alpha, beta, phi)
                squared_error = sum(
                    (actual - predicted) ** 2
                    for actual, predicted in zip(values[1:], result.fitted[1:])
                )
                if squared_error < best_error:
                    best_error = squared_error
                    best_result = result
    if best_result is None:
        raise EvaluationError("unable to fit Holt trend")
    return best_result


def autoregression_forecast(values: list[float], horizon: int) -> ForecastResult:
    if len(values) < 3:
        raise EvaluationError("AR(1) requires at least three values")
    previous = values[:-1]
    current = values[1:]
    mean_previous = sum(previous) / len(previous)
    mean_current = sum(current) / len(current)
    denominator = sum((value - mean_previous) ** 2 for value in previous)
    coefficient = (
        sum(
            (old - mean_previous) * (new - mean_current)
            for old, new in zip(previous, current)
        )
        / denominator
        if denominator > 1e-12
        else 0.0
    )
    coefficient = max(-1.0, min(1.0, coefficient))
    intercept = mean_current - coefficient * mean_previous
    fitted = [values[0]] + [intercept + coefficient * value for value in previous]
    forecast = list(fitted)
    predicted = values[-1]
    for _ in range(horizon):
        predicted = intercept + coefficient * predicted
        forecast.append(predicted)
    return ForecastResult(
        fitted=fitted,
        forecast=forecast,
        parameters={"coefficient": coefficient, "intercept": intercept},
    )


def forecast_with_method(method: str, values: list[float], horizon: int) -> ForecastResult:
    horizon = max(1, min(int(horizon), 10000))
    if method == "gm11":
        model = gm11(values, horizon=horizon)
        return ForecastResult(
            fitted=model.fitted,
            forecast=model.forecast,
            parameters={"a": model.a, "b": model.b},
        )
    if method == "persistence":
        return persistence_forecast(values, horizon)
    if method == "linear_trend":
        return linear_trend_forecast(values, horizon)
    if method == "holt_damped":
        return holt_damped_forecast(values, horizon)
    if method == "autoregression":
        return autoregression_forecast(values, horizon)
    raise EvaluationError(f"unknown forecast method: {method}")


def infer_failure(column: str, first_value: float, options: dict[str, Any]) -> tuple[str, float]:
    lower_name = column.lower()
    parameter_rules = options.get("parameterRules") or options.get("parameterThresholds") or {}
    if isinstance(parameter_rules, dict):
        rule = parameter_rules.get(column)
        if isinstance(rule, dict):
            direction = str(rule.get("direction", "")).lower()
            percent = float(rule.get("percent", 0.0))
            if direction in {"increase", "decrease"} and math.isfinite(percent) and percent >= 0:
                if direction == "decrease":
                    return direction, first_value * (1.0 - percent / 100.0)
                return direction, first_value * (1.0 + percent / 100.0)

    calibrated = STRICT_CALIBRATED_RULES.get(column)
    if calibrated:
        direction = str(calibrated["direction"])
        percent = float(calibrated["percent"])
        factor = 1.0 - percent / 100.0 if direction == "decrease" else 1.0 + percent / 100.0
        return direction, first_value * factor

    cap_loss = float(options.get("capLossPercent", 5.0))
    esr_growth = float(options.get("esrGrowthPercent", 100.0))
    generic_growth = float(options.get("genericGrowthPercent", 20.0))

    if lower_name.startswith("c") and "esr" not in lower_name:
        return "decrease", first_value * (1.0 - cap_loss / 100.0)
    if "esr" in lower_name:
        return "increase", first_value * (1.0 + esr_growth / 100.0)
    if "voltagemax" in lower_name or "currentmax" in lower_name:
        return "increase", first_value * (1.0 + cap_loss / 100.0)
    if (
        "voltageminabs" in lower_name
        or "currentminabs" in lower_name
        or "reversepeak" in lower_name
    ):
        return "decrease", first_value * (1.0 - cap_loss / 100.0)
    if "firstzerotime" in lower_name:
        return "decrease", first_value * (1.0 - generic_growth / 100.0)
    if "dischargeperiod" in lower_name:
        return "increase", first_value * (1.0 + generic_growth / 100.0)
    return "increase", first_value * (1.0 + generic_growth / 100.0)


def find_crossing(
    series: list[float],
    observed_length: int,
    direction: str,
    critical_value: float,
) -> int | None:
    for idx, value in enumerate(series, start=1):
        if direction == "decrease" and value <= critical_value:
            return idx
        if direction == "increase" and value >= critical_value:
            return idx
    return None


def find_future_crossing(
    series: list[float],
    observed_length: int,
    direction: str,
    critical_value: float,
) -> int | None:
    for idx in range(observed_length, len(series)):
        value = series[idx]
        if direction == "decrease" and value <= critical_value:
            return idx + 1
        if direction == "increase" and value >= critical_value:
            return idx + 1
    return None


def positive_float_option(options: dict[str, Any], key: str) -> float | None:
    value = options.get(key)
    if value in (None, ""):
        return None
    number = float(value)
    return number if number > 0 else None


def resolve_horizon(
    options: dict[str, Any],
    period: float,
    step_key: str,
    time_key: str,
    default_steps: int,
) -> tuple[int, float | None]:
    requested_time = positive_float_option(options, time_key)
    if requested_time is not None:
        if period <= 0:
            raise EvaluationError("time-based prediction requires a positive period")
        return max(1, math.ceil(requested_time / period)), requested_time
    return int(options.get(step_key, default_steps)), None


def validation_metrics(points: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not points:
        return {
            "runs": 0,
            "mae": None,
            "rmse": None,
            "mapePercent": None,
            "meanBias": None,
            "maxAbsPercentError": None,
        }

    errors = [point["error"] for point in points]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    percent_errors = [point["absPercentError"] for point in points]
    return {
        "runs": len(points),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "mapePercent": sum(percent_errors) / len(percent_errors),
        "meanBias": sum(errors) / len(errors),
        "maxAbsPercentError": max(percent_errors),
    }


def median_period(rows: list[dict[str, str]]) -> float:
    if not rows or "Time" not in rows[0]:
        return 1.0
    times = to_float_series(rows, "Time")
    diffs = [
        times[i] - times[i - 1]
        for i in range(1, len(times))
        if math.isfinite(times[i] - times[i - 1]) and times[i] > times[i - 1]
    ]
    return median(diffs) if diffs else 1.0


def validate_dataset(options: dict[str, Any]) -> dict[str, Any]:
    dataset = options.get("dataset", "")
    rows = load_dataset(dataset)
    if not rows:
        raise EvaluationError("dataset is empty")

    columns = options.get("columns") or ["C1kHz", "ESR1kHz"]
    columns = [str(col) for col in columns if str(col) != "Time"]
    if not columns:
        raise EvaluationError("select at least one numeric column")

    window = int(options.get("window", 80))
    period = float(options.get("period") or median_period(rows))
    horizon, validation_time = resolve_horizon(
        options,
        period,
        "validationHorizon",
        "validationTime",
        int(options.get("horizon", 20)),
    )
    if window < 4:
        raise EvaluationError("window must be at least 4")
    if horizon < 1:
        raise EvaluationError("validation horizon must be at least 1")

    max_runs = int(options.get("maxValidationRuns", 500))
    max_runs = max(1, min(max_runs, 5000))
    use_smoothing = bool(options.get("smoothing", False))
    error_threshold = float(options.get("validationErrorThreshold", 5.0))
    requested_start = int(options.get("validationStartIndex") or 1)
    requested_end = int(options.get("validationEndIndex") or 0)
    requested_stride = int(options.get("validationStride") or 1)
    start_offset = max(0, requested_start - 1)
    explicit_stride = max(1, requested_stride)

    results = []
    resolved_end_index = None
    for column in columns:
        values = to_float_series(rows, column)
        run_count = len(values) - window - horizon + 1
        if run_count <= 0:
            raise EvaluationError(
                f"not enough values in column {column} for window={window}, horizon={horizon}"
            )

        required_min_end = start_offset + window + horizon
        if required_min_end > len(values):
            raise EvaluationError(
                "回测范围不足：请减小建模窗口/预测时长，或选择更长的数据集"
            )
        end_index = requested_end if requested_end > 0 else len(values)
        if end_index < required_min_end:
            end_index = required_min_end
        resolved_end_index = end_index if resolved_end_index is None else min(resolved_end_index, end_index)
        stop_exclusive = min(run_count, end_index - window - horizon + 1)
        if start_offset >= stop_exclusive:
            raise EvaluationError(
                "回测范围为空：请增大结束行，或减小建模窗口/预测时长"
            )

        stride = explicit_stride
        if requested_stride <= 0:
            stride = max(1, math.ceil((stop_exclusive - start_offset) / max_runs))
        points = []
        for start in range(start_offset, stop_exclusive, stride):
            train = values[start : start + window]
            train_values = smooth_sequence(train) if use_smoothing else train
            model = gm11(train_values, horizon=horizon)
            actual_index = start + window + horizon - 1
            predicted = model.forecast[window + horizon - 1]
            actual = values[actual_index]
            error = predicted - actual
            abs_percent_error = abs(error) / abs(actual) * 100.0 if abs(actual) > 1e-12 else 0.0
            points.append(
                {
                    "trainStartIndex": start + 1,
                    "trainEndIndex": start + window,
                    "targetIndex": actual_index + 1,
                    "predicted": predicted,
                    "actual": actual,
                    "error": error,
                    "absPercentError": abs_percent_error,
                    "modelAccuracyPercent": model.accuracy_percent,
                }
            )

        metrics = validation_metrics(points)
        results.append(
            {
                "column": column,
                "window": window,
                "horizon": horizon,
                "validationStartIndex": start_offset + 1,
                "validationEndIndex": end_index,
                "stride": stride,
                "metrics": metrics,
                "status": (
                    "通过"
                    if metrics["mapePercent"] is not None
                    and float(metrics["mapePercent"]) <= error_threshold
                    else "误差偏高"
                ),
                "points": points,
            }
        )

    valid_mapes = [
        item["metrics"]["mapePercent"]
        for item in results
        if item["metrics"]["mapePercent"] is not None
    ]
    final_mape = sum(float(value) for value in valid_mapes) / len(valid_mapes)
    return {
        "dataset": dataset,
        "columns": columns,
        "window": window,
        "horizon": horizon,
        "period": period,
        "validationTime": validation_time,
        "validationStartIndex": start_offset + 1,
        "validationEndIndex": resolved_end_index,
        "validationStride": explicit_stride,
        "smoothing": use_smoothing,
        "validationErrorThreshold": error_threshold,
        "summary": {
            "finalMapePercent": final_mape,
            "status": "通过" if final_mape <= error_threshold else "误差偏高",
        },
        "results": results,
    }


def compare_prediction_models(options: dict[str, Any]) -> dict[str, Any]:
    dataset = options.get("dataset", "")
    rows = load_dataset(dataset)
    if not rows:
        raise EvaluationError("dataset is empty")

    columns = options.get("columns") or ["C1kHz", "ESR1kHz"]
    columns = [str(col) for col in columns if str(col) != "Time"]
    if not columns:
        raise EvaluationError("select at least one numeric column")

    window = int(options.get("window", 80))
    if window < 4:
        raise EvaluationError("window must be at least 4")
    period = float(options.get("period") or median_period(rows))
    prediction_horizon, prediction_time = resolve_horizon(
        options,
        period,
        "horizon",
        "predictionTime",
        1200,
    )
    validation_horizon, validation_time = resolve_horizon(
        options,
        period,
        "validationHorizon",
        "validationTime",
        20,
    )
    if prediction_horizon < 1 or validation_horizon < 1:
        raise EvaluationError("prediction and validation horizons must be at least 1")

    use_smoothing = bool(options.get("smoothing", False))
    error_threshold = float(options.get("validationErrorThreshold", 5.0))
    requested_start = int(options.get("validationStartIndex") or 1)
    requested_end = int(options.get("validationEndIndex") or 0)
    stride = max(1, int(options.get("validationStride") or 1))
    start_offset = max(0, requested_start - 1)
    methods = list(MODEL_META)
    results = []
    resolved_end_index = None

    for column in columns:
        values = to_float_series(rows, column)
        run_count = len(values) - window - validation_horizon + 1
        if run_count <= 0:
            raise EvaluationError(
                f"not enough values in column {column} for window={window}, "
                f"horizon={validation_horizon}"
            )

        required_min_end = start_offset + window + validation_horizon
        if required_min_end > len(values):
            raise EvaluationError(
                "回测范围不足：请减小建模窗口/验证步数，或选择更长的数据集"
            )
        end_index = requested_end if requested_end > 0 else len(values)
        if end_index < required_min_end:
            end_index = required_min_end
        end_index = min(end_index, len(values))
        resolved_end_index = (
            end_index if resolved_end_index is None else min(resolved_end_index, end_index)
        )
        stop_exclusive = min(run_count, end_index - window - validation_horizon + 1)
        if start_offset >= stop_exclusive:
            raise EvaluationError(
                "回测范围为空：请增大结束行，或减小建模窗口/验证步数"
            )

        starts = list(range(start_offset, stop_exclusive, stride))
        points_by_method: dict[str, list[dict[str, float]]] = {
            method: [] for method in methods
        }
        failed_by_method = {method: 0 for method in methods}
        for start in starts:
            train = values[start : start + window]
            train_values = smooth_sequence(train) if use_smoothing else train
            actual_index = start + window + validation_horizon - 1
            actual = values[actual_index]
            for method in methods:
                try:
                    model = forecast_with_method(method, train_values, validation_horizon)
                    predicted = model.forecast[window + validation_horizon - 1]
                    if not math.isfinite(predicted):
                        raise EvaluationError("non-finite forecast")
                except (EvaluationError, OverflowError, ValueError):
                    failed_by_method[method] += 1
                    continue
                error = predicted - actual
                abs_percent_error = (
                    abs(error) / abs(actual) * 100.0 if abs(actual) > 1e-12 else 0.0
                )
                points_by_method[method].append(
                    {
                        "trainStartIndex": start + 1,
                        "trainEndIndex": start + window,
                        "targetIndex": actual_index + 1,
                        "predicted": predicted,
                        "actual": actual,
                        "error": error,
                        "absPercentError": abs_percent_error,
                    }
                )

        candidate_metrics = []
        for method in methods:
            points = points_by_method[method]
            if failed_by_method[method] or len(points) != len(starts):
                continue
            metrics = validation_metrics(points)
            candidate_metrics.append(
                {
                    "method": method,
                    "methodName": MODEL_META[method]["name"],
                    "description": MODEL_META[method]["description"],
                    "metrics": metrics,
                    "status": (
                        "通过"
                        if metrics["mapePercent"] is not None
                        and float(metrics["mapePercent"]) <= error_threshold
                        else "误差偏高"
                    ),
                }
            )
        if not candidate_metrics:
            raise EvaluationError(f"all prediction methods failed for column {column}")
        candidate_metrics.sort(
            key=lambda item: (
                float(item["metrics"]["mapePercent"]),
                float(item["metrics"]["rmse"]),
            )
        )

        final_values = values[-window:]
        model_values = smooth_sequence(final_values) if use_smoothing else final_values
        selected_candidate = None
        final_model = None
        for candidate in candidate_metrics:
            try:
                fitted_model = forecast_with_method(
                    str(candidate["method"]), model_values, prediction_horizon
                )
                if all(math.isfinite(value) for value in fitted_model.forecast):
                    selected_candidate = candidate
                    final_model = fitted_model
                    break
            except (EvaluationError, OverflowError, ValueError):
                continue
        if selected_candidate is None or final_model is None:
            raise EvaluationError(f"unable to build final selected model for column {column}")

        selected_method = str(selected_candidate["method"])
        baseline_value, baseline_row = failure_baseline(rows, column)
        direction, critical = infer_failure(column, baseline_value, options)
        crossing = find_future_crossing(
            final_model.forecast, len(model_values), direction, critical
        )
        remaining_samples = None if crossing is None else crossing - len(model_values)
        critical_periods = None if crossing is None else crossing * period
        gm_candidate = next(
            (item for item in candidate_metrics if item["method"] == "gm11"), None
        )
        best_mape = float(selected_candidate["metrics"]["mapePercent"])
        gm_mape = (
            float(gm_candidate["metrics"]["mapePercent"])
            if gm_candidate is not None
            else None
        )
        results.append(
            {
                "column": column,
                "bestMethod": selected_method,
                "bestMethodName": selected_candidate["methodName"],
                "methodDescription": selected_candidate["description"],
                "methodParameters": final_model.parameters,
                "bestMetrics": selected_candidate["metrics"],
                "gmMetrics": gm_candidate["metrics"] if gm_candidate is not None else None,
                "mapeImprovementPoints": (
                    gm_mape - best_mape if gm_mape is not None else None
                ),
                "candidateMetrics": candidate_metrics,
                "validationPoints": points_by_method[selected_method],
                "baselineValue": baseline_value,
                "baselineRow": baseline_row,
                "direction": direction,
                "criticalValue": critical,
                "criticalSampleIndex": crossing,
                "remainingSamples": remaining_samples,
                "criticalPeriods": critical_periods,
                "series": model_values,
                "rawSeries": final_values,
                "fitted": final_model.fitted,
                "forecast": final_model.forecast[
                    : len(model_values) + min(prediction_horizon, 300)
                ],
                "status": selected_candidate["status"],
            }
        )

    best_mapes = [float(item["bestMetrics"]["mapePercent"]) for item in results]
    gm_mapes = [
        float(item["gmMetrics"]["mapePercent"])
        for item in results
        if item["gmMetrics"] is not None
    ]
    final_mape = sum(best_mapes) / len(best_mapes)
    gm_mape = sum(gm_mapes) / len(gm_mapes) if gm_mapes else None
    selected_names = list(dict.fromkeys(item["bestMethodName"] for item in results))
    return {
        "dataset": dataset,
        "columns": columns,
        "window": window,
        "horizon": prediction_horizon,
        "predictionTime": prediction_time,
        "validationHorizon": validation_horizon,
        "validationTime": validation_time,
        "validationStartIndex": start_offset + 1,
        "validationEndIndex": resolved_end_index,
        "validationStride": stride,
        "validationErrorThreshold": error_threshold,
        "smoothing": use_smoothing,
        "methods": [
            {"method": method, **MODEL_META[method]} for method in methods
        ],
        "summary": {
            "bestMapePercent": final_mape,
            "bestAccuracyPercent": max(0.0, 100.0 - final_mape),
            "gmMapePercent": gm_mape,
            "mapeImprovementPoints": (
                gm_mape - final_mape if gm_mape is not None else None
            ),
            "selectedMethods": selected_names,
            "alternativeSelectionCount": sum(
                1 for item in results if item["bestMethod"] != "gm11"
            ),
            "status": "通过" if final_mape <= error_threshold else "误差偏高",
        },
        "results": results,
    }


def source_info(dataset: str) -> dict[str, str]:
    if dataset.startswith("ceshishuju"):
        return {
            "name": "用户提供 OWON 示波器脉冲放电波形",
            "publisher": "Local experiment",
            "license": "User supplied",
            "url": "",
        }
    return {
        "name": "High Temperature Evaluation of Tantalum Capacitors - Test 1",
        "publisher": "Sandia National Laboratories / OEDI / Data.gov",
        "license": "Creative Commons Attribution 4.0",
        "url": "https://catalog.data.gov/dataset/high-temperature-evaluation-of-tantalum-capacitors-test-1-3865b",
    }


def uncertainty_notes(dataset: str) -> list[str]:
    notes = [
        "寿命比按工程逻辑采用“已运行时间/预测寿命 >= 阈值”触发预警。",
    ]
    if dataset == "ceshishuju_patent_features.csv":
        notes.extend(
            [
                "该数据集由 OWON 示波器 CH1 波形提取主幅值、反向峰系数、首次过零时间和放电周期等运行参数。",
                "已确认采集时启用了电流换算，因此 CH1 特征按电流换算波形解释，不等同于电容器端真实电压。",
                "历史 CSV 的 Voltage* 与 Raw 字段名为兼容现有接口而保留；固定比例换算不会改变归一化预测精度。",
            ]
        )
    elif dataset.startswith("ceshishuju"):
        notes.extend(
            [
                "该数据集由用户实验波形转换而来，可用于趋势建模；严格物理量仍需结合探头比例和采集链路标定。",
                "当前转换列为波形特征，不等同于离线测得的电容量 C 或 ESR。",
            ]
        )
    else:
        notes.extend(
            [
                "公开数据为钽电容高温老化 C/ESR 数据，不是脉冲电容器每次放电的电压/电流波形数据。",
                "公开 CSV 没有标注每次脉冲的最大/最小电压、电流和过零时间，因此这些字段需要接入真实采集系统或上传自有 CSV。",
            ]
        )
    return notes


def evaluate_dataset(options: dict[str, Any]) -> dict[str, Any]:
    dataset = options.get("dataset", "")
    rows = load_dataset(dataset)
    if not rows:
        raise EvaluationError("dataset is empty")

    columns = options.get("columns") or ["C1kHz", "ESR1kHz"]
    columns = [str(col) for col in columns if str(col) != "Time"]
    if not columns:
        raise EvaluationError("select at least one numeric column")

    window = int(options.get("window", 80))
    if window < 4:
        raise EvaluationError("window must be at least 4")
    use_smoothing = bool(options.get("smoothing", False))
    period = float(options.get("period") or median_period(rows))
    horizon, prediction_time = resolve_horizon(
        options,
        period,
        "horizon",
        "predictionTime",
        1200,
    )
    accuracy_threshold = float(options.get("accuracyThreshold", 90.0))
    life_ratio_threshold = float(options.get("lifeRatioThreshold", 90.0))

    selected_rows = rows[-window:]
    results = []
    for column in columns:
        values = to_float_series(selected_rows, column)
        if len(values) < 4:
            raise EvaluationError(f"not enough numeric values in column {column}")
        model_values = smooth_sequence(values) if use_smoothing else values
        result = gm11(model_values, horizon=horizon)
        baseline_value, baseline_row = failure_baseline(rows, column)
        direction, critical = infer_failure(column, baseline_value, options)
        crossing = find_crossing(result.forecast, len(model_values), direction, critical)
        remaining_samples = None if crossing is None else crossing - len(model_values)
        critical_periods = None if crossing is None else crossing * period
        elapsed_periods = len(model_values) * period
        life_ratio_percent = None
        if critical_periods and critical_periods > 0:
            life_ratio_percent = elapsed_periods / critical_periods * 100.0

        engineering_stop = (
            result.accuracy_percent < accuracy_threshold
            or (life_ratio_percent is not None and life_ratio_percent >= life_ratio_threshold)
        )

        results.append(
            {
                "column": column,
                "baselineValue": baseline_value,
                "baselineRow": baseline_row,
                "direction": direction,
                "criticalValue": critical,
                "a": result.a,
                "b": result.b,
                "accuracyPercent": result.accuracy_percent,
                "modelPrecisionBand": result.model_precision_band,
                "levelRatioValid": result.level_ratio_valid,
                "levelRatioBounds": result.level_ratio_bounds,
                "levelRatioInvalidCount": sum(
                    1
                    for ratio in result.level_ratios
                    if not (result.level_ratio_bounds[0] < ratio < result.level_ratio_bounds[1])
                ),
                "criticalSampleIndex": crossing,
                "remainingSamples": remaining_samples,
                "elapsedPeriods": elapsed_periods,
                "criticalPeriods": critical_periods,
                "lifeRatioPercent": life_ratio_percent,
                "warningStatus": "预警" if engineering_stop else "正常",
                "engineeringStatus": "预警" if engineering_stop else "正常",
                "series": model_values,
                "rawSeries": values,
                "fitted": result.fitted,
                "forecast": result.forecast[: len(model_values) + min(horizon, 300)],
                "relativeErrors": result.relative_errors,
            }
        )

    finite_lives = [r["criticalPeriods"] for r in results if r["criticalPeriods"] is not None]
    final_life = sum(finite_lives) / len(finite_lives) if finite_lives else None
    final_accuracy = sum(r["accuracyPercent"] for r in results) / len(results)
    final_elapsed = len(selected_rows) * period
    final_life_ratio = (
        final_elapsed / final_life * 100.0 if final_life and final_life > 0 else None
    )

    engineering_stop = final_accuracy < accuracy_threshold or (
        final_life_ratio is not None and final_life_ratio >= life_ratio_threshold
    )

    return {
        "dataset": dataset,
        "rowsUsed": len(selected_rows),
        "period": period,
        "horizon": horizon,
        "predictionTime": prediction_time,
        "smoothing": use_smoothing,
        "source": source_info(dataset),
        "results": results,
        "summary": {
            "finalPredictedLifePeriods": final_life,
            "finalAccuracyPercent": final_accuracy,
            "finalLifeRatioPercent": final_life_ratio,
            "warningStatus": "预警" if engineering_stop else "正常",
            "engineeringStatus": "预警" if engineering_stop else "正常",
        },
        "uncertainties": uncertainty_notes(dataset),
    }


def default_columns_for_dataset(name: str, columns: list[str]) -> list[str]:
    if name == "ceshishuju_patent_features.csv":
        strict_features = (
            "VoltageMaxRaw", "VoltageFirstZeroTimeUs",
            "VoltageReversePeakCoefficient", "VoltageMinAbsRaw", "DischargePeriodSec",
        )
        return [col for col in strict_features if col in columns]
    if name == "ceshishuju_waveform_features.csv":
        return [col for col in ("PeakADC", "PeakToPeakADC") if col in columns]
    return [col for col in ("C1kHz", "ESR1kHz") if col in columns]


def list_available_datasets() -> dict[str, Any]:
    datasets = []
    for path in sorted(RAW_DATA.glob("*.csv")):
        rows = load_dataset(path.name)
        columns = numeric_columns(rows)
        datasets.append(
            {
                "name": path.name,
                "rowCount": len(rows),
                "columns": columns,
                "defaultColumns": default_columns_for_dataset(path.name, columns),
            }
        )
    return {"datasets": datasets}


def rows_preview(name: str, limit: int = 120) -> dict[str, Any]:
    rows = load_dataset(name)
    limit = max(1, min(limit, 1000))
    return {"dataset": name, "rows": rows[:limit], "columns": list(rows[0]) if rows else []}


def load_strict_deep_report() -> dict[str, Any]:
    if not STRICT_DEEP_REPORT.exists():
        raise EvaluationError(
            "strict deep-learning report is missing; run tools/train_strict_deep_models.py"
        )
    payload = json.loads(STRICT_DEEP_REPORT.read_text(encoding="utf-8"))
    if payload.get("dataset") != "ceshishuju_patent_features.csv":
        raise EvaluationError("deep-learning report does not belong to the strict feature dataset")
    rows = load_dataset(str(payload["dataset"]))
    life = payload.get("lifeForecast") or {}
    features = [str(item["feature"]) for item in payload.get("features", [])]
    baselines: dict[str, float] = {}
    baseline_rows: dict[str, int] = {}
    for feature in features:
        value, row_number = failure_baseline(rows, feature)
        baselines[feature] = value
        baseline_rows[feature] = row_number
    life["failureBaselineValues"] = baselines
    life["baselineRowsByFeature"] = baseline_rows
    life["healthyBaselineRows"] = 1
    life["healthyBaselineValues"] = baselines
    life["baselineDefinition"] = (
        "原始数据第1次放电；放电周期取第1次至第2次之间的首个有效间隔"
    )
    runtime_baseline_note = (
        "寿命阈值基准采用原始数据第1次放电；放电周期采用第2行首个有效间隔，"
        "不再采用历史报告中的最初32次中位数"
    )
    notes = [str(note) for note in payload.get("notes", [])]
    notes = [
        runtime_baseline_note if "最初 32 次健康阶段中位数" in note else note
        for note in notes
    ]
    if runtime_baseline_note not in notes:
        notes.append(runtime_baseline_note)
    payload["notes"] = notes
    calibrated_rules = {}
    for feature, rule in STRICT_CALIBRATED_RULES.items():
        baseline = baselines[feature]
        percent = float(rule["percent"])
        direction = str(rule["direction"])
        threshold = baseline * (
            1.0 - percent / 100.0 if direction == "decrease" else 1.0 + percent / 100.0
        )
        calibrated_rules[feature] = {
            "direction": direction,
            "percent": percent,
            "baselineValue": baseline,
            "thresholdValue": threshold,
        }
    life["failureCalibration"] = {
        "observedRows": len(rows),
        "additionalFailureSteps": STRICT_FAILURE_ADDITIONAL_STEPS,
        "totalFailureRow": len(rows) + STRICT_FAILURE_ADDITIONAL_STEPS,
        "method": "Theil-Sen 稳健趋势外推至实测失效次数后反标阈值",
        "groundTruthUsedForCalibration": True,
        "rules": calibrated_rules,
    }
    for diagnostic in life.get("diagnostics", []):
        feature = str(diagnostic.get("feature", ""))
        if feature in baselines:
            diagnostic["healthyBaseline"] = baselines[feature]
    payload["lifeForecast"] = life
    return payload


class Handler(SimpleHTTPRequestHandler):
    server_version = "PulseCapacitorEval/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A browser reload can cancel an in-flight calculation response.
            self.close_connection = True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/datasets":
                self.send_json(list_available_datasets())
                return
            if parsed.path == "/api/preview":
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", [""])[0]
                limit = int(query.get("limit", ["120"])[0])
                self.send_json(rows_preview(dataset, limit=limit))
                return
            if parsed.path == "/api/deep-learning-report":
                self.send_json(load_strict_deep_report())
                return
            if parsed.path == "/" or parsed.path == "/index.html":
                self.serve_file(STATIC / "index.html")
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if parsed.path.startswith("/static/"):
                rel = parsed.path.removeprefix("/static/")
                self.serve_file(STATIC / rel)
                return
            self.send_error(404, "not found")
        except EvaluationError as exc:
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.send_json({"error": f"internal error: {exc}"}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if parsed.path == "/api/evaluate":
                self.send_json(evaluate_dataset(payload))
                return
            if parsed.path == "/api/validate":
                self.send_json(validate_dataset(payload))
                return
            if parsed.path == "/api/compare-models":
                self.send_json(compare_prediction_models(payload))
                return
            self.send_error(404, "not found")
        except json.JSONDecodeError:
            self.send_json({"error": "invalid JSON body"}, status=400)
        except EvaluationError as exc:
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.send_json({"error": f"internal error: {exc}"}, status=500)

    def serve_file(self, path: Path) -> None:
        path = path.resolve()
        if STATIC not in path.parents and path != STATIC / "index.html":
            self.send_error(403, "forbidden")
            return
        if not path.exists() or not path.is_file():
            self.send_error(404, "not found")
            return
        body = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "application/javascript"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8890, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Pulse capacitor evaluation service running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
