#!/usr/bin/env python3
"""Generate README figures directly from the committed model reports."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "model-docs"

BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#16A34A"
ORANGE = "#EA580C"
SLATE = "#64748B"
RED = "#DC2626"
ACTIVE_FREEPATH_SHA256 = "741ffa62ec7f902fafacfc4c1f84984ebe13ada73024f3da6ea4ce182ab7b1af"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metric_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["element"]: {
                key: float(value) for key, value in row.items() if key != "element"
            }
            for row in csv.DictReader(handle)
        }


def label_vertical_bars(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def generate_freepath() -> None:
    model_dir = ROOT / "freepath_service" / "free_path_model_outputs"
    benchmark = load_metric_rows(model_dir / "unified_free_path_benchmark_metrics.csv")
    extrapolation = load_metric_rows(model_dir / "unified_free_path_extrapolation_metrics.csv")
    model_path = model_dir / "unified_free_path_model.npz"
    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_sha256 != ACTIVE_FREEPATH_SHA256:
        raise RuntimeError(f"unexpected free-path artifact SHA256: {actual_sha256}")
    expected_counts = {"benchmark": 30_190, "extrapolation": 74_975}
    actual_counts = {
        "benchmark": int(benchmark["overall"]["n"]),
        "extrapolation": int(extrapolation["overall"]["n"]),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"unexpected free-path evaluation counts: {actual_counts}")
    keys = ["overall", "Z_4", "Z_13", "Z_79"]
    labels = ["Overall", "Be (Z=4)", "Al (Z=13)", "Au (Z=79)"]
    x = np.arange(len(keys))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title, ylabel, fmt in [
        (axes[0], "smape_percent", "SMAPE by element", "SMAPE (%)", "{:.2f}"),
        (axes[1], "p90_factor_error", "90th-percentile factor error", "Factor", "{:.3f}x"),
    ]:
        bench_values = [benchmark[key][metric] for key in keys]
        extra_values = [extrapolation[key][metric] for key in keys]
        bars_a = ax.bar(x - width / 2, bench_values, width, label="Frozen random benchmark", color=BLUE)
        bars_b = ax.bar(x + width / 2, extra_values, width, label="Frozen low/high-rod benchmark", color=ORANGE)
        label_vertical_bars(ax, bars_a, fmt)
        label_vertical_bars(ax, bars_b, fmt)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.margins(y=0.16)

    fig.suptitle("Unified free-path active artifact: frozen evaluation snapshot", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Active release snapshot: 30,190 frozen random-benchmark points and 74,975 low/high-rod points",
        ha="center",
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT / "freepath-evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_zpinch() -> None:
    summary = load_json(ROOT / "zpinch" / "ai_training_outputs" / "velocity_first_optimized_summary.json")
    current = summary["final_test_metrics"]
    baseline = summary["baseline_existing_best_final_metrics"]
    improvement = summary["improvement_vs_existing_best_final"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    specs = [
        ("E_MAE", "Energy MAE", "MJ/cm", "{:.6f}"),
        ("v_MAE", "Inward velocity MAE", "cm/s", "{:,.0f}"),
    ]
    for ax, (key, title, ylabel, fmt) in zip(axes, specs):
        values = [baseline[key], current[key]]
        bars = ax.bar(
            ["Energy-first\nPoly2Ridge baseline", "Current log(E)\nPoly4 Ridge"],
            values,
            color=[SLATE, GREEN],
            width=0.62,
        )
        label_vertical_bars(ax, bars, fmt)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.20)
        ax.text(
            0.5,
            0.92,
            f"{improvement[key]:.2f}x lower",
            transform=ax.transAxes,
            ha="center",
            color=GREEN,
            fontweight="bold",
        )

    fig.suptitle("Z-pinch internal random holdout (seed 2026, n=411)", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Degree-4 polynomial Ridge fits log(E), then the physical relation returns negative inward velocity",
        ha="center",
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT / "zpinch-velocity-first-improvement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_zpinch_diagnostics() -> None:
    zpinch_dir = ROOT / "zpinch"
    sys.path.insert(0, str(zpinch_dir))
    from train_zpinch_surrogate import V_FORMULA_COEF, make_features  # noqa: PLC0415

    artifact = zpinch_dir / "ai_training_outputs" / "best_velocity_first_optimized_model_artifact.pkl"
    with artifact.open("rb") as handle:
        payload = pickle.load(handle)
    columns = ["I_MA", "tr_ns", "liner_r_cm", "foam_r_cm", "m_mg_per_cm"]
    base_rows = []
    energy = []
    velocity = []
    data_path = zpinch_dir / "source_data" / "zpinch_training_data.csv"
    with data_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            base_rows.append([float(row[name]) for name in columns] + [13.0])
            energy.append(float(row["E_MJ_per_cm"]))
            velocity.append(float(row["v_cm_per_s"]))
    x_base = np.asarray(base_rows, dtype=float)
    x, _ = make_features(x_base)
    energy = np.asarray(energy, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    test_idx = np.asarray(payload["test_idx"], dtype=int)
    pred_velocity = np.asarray(payload["model"].predict(x[test_idx]), dtype=float)
    pred_energy = pred_velocity**2 * x[test_idx, 4] / V_FORMULA_COEF

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [
        (energy[test_idx], pred_energy, "Energy: predicted vs reference", "MJ/cm"),
        (velocity[test_idx], pred_velocity, "Velocity: predicted vs reference", "cm/s"),
    ]
    for ax, (truth, prediction, title, units) in zip(axes, panels):
        ax.scatter(truth, prediction, s=18, alpha=0.55, color=BLUE, edgecolors="none")
        lower = min(float(truth.min()), float(prediction.min()))
        upper = max(float(truth.max()), float(prediction.max()))
        ax.plot([lower, upper], [lower, upper], linestyle="--", color=RED, linewidth=1.4, label="Ideal y=x")
        ax.set_xlabel(f"Reference ({units})")
        ax.set_ylabel(f"Prediction ({units})")
        ax.set_title(title)
        ax.legend(frameon=False)

    fig.suptitle("Current velocity-first model diagnostics (fixed test split, n=411)", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Internal random holdout on the 2,730-row zero-dimensional simulation table; not an external experiment",
        ha="center",
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT / "zpinch-velocity-first-diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_capacitor_horizon() -> None:
    report_path = ROOT / "pulse_capacitor_online_eval" / "data" / "models" / "strict_deep_report.json"
    report = load_json(report_path)
    validation = report["lifeForecast"]["horizonValidation"]
    results = validation["results"]
    horizons = [int(item["horizon"]) for item in results]
    accuracy = [float(item["accuracyPercent"]) for item in results]
    threshold = float(validation["minimumAccuracyPercent"])
    validated = int(validation["validatedHorizonSteps"])

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(horizons, accuracy, marker="o", linewidth=2.2, color=BLUE, label="Independent-test accuracy")
    ax.axhline(threshold, color=RED, linestyle="--", linewidth=1.6, label=f"Reliability threshold = {threshold:.0f}%")
    ax.axvline(validated, color=GREEN, linestyle=":", linewidth=1.6, label=f"Longest audited pass = {validated} steps")
    ax.scatter([validated], [accuracy[horizons.index(validated)]], color=GREEN, s=80, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(horizons, [str(value) for value in horizons], rotation=35)
    ax.set_xlabel("Forecast horizon (discharge steps, log scale)")
    ax.set_ylabel("Normalized accuracy (%)")
    ax.set_title("Pulse-capacitor horizon reliability audit")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(max(0.0, min(accuracy) - 8.0), 100.0)
    fig.text(
        0.5,
        0.01,
        "The UI can explore 5,000 steps; among discrete audited candidates, the farthest >=85% pass is 120 steps",
        ha="center",
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT / "capacitor-horizon-validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_capacitor() -> None:
    report = load_json(ROOT / "pulse_capacitor_online_eval" / "data" / "models" / "strict_deep_report.json")
    model_labels = {
        "gm11": "GM(1,1)",
        "linear_trend": "Linear trend",
        "holt_damped": "Damped Holt",
        "autoregression": "AR(1)",
        "persistence": "Persistence",
        "ridge": "Ridge",
        "quantized_hybrid": "Quantized hybrid",
        "gru": "GRU",
        "lstm": "LSTM",
        "tcn": "TCN",
        "transformer": "Transformer",
    }
    models = sorted(report["lookbackSearch"]["perModel"], key=lambda item: item["testNmaePercent"], reverse=True)
    lookbacks = report["lookbackSearch"]["candidates"]
    selected = int(report["lookbackSearch"]["bestLookback"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = [model_labels[item["modelId"]] for item in models]
    values = [float(item["testNmaePercent"]) for item in models]
    colors = [GREEN if item["modelId"] == "quantized_hybrid" else BLUE if item["modelType"] == "deep_learning" else SLATE for item in models]
    bars = axes[0].barh(labels, values, color=colors)
    axes[0].set_xlabel("Independent test NMAE (%) - lower is better")
    axes[0].set_title("All candidate models")
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[0].margins(x=0.15)

    x = [int(item["lookback"]) for item in lookbacks]
    validation = [float(item["validationNmaePercent"]) for item in lookbacks]
    test = [float(item["testNmaePercent"]) for item in lookbacks]
    axes[1].plot(x, validation, marker="o", linewidth=2, color=CYAN, label="Validation")
    axes[1].plot(x, test, marker="o", linewidth=2, color=ORANGE, label="Independent test")
    axes[1].axvline(selected, linestyle="--", linewidth=1.5, color=GREEN, label=f"Selected lookback={selected}")
    axes[1].scatter([selected], [validation[x.index(selected)]], s=75, color=GREEN, zorder=5)
    axes[1].set_xlabel("Lookback steps")
    axes[1].set_ylabel("Mean NMAE (%)")
    axes[1].set_title("Quantized-hybrid window audit")
    axes[1].legend(frameon=False)

    fig.suptitle("Pulse-capacitor chronological evaluation", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Window selection uses validation only (24/32/48 tie -> predefined middle 32); test curve is audit-only",
        ha="center",
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT / "capacitor-model-selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    generate_freepath()
    generate_zpinch()
    generate_zpinch_diagnostics()
    generate_capacitor()
    generate_capacitor_horizon()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
