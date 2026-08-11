#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict free path using the two-dataset hybrid artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from train_two_dataset_free_path_model import DEFAULT_OUTPUT, apply_coordinate_transform


DEFAULT_ARTIFACT = DEFAULT_OUTPUT / "free_path_two_dataset_hybrid_model.npz"


def load_artifact(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata_json"]))
    datasets = {}
    for name in metadata["datasets"]:
        datasets[name] = {
            "axes": [
                payload[f"{name}_axis_rod"],
                payload[f"{name}_axis_tep"],
                payload[f"{name}_axis_tgama"],
            ],
            "log_lnu_grid": payload[f"{name}_log_lnu_grid"],
        }
    return metadata, datasets


def make_interpolator(
    axes: list[np.ndarray],
    log_grid: np.ndarray,
    method: str,
    allow_extrapolate: bool,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        axes,
        log_grid,
        method=method,
        bounds_error=not allow_extrapolate,
        fill_value=None if allow_extrapolate else np.nan,
    )


def inside_domain(point: np.ndarray, axes: list[np.ndarray]) -> tuple[bool, list[str]]:
    messages = []
    for value, axis, name in zip(point, axes, ["rod", "tep", "tgama"]):
        if value < axis.min() or value > axis.max():
            messages.append(f"{name}={value:.12g} outside [{axis.min():.12g}, {axis.max():.12g}]")
    return not messages, messages


def predict_one(
    point: np.ndarray,
    dataset_meta: dict,
    selected: dict,
    axes: list[np.ndarray],
    log_lnu_grid: np.ndarray,
    allow_extrapolate: bool,
) -> tuple[float, float, str, list[str]]:
    ok, warnings = inside_domain(point, axes)
    if ok:
        config = selected["interpolation"]
        mode = "interpolation"
        sigma = float(config.get("smoothing_sigma", 0.0) or 0.0)
        grid = gaussian_filter(log_lnu_grid, sigma=sigma, mode="nearest") if sigma > 0.0 else log_lnu_grid
        if config["method"] == "ensemble_linear_cubic":
            weight_cubic = float(config["ensemble_weight_cubic"])
            linear = make_interpolator(axes, grid, method="linear", allow_extrapolate=False)
            cubic = make_interpolator(axes, grid, method="cubic", allow_extrapolate=False)
            log_linear = float(linear(point.reshape(1, 3))[0])
            log_cubic = float(cubic(point.reshape(1, 3))[0])
            log_lnu = weight_cubic * log_cubic + (1.0 - weight_cubic) * log_linear
            lnu = float(10.0**log_lnu)
            return lnu, float(log_lnu), mode, warnings
        interp = make_interpolator(axes, grid, method=config["method"], allow_extrapolate=False)
    else:
        if not allow_extrapolate:
            raise ValueError(
                "Input outside model range: "
                + "; ".join(warnings)
                + ". Use --allow-extrapolate if this is intentional."
            )
        config = selected["extrapolation"]
        sigma = float(config["smoothing_sigma"])
        grid = gaussian_filter(log_lnu_grid, sigma=sigma, mode="nearest") if sigma > 0.0 else log_lnu_grid
        interp = make_interpolator(axes, grid, method=config["method"], allow_extrapolate=True)
        mode = "extrapolation"

    log_lnu = float(interp(point.reshape(1, 3))[0])
    if mode == "extrapolation" and config["clip_margin_log10"] != "":
        margin = float(config["clip_margin_log10"])
        log_lnu = float(np.clip(log_lnu, float(grid.min()) - margin, float(grid.max()) + margin))
    lnu = float(10.0**log_lnu)
    return lnu, log_lnu, mode, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Rosseland free path from a two-dataset artifact.")
    parser.add_argument("rod", type=float, nargs="?", help="first input coordinate")
    parser.add_argument("tep", type=float, nargs="?", help="second input coordinate")
    parser.add_argument("tgama", type=float, nargs="?", help="third input coordinate")
    parser.add_argument("--dataset", help="dataset name, e.g. data_Al or data")
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--model-coordinates",
        action="store_true",
        help="inputs are already in the stored model coordinate system; skip automatic transform",
    )
    parser.add_argument(
        "--log10-input",
        action="store_true",
        help="apply log10 to the three inputs before prediction, useful for data_Al physical inputs",
    )
    parser.add_argument("--allow-extrapolate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    metadata, datasets = load_artifact(args.model)

    if args.list_datasets:
        for name, info in metadata["datasets"].items():
            print(
                f"{name}: transform={info['coordinate_transform']}, "
                f"grid={info['axis_sizes']}, repaired={info['repaired_lnu_count']}"
            )
        return

    if not args.dataset:
        raise ValueError("Please pass --dataset. Use --list-datasets to see available names.")
    if args.dataset not in datasets:
        raise ValueError(f"Unknown dataset {args.dataset!r}. Available: {sorted(datasets)}")
    if args.rod is None or args.tep is None or args.tgama is None:
        raise ValueError("rod, tep, and tgama are required unless --list-datasets is used")

    dataset_meta = metadata["datasets"][args.dataset]
    selected = metadata["selected_models"][args.dataset]
    axes = datasets[args.dataset]["axes"]
    log_lnu_grid = datasets[args.dataset]["log_lnu_grid"]

    raw_point = np.array([args.rod, args.tep, args.tgama], dtype=float)
    if args.model_coordinates:
        point = raw_point.copy()
        applied_transform = "model_coordinates"
    elif args.log10_input:
        point = apply_coordinate_transform(raw_point, "log10")
        applied_transform = "manual_log10"
    else:
        transform = dataset_meta["coordinate_transform"]
        point = apply_coordinate_transform(raw_point, transform)
        applied_transform = transform

    lnu, log_lnu, mode, range_warnings = predict_one(
        point=point,
        dataset_meta=dataset_meta,
        selected=selected,
        axes=axes,
        log_lnu_grid=log_lnu_grid,
        allow_extrapolate=args.allow_extrapolate,
    )

    result = {
        "artifact": str(args.model),
        "dataset": args.dataset,
        "mode": mode,
        "input": {
            "rod": float(raw_point[0]),
            "tep": float(raw_point[1]),
            "tgama": float(raw_point[2]),
            "applied_transform": applied_transform,
        },
        "model_coordinates": {
            "rod": float(point[0]),
            "tep": float(point[1]),
            "tgama": float(point[2]),
        },
        "lnu": lnu,
        "log10_lnu": log_lnu,
        "range_warnings": range_warnings,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("===== Two-dataset free-path prediction =====")
    print(f"artifact: {args.model}")
    print(f"dataset: {args.dataset}")
    print(f"mode: {mode}")
    print(
        "input: "
        f"rod={raw_point[0]:.12g}, tep={raw_point[1]:.12g}, tgama={raw_point[2]:.12g}, "
        f"transform={applied_transform}"
    )
    print(
        "model coordinates: "
        f"rod={point[0]:.12g}, tep={point[1]:.12g}, tgama={point[2]:.12g}"
    )
    if range_warnings:
        print("warning: " + "; ".join(range_warnings))
    print(f"lnu: {lnu:.12e}")
    print(f"log10(lnu): {log_lnu:.12f}")


if __name__ == "__main__":
    main()
