#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict Rosseland free path from rod, tep, and tgama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_free_path_model import DEFAULT_OUTPUT, make_interpolator


def find_default_model() -> Path:
    candidates = sorted(
        DEFAULT_OUTPUT.glob("free_path_model_*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No model artifact found. Run `python train_free_path_model.py` first."
        )
    return candidates[0]


def load_artifact(path: Path) -> tuple[list[np.ndarray], np.ndarray, dict]:
    payload = np.load(path, allow_pickle=False)
    axes = [payload["axis_rod"], payload["axis_tep"], payload["axis_tgama"]]
    log_lnu_grid = payload["log_lnu_grid"]
    metadata = json.loads(str(payload["metadata_json"]))
    return axes, log_lnu_grid, metadata


def in_range(point: np.ndarray, axes: list[np.ndarray]) -> tuple[bool, list[str]]:
    messages = []
    for value, axis, name in zip(point, axes, ["rod", "tep", "tgama"]):
        if value < axis.min() or value > axis.max():
            messages.append(f"{name}={value:.12g} outside [{axis.min():.12g}, {axis.max():.12g}]")
    return not messages, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict free path lnu from rod, tep, tgama.")
    parser.add_argument("rod", type=float, help="density coordinate, or physical density with --log10-input")
    parser.add_argument("tep", type=float, help="electron temperature coordinate, or physical temperature with --log10-input")
    parser.add_argument("tgama", type=float, help="photon temperature coordinate, or physical temperature with --log10-input")
    parser.add_argument("--model", type=Path, default=None, help="path to free_path_model_*.npz")
    parser.add_argument(
        "--log10-input",
        action="store_true",
        help="apply log10 to rod/tep/tgama before prediction",
    )
    parser.add_argument(
        "--allow-extrapolate",
        action="store_true",
        help="allow extrapolation outside the training grid",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    axes, log_lnu_grid, metadata = load_artifact(model_path)
    method = metadata.get("method", "cubic")

    point = np.array([args.rod, args.tep, args.tgama], dtype=float)
    original_point = point.copy()
    if args.log10_input:
        if np.any(point <= 0):
            raise ValueError("--log10-input requires all three inputs to be positive")
        point = np.log10(point)

    ok, range_messages = in_range(point, axes)
    if not ok and not args.allow_extrapolate:
        detail = "; ".join(range_messages)
        raise ValueError(f"Input outside model range: {detail}. Use --allow-extrapolate if intentional.")

    interpolator = make_interpolator(
        axes,
        log_lnu_grid,
        method=method,
        allow_extrapolate=args.allow_extrapolate,
    )
    log_lnu = float(interpolator(point.reshape(1, 3))[0])
    lnu = float(10.0**log_lnu)

    result = {
        "model": str(model_path),
        "method": method,
        "input": {
            "rod": float(original_point[0]),
            "tep": float(original_point[1]),
            "tgama": float(original_point[2]),
            "log10_input": bool(args.log10_input),
        },
        "model_coordinates": {
            "rod": float(point[0]),
            "tep": float(point[1]),
            "tgama": float(point[2]),
        },
        "lnu": lnu,
        "log10_lnu": log_lnu,
        "range_warnings": range_messages,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("===== Free-path prediction =====")
    print(f"model: {model_path}")
    print(f"method: {method}")
    print(
        "input: "
        f"rod={original_point[0]:.12g}, tep={original_point[1]:.12g}, "
        f"tgama={original_point[2]:.12g}"
    )
    if args.log10_input:
        print(
            "model coordinates: "
            f"rod={point[0]:.12g}, tep={point[1]:.12g}, tgama={point[2]:.12g}"
        )
    if range_messages:
        print("warning: " + "; ".join(range_messages))
    print(f"lnu: {lnu:.12e}")
    print(f"log10(lnu): {log_lnu:.12f}")


if __name__ == "__main__":
    main()
