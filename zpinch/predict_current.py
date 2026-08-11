#!/usr/bin/env python3
"""Standalone inference for the current velocity-first Z-pinch artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_zpinch_surrogate import V_FORMULA_COEF, calc_v_from_E_m, load_model_artifact, make_features


ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = ROOT / "ai_training_outputs" / "best_velocity_first_optimized_model_artifact.pkl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--I", type=float, default=50.0, help="drive current, MA")
    parser.add_argument("--tr", type=float, default=300.0, help="rise time, ns")
    parser.add_argument("--liner-r", type=float, default=5.0, help="liner radius, cm")
    parser.add_argument("--foam-r", type=float, default=0.6, help="foam radius, cm")
    parser.add_argument("--m", type=float, default=30.0, help="mass, mg/cm")
    parser.add_argument("--Z", type=float, default=13.0, help="atomic number")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()

    values = np.array([[args.I, args.tr, args.liner_r, args.foam_r, args.m, args.Z]], dtype=float)
    if not np.all(np.isfinite(values)) or args.m <= 0:
        raise SystemExit("all inputs must be finite and m must be positive")

    model_payload = load_model_artifact(str(args.artifact.resolve()))
    features, feature_names = make_features(values)
    raw = float(np.asarray(model_payload["model"].predict(features)).reshape(-1)[0])
    prediction_target = model_payload.get("prediction_target", "E_MJ_per_cm")
    if prediction_target == "v_cm_per_s":
        velocity = raw
        energy = float((velocity * velocity) * args.m / V_FORMULA_COEF)
        pipeline = "velocity_first"
    else:
        energy = max(raw, 1e-12)
        velocity = float(calc_v_from_E_m(np.array([energy]), np.array([args.m]))[0])
        pipeline = "energy_first"

    print(json.dumps({
        "artifact": str(args.artifact.resolve()),
        "model_name": model_payload.get("model_name"),
        "prediction_target": prediction_target,
        "pipeline": pipeline,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "inputs": {"I_MA": args.I, "tr_ns": args.tr, "liner_r_cm": args.liner_r, "foam_r_cm": args.foam_r, "m_mg_per_cm": args.m, "Z": args.Z},
        "E_MJ_per_cm": energy,
        "v_cm_per_s": velocity,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
