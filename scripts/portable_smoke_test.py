#!/usr/bin/env python3
"""Offline integrity and inference checks for the portable bundle."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
checks: list[dict] = []


def record(name: str, ok: bool, detail: object = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def first_numeric_row(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                values = np.fromstring(stripped, sep=" ")
                if values.size:
                    return values
    raise RuntimeError(f"no numeric row in {path}")


required = [
    "freepath_service/free_path_web_app.py",
    "freepath_service/free_path_model_outputs/unified_free_path_model.npz",
    "freepath_service/free_path_model_outputs/unified_standard_training_data.txt",
    "pulse_capacitor_online_eval/app.py",
    "pulse_capacitor_online_eval/data/models/strict_deep_report.json",
    "zpinch/train_zpinch_surrogate.py",
    "zpinch/ai_training_outputs/best_velocity_first_optimized_model_artifact.pkl",
    "zpinch/source_data/zpinch_training_data.csv",
    "Cuba-4.2.2/libcuba.a",
    "Rosseland_opa_low_Z.zip",
]
missing = [rel for rel in required if not (ROOT / rel).exists()]
record("required_files", not missing, {"missing": missing})

flash_required = [
    "FLASH4.8/source/Simulation/Simulation_init.F90",
    "FLASH4.8/improved_MRT/flash.par",
    "FLASH4.8/improved_MRT/flash4",
]
flash_root = ROOT / "FLASH4.8"
if flash_root.is_dir():
    flash_missing = [rel for rel in flash_required if not (ROOT / rel).exists()]
    record(
        "optional_flash",
        not flash_missing,
        {"present": True, "missing": flash_missing},
    )
else:
    record(
        "optional_flash",
        True,
        {"present": False, "detail": "not distributed through GitHub; obtain separately from the Flash Center"},
    )

freepath_dir = ROOT / "freepath_service"
sys.path.insert(0, str(freepath_dir))
from high_z_free_path_model import HighZModel  # noqa: E402
from unified_free_path_core import UnifiedModel  # noqa: E402

artifact = freepath_dir / "free_path_model_outputs" / "unified_free_path_model.npz"
model = UnifiedModel.load(artifact)
row = first_numeric_row(freepath_dir / "free_path_model_outputs" / "unified_standard_training_data.txt")
result = model.predict(float(row[0]), row[1:4])
record("unified_model", math.isfinite(result["lnu"]) and result["lnu"] > 0, result)
record("disabled_high_z", HighZModel.load_if_available() is None, "route stays on unified model")

zpinch_dir = ROOT / "zpinch"
sys.path.insert(0, str(zpinch_dir))
from train_zpinch_surrogate import V_FORMULA_COEF, load_model_artifact, make_features  # noqa: E402

payload = load_model_artifact(str(zpinch_dir / "ai_training_outputs" / "best_velocity_first_optimized_model_artifact.pkl"))
x_base = np.array([[50.0, 300.0, 5.0, 0.6, 30.0, 13.0]], dtype=float)
x_feat, feature_names = make_features(x_base)
velocity = float(np.asarray(payload["model"].predict(x_feat)).reshape(-1)[0])
energy = float(velocity * velocity * 30.0 / V_FORMULA_COEF)
record("zpinch_feature_count", len(feature_names) == 14, len(feature_names))
record(
    "zpinch_model",
    np.isclose(energy, 5.030381733204523, rtol=1e-9, atol=1e-12)
    and np.isclose(velocity, -57910170.28239814, rtol=1e-9, atol=1e-6),
    {"E_MJ_per_cm": energy, "v_cm_per_s": velocity},
)

pulse_report = json.loads(
    (ROOT / "pulse_capacitor_online_eval" / "data" / "models" / "strict_deep_report.json").read_text(encoding="utf-8")
)
record("pulse_report", isinstance(pulse_report, dict) and bool(pulse_report), {"keys": sorted(pulse_report)[:10]})

print(json.dumps({"ok": all(item["ok"] for item in checks), "checks": checks}, ensure_ascii=False, indent=2))
