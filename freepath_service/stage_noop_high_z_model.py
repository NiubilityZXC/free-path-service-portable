#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage a disabled high-Z wrapper for a single unified model release."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from high_z_free_path_model import HIGH_Z_METADATA_PATH, HIGH_Z_MODEL_PATH
from unified_free_path_core import OUTPUT_DIR, STAGED_UNIFIED_ARTIFACT


STAGED_HIGH_Z_MODEL = OUTPUT_DIR / "staged_high_z_au_catboost_model.cbm"
STAGED_HIGH_Z_METRICS = OUTPUT_DIR / "staged_high_z_au_catboost_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=STAGED_UNIFIED_ARTIFACT)
    parser.add_argument("--model-path", type=Path, default=STAGED_HIGH_Z_MODEL)
    parser.add_argument("--metrics-path", type=Path, default=STAGED_HIGH_Z_METRICS)
    args = parser.parse_args()

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    previous = {}
    if HIGH_Z_METADATA_PATH.exists():
        try:
            previous = json.loads(HIGH_Z_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    metadata = {
        "artifact_type": "disabled_high_z_wrapper",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": None,
        "base_model_path": str(args.base_model),
        "route_enabled": False,
        "route_policy": "disabled_single_unified_model",
        "z_values": previous.get("z_values", [79.0]),
        "train_ranges": previous.get("train_ranges", {}),
        "standard_prediction_format": "Z rod tep tgama",
        "standard_training_format": "Z rod tep tgama lnu",
        "notes": [
            "High-Z route intentionally disabled.",
            "All positive old-Au and Au2 rows outside permanent tests are handled by the single staged unified model.",
        ],
    }
    args.metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"staged disabled high-Z wrapper: {args.metrics_path}")


if __name__ == "__main__":
    main()
