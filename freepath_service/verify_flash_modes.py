#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Verify free-path source modes used by the web and FLASH control path."""

from __future__ import annotations

import numpy as np

from free_path_web_app import ACTIVE_ARTIFACT, FreePathWebApp
from unified_free_path_core import STANDARD_TRAINING_DATA_PATH


def main() -> None:
    app = FreePathWebApp(ACTIVE_ARTIFACT)
    data = np.loadtxt(STANDARD_TRAINING_DATA_PATH, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    row = data[np.where(data[:, 0] == 13)[0][0]]
    base = {
        "Z": float(row[0]),
        "rod": float(row[1]),
        "tep": float(row[2]),
        "tgama": float(row[3]),
        "modelElements": [4, 13, 79],
    }

    expectations = {
        "model": "model",
        "hybrid": "table_truth",
        "truth": "table_truth",
    }
    for mode, expected_source in expectations.items():
        result = app.predict({**base, "predictionMode": mode})
        actual = result["selected_source"]
        if actual != expected_source:
            raise SystemExit(f"{mode}: expected {expected_source}, got {actual}")
        print(f"{mode}: {actual} ({result['mode']})")

    miss = {**base, "rod": base["rod"] + 0.00037, "predictionMode": "hybrid"}
    result = app.predict(miss)
    if result["selected_source"] != "model":
        raise SystemExit(f"hybrid miss: expected model, got {result['selected_source']}")
    print(f"hybrid miss: {result['selected_source']} ({result['mode']})")

    try:
        app.predict({**miss, "predictionMode": "truth"})
    except Exception as exc:
        print(f"truth miss: expected error ({exc})")
    else:
        raise SystemExit("truth miss: expected an error")


if __name__ == "__main__":
    main()
