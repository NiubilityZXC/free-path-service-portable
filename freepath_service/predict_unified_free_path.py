#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict free path with the single unified model.

Fixed input format:
    Z rod tep tgama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unified_free_path_core import ACTIVE_UNIFIED_ARTIFACT, UnifiedModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Rosseland free path with one unified model.")
    parser.add_argument("--model", type=Path, default=ACTIVE_UNIFIED_ARTIFACT)
    parser.add_argument("Z", type=float)
    parser.add_argument("rod", type=float)
    parser.add_argument("tep", type=float)
    parser.add_argument("tgama", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model = UnifiedModel.load(args.model)
    result = model.predict(
        z_value=args.Z,
        xyz=np.array([args.rod, args.tep, args.tgama], dtype=float),
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("===== Unified free-path prediction =====")
    print(f"model: {args.model}")
    print(f"Z: {args.Z:g}")
    print(f"lnu: {result['lnu']:.12e}")
    print(f"log10(lnu): {result['log10_lnu']:.12f}")


if __name__ == "__main__":
    main()
