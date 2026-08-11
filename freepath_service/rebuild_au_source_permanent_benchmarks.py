#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rebuild permanent Au benchmarks with both old-Au and Au2 sources."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from unified_free_path_core import (
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    OUTPUT_DIR,
    stable_mask_from_keys,
    standard_row_key,
)


ROOT = Path(__file__).resolve().parent
OLD_AU_STANDARD = OUTPUT_DIR / "unified_standard_training_data.bak_20260702_170959.txt"
AU2_RAW = ROOT / "data_Au2.txt"
SUMMARY_PATH = OUTPUT_DIR / "unified_permanent_benchmark_source_summary.json"


def backup(path: Path, stamp: str) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(f"{path.stem}.bak_source_au_{stamp}{path.suffix}"))


def load_existing_non_au(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)
    element = payload["element"].astype(str)
    mask = element != "Z_79"
    source = payload["source"].astype(str) if "source" in payload.files else element
    return {
        "x_model": payload["x_model"][mask],
        "y_log": payload["y_log"][mask],
        "element": element[mask],
        "source": source[mask],
        "keys": [standard_row_key(row[0], row[1:4]) for row in payload["x_model"][mask]],
    }


def load_old_au_rows() -> np.ndarray:
    data = np.loadtxt(OLD_AU_STANDARD, dtype=float)
    data = data[data[:, 0] == 79.0]
    if np.any(data[:, 4] <= 0.0):
        data = data[data[:, 4] > 0.0]
    return data


def load_au2_rows() -> np.ndarray:
    raw = np.loadtxt(AU2_RAW, dtype=float)
    raw = raw[np.isfinite(raw).all(axis=1)]
    raw = raw[raw[:, 3] > 0.0]
    return np.column_stack([np.full(raw.shape[0], 79.0), raw])


def source_masks(rows: np.ndarray, benchmark_ratio: float, extrapolation_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    keys = np.array([standard_row_key(row[0], row[1:4]) for row in rows], dtype=object)
    rod_values = np.unique(rows[:, 1])
    n_holdout = max(1, int(np.ceil(len(rod_values) * extrapolation_ratio)))
    cutoff_values = set(rod_values[:n_holdout].tolist()) | set(rod_values[-n_holdout:].tolist())
    extrap_mask = np.array([float(value) in cutoff_values for value in rows[:, 1]], dtype=bool)
    random_mask = (~extrap_mask) & stable_mask_from_keys(keys, benchmark_ratio)
    return random_mask, extrap_mask


def au_payload(rows: np.ndarray, mask: np.ndarray, source_name: str) -> dict:
    selected = rows[mask]
    return {
        "keys": [standard_row_key(row[0], row[1:4]) for row in selected],
        "x_model": selected[:, :4],
        "y_log": np.log10(selected[:, 4]),
        "element": np.array(["Z_79"] * selected.shape[0], dtype=object),
        "source": np.array([source_name] * selected.shape[0], dtype=object),
    }


def merge_payloads(parts: list[dict]) -> dict:
    return {
        "keys": [key for part in parts for key in part["keys"]],
        "x_model": np.vstack([part["x_model"] for part in parts]),
        "y_log": np.concatenate([part["y_log"] for part in parts]),
        "element": np.concatenate([part["element"] for part in parts]),
        "source": np.concatenate([part["source"] for part in parts]),
    }


def save_payload(keys_path: Path, data_path: Path, payload: dict) -> None:
    keys_path.write_text("\n".join(payload["keys"]) + "\n", encoding="utf-8")
    np.savez_compressed(
        data_path,
        x_model=payload["x_model"],
        y_log=payload["y_log"],
        element=payload["element"],
        source=payload["source"],
    )


def counts(payload: dict) -> dict:
    out = {
        "total": int(len(payload["keys"])),
        "by_element": {},
        "by_source": {},
    }
    for key, values in (("by_element", payload["element"]), ("by_source", payload["source"])):
        for value in sorted(set(values.tolist())):
            out[key][str(value)] = int(np.sum(values == value))
    return out


def main() -> None:
    benchmark_ratio = 0.10
    extrapolation_ratio = 0.10
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for path in [BENCHMARK_KEYS_PATH, BENCHMARK_DATA_PATH, EXTRAPOLATION_KEYS_PATH, EXTRAPOLATION_DATA_PATH]:
        backup(path, stamp)

    existing_random = load_existing_non_au(BENCHMARK_DATA_PATH)
    existing_extrap = load_existing_non_au(EXTRAPOLATION_DATA_PATH)

    old_rows = load_old_au_rows()
    au2_rows = load_au2_rows()
    old_random, old_extrap = source_masks(old_rows, benchmark_ratio, extrapolation_ratio)
    au2_random, au2_extrap = source_masks(au2_rows, benchmark_ratio, extrapolation_ratio)

    random_payload = merge_payloads(
        [
            existing_random,
            au_payload(old_rows, old_random, "Au_old_data.txt"),
            au_payload(au2_rows, au2_random, "Au2_data_Au2.txt"),
        ]
    )
    extrap_payload = merge_payloads(
        [
            existing_extrap,
            au_payload(old_rows, old_extrap, "Au_old_data.txt"),
            au_payload(au2_rows, au2_extrap, "Au2_data_Au2.txt"),
        ]
    )
    if len(set(random_payload["keys"])) != len(random_payload["keys"]):
        raise ValueError("random benchmark contains duplicate keys")
    if len(set(extrap_payload["keys"])) != len(extrap_payload["keys"]):
        raise ValueError("extrapolation benchmark contains duplicate keys")
    overlap = set(random_payload["keys"]) & set(extrap_payload["keys"])
    if overlap:
        raise ValueError(f"random/extrapolation benchmark overlap: {len(overlap)} keys")

    save_payload(BENCHMARK_KEYS_PATH, BENCHMARK_DATA_PATH, random_payload)
    save_payload(EXTRAPOLATION_KEYS_PATH, EXTRAPOLATION_DATA_PATH, extrap_payload)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": {
            "Be_Al": "kept from previous permanent benchmark files",
            "Au": "rebuilt per source with contiguous low/high rod extrapolation and stable random holdout",
            "benchmark_ratio": benchmark_ratio,
            "extrapolation_ratio_per_side_rod_axis": extrapolation_ratio,
        },
        "inputs": {
            "old_au_standard": str(OLD_AU_STANDARD),
            "au2_raw": str(AU2_RAW),
            "old_au_rows": int(old_rows.shape[0]),
            "au2_positive_rows": int(au2_rows.shape[0]),
        },
        "random_benchmark": counts(random_payload),
        "extrapolation_benchmark": counts(extrap_payload),
        "backup_suffix": f"bak_source_au_{stamp}",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
