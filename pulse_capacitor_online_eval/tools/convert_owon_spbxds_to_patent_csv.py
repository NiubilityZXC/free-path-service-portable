#!/usr/bin/env python3
"""Convert OWON SPBXDS waveform captures into patent-style feature CSV.

The patent method does not use the full waveform directly.  Each pulse waveform
is first reduced to an operation-parameter sequence such as maximum voltage,
reverse peak coefficient, first zero-crossing time, and discharge period.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import struct
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


NAME_TIME_RE = re.compile(
    r"(\d{2})-(\d{2})-(\d{2}) (\d{2})_(\d{2})_(\d{2})_(\d+)\.bin$"
)


def parse_capture_time(name: str) -> datetime:
    match = NAME_TIME_RE.search(Path(name).name)
    if not match:
        raise ValueError(f"cannot parse capture time from {name}")
    yy, month, day, hour, minute, second, millisecond = match.groups()
    millis = int(millisecond.ljust(3, "0")[:3])
    return datetime(
        2000 + int(yy),
        int(month),
        int(day),
        int(hour),
        int(minute),
        int(second),
        millis * 1000,
    )


def sample_rate_hz(header: dict[str, Any]) -> float:
    raw = str(header.get("SAMPLE", {}).get("SAMPLERATE", ""))
    match = re.search(r"([0-9.]+)\s*([GMK]?S/s)", raw, flags=re.IGNORECASE)
    if not match:
        return 500_000_000.0
    value = float(match.group(1))
    unit = match.group(2).upper()
    factor = {"GS/S": 1_000_000_000.0, "MS/S": 1_000_000.0, "KS/S": 1_000.0, "S/S": 1.0}
    return value * factor.get(unit, 1.0)


def parse_spbxds(data: bytes) -> tuple[dict[str, Any], list[int]]:
    if data[:6] != b"SPBXDS":
        raise ValueError("not an OWON SPBXDS capture")
    header_len = struct.unpack("<I", data[6:10])[0]
    header = json.loads(data[10 : 10 + header_len])
    pos = 10 + header_len
    data_len = struct.unpack("<I", data[pos : pos + 4])[0]
    raw = data[pos + 4 : pos + 4 + data_len]
    if len(raw) % 2:
        raw = raw[:-1]
    samples = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
    return header, samples


def first_threshold_index(values: list[float], threshold: float, fallback: int) -> int:
    for idx, value in enumerate(values):
        if value >= threshold:
            return idx
    return fallback


def first_zero_after_peak(values: list[float], peak_index: int) -> int:
    for idx in range(peak_index + 1, len(values)):
        if values[idx] <= 0:
            return idx
    return len(values) - 1


def waveform_features(
    header: dict[str, Any],
    samples: list[int],
    capture_time: datetime,
    first_time: datetime,
    previous_time: datetime | None,
) -> dict[str, str | float]:
    if len(samples) < 4:
        raise ValueError("waveform has too few samples")

    baseline_count = max(20, len(samples) // 10)
    baseline = statistics.median(samples[:baseline_count])
    signal = [float(value - baseline) for value in samples]
    peak_index = max(range(len(signal)), key=signal.__getitem__)
    voltage_max = max(signal)
    after_peak = signal[peak_index:]
    voltage_min = min(after_peak)
    voltage_min_abs = abs(voltage_min)
    reverse_peak_coeff = voltage_min_abs / voltage_max if voltage_max > 0 else 0.0

    threshold = max(1.0, 0.1 * voltage_max)
    pulse_start_index = first_threshold_index(signal, threshold, peak_index)
    zero_index = first_zero_after_peak(signal, peak_index)
    sample_period_us = 1_000_000.0 / sample_rate_hz(header)
    first_zero_time_us = max(sample_period_us, (zero_index - pulse_start_index) * sample_period_us)

    if previous_time is None:
        discharge_period_sec = 0.0
    else:
        discharge_period_sec = (capture_time - previous_time).total_seconds()

    # GM(1,1) requires positive sequences.  For unipolar captures with no visible
    # reverse peak, keep the parameter modelable by using a tiny positive floor.
    return {
        "Time": f"{(capture_time - first_time).total_seconds():.3f}",
        "VoltageMaxRaw": round(max(1.0, voltage_max), 6),
        "VoltageMinAbsRaw": round(max(1.0, voltage_min_abs), 6),
        "VoltageReversePeakCoefficient": f"{max(1e-6, reverse_peak_coeff):.9f}",
        "VoltageFirstZeroTimeUs": f"{first_zero_time_us:.6f}",
        "DischargePeriodSec": f"{max(1e-6, discharge_period_sec):.6f}",
    }


def convert_zip(zip_path: Path, output_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        captures = sorted(
            [info for info in archive.infolist() if info.filename.lower().endswith(".bin")],
            key=lambda info: parse_capture_time(info.filename),
        )
        if not captures:
            raise ValueError("no .bin captures found")

        first_time = parse_capture_time(captures[0].filename)
        previous_time = None
        rows = []
        for info in captures:
            capture_time = parse_capture_time(info.filename)
            header, samples = parse_spbxds(archive.read(info.filename))
            rows.append(waveform_features(header, samples, capture_time, first_time, previous_time))
            previous_time = capture_time

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Time",
        "VoltageMaxRaw",
        "VoltageFirstZeroTimeUs",
        "VoltageReversePeakCoefficient",
        "VoltageMinAbsRaw",
        "DischargePeriodSec",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    rows = convert_zip(args.zip_path, args.output_path)
    print(f"wrote {rows} rows to {args.output_path}")


if __name__ == "__main__":
    main()
