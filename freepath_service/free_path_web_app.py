#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small intranet web app for Rosseland free-path prediction."""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from high_z_free_path_model import HIGH_Z_METADATA_PATH, HIGH_Z_MODEL_PATH, HighZModel
from unified_free_path_core import (
    ACTIVE_UNIFIED_ARTIFACT,
    ACTIVE_UNIFIED_BENCHMARK_CSV,
    ACTIVE_UNIFIED_EXTRAPOLATION_CSV,
    ACTIVE_UNIFIED_METRICS,
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_KEYS_PATH,
    STANDARD_INPUT_COLUMNS,
    STANDARD_TRAINING_COLUMNS,
    STANDARD_TRAINING_DATA_PATH,
    STAGED_UNIFIED_ARTIFACT,
    STAGED_UNIFIED_METRICS,
    UnifiedModel,
    apply_coordinate_transform,
    infer_coordinate_transform,
    metric_dict,
    standard_row_key,
)

from build_single_point_simulators import ELEMENTS as SIMULATOR_ELEMENTS
from build_single_point_simulators import build_one, discover_data_elements, is_build_current, simulator_source_label


ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = Path(os.environ.get("FREE_PATH_BUNDLE_ROOT", ROOT.parent)).expanduser().resolve()
OUTPUT_DIR = ROOT / "free_path_model_outputs"
ACTIVE_ARTIFACT = ACTIVE_UNIFIED_ARTIFACT
ACTIVE_METRICS = ACTIVE_UNIFIED_METRICS
ACTIVE_BENCHMARK = ACTIVE_UNIFIED_BENCHMARK_CSV
ACTIVE_EXTRAPOLATION_BENCHMARK = ACTIVE_UNIFIED_EXTRAPOLATION_CSV
STAGED_ARTIFACT = STAGED_UNIFIED_ARTIFACT
STAGED_METRICS = STAGED_UNIFIED_METRICS
STAGED_BENCHMARK = OUTPUT_DIR / "staged_unified_free_path_benchmark_metrics.csv"
STAGED_EXTRAPOLATION_BENCHMARK = OUTPUT_DIR / "staged_unified_free_path_extrapolation_metrics.csv"
STAGED_HIGH_Z_MODEL = OUTPUT_DIR / "staged_high_z_au_catboost_model.cbm"
STAGED_HIGH_Z_METRICS = OUTPUT_DIR / "staged_high_z_au_catboost_metrics.json"
SIMULATOR_DIR = OUTPUT_DIR / "simulators"
SIMULATOR_TIMEOUT_SECONDS = 180.0
CAPACITOR_SERVICE_PORT = 8890
CAPACITOR_APP_DIR = Path(
    os.environ.get("CAPACITOR_APP_DIR", BUNDLE_ROOT / "pulse_capacitor_online_eval")
).expanduser().resolve()
ZPINCH_ROOT = Path(os.environ.get("ZPINCH_ROOT", BUNDLE_ROOT / "zpinch")).expanduser().resolve()
ZPINCH_PYTHON = Path(os.environ.get("FREE_PATH_PYTHON", sys.executable)).expanduser().resolve()
ZPINCH_ARTIFACT = ZPINCH_ROOT / "ai_training_outputs" / "best_velocity_first_optimized_model_artifact.pkl"
ZPINCH_SUMMARY = ZPINCH_ROOT / "ai_training_outputs" / "velocity_first_optimized_summary.json"
ZPINCH_LEGACY_SUMMARY = ZPINCH_ROOT / "ai_training_outputs" / "summary_metrics.json"
FLASH_ROOT = Path(os.environ.get("FLASH_ROOT", BUNDLE_ROOT / "FLASH4.8")).expanduser().resolve()
FLASH_RUN_DIR_NAMES = ("improved_MRT", "improved_MRT_AMR", "improved_MRT_long")
FLASH_WEB_LOG = "web_flash_run.log"
FLASH_WEB_PID = ".web_flash.pid"
FLASH_TEXT_SUFFIXES = {".log", ".out", ".err", ".txt", ".par", ".dat", ".csv", ".ini"}
FLASH_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
FLASH_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
FLASH_PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FLASH_PROJECT_LIGHT_IGNORE = shutil.ignore_patterns(
    "*.o",
    "*.mod",
    "*.log",
    "*.out",
    "*.err",
    "*.h5",
    "*plt*",
    "*chk*",
    FLASH_WEB_LOG,
    FLASH_WEB_PID,
    ".success",
)
FLASH_PARAM_SPECS = [
    {"key": "useRadTrans", "label": "启用辐射输运", "type": "bool"},
    {"key": "useOpacity", "label": "启用 opacity", "type": "bool"},
    {"key": "opacity_useFreePathModel", "label": "启用网页自由程模型", "type": "bool"},
    {"key": "opacity_freePathTimeout", "label": "模型 helper 超时秒数", "type": "number"},
    {"key": "rt_mgdNumGroups", "label": "MGD 能群数", "type": "number"},
    {"key": "op_fillTrans", "label": "fill transport opacity", "type": "string"},
    {"key": "op_lineTrans", "label": "line transport opacity", "type": "string"},
    {"key": "op_vacuTrans", "label": "vacu transport opacity", "type": "string"},
]

DATA_SOURCE_AUDIT = [
    {
        "name": "data_Al.txt",
        "Z": 13,
        "role": "原始低 Z 程序计算结果表",
        "path": str(ROOT / "data_Al.txt"),
        "coordinate_format": "log10(rod), log10(tep), log10(tgama)，表中只保留 3 位小数",
        "target_format": "lnu 以科学计数法保存，有限有效数字",
        "training_status": "已进入当前标准训练数据；部分永久测试样本会自动跳过训练",
        "note": "网页真实程序调用会把 3 位小数网格点还原成原程序完整 log 网格坐标。",
    },
    {
        "name": "data_Be.txt",
        "Z": 4,
        "role": "原始低 Z 程序计算结果表",
        "path": str(ROOT / "data_Be.txt"),
        "coordinate_format": "log10(rod), log10(tep), log10(tgama)，表中只保留 3 位小数",
        "target_format": "lnu 以科学计数法保存，有限有效数字",
        "training_status": "已进入当前暂存/网页运行模型；部分永久测试样本会自动跳过训练",
        "note": "网页真实程序调用会把 3 位小数网格点还原成原程序完整 log 网格坐标。",
    },
    {
        "name": "data.txt",
        "Z": 79,
        "role": "历史 Au 原始程序计算结果表",
        "path": str(ROOT / "data.txt"),
        "coordinate_format": "物理坐标 rod, tep, tgama，保留较完整小数",
        "target_format": "lnu 小数保存，和程序现场计算通常只差输出舍入/积分微扰",
        "training_status": "已合并进当前标准训练数据；永久随机/外推测试样本自动跳过训练",
        "note": "原始前三列为物理坐标，整理时统一转换为 log10 坐标；除永久测试 key 外，其余旧 Au 行参与训练。",
    },
    {
        "name": "data_Au2.txt",
        "Z": 79,
        "role": "新上传 Au 计算结果表",
        "path": str(ROOT / "data_Au2.txt"),
        "coordinate_format": "log10(rod), log10(tep), log10(tgama)，4 列旧格式不含 Z",
        "target_format": "lnu；导入时自动检查并剔除 lnu<=0 的无效自由程行",
        "training_status": "已合并进当前标准训练数据；永久随机/外推测试样本自动跳过训练",
        "note": "本次导入 125000 行，剔除 7397 行非正自由程，117603 行进入统一标准训练文件；后续新 Au 数据继续按同一标准格式合并。",
    },
    {
        "name": "unified_standard_training_data.txt",
        "Z": "mixed",
        "role": "标准化训练数据，不是原始程序输出",
        "path": str(STANDARD_TRAINING_DATA_PATH),
        "coordinate_format": "统一五列：Z log10(rod) log10(tep) log10(tgama) lnu",
        "target_format": "lnu",
        "training_status": "网页继续训练默认读取此文件",
        "note": "它由 Be、Al、旧 Au、Au2 原始表整理/合并得到；上传新数据时前三个参数必须已经是 log10 后的模型坐标。训练脚本会自动跳过永久随机/外推测试 key。",
    },
]


def _log_grid_axis(low: float, high: float, count: int) -> np.ndarray:
    return np.array(
        [
            (np.log10(high) - np.log10(low)) / float(count) * idx + np.log10(low)
            for idx in range(1, count + 1)
        ],
        dtype=float,
    )


LOW_Z_SIMULATOR_GRID = {
    z: {
        "rod": _log_grid_axis(1.0e-2, 3.0e4, 50),
        "tep": _log_grid_axis(50.0, 2.0e5, 50),
        "tgama": _log_grid_axis(50.0, 3.0e4, 50),
    }
    for z in (4, 13)
}
LOW_Z_GRID_DISPLAY_DECIMALS = 3
LOW_Z_GRID_SNAP_TOL = 5.0e-7

SIMULATION_THRESHOLDS = {
    "overall": {
        "smape_percent": 5.0,
        "log10_mae": 0.025,
        "p90_factor_error": 1.20,
        "p99_factor_error": 1.80,
    },
    "element": {
        "smape_percent": 8.0,
        "log10_mae": 0.040,
        "p90_factor_error": 1.25,
        "p99_factor_error": 2.00,
    },
}

THRESHOLD_LABELS = {
    "smape_percent": "SMAPE",
    "log10_mae": "log10 MAE",
    "p90_factor_error": "P90 倍数误差",
    "p99_factor_error": "P99 倍数误差",
}

BATCH_PREDICTION_MAX_ROWS = 20000

PREDICTION_MODE_LABELS = {
    "model": "只用模型",
    "hybrid": "混合：真值优先，模型兜底",
    "truth": "只用已计算真值",
}

SELECTED_SOURCE_LABELS = {
    "model": "模型预测",
    "table_truth": "标准数据表真值",
    "simulator_truth": "现场真实程序",
    "error": "不可用",
}


def threshold_status(metrics: dict, profile: str = "element") -> dict:
    limits = SIMULATION_THRESHOLDS[profile]
    checks = []
    for key, limit in limits.items():
        value = float(metrics.get(key, float("inf")))
        checks.append(
            {
                "metric": key,
                "label": THRESHOLD_LABELS[key],
                "value": value,
                "limit": limit,
                "passed": value <= limit,
            }
        )
    return {
        "profile": profile,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ai平台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #16202a;
      --muted: #607080;
      --line: #d7e0ea;
      --accent: #126b73;
      --accent-strong: #0a4f55;
      --warn: #a05a00;
      --bad: #b42318;
      --good: #1f7a4d;
      --shadow: 0 12px 28px rgba(20, 38, 55, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
    }
    .topbar {
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .app-tabs {
      display: flex;
      gap: 8px;
      padding: 0 0 12px;
      overflow-x: auto;
    }
    .tab-btn {
      min-height: 36px;
      border-color: #c8d3df;
      background: #ffffff;
      color: #2d3c49;
      white-space: nowrap;
    }
    .tab-btn.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
    }
    .app-view { display: none; }
    .app-view.active { display: block; }
    .embedded-frame {
      width: 100%;
      min-height: 820px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      box-shadow: var(--shadow);
    }
    .note {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .note strong { color: #273744; }
    .mini-note {
      margin-top: 8px;
      padding: 9px 10px;
      border: 1px solid #cbd7e2;
      border-radius: 6px;
      background: #fbfcfe;
      color: #4d6070;
      font-size: 13px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .status {
      min-height: 28px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      background: #fbfcfe;
      white-space: normal;
      max-width: min(720px, 100%);
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--good);
    }
    main {
      padding: 24px 0 34px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(340px, 430px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel-head {
      min-height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .panel-title {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .panel-body { padding: 16px; }
    .inline-section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }
    .inline-section-head h3 {
      margin: 0;
      font-size: 15px;
      font-weight: 730;
      letter-spacing: 0;
    }
    label {
      display: block;
      margin: 0 0 7px;
      font-weight: 650;
      color: #24313e;
    }
    .field { margin-bottom: 15px; }
    select,
    input[type="text"],
    input[type="number"] {
      width: 100%;
      height: 42px;
      border: 1px solid #c8d3df;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      padding: 0 11px;
      font: inherit;
      outline: none;
    }
    select:focus,
    input[type="text"]:focus,
    input[type="number"]:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(18, 107, 115, 0.14);
    }
    .inputs {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .segmented {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
      gap: 4px;
      padding: 4px;
      border: 1px solid #c8d3df;
      border-radius: 7px;
      background: #eef3f7;
    }
    .segmented input { position: absolute; opacity: 0; pointer-events: none; }
    .segmented span {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 32px;
      padding: 0 8px;
      border-radius: 5px;
      color: #4e5f6f;
      font-weight: 650;
      line-height: 1.25;
      text-align: center;
      white-space: normal;
      cursor: pointer;
    }
    .segmented input:checked + span {
      background: #ffffff;
      color: var(--accent-strong);
      box-shadow: 0 1px 3px rgba(22, 32, 42, 0.12);
    }
    .checkbox-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .check-chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 36px;
      margin: 0;
      padding: 0 10px;
      border: 1px solid #c8d3df;
      border-radius: 6px;
      background: #ffffff;
      color: #334554;
      font-size: 13px;
      font-weight: 680;
      cursor: pointer;
    }
    .check-chip input {
      width: 15px;
      height: 15px;
      accent-color: var(--accent);
    }
    .wide-textarea {
      width: 100%;
      border: 1px solid #c8d3df;
      border-radius: 6px;
      padding: 10px;
      color: var(--ink);
      font: inherit;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      resize: vertical;
      min-height: 156px;
      outline: none;
    }
    .wide-textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(18, 107, 115, 0.14);
    }
    .flash-control-grid {
      display: grid;
      grid-template-columns: minmax(280px, 0.82fr) minmax(360px, 1.18fr);
      gap: 16px;
      align-items: start;
    }
    .flash-usage-list {
      margin: 8px 0 0;
      padding-left: 20px;
      color: #4d6070;
      font-size: 13px;
    }
    .flash-usage-list li { margin: 6px 0; }
    .flash-console-grid {
      display: grid;
      grid-template-columns: minmax(360px, 0.95fr) minmax(420px, 1.05fr);
      gap: 16px;
      align-items: start;
    }
    .flash-param-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .flash-project-grid,
    .flash-param-toolbar,
    .flash-new-param-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-items: end;
    }
    .flash-param-toolbar {
      grid-template-columns: minmax(180px, 1fr) minmax(140px, 220px) auto;
      margin-bottom: 12px;
    }
    .flash-new-param-row {
      grid-template-columns: minmax(180px, 0.8fr) minmax(220px, 1.2fr) auto;
      margin-top: 12px;
    }
    .flash-param-table {
      min-width: 980px;
    }
    .flash-param-table input[type="text"],
    .flash-param-table input[type="number"] {
      height: 34px;
      min-width: 180px;
      font-size: 13px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .flash-param-table .toggle {
      min-height: 34px;
    }
    .flash-param-table tr.changed td {
      background: #fff9ec;
    }
    .flash-param-name {
      font-weight: 720;
      color: #24313e;
    }
    .flash-param-raw,
    .flash-param-comment {
      color: #657789;
      max-width: 260px;
      overflow-wrap: anywhere;
    }
    .flash-command-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }
    .command-output {
      margin: 12px 0 0;
      min-height: 220px;
      max-height: 520px;
      overflow: auto;
      border: 1px solid #c8d3df;
      border-radius: 6px;
      background: #111b24;
      color: #edf7f3;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .flash-plot-controls {
      display: grid;
      grid-template-columns: minmax(220px, 1.1fr) minmax(100px, 0.42fr) minmax(140px, 0.75fr) minmax(110px, 0.42fr) minmax(120px, 0.42fr) auto;
      gap: 10px;
      align-items: end;
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }
    .flash-plot-view {
      margin-top: 12px;
      min-height: 240px;
      border: 1px solid #c8d3df;
      border-radius: 6px;
      background: #f8fafc;
      overflow: auto;
      padding: 10px;
    }
    .flash-plot-view img {
      display: block;
      width: 100%;
      max-width: 1100px;
      height: auto;
      margin: 0 auto;
    }
    .flash-plot-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .file-action {
      min-height: 30px;
      padding: 0 9px;
      font-size: 12px;
    }
    .config-preview {
      margin: 12px 0 0;
      max-height: 300px;
      overflow: auto;
      border: 1px solid #c8d3df;
      border-radius: 6px;
      background: #0f1f2c;
      color: #eef7f4;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .switch-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 42px;
      padding: 0 2px;
    }
    .switch-row label {
      margin: 0;
      font-weight: 650;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      cursor: pointer;
    }
    .toggle input { position: absolute; opacity: 0; pointer-events: none; }
    .track {
      width: 46px;
      height: 26px;
      border-radius: 999px;
      background: #c7d2dc;
      position: relative;
      transition: background 0.15s ease;
    }
    .track::after {
      content: "";
      position: absolute;
      width: 20px;
      height: 20px;
      left: 3px;
      top: 3px;
      border-radius: 999px;
      background: #ffffff;
      box-shadow: 0 1px 3px rgba(20, 30, 40, 0.22);
      transition: transform 0.15s ease;
    }
    .toggle input:checked + .track { background: var(--accent); }
    .toggle input:checked + .track::after { transform: translateX(20px); }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 18px;
    }
    button {
      min-height: 42px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 0 15px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .primary {
      flex: 1;
      background: var(--accent);
      color: #ffffff;
    }
    .primary:hover { background: var(--accent-strong); }
    .secondary {
      background: #ffffff;
      color: #2d3c49;
      border-color: #c8d3df;
    }
    button:disabled {
      opacity: 0.58;
      cursor: wait;
    }
    .result {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 13px;
      min-height: 92px;
      background: #fbfcfe;
    }
    .metric .name {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }
    .metric .value {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 20px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .help {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      margin-left: 5px;
      flex: 0 0 auto;
      border: 1px solid #a9b8c6;
      border-radius: 999px;
      color: #4e6172;
      background: #ffffff;
      font-size: 0;
      font-weight: 800;
      line-height: 1;
      cursor: help;
      user-select: none;
      vertical-align: middle;
    }
    .help::before {
      content: "?";
      font-size: 11px;
      line-height: 1;
    }
    .tooltip-bubble {
      position: fixed;
      z-index: 10000;
      left: 0;
      top: 0;
      max-width: min(360px, calc(100vw - 24px));
      opacity: 0;
      pointer-events: none;
      padding: 9px 10px;
      border: 1px solid #b9c6d2;
      border-radius: 6px;
      background: #10202d;
      color: #ffffff;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
      font-weight: 520;
      line-height: 1.45;
      white-space: normal;
      box-shadow: 0 8px 24px rgba(16, 32, 45, 0.22);
      transition: opacity 0.08s ease;
    }
    .tooltip-bubble.show {
      opacity: 1;
    }
    .mode {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 2px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #eef7f4;
      color: var(--good);
      font-weight: 700;
    }
    .mode.extrapolation {
      background: #fff6e5;
      color: var(--warn);
      border-color: #f2d59a;
    }
    .notice {
      display: none;
      margin-top: 13px;
      padding: 10px 12px;
      border-radius: 6px;
      border: 1px solid #f0c7c0;
      background: #fff5f3;
      color: var(--bad);
    }
    .notice.show { display: block; }
    .progress-box {
      display: none;
      margin-top: 12px;
      padding: 11px 12px;
      border: 1px solid #cbd7e2;
      border-radius: 7px;
      background: #fbfcfe;
    }
    .progress-box.show { display: block; }
    .progress-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      color: #334554;
      font-size: 13px;
      font-weight: 700;
    }
    .progress-track {
      width: 100%;
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #dce5ed;
    }
    .progress-fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
      transition: width 0.22s ease;
    }
    .progress-detail {
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .pass {
      color: var(--good);
      font-weight: 760;
    }
    .fail {
      color: var(--bad);
      font-weight: 760;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 7px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      background: #ffffff;
    }
    th,
    td {
      padding: 10px 11px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      font-size: 13px;
      color: #4f6171;
      background: #f2f5f8;
      font-weight: 720;
    }
    td {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }
    tr:last-child td { border-bottom: 0; }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .meta div {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fbfcfe;
      min-height: 58px;
    }
    .meta strong {
      display: block;
      color: #293847;
      margin-bottom: 3px;
    }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .inputs,
      .result,
      .meta,
      .flash-control-grid,
      .flash-console-grid,
      .flash-param-grid,
      .flash-project-grid,
      .flash-param-toolbar,
      .flash-new-param-row,
      .flash-plot-controls,
      .flash-command-row { grid-template-columns: 1fr; }
      .topbar {
        align-items: flex-start;
        flex-direction: column;
        padding: 14px 0;
      }
      .status { white-space: normal; }
    }
  </style>
</head>
<body>
  <header>
    <div class="shell topbar">
      <h1>ai平台</h1>
    </div>
    <div class="shell app-tabs" role="tablist" aria-label="预测工具标签">
      <button class="tab-btn active" data-view="freePathView" type="button">不透明度自由程</button>
      <button class="tab-btn" data-view="flashView" type="button">FLASH 控制台</button>
      <button class="tab-btn" data-view="capacitorView" type="button">电容器寿命</button>
      <button class="tab-btn" data-view="zpinchView" type="button">套筒速度</button>
    </div>
  </header>
  <main>
    <section class="app-view active" id="freePathView">
    <div class="shell grid">
      <section class="panel">
        <div class="panel-head">
          <h2 class="panel-title">输入参数</h2>
        </div>
        <div class="panel-body">
          <div class="field">
            <div class="inputs">
              <div>
                <label for="zValue">Z</label>
                <input id="zValue" type="number" step="1" inputmode="numeric" value="13" />
              </div>
              <div>
                <label for="rod">rod</label>
                <input id="rod" type="number" step="any" inputmode="decimal" value="-1.870" />
              </div>
              <div>
                <label for="tep">tep</label>
                <input id="tep" type="number" step="any" inputmode="decimal" value="1.771" />
              </div>
              <div>
                <label for="tgama">tgama</label>
                <input id="tgama" type="number" step="any" inputmode="decimal" value="1.755" />
              </div>
            </div>
          </div>
          <div hidden aria-hidden="true">
            <div id="sourceMode" role="radiogroup" aria-label="自由程来源策略"></div>
            <div id="modelElementControls"></div>
            <div id="sourceControlNote"></div>
          </div>
          <div class="actions">
            <button class="primary" id="predictBtn">预测</button>
            <button class="secondary" id="fillSampleBtn">样本</button>
          </div>
          <div class="switch-row" style="margin-top: 12px;">
            <label for="runSimulationToggle">同时调用真实程序</label>
            <label class="toggle">
              <input id="runSimulationToggle" type="checkbox" />
              <span class="track"></span>
            </label>
          </div>
          <div class="progress-box" id="predictProgressBox">
            <div class="progress-head">
              <span id="predictProgressText">等待预测</span>
              <span id="predictProgressPercent">0%</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="predictProgressFill"></div>
            </div>
            <div class="progress-detail" id="predictProgressDetail">-</div>
          </div>
          <div class="notice" id="errorBox"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2 class="panel-title">预测结果</h2>
          <span class="mode" id="modeBadge">等待输入</span>
        </div>
        <div class="panel-body">
          <div class="result">
            <div class="metric">
              <div class="name">lnu<span class="help" tabindex="0" aria-label="指标说明" data-tip="自由程预测值。模型实际预测 log10(lnu)，再还原为 lnu=10^log10(lnu)。"></span></div>
              <div class="value" id="lnuValue">-</div>
            </div>
            <div class="metric">
              <div class="name">log10(lnu)<span class="help" tabindex="0" aria-label="指标说明" data-tip="自由程跨数量级变化，先在 log10 空间建模。例：lnu=100 时 log10(lnu)=2。"></span></div>
              <div class="value" id="logValue">-</div>
            </div>
          </div>
          <div class="meta" style="margin-top: 12px;">
            <div><strong>模型坐标</strong><span id="coordValue">-</span></div>
            <div><strong>范围提示</strong><span id="rangeValue">-</span></div>
            <div><strong>数据命中</strong><span id="membershipValue">-</span></div>
            <div><strong>预测耗时</strong><span id="modelTimeValue">-</span></div>
            <div><strong>最终来源</strong><span id="selectedSourceValue">-</span></div>
            <div><strong>来源策略</strong><span id="policyValue">-</span></div>
          </div>
          <div class="result" style="margin-top: 12px;">
            <div class="metric">
              <div class="name">数据表真值</div>
              <div class="value" id="tableTruthValue">-</div>
            </div>
            <div class="metric">
              <div class="name">预测/表值误差</div>
              <div class="value" id="tableErrorValue">-</div>
            </div>
            <div class="metric">
              <div class="name">表值来源</div>
              <div class="value" id="tableSourceValue">-</div>
            </div>
            <div class="metric">
              <div class="name">表值说明</div>
              <div class="value" id="tableNoteValue">-</div>
            </div>
          </div>
          <div class="result" style="margin-top: 12px;">
            <div class="metric">
              <div class="name">不透明度程序真值</div>
              <div class="value" id="truthValue">-</div>
            </div>
            <div class="metric">
              <div class="name">预测/真值误差</div>
              <div class="value" id="truthErrorValue">-</div>
            </div>
            <div class="metric">
              <div class="name">真值来源</div>
              <div class="value" id="truthSourceValue">-</div>
            </div>
            <div class="metric">
              <div class="name">用时对比</div>
              <div class="value" id="timingCompareValue">-</div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">FLASH 接入控制</h2>
          <span class="mode" id="flashBadge">未开启</span>
        </div>
        <div class="panel-body">
          <div class="flash-control-grid">
            <div>
              <div class="switch-row">
                <label for="flashEnabledToggle">允许 FLASH 使用网页自由程模型</label>
                <label class="toggle">
                  <input id="flashEnabledToggle" type="checkbox" />
                  <span class="track"></span>
                </label>
              </div>
              <div class="field" style="margin-top: 14px;">
                <label>FLASH 自由程来源策略</label>
                <div class="segmented" id="flashMode" role="radiogroup" aria-label="FLASH 自由程来源策略"></div>
              </div>
              <div class="field">
                <label>FLASH 允许模型兜底的元素</label>
                <div class="checkbox-grid" id="flashElementControls"></div>
              </div>
              <div class="actions">
                <button class="primary" id="saveFlashControlBtn" type="button">保存 FLASH 控制</button>
              </div>
              <div class="notice" id="flashNotice"></div>
            </div>
            <div>
              <div class="mini-note">
                <strong>后续使用 FLASH 的方式：</strong>
                <ol class="flash-usage-list">
                  <li>FLASH 侧先读取 <strong>GET /api/flash/control</strong>；如果 enabled=false，继续使用原 IONMIX/查表路径。</li>
                  <li>enabled=true 时，Opacity 里把每个材料状态转换成 <strong>Z rod tep tgama</strong>，其中 rod=log10(rho[g/cm^3])，tep=log10(Te[eV])，tgama=log10(Trad[eV])。</li>
                  <li>不要在每个 cell/能群逐点 HTTP 调用；先按时间步或 opacity 调用批量收集点，再调用 <strong>POST /api/predict/batch</strong>，并在 FLASH 侧做缓存。</li>
                  <li>拿到 lnu 后按 <strong>opacityRO = 1 / (rho * lnu)</strong> 转回 Rosseland opacity；rho 必须是材料质量密度 g/cm^3。</li>
                  <li>遇到未开启元素、超出训练范围、接口失败、DD fill 等未训练材料时，必须回退原表值，不要强行外推。</li>
                </ol>
              </div>
              <pre class="config-preview" id="flashConfigPreview">正在加载 FLASH 控制配置...</pre>
            </div>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">批量读取点</h2>
          <span class="mode" id="batchBadge">等待数据</span>
        </div>
        <div class="panel-body">
          <div class="inputs">
            <div>
              <label for="batchFile">数据文件</label>
              <input id="batchFile" type="file" accept=".txt,.csv,.dat" />
            </div>
            <div>
              <label for="batchDefaultZ">默认 Z</label>
              <input id="batchDefaultZ" type="number" step="1" placeholder="三列输入时使用" />
            </div>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="batchText">批量点内容</label>
            <textarea id="batchText" class="wide-textarea" rows="8" placeholder="4 列：Z rod tep tgama；5 列可带目标 lnu；填写默认 Z 后可用 3 列：rod tep tgama"></textarea>
          </div>
          <div class="actions">
            <button class="primary" id="batchPredictBtn">批量预测</button>
            <button class="secondary" id="batchSampleBtn">样本</button>
          </div>
          <div class="notice" id="batchNotice"></div>
          <div class="meta" style="margin-top: 12px;">
            <div><strong>总点数</strong><span id="batchTotal">-</span></div>
            <div><strong>表值命中</strong><span id="batchTableRows">-</span></div>
            <div><strong>模型兜底</strong><span id="batchModelRows">-</span></div>
            <div><strong>失败行</strong><span id="batchErrorRows">-</span></div>
          </div>
          <div class="table-wrap" style="margin-top: 12px;">
            <table>
              <thead>
                <tr>
                  <th>行</th>
                  <th>Z</th>
                  <th>来源</th>
                  <th>rod</th>
                  <th>tep</th>
                  <th>tgama</th>
                  <th>lnu</th>
                  <th>log10(lnu)</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody id="batchBody">
                <tr><td colspan="9">暂无批量结果</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">最近预测</h2>
          <button class="secondary" id="clearBtn">清空</button>
        </div>
        <div class="panel-body">
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Z</th>
                  <th>模式</th>
                  <th>rod</th>
                  <th>tep</th>
                  <th>tgama</th>
                  <th>lnu</th>
                  <th>log10(lnu)</th>
                </tr>
              </thead>
              <tbody id="historyBody">
                <tr><td colspan="7">暂无记录</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">继续训练</h2>
          <span class="mode" id="trainBadge">未开始</span>
        </div>
        <div class="panel-body">
          <div class="inputs">
            <div>
              <label for="trainData">标准训练数据路径</label>
              <input id="trainData" type="text" value="free_path_model_outputs/unified_standard_training_data.txt" />
            </div>
          </div>
          <div class="actions">
            <button class="primary" id="startTrainBtn">开始暂存训练</button>
            <button class="secondary" id="refreshTrainBtn">刷新状态</button>
            <button class="secondary" id="confirmTrainBtn">确认更新模型权重</button>
          </div>
          <div class="notice" id="trainNotice"></div>
          <div class="inline-section-head">
            <h3>训练数据导入/导出</h3>
            <span class="mode" id="trainDataBadge">标准格式</span>
          </div>
          <div class="mini-note">
            优先上传完整精度的标准五列数据：<strong>Z rod tep tgama lnu</strong>，其中 <strong>rod/tep/tgama 必须是 log10 后的模型坐标</strong>，<strong>lnu 必须为正自由程</strong>。若上传数据中存在 lnu<=0，网页会先询问是否剔除这些行。后续新上传的 Au 数据会默认合并进当前统一训练数据；永久随机测试和连续外推测试中的样本会自动跳过训练。
          </div>
          <div class="inputs" style="margin-top: 12px;">
            <div>
              <label for="trainIoPath">训练数据路径</label>
              <input id="trainIoPath" type="text" value="free_path_model_outputs/unified_standard_training_data.txt" />
            </div>
            <div>
              <label for="trainImportMode">导入方式</label>
              <select id="trainImportMode">
                <option value="append" selected>追加合并</option>
                <option value="replace_element">替换同 Z 元素</option>
                <option value="replace">覆盖替换</option>
              </select>
            </div>
            <div>
              <label for="trainImportZ">4 列数据对应 Z</label>
              <input id="trainImportZ" type="number" step="1" value="79" />
            </div>
            <div>
              <label for="trainImportFile">导入文件</label>
              <input id="trainImportFile" type="file" accept=".txt,.csv,.dat" />
            </div>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="trainImportText">训练数据内容</label>
            <textarea id="trainImportText" rows="7" style="width:100%;border:1px solid #c8d3df;border-radius:6px;padding:10px;font:inherit;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;" placeholder="标准 5 列：Z rod tep tgama lnu；或 4 列：rod tep tgama lnu，并在上方填写 Z"></textarea>
          </div>
          <div class="actions">
            <button class="primary" id="importTrainDataBtn">导入训练数据</button>
            <button class="secondary" id="exportTrainDataBtn">导出当前训练数据</button>
          </div>
          <div class="notice" id="trainDataNotice"></div>
          <div class="meta" style="margin-top: 12px;">
            <div><strong>导入行数</strong><span id="trainImportRows">-</span></div>
            <div><strong>最终行数</strong><span id="trainFinalRows">-</span></div>
            <div><strong>去重行数</strong><span id="trainDedupRows">-</span></div>
            <div><strong>备份文件</strong><span id="trainBackupPath">-</span></div>
          </div>
          <div class="table-wrap" style="margin-top: 12px;">
            <table>
              <thead>
                <tr>
                  <th>训练文件</th>
                  <th>总行数</th>
                  <th>参与训练</th>
                  <th>随机测试跳过</th>
                  <th>外推测试跳过</th>
                </tr>
              </thead>
              <tbody id="trainSplitBody">
                <tr><td colspan="5">暂无训练数据审计</td></tr>
              </tbody>
            </table>
          </div>
          <div class="table-wrap" style="margin-top: 12px;">
            <table>
              <thead>
                <tr>
                  <th>元素</th>
                  <th>benchmark 样本数</th>
                  <th>SMAPE<span class="help" tabindex="0" aria-label="指标说明" data-tip="SMAPE=mean(2*|pred-true|/(|pred|+|true|))*100%。例：true=100,pred=110，SMAPE≈9.52%。"></span></th>
                  <th>log10 MAE<span class="help" tabindex="0" aria-label="指标说明" data-tip="log10 MAE=mean(|log10(pred)-log10(true)|)。例：0.03 约等于典型 10^0.03≈1.07 倍误差。"></span></th>
                  <th>P90 倍数误差<span class="help" tabindex="0" aria-label="指标说明" data-tip="先算 factor=10^|log10(pred)-log10(true)|，P90 是 90% 样本不超过的倍数。例：1.20 表示 90% 样本在 1.2 倍内。"></span></th>
                  <th>P99 倍数误差<span class="help" tabindex="0" aria-label="指标说明" data-tip="P99 是 99% 样本不超过的倍数误差，用来观察尾部误差。例：2.0 表示 99% 样本在 2 倍内。"></span></th>
                  <th>达标</th>
                </tr>
              </thead>
              <tbody id="trainMetricsBody">
                <tr><td colspan="7">暂无暂存训练结果</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">仿真可用阈值</h2>
          <span class="mode" id="thresholdBadge">加载中</span>
        </div>
        <div class="panel-body">
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>范围</th>
                  <th>SMAPE<span class="help" tabindex="0" aria-label="指标说明" data-tip="SMAPE=mean(2*|pred-true|/(|pred|+|true|))*100%。例：true=100,pred=110，SMAPE≈9.52%。"></span></th>
                  <th>log10 MAE<span class="help" tabindex="0" aria-label="指标说明" data-tip="log10 MAE=mean(|log10(pred)-log10(true)|)。例：0.03 约等于典型 10^0.03≈1.07 倍误差。"></span></th>
                  <th>P90 倍数误差<span class="help" tabindex="0" aria-label="指标说明" data-tip="先算 factor=10^|log10(pred)-log10(true)|，P90 是 90% 样本不超过的倍数。例：1.20 表示 90% 样本在 1.2 倍内。"></span></th>
                  <th>P99 倍数误差<span class="help" tabindex="0" aria-label="指标说明" data-tip="P99 是 99% 样本不超过的倍数误差，用来观察尾部误差。例：2.0 表示 99% 样本在 2 倍内。"></span></th>
                </tr>
              </thead>
              <tbody id="thresholdBody">
                <tr><td colspan="5">加载中</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">数据表来源说明</h2>
          <span class="mode">原始/整理数据</span>
        </div>
        <div class="panel-body">
          <div class="mini-note">
            这里区分“原始计算结果表”和“整理后的训练表”。继续训练、上传评测建议优先使用完整精度的标准五列数据：<strong>Z rod tep tgama lnu</strong>；其中 <strong>rod/tep/tgama 统一为 log10 坐标</strong>，<strong>lnu 为正自由程</strong>。如果只有 Al/Be 这种 3 位 log 坐标表，真实程序对比会自动还原低 Z 原始网格点。
          </div>
          <div class="table-wrap" style="margin-top: 12px;">
            <table>
              <thead>
                <tr>
                  <th>数据表</th>
                  <th>Z</th>
                  <th>性质</th>
                  <th>坐标格式</th>
                  <th>训练状态</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody id="dataSourceBody">
                <tr><td colspan="6">加载中</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">上传评测</h2>
          <span class="mode" id="evalBadge">等待数据</span>
        </div>
        <div class="panel-body">
          <div class="inputs">
            <div>
              <label for="evalFile">数据文件</label>
              <input id="evalFile" type="file" accept=".txt,.csv,.dat" />
            </div>
          </div>
          <div class="field">
            <label for="evalText">评测数据</label>
            <textarea id="evalText" rows="7" style="width:100%;border:1px solid #c8d3df;border-radius:6px;padding:10px;font:inherit;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;" placeholder="每行 5 列：Z log10(rod) log10(tep) log10(tgama) lnu"></textarea>
          </div>
          <div class="actions">
            <button class="primary" id="evalBtn">上传评测</button>
            <button class="secondary" id="evalSampleBtn">样本</button>
          </div>
          <div class="notice" id="evalNotice"></div>
          <div class="result" style="margin-top: 12px;">
            <div class="metric">
              <div class="name">SMAPE<span class="help" tabindex="0" aria-label="指标说明" data-tip="SMAPE=mean(2*|pred-true|/(|pred|+|true|))*100%。例：true=100,pred=110，SMAPE≈9.52%。"></span></div>
              <div class="value" id="evalSmape">-</div>
            </div>
            <div class="metric">
              <div class="name">log10 MAE<span class="help" tabindex="0" aria-label="指标说明" data-tip="log10 MAE=mean(|log10(pred)-log10(true)|)。例：0.03 约等于典型 10^0.03≈1.07 倍误差。"></span></div>
              <div class="value" id="evalMae">-</div>
            </div>
            <div class="metric">
              <div class="name">达标</div>
              <div class="value" id="evalPass">-</div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel" style="grid-column: 1 / -1;">
        <div class="panel-head">
          <h2 class="panel-title">训练元素 Benchmark</h2>
          <span class="mode" id="liveBenchBadge">等待数据</span>
        </div>
        <div class="panel-body">
          <div class="inputs">
            <div>
              <label for="liveBenchZ">Z</label>
              <input id="liveBenchZ" type="number" step="1" value="13" />
            </div>
            <div>
              <label for="liveBenchRows">抽样行数</label>
              <input id="liveBenchRows" type="number" step="1" min="1" value="3000" />
            </div>
            <div>
              <label for="liveBenchPath">仿真数据路径</label>
              <input id="liveBenchPath" type="text" value="data_Al.txt" />
            </div>
          </div>
          <div class="actions">
            <button class="primary" id="liveBenchBtn">现场评测</button>
          </div>
          <div class="notice" id="liveBenchNotice"></div>
          <div class="result" style="margin-top: 12px;">
            <div class="metric">
              <div class="name">SMAPE<span class="help" tabindex="0" aria-label="指标说明" data-tip="SMAPE=mean(2*|pred-true|/(|pred|+|true|))*100%。例：true=100,pred=110，SMAPE≈9.52%。"></span></div>
              <div class="value" id="liveBenchSmape">-</div>
            </div>
            <div class="metric">
              <div class="name">log10 MAE<span class="help" tabindex="0" aria-label="指标说明" data-tip="log10 MAE=mean(|log10(pred)-log10(true)|)。例：0.03 约等于典型 10^0.03≈1.07 倍误差。"></span></div>
              <div class="value" id="liveBenchMae">-</div>
            </div>
            <div class="metric">
              <div class="name">P90 倍数误差<span class="help" tabindex="0" aria-label="指标说明" data-tip="先算 factor=10^|log10(pred)-log10(true)|，P90 是 90% 样本不超过的倍数。例：1.20 表示 90% 样本在 1.2 倍内。"></span></div>
              <div class="value" id="liveBenchP90">-</div>
            </div>
            <div class="metric">
              <div class="name">达标</div>
              <div class="value" id="liveBenchPass">-</div>
            </div>
          </div>
        </div>
      </section>
    </div>
    </section>

    <section class="app-view" id="flashView">
      <div class="shell flash-console-grid">
        <section class="panel">
          <div class="panel-head">
            <h2 class="panel-title">运行状态</h2>
            <span class="mode" id="flashRunBadge">加载中</span>
          </div>
          <div class="panel-body">
            <div class="inputs">
              <div>
                <label for="flashRunDir">运行目录</label>
                <select id="flashRunDir"></select>
              </div>
              <div>
                <label for="flashStartCommand">启动命令</label>
                <input id="flashStartCommand" type="text" value="./flash4" />
              </div>
            </div>
            <div class="actions">
              <button class="primary" id="flashStartBtn" type="button">启动</button>
              <button class="secondary" id="flashStopBtn" type="button">停止</button>
              <button class="secondary" id="flashRefreshBtn" type="button">刷新</button>
              <button class="secondary" id="flashBuildBtn" type="button">编译</button>
            </div>
            <div class="notice" id="flashConsoleNotice"></div>
            <div class="meta" style="margin-top: 12px;">
              <div><strong>FLASH 根目录</strong><span id="flashRootValue">-</span></div>
              <div><strong>运行目录</strong><span id="flashRunDirValue">-</span></div>
              <div><strong>进程</strong><span id="flashPidValue">-</span></div>
              <div><strong>可执行文件</strong><span id="flashExeValue">-</span></div>
            </div>
            <div class="inline-section-head">
              <h3>新建项目</h3>
            </div>
            <div class="flash-project-grid">
              <div>
                <label for="flashProjectName">项目名</label>
                <input id="flashProjectName" type="text" placeholder="例如 MRT_test_01" />
              </div>
              <div>
                <label for="flashProjectTemplate">模板</label>
                <select id="flashProjectTemplate"></select>
              </div>
              <div>
                <label for="flashProjectMode">复制模式</label>
                <select id="flashProjectMode">
                  <option value="light">轻量：跳过日志和编译产物</option>
                  <option value="full">完整：复制全部文件</option>
                </select>
              </div>
            </div>
            <div class="actions">
              <button class="primary" id="flashCreateProjectBtn" type="button">新建项目</button>
            </div>
          </div>
        </section>

        <section class="panel" style="grid-column: 1 / -1;">
          <div class="panel-head">
            <h2 class="panel-title">全部 flash.par 参数</h2>
            <span class="mode" id="flashParamBadge">flash.par</span>
          </div>
          <div class="panel-body">
            <div class="flash-param-toolbar">
              <div>
                <label for="flashParamSearch">搜索参数</label>
                <input id="flashParamSearch" type="text" placeholder="参数名、原始值、注释" />
              </div>
              <div>
                <label for="flashParamGroup">分组</label>
                <select id="flashParamGroup"></select>
              </div>
              <label class="check-chip" style="margin-bottom: 0;">
                <input id="flashOnlyChanged" type="checkbox" />
                <span>只看已修改</span>
              </label>
            </div>
            <div class="table-wrap">
              <table class="flash-param-table">
                <thead>
                  <tr>
                    <th>参数</th>
                    <th>分组</th>
                    <th>类型</th>
                    <th>可视化控制</th>
                    <th>原始值</th>
                    <th>注释</th>
                  </tr>
                </thead>
                <tbody id="flashParamsBody">
                  <tr><td colspan="6">暂无参数</td></tr>
                </tbody>
              </table>
            </div>
            <div class="flash-new-param-row">
              <div>
                <label for="flashNewParamKey">新增参数名</label>
                <input id="flashNewParamKey" type="text" placeholder="例如 customParam" />
              </div>
              <div>
                <label for="flashNewParamValue">新增参数值</label>
                <input id="flashNewParamValue" type="text" placeholder="数字、.true.、或字符串" />
              </div>
              <button class="secondary" id="flashAddParamBtn" type="button">加入参数表</button>
            </div>
            <div class="actions">
              <button class="primary" id="flashSaveParamsBtn" type="button">保存已修改参数</button>
            </div>
            <div class="mini-note" id="flashParamNote">当前显示 <span id="flashParamCount">0/0</span> 个参数。保存时会先备份 flash.par；真正是否使用自由程模型还同时受“不透明度自由程”页的 FLASH 接入控制开关影响。</div>
          </div>
        </section>

        <section class="panel" style="grid-column: 1 / -1;">
          <div class="panel-head">
            <h2 class="panel-title">命令行</h2>
            <span class="mode" id="flashCommandBadge">就绪</span>
          </div>
          <div class="panel-body">
            <div class="flash-command-row">
              <div>
                <label for="flashCommandInput">FLASH 运行目录内执行</label>
                <input id="flashCommandInput" type="text" value="tail -n 80 ZPinch_2D.log" />
              </div>
              <button class="primary" id="flashCommandBtn" type="button">执行</button>
            </div>
            <pre class="command-output" id="flashCommandOutput">等待命令</pre>
          </div>
        </section>

        <section class="panel" style="grid-column: 1 / -1;">
          <div class="panel-head">
            <h2 class="panel-title">日志和结果</h2>
            <span class="mode" id="flashFilesBadge">等待刷新</span>
          </div>
          <div class="panel-body">
            <div class="inputs">
              <div>
                <label for="flashLogSelect">日志文件</label>
                <select id="flashLogSelect"></select>
              </div>
              <div>
                <label>文件操作</label>
                <button class="secondary" id="flashOpenLogBtn" type="button">查看日志</button>
              </div>
            </div>
            <pre class="command-output" id="flashLogOutput">暂无日志</pre>
            <div class="flash-plot-controls">
              <div>
                <label for="flashPlotFile">结果文件</label>
                <select id="flashPlotFile"></select>
              </div>
              <div>
                <label for="flashPlotXColumn">X 列</label>
                <select id="flashPlotXColumn"></select>
              </div>
              <div>
                <label for="flashPlotVariable">变量 / Y 列</label>
                <select id="flashPlotVariable"></select>
              </div>
              <div>
                <label for="flashPlotScale">尺度</label>
                <select id="flashPlotScale">
                  <option value="linear">线性</option>
                  <option value="log10">log10</option>
                </select>
              </div>
              <div>
                <label for="flashPlotRange">色标</label>
                <select id="flashPlotRange">
                  <option value="robust">稳健 2%-98%</option>
                  <option value="full">全范围</option>
                </select>
              </div>
              <button class="primary" id="flashRenderPlotBtn" type="button">显示图像</button>
            </div>
            <div class="flash-plot-view" id="flashPlotView">
              <span class="mini-note">选择 FLASH HDF5 plot/checkpoint 文件显示二维场；选择 DAT/CSV/TXT 数字表显示曲线图。</span>
            </div>
            <div class="flash-plot-meta" id="flashPlotMeta">-</div>
            <div class="table-wrap" style="margin-top: 12px;">
              <table>
                <thead>
                  <tr>
                    <th>文件</th>
                    <th>类型</th>
                    <th>大小</th>
                    <th>时间</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody id="flashFilesBody">
                  <tr><td colspan="5">暂无结果文件</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </section>

    <section class="app-view" id="capacitorView">
      <div class="shell">
        <section class="panel">
          <div class="panel-head">
            <h2 class="panel-title">脉冲电容器寿命预测</h2>
            <span class="mode" id="capacitorBadge">8890</span>
          </div>
          <div class="panel-body">
            <div class="mini-note" id="capacitorNote">
              正在嵌入便携包内的电容器寿命网页。该网页使用 Sandia/OEDI 钽电容高温老化 CSV，是实测老化数据，不是不透明度计算程序输出。
            </div>
            <div class="actions">
              <button class="secondary" id="openCapacitorBtn" type="button">新窗口打开</button>
            </div>
            <iframe class="embedded-frame" id="capacitorFrame" title="脉冲电容器状态在线评估"></iframe>
          </div>
        </section>
      </div>
    </section>

    <section class="app-view" id="zpinchView">
      <div class="shell grid">
        <section class="panel">
          <div class="panel-head">
            <h2 class="panel-title">套筒速度输入</h2>
          </div>
          <div class="panel-body">
            <div class="inputs">
              <div>
                <label for="zpinchI">I (MA)</label>
                <input id="zpinchI" type="number" step="any" value="50" />
              </div>
              <div>
                <label for="zpinchTr">tr (ns)</label>
                <input id="zpinchTr" type="number" step="any" value="300" />
              </div>
              <div>
                <label for="zpinchLinerR">套筒半径 (cm)</label>
                <input id="zpinchLinerR" type="number" step="any" value="5" />
              </div>
              <div>
                <label for="zpinchFoamR">泡沫半径 (cm)</label>
                <input id="zpinchFoamR" type="number" step="any" value="0.6" />
              </div>
              <div>
                <label for="zpinchM">套筒质量 (mg/cm)</label>
                <input id="zpinchM" type="number" step="any" value="30" />
              </div>
              <div>
                <label for="zpinchZ">Z</label>
                <input id="zpinchZ" type="number" step="any" value="13" />
              </div>
            </div>
            <div class="actions">
              <button class="primary" id="zpinchPredictBtn" type="button">预测速度</button>
            </div>
            <div class="notice" id="zpinchNotice"></div>
            <div class="mini-note" id="zpinchDataNote">
              当前 Z-pinch 使用便携包 zpinch/ai_training_outputs 中的优化速度优先模型：模型输出速度 v，再由 v 和质量 m 反算动能 E；默认示例在训练范围内，训练数据是零维程序/表格结果，不是不透明度自由程数据。
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2 class="panel-title">套筒速度结果</h2>
            <span class="mode" id="zpinchBadge">待预测</span>
          </div>
          <div class="panel-body">
            <div class="result">
              <div class="metric">
                <div class="name">由速度反算动能 E</div>
                <div class="value" id="zpinchEnergy">-</div>
              </div>
              <div class="metric">
                <div class="name">预测速度 v</div>
                <div class="value" id="zpinchVelocity">-</div>
              </div>
              <div class="metric">
                <div class="name">模型</div>
                <div class="value" id="zpinchModel">-</div>
              </div>
              <div class="metric">
                <div class="name">用时</div>
                <div class="value" id="zpinchTime">-</div>
              </div>
            </div>
            <div class="table-wrap" style="margin-top: 12px;">
              <table>
                <thead>
                  <tr>
                    <th>训练摘要</th>
                    <th>数值</th>
                  </tr>
                </thead>
                <tbody id="zpinchSummaryBody">
                  <tr><td colspan="2">加载中</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </section>
  </main>
  <div class="tooltip-bubble" id="tooltipBubble" role="tooltip"></div>
  <script>
    const zValueEl = document.getElementById("zValue");
    const rodEl = document.getElementById("rod");
    const tepEl = document.getElementById("tep");
    const tgamaEl = document.getElementById("tgama");
    const predictBtn = document.getElementById("predictBtn");
    const fillSampleBtn = document.getElementById("fillSampleBtn");
    const runSimulationToggle = document.getElementById("runSimulationToggle");
    const sourceMode = document.getElementById("sourceMode");
    const modelElementControls = document.getElementById("modelElementControls");
    const sourceControlNote = document.getElementById("sourceControlNote");
    const predictProgressBox = document.getElementById("predictProgressBox");
    const predictProgressText = document.getElementById("predictProgressText");
    const predictProgressPercent = document.getElementById("predictProgressPercent");
    const predictProgressFill = document.getElementById("predictProgressFill");
    const predictProgressDetail = document.getElementById("predictProgressDetail");
    const clearBtn = document.getElementById("clearBtn");
    const startTrainBtn = document.getElementById("startTrainBtn");
    const refreshTrainBtn = document.getElementById("refreshTrainBtn");
    const confirmTrainBtn = document.getElementById("confirmTrainBtn");
    const trainData = document.getElementById("trainData");
    const trainIoPath = document.getElementById("trainIoPath");
    const trainImportMode = document.getElementById("trainImportMode");
    const trainImportZ = document.getElementById("trainImportZ");
    const trainImportFile = document.getElementById("trainImportFile");
    const trainImportText = document.getElementById("trainImportText");
    const importTrainDataBtn = document.getElementById("importTrainDataBtn");
    const exportTrainDataBtn = document.getElementById("exportTrainDataBtn");
    const trainDataNotice = document.getElementById("trainDataNotice");
    const trainDataBadge = document.getElementById("trainDataBadge");
    const trainImportRows = document.getElementById("trainImportRows");
    const trainFinalRows = document.getElementById("trainFinalRows");
    const trainDedupRows = document.getElementById("trainDedupRows");
    const trainBackupPath = document.getElementById("trainBackupPath");
    const evalFile = document.getElementById("evalFile");
    const evalText = document.getElementById("evalText");
    const evalBtn = document.getElementById("evalBtn");
    const evalSampleBtn = document.getElementById("evalSampleBtn");
    const errorBox = document.getElementById("errorBox");
    const trainNotice = document.getElementById("trainNotice");
    const evalNotice = document.getElementById("evalNotice");
    const lnuValue = document.getElementById("lnuValue");
    const logValue = document.getElementById("logValue");
    const coordValue = document.getElementById("coordValue");
    const rangeValue = document.getElementById("rangeValue");
    const membershipValue = document.getElementById("membershipValue");
    const modelTimeValue = document.getElementById("modelTimeValue");
    const selectedSourceValue = document.getElementById("selectedSourceValue");
    const policyValue = document.getElementById("policyValue");
    const tableTruthValue = document.getElementById("tableTruthValue");
    const tableErrorValue = document.getElementById("tableErrorValue");
    const tableSourceValue = document.getElementById("tableSourceValue");
    const tableNoteValue = document.getElementById("tableNoteValue");
    const truthValue = document.getElementById("truthValue");
    const truthErrorValue = document.getElementById("truthErrorValue");
    const truthSourceValue = document.getElementById("truthSourceValue");
    const timingCompareValue = document.getElementById("timingCompareValue");
    const modeBadge = document.getElementById("modeBadge");
    const flashEnabledToggle = document.getElementById("flashEnabledToggle");
    const flashMode = document.getElementById("flashMode");
    const flashElementControls = document.getElementById("flashElementControls");
    const saveFlashControlBtn = document.getElementById("saveFlashControlBtn");
    const flashNotice = document.getElementById("flashNotice");
    const flashBadge = document.getElementById("flashBadge");
    const flashConfigPreview = document.getElementById("flashConfigPreview");
    const flashRunDir = document.getElementById("flashRunDir");
    const flashStartCommand = document.getElementById("flashStartCommand");
    const flashStartBtn = document.getElementById("flashStartBtn");
    const flashStopBtn = document.getElementById("flashStopBtn");
    const flashRefreshBtn = document.getElementById("flashRefreshBtn");
    const flashBuildBtn = document.getElementById("flashBuildBtn");
    const flashRunBadge = document.getElementById("flashRunBadge");
    const flashConsoleNotice = document.getElementById("flashConsoleNotice");
    const flashRootValue = document.getElementById("flashRootValue");
    const flashRunDirValue = document.getElementById("flashRunDirValue");
    const flashPidValue = document.getElementById("flashPidValue");
    const flashExeValue = document.getElementById("flashExeValue");
    const flashProjectName = document.getElementById("flashProjectName");
    const flashProjectTemplate = document.getElementById("flashProjectTemplate");
    const flashProjectMode = document.getElementById("flashProjectMode");
    const flashCreateProjectBtn = document.getElementById("flashCreateProjectBtn");
    const flashParamSearch = document.getElementById("flashParamSearch");
    const flashParamGroup = document.getElementById("flashParamGroup");
    const flashOnlyChanged = document.getElementById("flashOnlyChanged");
    const flashParamsBody = document.getElementById("flashParamsBody");
    const flashParamCount = document.getElementById("flashParamCount");
    const flashNewParamKey = document.getElementById("flashNewParamKey");
    const flashNewParamValue = document.getElementById("flashNewParamValue");
    const flashAddParamBtn = document.getElementById("flashAddParamBtn");
    const flashSaveParamsBtn = document.getElementById("flashSaveParamsBtn");
    const flashParamBadge = document.getElementById("flashParamBadge");
    const flashCommandInput = document.getElementById("flashCommandInput");
    const flashCommandBtn = document.getElementById("flashCommandBtn");
    const flashCommandBadge = document.getElementById("flashCommandBadge");
    const flashCommandOutput = document.getElementById("flashCommandOutput");
    const flashLogSelect = document.getElementById("flashLogSelect");
    const flashOpenLogBtn = document.getElementById("flashOpenLogBtn");
    const flashLogOutput = document.getElementById("flashLogOutput");
    const flashPlotFile = document.getElementById("flashPlotFile");
    const flashPlotXColumn = document.getElementById("flashPlotXColumn");
    const flashPlotVariable = document.getElementById("flashPlotVariable");
    const flashPlotScale = document.getElementById("flashPlotScale");
    const flashPlotRange = document.getElementById("flashPlotRange");
    const flashRenderPlotBtn = document.getElementById("flashRenderPlotBtn");
    const flashPlotView = document.getElementById("flashPlotView");
    const flashPlotMeta = document.getElementById("flashPlotMeta");
    const flashFilesBadge = document.getElementById("flashFilesBadge");
    const flashFilesBody = document.getElementById("flashFilesBody");
    const trainBadge = document.getElementById("trainBadge");
    const evalBadge = document.getElementById("evalBadge");
    const historyBody = document.getElementById("historyBody");
    const trainSplitBody = document.getElementById("trainSplitBody");
    const trainMetricsBody = document.getElementById("trainMetricsBody");
    const thresholdBadge = document.getElementById("thresholdBadge");
    const thresholdBody = document.getElementById("thresholdBody");
    const dataSourceBody = document.getElementById("dataSourceBody");
    const batchFile = document.getElementById("batchFile");
    const batchDefaultZ = document.getElementById("batchDefaultZ");
    const batchText = document.getElementById("batchText");
    const batchPredictBtn = document.getElementById("batchPredictBtn");
    const batchSampleBtn = document.getElementById("batchSampleBtn");
    const batchNotice = document.getElementById("batchNotice");
    const batchBadge = document.getElementById("batchBadge");
    const batchTotal = document.getElementById("batchTotal");
    const batchTableRows = document.getElementById("batchTableRows");
    const batchModelRows = document.getElementById("batchModelRows");
    const batchErrorRows = document.getElementById("batchErrorRows");
    const batchBody = document.getElementById("batchBody");
    const evalSmape = document.getElementById("evalSmape");
    const evalMae = document.getElementById("evalMae");
    const evalPass = document.getElementById("evalPass");
    const liveBenchZ = document.getElementById("liveBenchZ");
    const liveBenchRows = document.getElementById("liveBenchRows");
    const liveBenchPath = document.getElementById("liveBenchPath");
    const liveBenchBtn = document.getElementById("liveBenchBtn");
    const liveBenchNotice = document.getElementById("liveBenchNotice");
    const liveBenchBadge = document.getElementById("liveBenchBadge");
    const liveBenchSmape = document.getElementById("liveBenchSmape");
    const liveBenchMae = document.getElementById("liveBenchMae");
    const liveBenchP90 = document.getElementById("liveBenchP90");
    const liveBenchPass = document.getElementById("liveBenchPass");
    const capacitorFrame = document.getElementById("capacitorFrame");
    const capacitorNote = document.getElementById("capacitorNote");
    const capacitorBadge = document.getElementById("capacitorBadge");
    const openCapacitorBtn = document.getElementById("openCapacitorBtn");
    const zpinchI = document.getElementById("zpinchI");
    const zpinchTr = document.getElementById("zpinchTr");
    const zpinchLinerR = document.getElementById("zpinchLinerR");
    const zpinchFoamR = document.getElementById("zpinchFoamR");
    const zpinchM = document.getElementById("zpinchM");
    const zpinchZ = document.getElementById("zpinchZ");
    const zpinchPredictBtn = document.getElementById("zpinchPredictBtn");
    const zpinchNotice = document.getElementById("zpinchNotice");
    const zpinchDataNote = document.getElementById("zpinchDataNote");
    const zpinchBadge = document.getElementById("zpinchBadge");
    const zpinchEnergy = document.getElementById("zpinchEnergy");
    const zpinchVelocity = document.getElementById("zpinchVelocity");
    const zpinchModel = document.getElementById("zpinchModel");
    const zpinchTime = document.getElementById("zpinchTime");
    const zpinchSummaryBody = document.getElementById("zpinchSummaryBody");
    const tooltipBubble = document.getElementById("tooltipBubble");

    let history = [];
    let thresholds = {};
    let predictionControls = null;
    let flashControl = null;
    let flashConsole = null;
    let flashParamRows = [];

    function formatNum(value, digits = 12) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      const n = Number(value);
      if (n === 0) return "0";
      if (Math.abs(n) >= 1e4 || Math.abs(n) < 1e-3) return n.toExponential(8);
      return Number(n.toPrecision(digits)).toString();
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }
    function countNonpositiveTargets(text) {
      let count = 0;
      for (const rawLine of String(text || "").split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line || line.startsWith("#")) continue;
        const parts = line.replace(/,/g, " ").split(/\s+/);
        const lowered = parts.map(v => v.toLowerCase());
        if (lowered.slice(0, 5).join(" ") === "z rod tep tgama lnu") continue;
        if (lowered.slice(0, 4).join(" ") === "rod tep tgama lnu") continue;
        if (parts.length < 4) continue;
        const targetIndex = parts.length >= 5 ? 4 : 3;
        const target = Number(parts[targetIndex]);
        if (Number.isFinite(target) && target <= 0) count += 1;
      }
      return count;
    }
    function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
    function setPredictionProgress(percent, text, detail = "", show = true) {
      const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
      predictProgressBox.classList.toggle("show", Boolean(show));
      predictProgressText.textContent = text || "等待预测";
      predictProgressPercent.textContent = `${Math.round(safePercent)}%`;
      predictProgressFill.style.width = `${safePercent}%`;
      predictProgressDetail.textContent = detail || "-";
    }
    function renderPredictionJob(job) {
      if (!job) return;
      const elapsed = job.elapsed_ms ? `已用时 ${formatNum(job.elapsed_ms, 8)} ms` : "";
      const detailParts = [];
      if (job.job_id) detailParts.push(`任务 ${job.job_id}`);
      if (job.phase) detailParts.push(`阶段 ${job.phase}`);
      if (elapsed) detailParts.push(elapsed);
      if (job.runSimulation) detailParts.push("真实程序已勾选");
      setPredictionProgress(job.progress ?? 0, job.message || job.status || "预测中", detailParts.join(" / "));
    }
    function showTooltip(el) {
      const text = el && el.dataset ? el.dataset.tip : "";
      if (!text) return;
      tooltipBubble.textContent = text;
      tooltipBubble.classList.add("show");
      const rect = el.getBoundingClientRect();
      const bubbleRect = tooltipBubble.getBoundingClientRect();
      let left = rect.left + rect.width / 2 - bubbleRect.width / 2;
      left = Math.max(12, Math.min(left, window.innerWidth - bubbleRect.width - 12));
      let top = rect.top - bubbleRect.height - 8;
      if (top < 8) top = rect.bottom + 8;
      tooltipBubble.style.left = `${left}px`;
      tooltipBubble.style.top = `${top}px`;
    }
    function hideTooltip() {
      tooltipBubble.classList.remove("show");
    }
    function showError(message) {
      errorBox.textContent = message;
      errorBox.classList.add("show");
    }
    function clearError() {
      errorBox.textContent = "";
      errorBox.classList.remove("show");
    }
    function showTrainNotice(message, isError = false) {
      trainNotice.textContent = message;
      trainNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      trainNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      trainNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      trainNotice.classList.add("show");
    }
    function clearTrainNotice() {
      trainNotice.textContent = "";
      trainNotice.classList.remove("show");
    }
    function showTrainDataNotice(message, isError = false) {
      trainDataNotice.textContent = message;
      trainDataNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      trainDataNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      trainDataNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      trainDataNotice.classList.add("show");
    }
    function clearTrainDataNotice() {
      trainDataNotice.textContent = "";
      trainDataNotice.classList.remove("show");
    }
    function showEvalNotice(message, isError = false) {
      evalNotice.textContent = message;
      evalNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      evalNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      evalNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      evalNotice.classList.add("show");
    }
    function clearEvalNotice() {
      evalNotice.textContent = "";
      evalNotice.classList.remove("show");
    }
    function showBatchNotice(message, isError = false) {
      batchNotice.textContent = message;
      batchNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      batchNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      batchNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      batchNotice.classList.add("show");
    }
    function clearBatchNotice() {
      batchNotice.textContent = "";
      batchNotice.classList.remove("show");
    }
    function showFlashNotice(message, isError = false) {
      flashNotice.textContent = message;
      flashNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      flashNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      flashNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      flashNotice.classList.add("show");
    }
    function clearFlashNotice() {
      flashNotice.textContent = "";
      flashNotice.classList.remove("show");
    }
    function showLiveBenchNotice(message, isError = false) {
      liveBenchNotice.textContent = message;
      liveBenchNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      liveBenchNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      liveBenchNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      liveBenchNotice.classList.add("show");
    }
    function clearLiveBenchNotice() {
      liveBenchNotice.textContent = "";
      liveBenchNotice.classList.remove("show");
    }
    function showZpinchNotice(message, isError = false) {
      zpinchNotice.textContent = message;
      zpinchNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      zpinchNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      zpinchNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      zpinchNotice.classList.add("show");
    }
    function clearZpinchNotice() {
      zpinchNotice.textContent = "";
      zpinchNotice.classList.remove("show");
    }
    function switchAppView(viewId) {
      document.querySelectorAll(".app-view").forEach(view => {
        view.classList.toggle("active", view.id === viewId);
      });
      document.querySelectorAll(".tab-btn").forEach(button => {
        button.classList.toggle("active", button.dataset.view === viewId);
      });
      if (viewId === "capacitorView" && !capacitorFrame.src) {
        const url = `${window.location.protocol}//${window.location.hostname}:8890/`;
        capacitorFrame.src = url;
        openCapacitorBtn.onclick = () => window.open(url, "_blank", "noopener");
        capacitorBadge.textContent = `${window.location.hostname}:8890`;
      }
      if (viewId === "flashView") {
        loadFlashConsole().catch(err => showFlashConsoleNotice(err.message, true));
      }
    }
    function setMode(mode) {
      modeBadge.textContent = mode || "等待输入";
      modeBadge.className = "mode" + (mode === "extrapolation" ? " extrapolation" : "");
    }
    function selectedPredictionMode() {
      const checked = sourceMode.querySelector("input[name='predictionMode']:checked");
      return checked ? checked.value : "model";
    }
    function selectedModelElements() {
      return Array.from(modelElementControls.querySelectorAll(".model-element-toggle:checked"))
        .map(input => Number(input.value))
        .filter(Number.isFinite);
    }
    function predictionControlPayload() {
      return {
        predictionMode: selectedPredictionMode(),
        modelElements: selectedModelElements()
      };
    }
    function updateSourceControlNote() {
      if (!predictionControls) return;
      const modeValue = selectedPredictionMode();
      const mode = (predictionControls.modes || []).find(item => item.value === modeValue);
      const selected = selectedModelElements();
      const total = (predictionControls.model_elements || []).length;
      const selectedText = selected.length === total
        ? "全部已训练元素"
        : (selected.length ? selected.map(value => `Z=${formatNum(value, 6)}`).join("、") : "未启用任何元素");
      const reminders = predictionControls.reminders || [];
      sourceControlNote.innerHTML = `<strong>当前：</strong>${escapeHtml(mode ? mode.label : modeValue)}；模型兜底元素：${escapeHtml(selectedText)}。<br><strong>提醒：</strong>${escapeHtml(reminders.join(" "))}`;
    }
    function renderPredictionControls(controls) {
      predictionControls = controls || {
        modes: [
          { value: "model", label: "只用模型" },
          { value: "hybrid", label: "混合：真值优先，模型兜底" },
          { value: "truth", label: "只用已计算真值" }
        ],
        default_mode: "model",
        model_elements: [],
        default_model_elements: [],
        reminders: []
      };
      const defaultMode = predictionControls.default_mode || "model";
      sourceMode.innerHTML = (predictionControls.modes || []).map(item => `
        <label>
          <input type="radio" name="predictionMode" value="${escapeHtml(item.value)}" ${item.value === defaultMode ? "checked" : ""} />
          <span>${escapeHtml(item.label)}</span>
        </label>
      `).join("");
      const defaultSet = new Set((predictionControls.default_model_elements || []).map(value => String(Number(value))));
      const elements = predictionControls.model_elements || [];
      modelElementControls.innerHTML = elements.length ? elements.map(item => {
        const value = String(Number(item.Z));
        const checked = defaultSet.has(value) ? "checked" : "";
        return `
          <label class="check-chip">
            <input class="model-element-toggle" type="checkbox" value="${escapeHtml(value)}" ${checked} />
            <span>${escapeHtml(item.label)}</span>
          </label>
        `;
      }).join("") : "<span class='mini-note'>当前模型没有可用元素。</span>";
      updateSourceControlNote();
    }
    function selectedFlashMode() {
      const checked = flashMode.querySelector("input[name='flashPredictionMode']:checked");
      return checked ? checked.value : "hybrid";
    }
    function selectedFlashElements() {
      return Array.from(flashElementControls.querySelectorAll(".flash-element-toggle:checked"))
        .map(input => Number(input.value))
        .filter(Number.isFinite);
    }
    function flashControlPayload() {
      return {
        enabled: flashEnabledToggle.checked,
        predictionMode: selectedFlashMode(),
        modelElements: selectedFlashElements()
      };
    }
    function updateFlashPreview() {
      if (!flashControl) return;
      const body = flashControlPayload();
      const mode = (flashControl.modes || []).find(item => item.value === body.predictionMode);
      const config = {
        enabled: body.enabled,
        control_url: flashControl.control_url || "/api/flash/control",
        batch_predict_url: flashControl.batch_predict_url || "/api/predict/batch",
        predictionMode: body.predictionMode,
        predictionModeLabel: mode ? mode.label : body.predictionMode,
        modelElements: body.modelElements,
        flash_usage: [
          "FLASH 启动或每个大时间步读取 control_url；enabled=false 时使用原 IONMIX/查表。",
          "将材料状态转换为 Z rod tep tgama：rod=log10(rho[g/cm^3]), tep=log10(Te[eV]), tgama=log10(Trad[eV])。",
          "批量 POST batch_predict_url，body 里放 text 多行点和 predictionMode/modelElements。",
          "把返回 lnu[cm] 转换为 opacityRO[cm^2/g] = 1 / (rho[g/cm^3] * lnu[cm])。",
          "接口失败、元素未启用、范围外或未训练材料时回退原 opacity 表。"
        ],
        example_batch_body: {
          text: "4 0.2667 2.0 2.0\\n4 0.3000 2.1 2.1",
          predictionMode: body.predictionMode,
          modelElements: body.modelElements
        }
      };
      flashConfigPreview.textContent = JSON.stringify(config, null, 2);
      flashBadge.textContent = body.enabled ? "已开启" : "未开启";
      flashBadge.className = "mode" + (body.enabled ? "" : " extrapolation");
    }
    function renderFlashControl(control) {
      flashControl = control || {
        enabled: false,
        predictionMode: "hybrid",
        modes: predictionControls ? predictionControls.modes : [],
        model_elements: predictionControls ? predictionControls.model_elements : [],
        modelElements: [4]
      };
      flashEnabledToggle.checked = Boolean(flashControl.enabled);
      const selectedMode = flashControl.predictionMode || flashControl.mode || "hybrid";
      flashMode.innerHTML = (flashControl.modes || []).map(item => `
        <label>
          <input type="radio" name="flashPredictionMode" value="${escapeHtml(item.value)}" ${item.value === selectedMode ? "checked" : ""} />
          <span>${escapeHtml(item.label)}</span>
        </label>
      `).join("");
      const selectedSet = new Set((flashControl.modelElements || flashControl.model_elements_enabled || []).map(value => String(Number(value))));
      const elements = flashControl.model_elements || [];
      flashElementControls.innerHTML = elements.length ? elements.map(item => {
        const value = String(Number(item.Z));
        const checked = selectedSet.has(value) ? "checked" : "";
        return `
          <label class="check-chip">
            <input class="flash-element-toggle" type="checkbox" value="${escapeHtml(value)}" ${checked} />
            <span>${escapeHtml(item.label)}</span>
          </label>
        `;
      }).join("") : "<span class='mini-note'>当前没有可给 FLASH 使用的已训练元素。</span>";
      updateFlashPreview();
    }
    async function saveFlashControl() {
      clearFlashNotice();
      saveFlashControlBtn.disabled = true;
      try {
        const res = await fetch("/api/flash/control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(flashControlPayload())
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "FLASH 控制保存失败");
        renderFlashControl(payload);
        showFlashNotice(payload.enabled ? "FLASH 控制已开启；后续 FLASH 侧读取 /api/flash/control 后再批量调用 /api/predict/batch" : "FLASH 控制已关闭；FLASH 应回退原 opacity 查表路径");
      } catch (err) {
        showFlashNotice(err.message, true);
      } finally {
        saveFlashControlBtn.disabled = false;
      }
    }
    function showFlashConsoleNotice(message, isError = false) {
      flashConsoleNotice.textContent = message;
      flashConsoleNotice.style.color = isError ? "var(--bad)" : "var(--good)";
      flashConsoleNotice.style.borderColor = isError ? "#f0c7c0" : "#bfe1d2";
      flashConsoleNotice.style.background = isError ? "#fff5f3" : "#f2fbf6";
      flashConsoleNotice.classList.add("show");
    }
    function clearFlashConsoleNotice() {
      flashConsoleNotice.textContent = "";
      flashConsoleNotice.classList.remove("show");
    }
    function currentFlashRunDir() {
      return flashRunDir.value || (flashConsole && flashConsole.run_dir) || "improved_MRT";
    }
    function normalizeFlashParamInput(rawValue) {
      const text = String(rawValue ?? "").trim();
      const lowered = text.toLowerCase();
      if ([".true.", "true", "t"].includes(lowered)) return true;
      if ([".false.", "false", "f"].includes(lowered)) return false;
      if (text !== "" && Number.isFinite(Number(text))) return Number(text);
      return rawValue;
    }
    function flashParamInputHtml(row) {
      const disabled = row.editable === false ? "disabled" : "";
      const value = row.current_value !== undefined ? row.current_value : row.value;
      if (row.value_type === "bool") {
        const checked = Boolean(value) ? "checked" : "";
        return `<label class="toggle"><input class="flash-param-input" data-param-key="${escapeHtml(row.key)}" type="checkbox" ${checked} ${disabled} /><span class="track"></span></label>`;
      }
      if (row.value_type === "number") {
        return `<input class="flash-param-input" data-param-key="${escapeHtml(row.key)}" type="number" step="any" value="${escapeHtml(value ?? "")}" ${disabled} />`;
      }
      return `<input class="flash-param-input" data-param-key="${escapeHtml(row.key)}" type="text" value="${escapeHtml(value ?? "")}" ${disabled} />`;
    }
    function visibleFlashParamRows() {
      const query = (flashParamSearch.value || "").trim().toLowerCase();
      const group = flashParamGroup.value || "";
      return flashParamRows.filter(row => {
        if (group && row.group !== group) return false;
        if (flashOnlyChanged.checked && !row.changed) return false;
        if (!query) return true;
        const haystack = [row.key, row.group, row.value_type, row.raw, row.comment, row.current_value].join(" ").toLowerCase();
        return haystack.includes(query);
      });
    }
    function renderFlashParamGroups(groups) {
      const current = flashParamGroup.value;
      const options = [`<option value="">全部分组</option>`].concat((groups || []).map(item => (
        `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} (${escapeHtml(item.count)})</option>`
      )));
      flashParamGroup.innerHTML = options.join("");
      if (current && (groups || []).some(item => item.name === current)) {
        flashParamGroup.value = current;
      }
    }
    function flashParamGroupsFromRows() {
      const counts = {};
      for (const row of flashParamRows) {
        const group = row.group || "other";
        counts[group] = (counts[group] || 0) + 1;
      }
      return Object.keys(counts).sort().map(name => ({ name, count: counts[name] }));
    }
    function renderFlashParamTable() {
      const rows = visibleFlashParamRows();
      flashParamCount.textContent = `${rows.length}/${flashParamRows.length}`;
      if (!rows.length) {
        flashParamsBody.innerHTML = "<tr><td colspan='6'>没有匹配的参数</td></tr>";
        return;
      }
      flashParamsBody.innerHTML = rows.map(row => `
        <tr data-param-row="${escapeHtml(row.key)}" class="${row.changed ? "changed" : ""}">
          <td><span class="flash-param-name">${escapeHtml(row.key)}</span><br><span class="flash-param-raw">line ${escapeHtml(row.line || "-")}</span></td>
          <td>${escapeHtml(row.group || "-")}</td>
          <td>${escapeHtml(row.value_type || "-")}</td>
          <td>${flashParamInputHtml(row)}</td>
          <td class="flash-param-raw">${escapeHtml(row.raw ?? "")}</td>
          <td class="flash-param-comment">${escapeHtml(row.comment || "-")}</td>
        </tr>
      `).join("");
    }
    function setFlashParamChanged(key, value) {
      const row = flashParamRows.find(item => item.key === key);
      if (!row) return;
      row.current_value = value;
      row.changed = true;
      renderFlashParamTable();
      flashParamBadge.textContent = `${flashParamRows.filter(item => item.changed).length} 项已修改`;
      flashParamBadge.className = "mode extrapolation";
    }
    function addFlashParamRow() {
      const key = flashNewParamKey.value.trim();
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
        showFlashConsoleNotice("新增参数名只能包含字母、数字和下划线，且不能以数字开头", true);
        return;
      }
      if (flashParamRows.some(row => row.key === key)) {
        showFlashConsoleNotice(`参数 ${key} 已存在，直接在表中修改即可`, true);
        return;
      }
      const normalized = normalizeFlashParamInput(flashNewParamValue.value);
      let valueType = "string";
      if (typeof normalized === "boolean") valueType = "bool";
      else if (typeof normalized === "number") valueType = "number";
      flashParamRows.push({
        key,
        line: "new",
        raw: "",
        value: normalized,
        current_value: normalized,
        value_type: valueType,
        group: key.includes("_") ? key.split("_", 1)[0] : "custom",
        comment: "网页新增",
        editable: true,
        changed: true,
        is_new: true
      });
      flashNewParamKey.value = "";
      flashNewParamValue.value = "";
      renderFlashParamGroups(flashParamGroupsFromRows());
      renderFlashParamTable();
      flashParamBadge.textContent = `${flashParamRows.filter(item => item.changed).length} 项已修改`;
      flashParamBadge.className = "mode extrapolation";
    }
    function renderFlashConsole(status) {
      flashConsole = status || {};
      const previous = flashRunDir.value;
      const runDirs = flashConsole.run_dirs || [];
      flashRunDir.innerHTML = runDirs.map(item => `
        <option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>
      `).join("");
      flashRunDir.value = previous && runDirs.some(item => item.name === previous) ? previous : (flashConsole.run_dir || (runDirs[0] && runDirs[0].name) || "");
      flashProjectTemplate.innerHTML = runDirs.map(item => `
        <option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>
      `).join("");
      flashProjectTemplate.value = flashRunDir.value || flashConsole.run_dir || "";

      const pid = flashConsole.pid || {};
      flashRunBadge.textContent = pid.running ? `运行中 pid ${pid.pid}` : "未运行";
      flashRunBadge.className = "mode" + (pid.running ? "" : " extrapolation");
      flashRootValue.textContent = flashConsole.root || "-";
      flashRunDirValue.textContent = flashConsole.run_dir_path || "-";
      flashPidValue.textContent = pid.running ? `${pid.pid} / ${pid.command || "-"}` : "未运行";
      flashExeValue.textContent = flashConsole.flash4_exists ? "flash4 存在" : "缺少 flash4";

      flashParamRows = (flashConsole.param_rows || []).map(row => ({ ...row, current_value: row.value, changed: false }));
      renderFlashParamGroups(flashConsole.param_groups || []);
      renderFlashParamTable();
      flashParamBadge.textContent = flashConsole.flash_par_exists ? `${flashParamRows.length} 个参数` : "缺少 flash.par";
      flashParamBadge.className = "mode";

      const logs = flashConsole.logs || [];
      const selectedLog = flashLogSelect.value || flashConsole.selected_log || logs[0] || "";
      flashLogSelect.innerHTML = logs.length
        ? logs.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")
        : "<option value=''>暂无日志</option>";
      flashLogSelect.value = logs.includes(selectedLog) ? selectedLog : (logs[0] || "");
      flashLogOutput.textContent = flashConsole.log_tail || "暂无日志";

      const files = flashConsole.files || [];
      flashFilesBadge.textContent = `${files.length} 个文件`;
      flashFilesBody.innerHTML = files.length ? files.map(file => `
        <tr>
          <td>${escapeHtml(file.name)}</td>
          <td>${escapeHtml(file.kind || "-")}</td>
          <td>${formatNum(file.size, 8)}</td>
          <td>${escapeHtml(new Date(Number(file.mtime) * 1000).toLocaleString())}</td>
          <td><button class="secondary file-action" type="button" data-flash-file="${escapeHtml(file.name)}">查看</button></td>
        </tr>
      `).join("") : "<tr><td colspan='5'>暂无结果文件</td></tr>";
      const plotFiles = files.filter(file => file.kind === "hdf5" || /\.(dat|csv|txt)$/i.test(file.name || ""));
      const previousPlotFile = flashPlotFile.value;
      flashPlotFile.innerHTML = plotFiles.length
        ? plotFiles.map(file => `<option value="${escapeHtml(file.name)}">${escapeHtml(file.name)} (${escapeHtml(file.kind)})</option>`).join("")
        : "<option value=''>暂无可视化结果</option>";
      flashPlotFile.value = plotFiles.some(file => file.name === previousPlotFile)
        ? previousPlotFile
        : (plotFiles[0] ? plotFiles[0].name : "");
      if (flashPlotFile.value) {
        loadFlashPlotVariables(flashPlotFile.value).catch(err => {
          flashPlotVariable.innerHTML = "<option value=''>变量读取失败</option>";
          flashPlotXColumn.innerHTML = "<option value='0'>自动</option>";
          flashPlotMeta.textContent = err.message;
        });
      } else {
        flashPlotVariable.innerHTML = "<option value=''>暂无变量</option>";
        flashPlotXColumn.innerHTML = "<option value='0'>自动</option>";
        flashPlotMeta.textContent = "当前运行目录没有可视化结果文件";
      }
    }
    async function loadFlashConsole() {
      const params = new URLSearchParams({ runDir: currentFlashRunDir() });
      const res = await fetch(`/api/flash/status?${params.toString()}`);
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "FLASH 状态加载失败");
      renderFlashConsole(payload);
      return payload;
    }
    function flashParamsPayload() {
      const changed = {};
      let allowNew = false;
      for (const row of flashParamRows) {
        if (!row.changed) continue;
        changed[row.key] = row.current_value;
        if (row.is_new) allowNew = true;
      }
      return {
        runDir: currentFlashRunDir(),
        params: changed,
        allowNew
      };
    }
    async function saveFlashParams() {
      clearFlashConsoleNotice();
      const body = flashParamsPayload();
      const changedCount = Object.keys(body.params || {}).length;
      if (!changedCount) {
        showFlashConsoleNotice("没有需要保存的参数修改");
        return;
      }
      flashSaveParamsBtn.disabled = true;
      try {
        const res = await fetch("/api/flash/params", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "FLASH 参数保存失败");
        renderFlashConsole(payload);
        showFlashConsoleNotice(`${changedCount} 个参数已保存，备份：${payload.backup_path || "-"}`);
      } catch (err) {
        showFlashConsoleNotice(err.message, true);
      } finally {
        flashSaveParamsBtn.disabled = false;
      }
    }
    async function createFlashProject() {
      clearFlashConsoleNotice();
      flashCreateProjectBtn.disabled = true;
      try {
        const name = flashProjectName.value.trim();
        if (!name) throw new Error("请输入新项目名");
        const res = await fetch("/api/flash/project", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            template: flashProjectTemplate.value || currentFlashRunDir(),
            copyMode: flashProjectMode.value || "light"
          })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "新建项目失败");
        flashProjectName.value = "";
        renderFlashConsole(payload);
        showFlashConsoleNotice(`项目 ${payload.created_project.name} 已创建，用时 ${formatNum(payload.created_project.elapsed_ms, 8)} ms`);
      } catch (err) {
        showFlashConsoleNotice(err.message, true);
      } finally {
        flashCreateProjectBtn.disabled = false;
      }
    }
    async function startFlash() {
      clearFlashConsoleNotice();
      flashStartBtn.disabled = true;
      try {
        const res = await fetch("/api/flash/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ runDir: currentFlashRunDir(), command: flashStartCommand.value || "./flash4" })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "FLASH 启动失败");
        renderFlashConsole(payload);
        showFlashConsoleNotice("FLASH 已启动，日志写入 web_flash_run.log");
      } catch (err) {
        showFlashConsoleNotice(err.message, true);
      } finally {
        flashStartBtn.disabled = false;
      }
    }
    async function stopFlash() {
      clearFlashConsoleNotice();
      flashStopBtn.disabled = true;
      try {
        const res = await fetch("/api/flash/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ runDir: currentFlashRunDir() })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "FLASH 停止失败");
        renderFlashConsole(payload);
        showFlashConsoleNotice("已发送停止信号");
      } catch (err) {
        showFlashConsoleNotice(err.message, true);
      } finally {
        flashStopBtn.disabled = false;
      }
    }
    async function runFlashCommand(command = null, timeout = 120) {
      flashCommandBadge.textContent = "运行中";
      flashCommandBtn.disabled = true;
      const actualCommand = command || flashCommandInput.value;
      try {
        const res = await fetch("/api/flash/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ runDir: currentFlashRunDir(), command: actualCommand, timeout })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "命令执行失败");
        flashCommandOutput.textContent = [
          `$ ${payload.command}`,
          `returncode=${payload.returncode} elapsed=${formatNum(payload.elapsed_ms, 8)} ms`,
          payload.stdout ? `\n[stdout]\n${payload.stdout}` : "",
          payload.stderr ? `\n[stderr]\n${payload.stderr}` : ""
        ].join("\n");
        flashCommandBadge.textContent = payload.returncode === 0 ? "完成" : "失败";
        await loadFlashConsole();
      } catch (err) {
        flashCommandOutput.textContent = err.message;
        flashCommandBadge.textContent = "失败";
      } finally {
        flashCommandBtn.disabled = false;
      }
    }
    async function openFlashFile(name = null) {
      const fileName = name || flashLogSelect.value;
      if (!fileName) return;
      const params = new URLSearchParams({ runDir: currentFlashRunDir(), file: fileName });
      const res = await fetch(`/api/flash/file?${params.toString()}`);
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "文件读取失败");
      if (payload.kind === "text") {
        flashLogOutput.textContent = payload.content || "";
        if (payload.table && payload.table.available) {
          flashPlotFile.value = payload.name;
          populateFlashPlotVariables(payload);
        }
      } else if (payload.kind === "image" && payload.data_url) {
        flashLogOutput.textContent = JSON.stringify({ name: payload.name, kind: payload.kind, size: payload.size, mtime: payload.mtime }, null, 2);
        flashPlotView.innerHTML = `<img alt="${escapeHtml(payload.name)}" src="${payload.data_url}" />`;
        flashPlotMeta.textContent = `${payload.name} / ${formatNum(payload.size, 8)} bytes`;
      } else if (payload.kind === "hdf5") {
        flashLogOutput.textContent = JSON.stringify(payload, null, 2);
        flashPlotFile.value = payload.name;
        populateFlashPlotVariables(payload);
      } else {
        flashLogOutput.textContent = JSON.stringify(payload, null, 2);
      }
    }
    function populateFlashPlotVariables(payload) {
      if (payload.kind === "hdf5") {
        const variables = payload.variables || [];
        flashPlotXColumn.innerHTML = "<option value='0'>block x/y</option>";
        flashPlotXColumn.disabled = true;
        flashPlotVariable.innerHTML = variables.length
          ? variables.map(item => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} ${escapeHtml((item.shape || []).join("x"))}</option>`).join("")
          : "<option value=''>暂无可绘制变量</option>";
        flashPlotMeta.textContent = variables.length
          ? `${payload.name || flashPlotFile.value}：${variables.length} 个可绘制变量`
          : `${payload.name || flashPlotFile.value}：没有可绘制变量`;
        return;
      }
      const columns = payload.table && payload.table.columns ? payload.table.columns : [];
      flashPlotXColumn.disabled = false;
      flashPlotXColumn.innerHTML = columns.length
        ? columns.map(item => `<option value="col${escapeHtml(item.index)}">${escapeHtml(item.label || item.name)}</option>`).join("")
        : "<option value='0'>暂无 X 列</option>";
      flashPlotVariable.innerHTML = columns.length
        ? columns.map((item, idx) => `<option value="col${escapeHtml(item.index)}" ${idx === 1 ? "selected" : ""}>${escapeHtml(item.label || item.name)}</option>`).join("")
        : "<option value=''>暂无 Y 列</option>";
      if (columns.length > 1) flashPlotVariable.value = `col${columns[1].index}`;
      flashPlotMeta.textContent = columns.length
        ? `${payload.name || flashPlotFile.value}：${columns.length} 个数字列，可绘制曲线`
        : `${payload.name || flashPlotFile.value}：没有识别出数字列`;
    }
    async function loadFlashPlotVariables(name = null) {
      const fileName = name || flashPlotFile.value;
      if (!fileName) return null;
      const params = new URLSearchParams({ runDir: currentFlashRunDir(), file: fileName });
      const res = await fetch(`/api/flash/file?${params.toString()}`);
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "HDF5 变量读取失败");
      if (payload.kind !== "hdf5" && !(payload.table && payload.table.available)) {
        throw new Error("所选文件不是 HDF5，也没有识别出可绘制数字列");
      }
      populateFlashPlotVariables(payload);
      return payload;
    }
    async function renderFlashPlot() {
      clearFlashConsoleNotice();
      const fileName = flashPlotFile.value;
      const variable = flashPlotVariable.value;
      if (!fileName || !variable) {
        showFlashConsoleNotice("请选择 HDF5 文件和变量", true);
        return;
      }
      flashRenderPlotBtn.disabled = true;
      flashPlotMeta.textContent = "正在渲染图像...";
      try {
        const params = new URLSearchParams({
          runDir: currentFlashRunDir(),
          file: fileName,
          variable,
          xColumn: flashPlotXColumn.value || "col0",
          scale: flashPlotScale.value,
          range: flashPlotRange.value
        });
        const res = await fetch(`/api/flash/plot?${params.toString()}`);
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "图像渲染失败");
        flashPlotView.innerHTML = `<img alt="${escapeHtml(payload.variable)}" src="${payload.image}" />`;
        flashPlotMeta.textContent = [
          `${payload.file} / ${payload.variable}`,
          payload.plot_type === "table" ? `rows=${payload.rows_plotted}` : `blocks=${payload.blocks_plotted}`,
          `值域 ${formatNum(payload.value_min, 8)}..${formatNum(payload.value_max, 8)}`,
          payload.display_min !== undefined ? `显示 ${formatNum(payload.display_min, 8)}..${formatNum(payload.display_max, 8)}` : "",
          `${formatNum(payload.elapsed_ms, 8)} ms`
        ].filter(Boolean).join("；");
      } catch (err) {
        flashPlotMeta.textContent = err.message;
        showFlashConsoleNotice(err.message, true);
      } finally {
        flashRenderPlotBtn.disabled = false;
      }
    }
    function passText(status) {
      if (!status) return "-";
      return status.passed ? "<span class='pass'>达标</span>" : "<span class='fail'>未达标</span>";
    }
    function statusForMetrics(row, profile) {
      const limits = thresholds[profile] || {};
      const keys = ["smape_percent", "log10_mae", "p90_factor_error", "p99_factor_error"];
      return {
        passed: keys.every(key => Number(row[key]) <= Number(limits[key] ?? Infinity))
      };
    }
    function renderThresholds(payload) {
      thresholds = payload.metric_thresholds || {};
      const rows = [
        ["overall", "整体默认 benchmark"],
        ["element", "单元素 / 外部仿真"]
      ];
      thresholdBody.innerHTML = rows.map(([key, label]) => {
        const t = thresholds[key] || {};
        return `
          <tr>
            <td>${label}</td>
            <td>≤ ${formatNum(t.smape_percent)}%</td>
            <td>≤ ${formatNum(t.log10_mae)}</td>
            <td>≤ ${formatNum(t.p90_factor_error)}</td>
            <td>≤ ${formatNum(t.p99_factor_error)}</td>
          </tr>
        `;
      }).join("");
      thresholdBadge.textContent = "已加载";
    }
    function renderDataSources(items) {
      if (!items || !items.length) {
        dataSourceBody.innerHTML = "<tr><td colspan='6'>暂无数据来源信息</td></tr>";
        return;
      }
      dataSourceBody.innerHTML = items.map(item => `
        <tr>
          <td>${escapeHtml(item.name)}</td>
          <td>${escapeHtml(item.Z)}</td>
          <td>${escapeHtml(item.role)}</td>
          <td>${escapeHtml(item.coordinate_format)}</td>
          <td>${escapeHtml(item.training_status)}</td>
          <td>${escapeHtml(item.note)}</td>
        </tr>
      `).join("");
    }
    function renderZpinchInfo(info) {
      if (!info) return;
      zpinchDataNote.textContent = info.data_note || zpinchDataNote.textContent;
      const summary = info.summary || {};
      const metrics = summary.final_test_metrics || summary.fixed_split_best_metrics || {};
      const multiMetrics = summary.multi_seed_test_metrics || {};
      const ranges = summary.training_ranges || {};
      const rangeText = Object.entries(ranges).map(([key, range]) => {
        if (!range || range.min === undefined || range.max === undefined) return "";
        return `${key}: ${formatNum(range.min, 8)}-${formatNum(range.max, 8)}`;
      }).filter(Boolean).join("; ");
      const rows = [
        ["最新目录", info.root],
        ["权重文件", info.artifact],
        ["权重存在", info.artifact_exists ? "是" : "否"],
        ["预测目标", summary.prediction_target ?? "-"],
        ["流程", summary.pipeline ?? "-"],
        ["样本数", summary.n_samples ?? summary.n_total_samples ?? summary.combined_n_samples ?? "-"],
        ["最佳模型", summary.model_name ?? summary.fixed_split_best_model ?? metrics.model ?? "-"],
        ["E MAPE", metrics.E_MAPE_percent !== undefined ? `${formatNum(metrics.E_MAPE_percent, 8)}%` : "-"],
        ["v MAPE", metrics.v_MAPE_percent !== undefined ? `${formatNum(metrics.v_MAPE_percent, 8)}%` : "-"],
        ["多 seed E MAPE", multiMetrics.E_MAPE_percent !== undefined ? `${formatNum(multiMetrics.E_MAPE_percent, 8)}%` : "-"],
        ["多 seed v MAPE", multiMetrics.v_MAPE_percent !== undefined ? `${formatNum(multiMetrics.v_MAPE_percent, 8)}%` : "-"],
        ["训练范围", rangeText || "-"],
      ];
      zpinchSummaryBody.innerHTML = rows.map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`).join("");
    }
    function samplePoint() {
      return { Z: 13, rod: 0.720, tep: 1.771, tgama: 1.755 };
    }
    function fillSample() {
      const sample = samplePoint();
      zValueEl.value = sample.Z;
      rodEl.value = sample.rod;
      tepEl.value = sample.tep;
      tgamaEl.value = sample.tgama;
    }
    function renderHistory() {
      if (!history.length) {
        historyBody.innerHTML = "<tr><td colspan='7'>暂无记录</td></tr>";
        return;
      }
      historyBody.innerHTML = history.slice(0, 12).map(item => `
        <tr>
          <td>${formatNum(item.Z)}</td>
          <td>${item.mode}</td>
          <td>${formatNum(item.input.rod)}</td>
          <td>${formatNum(item.input.tep)}</td>
          <td>${formatNum(item.input.tgama)}</td>
          <td>${formatNum(item.lnu)}</td>
          <td>${formatNum(item.log10_lnu)}</td>
        </tr>
      `).join("");
    }
    function fillBatchSample() {
      batchDefaultZ.value = "13";
      batchText.value = [
        "13 0.720 1.771 1.755",
        "13 -1.870 1.771 1.755 0.0498257",
        "4 -1.870 1.771 1.755",
        "0.720 1.771 1.755"
      ].join("\n");
    }
    function batchRowNote(row) {
      if (!row) return "-";
      if (row.error) return row.error;
      const bits = [];
      if (row.data_membership && row.data_membership.label) bits.push(row.data_membership.label);
      if (row.range_warnings && row.range_warnings.length) bits.push(row.range_warnings.join("；"));
      if (row.target_error) {
        bits.push(`目标误差 ${formatNum(row.target_error.relative_percent, 8)}%, 倍数 ${formatNum(row.target_error.factor_error, 8)}`);
      }
      return bits.join("；") || "-";
    }
    function renderBatchResults(payload) {
      const summary = payload.summary || {};
      const rows = payload.rows || [];
      batchBadge.textContent = `${summary.rows_succeeded ?? 0}/${summary.rows_total ?? rows.length} 成功`;
      batchTotal.textContent = summary.rows_total ?? rows.length;
      batchTableRows.textContent = summary.table_truth_rows ?? 0;
      batchModelRows.textContent = summary.model_rows ?? 0;
      batchErrorRows.textContent = summary.rows_error ?? 0;
      if (!rows.length) {
        batchBody.innerHTML = "<tr><td colspan='9'>暂无批量结果</td></tr>";
      } else {
        const visibleRows = rows.slice(0, 300);
        batchBody.innerHTML = visibleRows.map(row => `
          <tr>
            <td>${escapeHtml(row.line ?? "-")}</td>
            <td>${formatNum(row.Z, 8)}</td>
            <td>${escapeHtml(row.selected_source_label || row.selected_source || "-")}</td>
            <td>${formatNum(row.input && row.input.rod)}</td>
            <td>${formatNum(row.input && row.input.tep)}</td>
            <td>${formatNum(row.input && row.input.tgama)}</td>
            <td>${formatNum(row.lnu)}</td>
            <td>${formatNum(row.log10_lnu)}</td>
            <td>${escapeHtml(batchRowNote(row))}</td>
          </tr>
        `).join("");
        if (rows.length > visibleRows.length) {
          batchBody.innerHTML += `<tr><td colspan="9">只显示前 ${visibleRows.length} 行，共 ${rows.length} 行</td></tr>`;
        }
      }
      const metric = summary.target_metrics;
      const metricText = metric
        ? `；目标列指标：SMAPE ${formatNum(metric.smape_percent, 8)}%，log10 MAE ${formatNum(metric.log10_mae, 8)}`
        : "";
      const notes = (payload.notes || []).join(" ");
      showBatchNotice(`批量预测完成，用时 ${formatNum(summary.elapsed_ms, 8)} ms${metricText}。${notes}`);
    }
    function renderMetricRows(tbody, metrics, emptyText) {
      if (!metrics) {
        tbody.innerHTML = `<tr><td colspan='7'>${emptyText}</td></tr>`;
        return;
      }
      const rows = [
        metrics.overall,
        ...(metrics.by_element || []),
        ...(metrics.by_source || []),
        ...(metrics.by_dataset || [])
      ];
      tbody.innerHTML = rows.map((row, idx) => {
        const name = idx === 0 ? "overall" : (row.element || row.source || row.dataset);
        const profile = idx === 0 ? "overall" : "element";
        const status = statusForMetrics(row, profile);
        return `
          <tr>
            <td>${name}</td>
            <td>${row.n}</td>
            <td>${formatNum(row.smape_percent)}%</td>
            <td>${formatNum(row.log10_mae)}</td>
            <td>${formatNum(row.p90_factor_error)}</td>
            <td>${formatNum(row.p99_factor_error)}</td>
            <td>${passText(status)}</td>
          </tr>
        `;
      }).join("");
    }
    function renderTrainSplits(summary) {
      const files = summary && summary.training_files ? Object.entries(summary.training_files) : [];
      if (!files.length) {
        trainSplitBody.innerHTML = "<tr><td colspan='5'>暂无训练数据审计</td></tr>";
        return;
      }
      trainSplitBody.innerHTML = files.map(([name, item]) => `
        <tr>
          <td>${escapeHtml(name)}</td>
          <td>${item.total_rows ?? "-"}</td>
          <td>${item.training_rows ?? "-"}</td>
          <td>${item.benchmark_excluded ?? 0}</td>
          <td>${item.extrapolation_excluded ?? 0}</td>
        </tr>
      `).join("");
    }
    function renderTrainMetrics(summary) {
      renderTrainSplits(summary);
      renderMetricRows(trainMetricsBody, summary && summary.benchmark_metrics, "暂无暂存训练结果");
    }
    function renderTrainStatus(status) {
      trainBadge.textContent = status.status || "unknown";
      trainBadge.className = "mode" + (status.status === "running" ? " extrapolation" : "");
      startTrainBtn.disabled = status.status === "running";
      confirmTrainBtn.disabled = status.status === "running" || !status.has_staged_artifact;
      if (status.status === "failed") {
        showTrainNotice(status.error || "训练失败", true);
      } else if (status.status === "succeeded") {
        showTrainNotice("暂存训练完成，确认后才会更新线上权重");
      }
      renderTrainMetrics(status.summary);
      if (status.status === "running") {
        setTimeout(() => refreshTrainStatus().catch(err => showTrainNotice(err.message, true)), 2000);
      }
    }
    function renderResult(result) {
      lnuValue.textContent = formatNum(result.lnu);
      logValue.textContent = formatNum(result.log10_lnu);
      coordValue.textContent = `rod=${formatNum(result.model_coordinates.rod)}, tep=${formatNum(result.model_coordinates.tep)}, tgama=${formatNum(result.model_coordinates.tgama)}`;
      rangeValue.textContent = result.range_warnings && result.range_warnings.length ? result.range_warnings.join("; ") : "范围内";
      membershipValue.textContent = result.data_membership ? result.data_membership.label : "-";
      if (result.timing && result.timing.model_ms !== null && result.timing.model_ms !== undefined) {
        modelTimeValue.textContent = `模型 ${formatNum(result.timing.model_ms, 8)} ms`;
      } else if (result.timing && result.timing.truth_lookup_ms !== undefined) {
        modelTimeValue.textContent = `查表 ${formatNum(result.timing.truth_lookup_ms, 8)} ms`;
      } else {
        modelTimeValue.textContent = "-";
      }
      selectedSourceValue.textContent = result.selected_source_label || (result.prediction_policy && result.prediction_policy.selected_source_label) || "-";
      if (result.prediction_policy) {
        const labels = result.prediction_policy.model_element_labels || [];
        policyValue.textContent = `${result.prediction_policy.mode_label}；模型元素 ${labels.length ? labels.join("、") : "无"}`;
      } else {
        policyValue.textContent = "-";
      }
      if (result.table_truth && result.table_truth.found) {
        const table = result.table_truth;
        tableTruthValue.textContent = `${formatNum(table.lnu)} / log=${formatNum(table.log10_lnu)}`;
        tableSourceValue.textContent = table.source_path || table.source_label || "-";
        tableNoteValue.textContent = table.note || "-";
        tableErrorValue.textContent = table.error
          ? `相对 ${formatNum(table.error.relative_percent, 8)}%, 倍数 ${formatNum(table.error.factor_error, 8)}`
          : "-";
      } else {
        tableTruthValue.textContent = "-";
        tableErrorValue.textContent = "-";
        tableSourceValue.textContent = "当前标准训练/测试表中没有这个精确点";
        tableNoteValue.textContent = "表值误差只在输入点命中标准数据时显示";
      }
      if (result.reference_truth && result.reference_truth.found) {
        const ref = result.reference_truth;
        truthValue.textContent = `${formatNum(ref.lnu)} / log=${formatNum(ref.log10_lnu)}`;
        const callBits = [];
        callBits.push(ref.invoked ? "真实程序已调用" : "真实程序未调用");
        if (ref.coordinate_snap && ref.coordinate_snap.applied) callBits.push("已还原原始网格坐标");
        if (ref.pid) callBits.push(`PID ${ref.pid}`);
        if (ref.program_path) callBits.push(ref.program_path);
        truthSourceValue.textContent = `${ref.source_label || ref.source || "-"}；${callBits.join("；")}`;
        if (ref.error) {
          truthErrorValue.textContent = `相对 ${formatNum(ref.error.relative_percent, 8)}%, 倍数 ${formatNum(ref.error.factor_error, 8)}`;
        } else {
          truthErrorValue.textContent = "-";
        }
        if (result.timing) {
          const simMs = Number(ref.wall_ms ?? (Number(ref.program_elapsed_s || 0) * 1000));
          if (result.timing.model_ms !== null && result.timing.model_ms !== undefined) {
            const speedup = simMs > 0 ? simMs / Number(result.timing.model_ms || 1) : null;
            timingCompareValue.textContent = `模型 ${formatNum(result.timing.model_ms, 8)} ms / 真实程序 ${formatNum(simMs, 8)} ms / 加速 ${formatNum(speedup, 8)}x`;
          } else {
            timingCompareValue.textContent = `查表 ${formatNum(result.timing.truth_lookup_ms, 8)} ms / 真实程序 ${formatNum(simMs, 8)} ms`;
          }
        } else {
          timingCompareValue.textContent = `真实程序 ${formatNum(ref.wall_ms, 8)} ms`;
        }
      } else {
        const reason = result.reference_truth && result.reference_truth.reason ? result.reference_truth.reason : "未命中已计算真值";
        truthValue.textContent = "-";
        truthErrorValue.textContent = "-";
        if (result.reference_truth) {
          const ref = result.reference_truth;
          const callText = ref.requested ? (ref.invoked ? "真实程序已调用但未返回有效真值" : "真实程序未启动") : "未请求真实程序";
          const pathText = ref.program_path ? `；${ref.program_path}` : "";
          truthSourceValue.textContent = `${callText}：${reason}${pathText}`;
        } else {
          truthSourceValue.textContent = reason;
        }
        if (result.timing && result.timing.model_ms !== null && result.timing.model_ms !== undefined) {
          timingCompareValue.textContent = `模型 ${formatNum(result.timing.model_ms, 8)} ms / 真实程序未完成`;
        } else if (result.timing && result.timing.truth_lookup_ms !== undefined) {
          timingCompareValue.textContent = `查表 ${formatNum(result.timing.truth_lookup_ms, 8)} ms / 真实程序未完成`;
        } else {
          timingCompareValue.textContent = "-";
        }
      }
      setMode(result.mode);
      history.unshift(result);
      history = history.slice(0, 30);
      renderHistory();
    }
    async function loadModelInfo() {
      const res = await fetch("/api/datasets");
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "模型加载失败");
      if (payload.standard_training_data_path) {
        trainData.value = payload.standard_training_data_path;
        trainIoPath.value = payload.standard_training_data_path;
      }
      renderThresholds(payload);
      renderDataSources(payload.data_source_audit);
      renderZpinchInfo(payload.zpinch_app);
      renderPredictionControls(payload.prediction_controls);
      renderFlashControl(payload.flash_control);
      if (payload.flash_console) renderFlashConsole(payload.flash_console);
      if (payload.capacitor_app) {
        capacitorNote.textContent = `${payload.capacitor_app.source}；${payload.capacitor_app.data_note}`;
        capacitorBadge.textContent = `端口 ${payload.capacitor_app.port}`;
      }
      fillSample();
    }
    async function predict() {
      clearError();
      predictBtn.disabled = true;
      setPredictionProgress(4, "准备预测", "正在检查输入");
      try {
        const body = {
          Z: Number(zValueEl.value),
          rod: Number(rodEl.value),
          tep: Number(tepEl.value),
          tgama: Number(tgamaEl.value),
          runSimulation: runSimulationToggle.checked,
          ...predictionControlPayload()
        };
        if (![body.Z, body.rod, body.tep, body.tgama].every(Number.isFinite)) {
          throw new Error("请输入完整数值");
        }
        const modeLabel = (predictionControls && (predictionControls.modes || []).find(item => item.value === body.predictionMode) || {}).label || body.predictionMode;
        setPredictionProgress(8, "提交预测任务", body.runSimulation ? `${modeLabel}；将启动真实不透明度单点程序` : modeLabel);
        const startRes = await fetch("/api/predict/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        let job = await startRes.json();
        if (!startRes.ok) throw new Error(job.error || "预测任务启动失败");
        renderPredictionJob(job);
        while (job.status === "queued" || job.status === "running") {
          await sleep(500);
          const statusRes = await fetch(`/api/predict/status?id=${encodeURIComponent(job.job_id)}`);
          job = await statusRes.json();
          if (!statusRes.ok) throw new Error(job.error || "预测状态读取失败");
          renderPredictionJob(job);
        }
        if (job.status !== "succeeded") {
          throw new Error(job.error || "预测失败");
        }
        setPredictionProgress(100, "预测完成", job.runSimulation ? "模型预测和真实程序调用均已结束" : "模型预测已结束");
        renderResult(job.result);
      } catch (err) {
        setPredictionProgress(100, "预测失败", err.message);
        showError(err.message);
      } finally {
        predictBtn.disabled = false;
      }
    }
    async function refreshTrainStatus() {
      const res = await fetch("/api/train/status");
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "训练状态读取失败");
      renderTrainStatus(payload);
      return payload;
    }
    async function startTraining() {
      clearTrainNotice();
      startTrainBtn.disabled = true;
      try {
        const res = await fetch("/api/train/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            standardDataPath: trainData.value.trim()
          })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "无法启动训练");
        renderTrainStatus(payload);
        showTrainNotice("训练已开始，完成前不会更新线上权重");
      } catch (err) {
        showTrainNotice(err.message, true);
      } finally {
        setTimeout(refreshTrainStatus, 1200);
      }
    }
    async function confirmTraining() {
      clearTrainNotice();
      if (!window.confirm("确认用暂存模型更新线上权重？")) return;
      const res = await fetch("/api/train/confirm", { method: "POST" });
      const payload = await res.json();
      if (!res.ok) {
        showTrainNotice(payload.error || "确认更新失败", true);
        return;
      }
      showTrainNotice("线上模型权重已更新并热加载");
      renderTrainStatus(payload.training_status);
      await loadModelInfo();
    }
    function exportTrainingData() {
      clearTrainDataNotice();
      const path = trainIoPath.value.trim() || trainData.value.trim();
      if (!path) {
        showTrainDataNotice("请输入训练数据路径", true);
        return;
      }
      window.location.href = `/api/training-data/export?path=${encodeURIComponent(path)}`;
    }
    async function importTrainingData() {
      clearTrainDataNotice();
      importTrainDataBtn.disabled = true;
      try {
        const text = trainImportText.value.trim();
        if (!text) throw new Error("请粘贴或选择训练数据");
        const mode = trainImportMode.value;
        const nonpositiveRows = countNonpositiveTargets(text);
        const removeNonpositive = nonpositiveRows > 0 && window.confirm(`检测到 ${nonpositiveRows} 行 lnu<=0。自由程必须为正，是否剔除这些行后继续导入？`);
        if (nonpositiveRows > 0 && !removeNonpositive) {
          throw new Error("已取消导入：训练数据中存在 lnu<=0 的行");
        }
        if (mode === "replace" && !window.confirm("确认覆盖当前训练数据文件？系统会先备份旧文件。")) {
          return;
        }
        if (mode === "replace_element" && !window.confirm(`确认替换导入数据中对应 Z 的现有训练行？系统会先备份旧文件。`)) {
          return;
        }
        const res = await fetch("/api/training-data/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            path: trainIoPath.value.trim() || trainData.value.trim(),
            mode,
            text,
            legacyZ: trainImportZ.value,
            removeNonpositive
          })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "训练数据导入失败");
        trainData.value = payload.path;
        trainIoPath.value = payload.path;
        trainImportRows.textContent = payload.imported_rows;
        trainFinalRows.textContent = payload.final_rows;
        trainDedupRows.textContent = payload.duplicate_rows_removed;
        trainBackupPath.textContent = payload.backup_path || "-";
        trainDataBadge.textContent = `${payload.final_rows} rows`;
        const removedText = payload.nonpositive_rows_removed ? `，已剔除 ${payload.nonpositive_rows_removed} 行 lnu<=0` : "";
        showTrainDataNotice(`导入完成：${payload.mode_label}${removedText}，后续训练会继续自动跳过永久测试集`);
        refreshTrainStatus().catch(() => {});
      } catch (err) {
        showTrainDataNotice(err.message, true);
      } finally {
        importTrainDataBtn.disabled = false;
      }
    }
    function fillEvalSample() {
      evalText.value = "13 -1.870 1.771 1.755 0.0498257";
    }
    async function runUploadEval() {
      clearEvalNotice();
      evalBtn.disabled = true;
      try {
        const text = evalText.value.trim();
        if (!text) throw new Error("请粘贴或选择评测数据");
        const nonpositiveRows = countNonpositiveTargets(text);
        const removeNonpositive = nonpositiveRows > 0 && window.confirm(`检测到 ${nonpositiveRows} 行 lnu<=0。评测指标需要正自由程，是否剔除这些行后继续评测？`);
        if (nonpositiveRows > 0 && !removeNonpositive) {
          throw new Error("已取消评测：评测数据中存在 lnu<=0 的行");
        }
        const res = await fetch("/api/evaluate/upload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            removeNonpositive
          })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "评测失败");
        evalBadge.textContent = `${payload.metrics.n} rows`;
        evalSmape.textContent = `${formatNum(payload.metrics.smape_percent)}%`;
        evalMae.textContent = formatNum(payload.metrics.log10_mae);
        evalPass.innerHTML = passText(payload.threshold_status);
        const removedText = payload.nonpositive_rows_removed ? `，已剔除 ${payload.nonpositive_rows_removed} 行 lnu<=0` : "";
        showEvalNotice(`评测完成${removedText}`);
      } catch (err) {
        showEvalNotice(err.message, true);
      } finally {
        evalBtn.disabled = false;
      }
    }
    async function runBatchPrediction() {
      clearBatchNotice();
      batchPredictBtn.disabled = true;
      batchBadge.textContent = "计算中";
      try {
        const text = batchText.value.trim();
        if (!text) throw new Error("请粘贴或选择批量点数据");
        const res = await fetch("/api/predict/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            defaultZ: batchDefaultZ.value.trim(),
            ...predictionControlPayload()
          })
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "批量预测失败");
        renderBatchResults(payload);
      } catch (err) {
        batchBadge.textContent = "失败";
        showBatchNotice(err.message, true);
      } finally {
        batchPredictBtn.disabled = false;
      }
    }
    async function runLiveBenchmark() {
      clearLiveBenchNotice();
      liveBenchBtn.disabled = true;
      try {
        const body = {
          Z: Number(liveBenchZ.value),
          path: liveBenchPath.value.trim(),
          sampleRows: Number(liveBenchRows.value || 3000)
        };
        if (!Number.isFinite(body.Z) || !body.path) throw new Error("请输入 Z 和仿真数据路径");
        const res = await fetch("/api/benchmark/file", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "现场评测失败");
        liveBenchBadge.textContent = `${payload.metrics.n} rows`;
        liveBenchSmape.textContent = `${formatNum(payload.metrics.smape_percent)}%`;
        liveBenchMae.textContent = formatNum(payload.metrics.log10_mae);
        liveBenchP90.textContent = formatNum(payload.metrics.p90_factor_error);
        liveBenchPass.innerHTML = passText(payload.threshold_status);
        showLiveBenchNotice(`评测完成：${payload.path}，坐标处理 ${payload.coordinate_transform}`);
      } catch (err) {
        showLiveBenchNotice(err.message, true);
      } finally {
        liveBenchBtn.disabled = false;
      }
    }
    async function runZpinchPrediction() {
      clearZpinchNotice();
      zpinchPredictBtn.disabled = true;
      zpinchBadge.textContent = "计算中";
      try {
        const body = {
          I: Number(zpinchI.value),
          tr: Number(zpinchTr.value),
          liner_r: Number(zpinchLinerR.value),
          foam_r: Number(zpinchFoamR.value),
          m: Number(zpinchM.value),
          Z: Number(zpinchZ.value)
        };
        if (!Object.values(body).every(Number.isFinite)) throw new Error("请输入完整的套筒速度参数");
        const res = await fetch("/api/zpinch/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "套筒速度预测失败");
        zpinchEnergy.textContent = `${formatNum(payload.E_MJ_per_cm, 10)} MJ/cm`;
        zpinchVelocity.textContent = `${formatNum(payload.v_cm_per_s, 10)} cm/s`;
        zpinchModel.textContent = [payload.model_name, payload.pipeline].filter(Boolean).join(" / ") || "-";
        zpinchTime.textContent = `${formatNum(payload.elapsed_ms, 8)} ms`;
        zpinchBadge.textContent = "完成";
        showZpinchNotice(`预测完成，权重文件：${payload.artifact}`);
      } catch (err) {
        zpinchBadge.textContent = "失败";
        showZpinchNotice(err.message, true);
      } finally {
        zpinchPredictBtn.disabled = false;
      }
    }
    document.querySelectorAll(".tab-btn").forEach(button => {
      button.addEventListener("click", () => switchAppView(button.dataset.view));
    });
    fillSampleBtn.addEventListener("click", fillSample);
    predictBtn.addEventListener("click", predict);
    sourceMode.addEventListener("change", updateSourceControlNote);
    modelElementControls.addEventListener("change", updateSourceControlNote);
    flashEnabledToggle.addEventListener("change", updateFlashPreview);
    flashMode.addEventListener("change", updateFlashPreview);
    flashElementControls.addEventListener("change", updateFlashPreview);
    saveFlashControlBtn.addEventListener("click", saveFlashControl);
    flashRunDir.addEventListener("change", () => loadFlashConsole().catch(err => showFlashConsoleNotice(err.message, true)));
    flashRefreshBtn.addEventListener("click", () => loadFlashConsole().catch(err => showFlashConsoleNotice(err.message, true)));
    flashSaveParamsBtn.addEventListener("click", saveFlashParams);
    flashCreateProjectBtn.addEventListener("click", createFlashProject);
    flashParamSearch.addEventListener("input", renderFlashParamTable);
    flashParamGroup.addEventListener("change", renderFlashParamTable);
    flashOnlyChanged.addEventListener("change", renderFlashParamTable);
    flashAddParamBtn.addEventListener("click", addFlashParamRow);
    flashParamsBody.addEventListener("change", event => {
      const input = event.target.closest && event.target.closest(".flash-param-input");
      if (!input) return;
      const key = input.dataset.paramKey;
      let value = input.type === "checkbox" ? input.checked : input.value;
      if (input.type === "number") value = Number(input.value);
      setFlashParamChanged(key, value);
    });
    flashParamsBody.addEventListener("input", event => {
      const input = event.target.closest && event.target.closest(".flash-param-input");
      if (!input || input.type === "checkbox") return;
      const key = input.dataset.paramKey;
      let value = input.type === "number" ? Number(input.value) : input.value;
      const row = flashParamRows.find(item => item.key === key);
      if (!row) return;
      row.current_value = value;
      row.changed = true;
      const tr = input.closest("tr");
      if (tr) tr.classList.add("changed");
      flashParamBadge.textContent = `${flashParamRows.filter(item => item.changed).length} 项已修改`;
      flashParamBadge.className = "mode extrapolation";
    });
    flashStartBtn.addEventListener("click", startFlash);
    flashStopBtn.addEventListener("click", stopFlash);
    flashBuildBtn.addEventListener("click", () => runFlashCommand("make -j2", 900));
    flashCommandBtn.addEventListener("click", () => runFlashCommand());
    flashOpenLogBtn.addEventListener("click", () => openFlashFile().catch(err => showFlashConsoleNotice(err.message, true)));
    flashPlotFile.addEventListener("change", () => loadFlashPlotVariables().catch(err => showFlashConsoleNotice(err.message, true)));
    flashRenderPlotBtn.addEventListener("click", renderFlashPlot);
    flashFilesBody.addEventListener("click", event => {
      const button = event.target.closest && event.target.closest("[data-flash-file]");
      if (button) openFlashFile(button.dataset.flashFile).catch(err => showFlashConsoleNotice(err.message, true));
    });
    startTrainBtn.addEventListener("click", startTraining);
    refreshTrainBtn.addEventListener("click", () => refreshTrainStatus().catch(err => showTrainNotice(err.message, true)));
    confirmTrainBtn.addEventListener("click", () => confirmTraining().catch(err => showTrainNotice(err.message, true)));
    exportTrainDataBtn.addEventListener("click", exportTrainingData);
    importTrainDataBtn.addEventListener("click", importTrainingData);
    evalBtn.addEventListener("click", runUploadEval);
    evalSampleBtn.addEventListener("click", fillEvalSample);
    batchPredictBtn.addEventListener("click", runBatchPrediction);
    batchSampleBtn.addEventListener("click", fillBatchSample);
    liveBenchBtn.addEventListener("click", runLiveBenchmark);
    zpinchPredictBtn.addEventListener("click", runZpinchPrediction);
    document.addEventListener("mouseover", event => {
      const el = event.target.closest && event.target.closest(".help");
      if (el) showTooltip(el);
    });
    document.addEventListener("mouseout", event => {
      const el = event.target.closest && event.target.closest(".help");
      if (el) hideTooltip();
    });
    document.addEventListener("focusin", event => {
      const el = event.target.closest && event.target.closest(".help");
      if (el) showTooltip(el);
    });
    document.addEventListener("focusout", event => {
      const el = event.target.closest && event.target.closest(".help");
      if (el) hideTooltip();
    });
    window.addEventListener("scroll", hideTooltip, true);
    window.addEventListener("resize", hideTooltip);
    evalFile.addEventListener("change", async () => {
      const file = evalFile.files && evalFile.files[0];
      if (file) evalText.value = await file.text();
    });
    batchFile.addEventListener("change", async () => {
      const file = batchFile.files && batchFile.files[0];
      if (file) batchText.value = await file.text();
    });
    trainImportFile.addEventListener("change", async () => {
      const file = trainImportFile.files && trainImportFile.files[0];
      if (file) trainImportText.value = await file.text();
    });
    clearBtn.addEventListener("click", () => {
      history = [];
      renderHistory();
    });
    [zValueEl, rodEl, tepEl, tgamaEl].forEach(el => {
      el.addEventListener("keydown", event => {
        if (event.key === "Enter") predict();
      });
    });
    loadModelInfo().then(() => refreshTrainStatus()).catch(err => showError(err.message));
  </script>
</body>
</html>
"""


class FreePathWebApp:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.model = UnifiedModel.load(artifact_path)
        self.high_z_model = HighZModel.load_if_available()
        self.metadata = copy.deepcopy(self.model.metadata)
        self.metadata["standard_training_data_path"] = str(STANDARD_TRAINING_DATA_PATH)
        for item in self.metadata.get("training_files", {}).values():
            original = Path(str(item.get("path", ""))).expanduser()
            local_candidate = ROOT / original.name
            if original.is_absolute() and not original.exists() and local_candidate.exists():
                item["path"] = str(local_candidate)
        self.model.metadata = self.metadata
        self._lock = threading.Lock()
        self._truth_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._flash_control_lock = threading.Lock()
        self._simulator_build_lock = threading.Lock()
        self._truth_cache: dict | None = None
        self._predict_jobs: dict[str, dict] = {}
        self._flash_control = self._default_flash_control()
        self._train_job = {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": "",
        }

    def _predict_log_routed(
        self,
        x_model: np.ndarray,
        data_source: str | list | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict, np.ndarray]:
        x = np.asarray(x_model, dtype=float)
        one = x.ndim == 1
        if one:
            x = x.reshape(1, -1)
        base_pred = np.asarray(self.model.predict_log(x), dtype=float).reshape(-1)
        pred = base_pred.copy()
        route = {
            "base_model": "unified_local_regression",
            "data_source": data_source if data_source is not None else "auto",
            "high_z_model_available": self.high_z_model is not None,
            "high_z_model_used_rows": 0,
            "high_z_direct_rows": 0,
            "high_z_boundary_rows": 0,
            "high_z_model": None,
        }
        if self.high_z_model is not None:
            pred, high_z_mask, route_details = self.high_z_model.predict_log(
                x,
                fallback_log=base_pred,
                data_source=data_source,
                return_details=True,
            )
            high_z_mask = np.asarray(high_z_mask, dtype=bool).reshape(-1)
            route.update(
                {
                    "high_z_model_used_rows": int(high_z_mask.sum()),
                    "high_z_direct_rows": int(route_details.get("direct_rows", 0)),
                    "high_z_boundary_rows": int(route_details.get("boundary_rows", 0)),
                    "high_z_route_policy": route_details.get("route_policy"),
                    "high_z_model": {
                        "artifact_type": self.high_z_model.metadata.get("artifact_type"),
                        "model_path": str(self.high_z_model.model_path),
                        "z_values": self.high_z_model.metadata.get("z_values", []),
                        "alpha_base": self.high_z_model.metadata.get("alpha_base", 0.0),
                        "train_ranges": self.high_z_model.metadata.get("train_ranges", {}),
                        "old_au_midrod_ranges": self.high_z_model.metadata.get("old_au_midrod_ranges", {}),
                        "old_au_full_ranges": self.high_z_model.metadata.get("old_au_full_ranges", {}),
                        "boundary_model_available": bool(route_details.get("boundary_model_available")),
                    },
                }
            )
        if one:
            return pred.reshape(-1), route, base_pred.reshape(-1)
        return np.asarray(pred, dtype=float).reshape(-1), route, base_pred

    @staticmethod
    def _normalize_data_source(value: object, default: str = "current") -> str:
        text = str(value or "").strip().lower()
        aliases = {
            "": default,
            "default": default,
            "current": "current",
            "new": "current",
            "new_upload": "current",
            "au2": "current",
            "old": "old_au",
            "old_au": "old_au",
            "fixed": "old_au",
            "fixed_benchmark": "old_au",
            "benchmark": "old_au",
            "auto": default,
        }
        return aliases.get(text, default)

    @staticmethod
    def _infer_benchmark_data_source(path: Path, explicit: object = None) -> str:
        if explicit not in (None, ""):
            return FreePathWebApp._normalize_data_source(explicit, default="current")
        name = path.name.lower()
        full = str(path).lower()
        if name == "data.txt":
            return "old_au"
        if "old_au" in full or "unified_permanent_benchmark" in full or "unified_permanent_extrapolation" in full:
            return "old_au"
        return "current"

    def _load_summary_for_path(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _summary_with_routed_high_z_metrics(self, summary: dict | None, high_z_metrics_path: Path) -> dict | None:
        if not summary:
            return summary
        high_z_summary = self._load_summary_for_path(high_z_metrics_path)
        if not high_z_summary:
            return summary
        routed_benchmark = high_z_summary.get("routed_benchmark_metrics")
        routed_extrapolation = high_z_summary.get("routed_extrapolation_metrics")
        if not routed_benchmark:
            return summary
        merged = copy.deepcopy(summary)
        merged["base_unified_benchmark_metrics"] = merged.get("benchmark_metrics")
        merged["base_unified_extrapolation_benchmark_metrics"] = merged.get("extrapolation_benchmark_metrics")
        merged["benchmark_metrics"] = routed_benchmark
        if routed_extrapolation:
            merged["extrapolation_benchmark_metrics"] = routed_extrapolation
        merged["high_z_routed_metrics_applied"] = True
        merged["high_z_routed_metrics_path"] = str(high_z_metrics_path)
        return merged

    @staticmethod
    def _summary_for_api(summary: dict | None) -> dict | None:
        if not summary:
            return summary
        out = copy.deepcopy(summary)
        for extension in out.get("boundary_extensions", []) or []:
            rows = extension.pop("rows", None)
            if rows is not None:
                extension["row_count"] = len(rows)
        return out

    def _benchmark_metrics_payload(self) -> dict:
        metadata_summary = self._summary_with_routed_high_z_metrics(self.metadata, HIGH_Z_METADATA_PATH)
        metrics = (metadata_summary or {}).get("benchmark_metrics")
        if metrics:
            return metrics
        if self.artifact_path.resolve() == STAGED_ARTIFACT.resolve():
            summary = self._load_summary_for_path(STAGED_METRICS)
        else:
            summary = self._load_summary_for_path(ACTIVE_METRICS)
        summary = self._summary_with_routed_high_z_metrics(summary, HIGH_Z_METADATA_PATH)
        return (summary or {}).get("benchmark_metrics", {})

    def _element_map(self) -> dict:
        if "elements" in self.metadata:
            return self.metadata["elements"]
        if "sources" in self.metadata:
            return self.metadata["sources"]
        elements = {}
        for name, info in self.metadata.get("datasets", {}).items():
            z_value = float(info.get("Z", info.get("source_id")))
            label = f"Z_{z_value:g}"
            elements[label] = {
                "Z": z_value,
                "legacy_name": name,
                "ranges": info["ranges"],
            }
        return elements

    def _element_info(self, z_value: float) -> dict | None:
        label = f"Z_{float(z_value):g}"
        return self._element_map().get(label)

    def _trained_z_values(self) -> set[float]:
        return {float(info["Z"]) for info in self._element_map().values()}

    def _require_trained_z(self, z_values: np.ndarray | list | float) -> None:
        trained = self._trained_z_values()
        arr = np.asarray(z_values, dtype=float).reshape(-1)
        unknown = sorted({float(v) for v in arr if float(v) not in trained})
        if unknown:
            trained_text = ", ".join(f"{value:g}" for value in sorted(trained))
            unknown_text = ", ".join(f"{value:g}" for value in unknown)
            raise ValueError(f"现场评测只允许已训练元素；未训练 Z: {unknown_text}；已训练 Z: {trained_text}")

    @staticmethod
    def _normalize_prediction_mode(value: object) -> str:
        text = str(value or "model").strip().lower()
        aliases = {
            "model": "model",
            "model_only": "model",
            "only_model": "model",
            "hybrid": "hybrid",
            "mixed": "hybrid",
            "truth_first": "hybrid",
            "real_first": "hybrid",
            "table_first": "hybrid",
            "truth": "truth",
            "truth_only": "truth",
            "table": "truth",
            "table_truth": "truth",
        }
        if text not in aliases:
            raise ValueError("自由程来源策略必须是 model、hybrid 或 truth")
        return aliases[text]

    @staticmethod
    def _parse_float_list(value: object) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"all", "*"}:
                return None
            raw_values = text.replace(",", " ").split()
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            raw_values = list(value)
        else:
            raw_values = [value]
        out = []
        for item in raw_values:
            if str(item).strip() == "":
                continue
            out.append(float(item))
        return out

    def _prediction_policy(self, payload: dict) -> dict:
        mode = self._normalize_prediction_mode(
            payload.get("predictionMode", payload.get("prediction_mode", payload.get("sourceMode")))
        )
        explicit = self._parse_float_list(payload.get("modelElements", payload.get("model_elements")))
        trained = sorted(self._trained_z_values())
        model_elements = trained if explicit is None else explicit
        model_elements = sorted({float(value) for value in model_elements})
        return {
            "mode": mode,
            "mode_label": PREDICTION_MODE_LABELS[mode],
            "model_elements": model_elements,
            "model_element_labels": [f"Z={value:g}" for value in model_elements],
        }

    @staticmethod
    def _model_allowed_for_z(z_value: float, policy: dict) -> bool:
        return any(np.isclose(float(z_value), float(item), rtol=0.0, atol=1e-8) for item in policy["model_elements"])

    def _source_control_payload(self) -> dict:
        elements = []
        simulator_info = {int(z): info for z, info in SIMULATOR_ELEMENTS.items()}
        for value in sorted(self._trained_z_values()):
            rounded = int(round(value))
            symbol = simulator_info.get(rounded, {}).get("symbol")
            elements.append({"Z": value, "label": f"Z={value:g}" + (f" {symbol}" if symbol else "")})
        return {
            "modes": [
                {"value": key, "label": label}
                for key, label in PREDICTION_MODE_LABELS.items()
            ],
            "default_mode": "model",
            "model_elements": elements,
            "default_model_elements": [item["Z"] for item in elements],
            "reminders": [
                "混合模式只在标准数据表中精确命中 Z|rod|tep|tgama 时直接使用已计算 lnu；没有命中才按元素开关使用模型。",
                "元素开关只控制模型兜底；表中已有真值仍可在混合/真值模式下使用。",
                "批量预测不会逐点调用 SNOP_op_Ross_single 真实程序；需要现场重算时使用单点预测的真实程序开关。",
                "输入坐标固定为 Z rod tep tgama，其中 rod/tep/tgama 是 log10 后的模型坐标。",
            ],
        }

    def _default_flash_control(self) -> dict:
        trained = sorted(self._trained_z_values())
        preferred = [value for value in trained if np.isclose(value, 4.0, rtol=0.0, atol=1e-8)]
        return {
            "enabled": False,
            "predictionMode": "hybrid",
            "modelElements": preferred or trained,
            "updated_at": None,
        }

    def flash_control_payload(self) -> dict:
        with self._flash_control_lock:
            state = copy.deepcopy(self._flash_control)
        source_controls = self._source_control_payload()
        mode = self._normalize_prediction_mode(state.get("predictionMode", "hybrid"))
        model_elements = sorted({float(value) for value in state.get("modelElements", [])})
        return {
            "enabled": bool(state.get("enabled", False)),
            "predictionMode": mode,
            "predictionModeLabel": PREDICTION_MODE_LABELS[mode],
            "modelElements": model_elements,
            "modelElementLabels": [f"Z={value:g}" for value in model_elements],
            "updated_at": state.get("updated_at"),
            "control_url": "/api/flash/control",
            "batch_predict_url": "/api/predict/batch",
            "modes": source_controls["modes"],
            "model_elements": source_controls["model_elements"],
            "usage": {
                "coordinate_format": "Z rod tep tgama，其中 rod=log10(rho[g/cm^3])，tep=log10(Te[eV])，tgama=log10(Trad[eV])",
                "opacity_formula": "opacityRO[cm^2/g] = 1 / (rho[g/cm^3] * lnu[cm])",
                "recommended_call": "FLASH 侧按时间步/opacity 调用批量收集点，再 POST /api/predict/batch；不要逐 cell 逐能群调用 HTTP。",
                "fallback": "enabled=false、元素未启用、点超出训练范围、接口失败或材料未训练时回退原 IONMIX/查表 opacity。",
            },
        }

    def update_flash_control(self, payload: dict) -> dict:
        trained = sorted(self._trained_z_values())
        trained_set = {float(value) for value in trained}
        with self._flash_control_lock:
            state = copy.deepcopy(self._flash_control)
            if "enabled" in payload:
                state["enabled"] = bool(payload.get("enabled"))
            mode_value = payload.get("predictionMode", payload.get("prediction_mode", payload.get("sourceMode")))
            if mode_value is not None:
                state["predictionMode"] = self._normalize_prediction_mode(mode_value)
            if "modelElements" in payload or "model_elements" in payload:
                explicit = self._parse_float_list(payload.get("modelElements", payload.get("model_elements")))
                elements = trained if explicit is None else explicit
                normalized = sorted({float(value) for value in elements})
                unknown = [value for value in normalized if value not in trained_set]
                if unknown:
                    trained_text = ", ".join(f"{value:g}" for value in trained)
                    unknown_text = ", ".join(f"{value:g}" for value in unknown)
                    raise ValueError(f"FLASH 模型元素必须来自已训练元素；未训练 Z: {unknown_text}；已训练 Z: {trained_text}")
                state["modelElements"] = normalized
            state["updated_at"] = time.time()
            self._flash_control = state
        return self.flash_control_payload()

    def _range_warnings_for_point(self, z_value: float, point: np.ndarray) -> list[str]:
        warnings = []
        element_info = self._element_info(z_value)
        if element_info and "ranges" in element_info:
            ranges = element_info["ranges"]
            for key, value in zip(("rod", "tep", "tgama"), point):
                low, high = ranges[key]
                if float(value) < low or float(value) > high:
                    warnings.append(f"{key} 超出 Z={z_value:g} 训练范围 [{low:.6g}, {high:.6g}]")
        elif element_info is None:
            trained = ", ".join(label.replace("Z_", "") for label in sorted(self._element_map()))
            warnings.append(f"Z={z_value:g} 当前没有直接训练样本；这是跨原子序数预测，已训练 Z: {trained}")
        return warnings

    @staticmethod
    def _prediction_error(pred_lnu: float, pred_log: float, true_lnu: float) -> dict:
        true_log = float(np.log10(true_lnu))
        log_error = float(pred_log - true_log)
        return {
            "log10_error": log_error,
            "abs_log10_error": abs(log_error),
            "relative_percent": float(abs(pred_lnu - true_lnu) / max(abs(true_lnu), 1e-300) * 100.0),
            "smape_percent": float(2.0 * abs(pred_lnu - true_lnu) / max(abs(pred_lnu) + abs(true_lnu), 1e-300) * 100.0),
            "factor_error": float(10.0 ** abs(log_error)),
        }

    def _standard_data_paths(self) -> list[Path]:
        paths = []
        for item in self.metadata.get("training_files", {}).values():
            path = Path(str(item.get("path", ""))).expanduser()
            if path:
                paths.append(path if path.is_absolute() else (ROOT / path).resolve())
        configured = Path(str(self.metadata.get("standard_training_data_path", STANDARD_TRAINING_DATA_PATH))).expanduser()
        paths.append(configured if configured.is_absolute() else (ROOT / configured).resolve())
        # Older artifacts embed the original machine's absolute training paths.
        # Always keep the portable canonical table as a final local fallback.
        paths.append(STANDARD_TRAINING_DATA_PATH)
        out = []
        seen = set()
        for path in paths:
            resolved = path if path.is_absolute() else (ROOT / path).resolve()
            if resolved not in seen and resolved.exists():
                out.append(resolved)
                seen.add(resolved)
        return out

    @staticmethod
    def _read_key_file(path: Path) -> set[str]:
        if not path.exists():
            return set()
        return set(path.read_text(encoding="utf-8").splitlines())

    def _load_truth_cache(self) -> dict:
        with self._truth_lock:
            if self._truth_cache is not None:
                return self._truth_cache
            truth = {}
            source_by_key = {}
            paths = self._standard_data_paths()
            for path in paths:
                arr = np.loadtxt(path, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim != 2 or arr.shape[1] != 5:
                    continue
                for row in arr:
                    key = standard_row_key(row[0], row[1:4])
                    truth[key] = float(row[4])
                    source_by_key[key] = str(path)
            self._truth_cache = {
                "truth": truth,
                "source_by_key": source_by_key,
                "benchmark_keys": self._read_key_file(BENCHMARK_KEYS_PATH),
                "extrapolation_keys": self._read_key_file(EXTRAPOLATION_KEYS_PATH),
                "paths": [str(path) for path in paths],
            }
            return self._truth_cache

    def _membership_and_truth(
        self,
        z_value: float,
        point: np.ndarray,
        pred_lnu: float | None = None,
        pred_log: float | None = None,
    ) -> dict:
        key = standard_row_key(z_value, point)
        t0 = time.perf_counter()
        cache = self._load_truth_cache()
        true_lnu = cache["truth"].get(key)
        lookup_ms = (time.perf_counter() - t0) * 1000.0
        in_random_benchmark = key in cache["benchmark_keys"]
        in_extrapolation_benchmark = key in cache["extrapolation_keys"]
        exists = true_lnu is not None
        in_training = bool(exists and not in_random_benchmark and not in_extrapolation_benchmark)
        if in_training:
            label = "训练集已存在"
        elif in_random_benchmark:
            label = "永久随机测试集，未参与训练"
        elif in_extrapolation_benchmark:
            label = "永久两端连续外推测试集，未参与训练"
        elif exists:
            label = "标准数据已存在，但不在当前训练集"
        else:
            label = "标准数据中不存在该精确点"

        membership = {
            "key": key,
            "exists_in_standard_data": bool(exists),
            "in_training_set": in_training,
            "in_random_benchmark": bool(in_random_benchmark),
            "in_extrapolation_benchmark": bool(in_extrapolation_benchmark),
            "label": label,
        }
        reference = {
            "found": False,
            "requested": False,
            "invoked": False,
            "lookup_ms": lookup_ms,
            "source": "SNOP_op_Ross_single",
            "source_label": "当前目录单点不透明度程序",
            "reason": "未勾选“同时调用真实程序”",
        }
        table_truth = {
            "found": bool(exists),
            "source": "training_table",
            "source_label": "训练/标准数据表中的 lnu",
            "source_path": cache["source_by_key"].get(key),
            "lnu": true_lnu,
            "log10_lnu": float(np.log10(true_lnu)) if true_lnu is not None and true_lnu > 0 else None,
            "lookup_ms": lookup_ms,
            "note": "这是数据表保存值；若原始表坐标或 lnu 被截断，它不一定逐位等于现场程序重算值。",
        }
        if true_lnu is not None and pred_lnu is not None and pred_log is not None:
            table_truth["error"] = self._prediction_error(float(pred_lnu), float(pred_log), float(true_lnu))
        return {"data_membership": membership, "table_truth": table_truth, "reference_truth": reference}

    @staticmethod
    def _simulator_executable(z_value: float) -> Path:
        return SIMULATOR_DIR / f"Z_{float(z_value):g}" / "SNOP_op_Ross_single"

    @staticmethod
    def _simulator_source_label(z_value: float) -> str:
        rounded = int(round(float(z_value)))
        if abs(float(z_value) - rounded) > 1e-9 or rounded not in SIMULATOR_ELEMENTS:
            return "单点不透明度程序"
        return simulator_source_label(rounded)

    def _ensure_simulator_executable(self, z_value: float, progress_callback=None) -> tuple[Path, str | None]:
        exe = self._simulator_executable(z_value)
        rounded = int(round(float(z_value)))
        if abs(float(z_value) - rounded) > 1e-9:
            return exe, f"Z={z_value:g} 不是整数原子序数，无法选择单点真实程序"
        if rounded not in SIMULATOR_ELEMENTS:
            return exe, f"当前真实程序模板尚不支持 Z={rounded:g}"
        with self._simulator_build_lock:
            exe = self._simulator_executable(rounded)
            if is_build_current(rounded):
                return exe, None
            if progress_callback:
                progress_callback(f"正在为 Z={rounded:g} 自动构建真实单点程序", 48.0, "simulator_build")
            try:
                built = build_one(rounded, force=False)
            except Exception as exc:
                return exe, f"自动构建 Z={rounded:g} 真实程序失败: {exc}"
            if not built.exists():
                return built, f"自动构建 Z={rounded:g} 后未找到可执行文件"
            return built, None

    @staticmethod
    def _snap_low_z_grid_point(z_value: float, point: np.ndarray) -> tuple[np.ndarray, dict]:
        rounded = int(round(float(z_value)))
        grid = LOW_Z_SIMULATOR_GRID.get(rounded)
        if grid is None:
            return point.astype(float), {"applied": False}

        exact_values = []
        grid_indices = {}
        input_log = {}
        simulator_log = {}
        for idx, name in enumerate(("rod", "tep", "tgama")):
            axis = grid[name]
            displayed_axis = np.round(axis, LOW_Z_GRID_DISPLAY_DECIMALS)
            distances = np.abs(displayed_axis - float(point[idx]))
            best = int(np.argmin(distances))
            if float(distances[best]) > LOW_Z_GRID_SNAP_TOL:
                return point.astype(float), {"applied": False}
            exact = float(axis[best])
            exact_values.append(exact)
            grid_indices[name] = best + 1
            input_log[name] = float(point[idx])
            simulator_log[name] = exact

        snapped = np.array(exact_values, dtype=float)
        if np.allclose(snapped, point.astype(float), rtol=0.0, atol=1.0e-12):
            return snapped, {"applied": False}
        return snapped, {
            "applied": True,
            "reason": "低 Z 表格只保存 3 位小数；真实程序调用已还原原始 log 网格坐标",
            "input_log": input_log,
            "simulator_log": simulator_log,
            "grid_indices": grid_indices,
        }

    @staticmethod
    def _simulator_physical_point(z_value: float, point: np.ndarray) -> tuple[np.ndarray, dict]:
        simulator_point, coordinate_snap = FreePathWebApp._snap_low_z_grid_point(z_value, point)
        if np.any(simulator_point > 300.0):
            raise ValueError("标准坐标过大，无法安全还原为物理坐标 10^x")
        return np.power(10.0, simulator_point.astype(float)), coordinate_snap

    @staticmethod
    def _parse_simulator_result(stdout: str) -> dict:
        for line in reversed(stdout.splitlines()):
            parts = line.strip().split()
            if parts and parts[0] == "SNOP_SINGLE_RESULT":
                if len(parts) < 10:
                    raise ValueError("真实程序结果行字段不足")
                return {
                    "Z": float(parts[1]),
                    "physical_rod": float(parts[2]),
                    "physical_tep": float(parts[3]),
                    "physical_tgama": float(parts[4]),
                    "lnu": float(parts[5]),
                    "log10_lnu": float(parts[6]),
                    "program_elapsed_s": float(parts[7]),
                    "neval": int(float(parts[8])),
                    "fail": int(float(parts[9])),
                }
        raise ValueError("真实程序没有输出 SNOP_SINGLE_RESULT")

    def _run_single_point_simulator(
        self,
        z_value: float,
        point: np.ndarray,
        pred_lnu: float,
        pred_log: float,
        progress_callback=None,
    ) -> dict:
        source_label = self._simulator_source_label(z_value)
        exe, missing_reason = self._ensure_simulator_executable(z_value, progress_callback)
        if missing_reason or not exe.exists():
            return {
                "found": False,
                "requested": True,
                "invoked": False,
                "source": "SNOP_op_Ross_single",
                "source_label": source_label,
                "reason": missing_reason or f"未找到 Z={z_value:g} 的单点程序，请先运行 python build_single_point_simulators.py --z {int(z_value)}",
                "program_path": str(exe),
            }
        physical, coordinate_snap = self._simulator_physical_point(z_value, point)
        command = [str(exe), *(f"{float(v):.17g}" for v in physical)]
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1"),
                "ROSS_USE_GPU": env.get("ROSS_USE_GPU", "1"),
                "ROSS_CUBAACCELS": env.get("ROSS_CUBAACCELS", "2"),
                "ROSS_CUBAACCELMAX": env.get("ROSS_CUBAACCELMAX", "1000000"),
                "CUBACORES": env.get("CUBACORES", "0"),
                "CUBAVERBOSE": env.get("CUBAVERBOSE", "0"),
            }
        )
        if progress_callback:
            progress_callback("正在启动真实不透明度程序", 52.0, "simulator")
        t0 = time.perf_counter()
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if progress_callback:
                progress_callback(f"真实程序已启动，PID={process.pid}", 58.0, "simulator")
            stdout, stderr = process.communicate(timeout=SIMULATOR_TIMEOUT_SECONDS)
            wall_ms = (time.perf_counter() - t0) * 1000.0
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                stdout, stderr = process.communicate()
                wall_ms = (time.perf_counter() - t0) * 1000.0
            else:
                stdout, stderr = "", ""
                wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "found": False,
                "requested": True,
                "invoked": process is not None,
                "source": "SNOP_op_Ross_single",
                "source_label": source_label,
                "reason": f"真实程序超过 {SIMULATOR_TIMEOUT_SECONDS:g}s 未完成",
                "program_path": str(exe),
                "command": command,
                "pid": process.pid if process is not None else None,
                "wall_ms": wall_ms,
                "physical_input": {
                    "rod": float(physical[0]),
                    "tep": float(physical[1]),
                    "tgama": float(physical[2]),
                },
                "coordinate_snap": coordinate_snap,
                "stdout_tail": stdout[-1000:],
                "stderr_tail": stderr[-1000:],
            }
        except OSError as exc:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "found": False,
                "requested": True,
                "invoked": False,
                "source": "SNOP_op_Ross_single",
                "source_label": source_label,
                "reason": f"真实程序启动失败: {exc}",
                "program_path": str(exe),
                "command": command,
                "wall_ms": wall_ms,
            }
        returncode = process.returncode if process is not None else -1
        if returncode != 0:
            return {
                "found": False,
                "requested": True,
                "invoked": True,
                "source": "SNOP_op_Ross_single",
                "source_label": source_label,
                "reason": f"真实程序退出码 {returncode}: {(stderr or stdout)[-1000:]}",
                "program_path": str(exe),
                "command": command,
                "pid": process.pid if process is not None else None,
                "wall_ms": wall_ms,
            }
        try:
            parsed = self._parse_simulator_result(stdout)
        except ValueError as exc:
            return {
                "found": False,
                "requested": True,
                "invoked": True,
                "source": "SNOP_op_Ross_single",
                "source_label": source_label,
                "reason": str(exc),
                "program_path": str(exe),
                "command": command,
                "pid": process.pid if process is not None else None,
                "wall_ms": wall_ms,
                "stdout_tail": stdout[-1000:],
                "stderr_tail": stderr[-1000:],
            }
        true_lnu = float(parsed["lnu"])
        true_log = float(parsed["log10_lnu"])
        log_error = float(pred_log - true_log)
        relative_percent = float(abs(pred_lnu - true_lnu) / max(abs(true_lnu), 1e-300) * 100.0)
        smape_percent = float(2.0 * abs(pred_lnu - true_lnu) / max(abs(pred_lnu) + abs(true_lnu), 1e-300) * 100.0)
        return {
            "found": True,
            "requested": True,
            "invoked": True,
            "source": "SNOP_op_Ross_single",
            "source_label": source_label,
            "program_path": str(exe),
            "command": command,
            "pid": process.pid if process is not None else None,
            "lnu": true_lnu,
            "log10_lnu": true_log,
            "wall_ms": wall_ms,
            "program_elapsed_s": float(parsed["program_elapsed_s"]),
            "neval": int(parsed["neval"]),
            "fail": int(parsed["fail"]),
            "physical_input": {
                "rod": float(parsed["physical_rod"]),
                "tep": float(parsed["physical_tep"]),
                "tgama": float(parsed["physical_tgama"]),
            },
            "coordinate_snap": coordinate_snap,
            "error": {
                "log10_error": log_error,
                "abs_log10_error": abs(log_error),
                "relative_percent": relative_percent,
                "smape_percent": smape_percent,
                "factor_error": float(10.0 ** abs(log_error)),
            },
        }

    def _resolve_training_data_path(self, path_text: str | None) -> Path:
        text = str(path_text or STANDARD_TRAINING_DATA_PATH).strip()
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        else:
            path = path.resolve()
        root = ROOT.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"训练数据路径必须在项目目录内: {root}") from exc
        return path

    @staticmethod
    def _parse_standard_training_text(
        text: str,
        remove_nonpositive: bool = False,
        legacy_z: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        rows = []
        nonpositive_rows = 0
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            lowered = [part.lower() for part in parts[:5]]
            if lowered == [name.lower() for name in STANDARD_TRAINING_COLUMNS]:
                continue
            if [part.lower() for part in parts[:4]] == ["rod", "tep", "tgama", "lnu"]:
                continue
            if len(parts) < 4:
                raise ValueError(f"第 {lineno} 行不足 4 列，需要 5 列标准格式或 4 列老格式")
            try:
                if len(parts) >= 5:
                    row = [float(parts[i]) for i in range(5)]
                else:
                    if legacy_z is None or not np.isfinite(float(legacy_z)):
                        raise ValueError("4 列训练数据需要提供对应的 Z")
                    row = [float(legacy_z)] + [float(parts[i]) for i in range(4)]
            except ValueError as exc:
                raise ValueError(f"第 {lineno} 行包含非数值内容") from exc
            if row[4] <= 0.0:
                nonpositive_rows += 1
                if remove_nonpositive:
                    continue
            rows.append(row)
        if not rows:
            raise ValueError("没有找到有效训练数据行")
        arr = np.asarray(rows, dtype=float)
        if np.any(~np.isfinite(arr)):
            raise ValueError("训练数据包含 NaN 或 Inf")
        if np.any(arr[:, 4] <= 0.0):
            raise ValueError(f"训练目标 lnu 必须为正数；检测到 {nonpositive_rows} 行 lnu<=0，可确认后剔除这些行再导入")
        return arr, {
            "parsed_rows": int(arr.shape[0] + nonpositive_rows),
            "nonpositive_rows_removed": int(nonpositive_rows if remove_nonpositive else 0),
        }

    @staticmethod
    def _load_training_rows(path: Path) -> np.ndarray:
        if not path.exists() or path.stat().st_size == 0:
            return np.empty((0, 5), dtype=float)
        arr = np.loadtxt(path, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(f"{path} 不是标准五列训练数据")
        if np.any(~np.isfinite(arr)) or np.any(arr[:, 4] <= 0.0):
            raise ValueError(f"{path} 包含非法训练数据")
        return arr

    @staticmethod
    def _dedupe_training_rows(rows: np.ndarray) -> np.ndarray:
        if rows.size == 0:
            return np.empty((0, 5), dtype=float)
        by_key = {}
        for row in rows:
            by_key[standard_row_key(row[0], row[1:4])] = np.asarray(row, dtype=float)
        return np.vstack(list(by_key.values()))

    @staticmethod
    def _save_training_rows(path: Path, rows: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        np.savetxt(tmp, rows, fmt="%.12e", header=" ".join(STANDARD_TRAINING_COLUMNS))
        tmp.replace(path)

    def import_training_data(self, payload: dict) -> dict:
        path = self._resolve_training_data_path(payload.get("path"))
        mode = str(payload.get("mode", "append")).strip().lower()
        if mode not in {"append", "replace", "replace_element"}:
            raise ValueError("导入方式必须是 append、replace 或 replace_element")
        remove_nonpositive = bool(payload.get("removeNonpositive", False))
        legacy_z_text = str(payload.get("legacyZ", "")).strip()
        legacy_z = float(legacy_z_text) if legacy_z_text else None
        incoming, incoming_stats = self._parse_standard_training_text(
            str(payload.get("text", "")),
            remove_nonpositive=remove_nonpositive,
            legacy_z=legacy_z,
        )
        previous = self._load_training_rows(path)
        if mode == "append":
            combined = np.vstack([previous, incoming]) if previous.size else incoming
            mode_label = "追加合并"
            raw_total = int(previous.shape[0] + incoming.shape[0])
        elif mode == "replace":
            combined = incoming
            mode_label = "覆盖替换"
            raw_total = int(incoming.shape[0])
        else:
            z_values = sorted({float(value) for value in incoming[:, 0]})
            if not z_values:
                raise ValueError("替换元素模式没有找到 Z")
            keep = np.ones(previous.shape[0], dtype=bool)
            for z_value in z_values:
                keep &= ~np.isclose(previous[:, 0], z_value, rtol=0.0, atol=1e-8)
            kept_previous = previous[keep]
            combined = np.vstack([kept_previous, incoming]) if kept_previous.size else incoming
            mode_label = "替换同 Z 元素"
            raw_total = int(kept_previous.shape[0] + incoming.shape[0])
        final_rows = self._dedupe_training_rows(combined)

        backup_path = None
        if path.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
            shutil.copy2(path, backup)
            backup_path = str(backup)
        self._save_training_rows(path, final_rows)
        with self._truth_lock:
            self._truth_cache = None

        return {
            "ok": True,
            "path": str(path),
            "mode": mode,
            "mode_label": mode_label,
            "previous_rows": int(previous.shape[0]),
            "imported_rows": int(incoming.shape[0]),
            "parsed_rows": int(incoming_stats["parsed_rows"]),
            "nonpositive_rows_removed": int(incoming_stats["nonpositive_rows_removed"]),
            "final_rows": int(final_rows.shape[0]),
            "duplicate_rows_removed": int(raw_total - final_rows.shape[0]),
            "backup_path": backup_path,
            "standard_training_format": " ".join(STANDARD_TRAINING_COLUMNS),
        }

    def export_training_data_path(self, query: dict) -> Path:
        value = ""
        if query.get("path"):
            value = query["path"][0]
        path = self._resolve_training_data_path(value)
        if not path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {path}")
        return path

    @staticmethod
    def _high_z_paths_from_metadata(metadata_path: Path, default_model_path: Path) -> list[Path]:
        paths = [Path(default_model_path)]
        if not metadata_path.exists():
            return paths
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return paths
        for key in ("model_path", "boundary_model_path"):
            if metadata.get(key):
                paths.append(Path(str(metadata[key])))
        for item in metadata.get("model_paths", []) or []:
            paths.append(Path(str(item)))
        for spec in metadata.get("model_specs", []) or []:
            if spec.get("path"):
                paths.append(Path(str(spec["path"])))
        resolved = []
        for item in paths:
            path = item if item.is_absolute() else metadata_path.parent / item
            if path not in resolved:
                resolved.append(path)
        return resolved

    @staticmethod
    def _staged_high_z_destination(path: Path) -> Path:
        if path == STAGED_HIGH_Z_MODEL:
            return HIGH_Z_MODEL_PATH
        name = path.name.replace(STAGED_HIGH_Z_MODEL.stem, HIGH_Z_MODEL_PATH.stem, 1)
        return HIGH_Z_MODEL_PATH.with_name(name)

    @staticmethod
    def _rewrite_high_z_metadata_paths(metadata: dict) -> dict:
        def rewrite(value: str) -> str:
            path = Path(value)
            if path.name.startswith(STAGED_HIGH_Z_MODEL.stem):
                return str(FreePathWebApp._staged_high_z_destination(path))
            return str(path)

        metadata = json.loads(json.dumps(metadata))
        metadata["model_path"] = str(HIGH_Z_MODEL_PATH)
        if metadata.get("model_paths"):
            metadata["model_paths"] = [rewrite(item) for item in metadata["model_paths"]]
        if metadata.get("model_specs"):
            for spec in metadata["model_specs"]:
                if spec.get("path"):
                    spec["path"] = rewrite(spec["path"])
        if metadata.get("boundary_model_path"):
            metadata["boundary_model_path"] = rewrite(metadata["boundary_model_path"])
        params = metadata.get("params", {})
        for key in ("members", "auxiliary_members"):
            for item in params.get(key, []) or []:
                if item.get("model_path"):
                    item["model_path"] = rewrite(item["model_path"])
        metadata["base_model_path"] = str(ACTIVE_ARTIFACT)
        return metadata

    @staticmethod
    def _remove_staged_high_z_files() -> None:
        for path in OUTPUT_DIR.glob(f"{STAGED_HIGH_Z_MODEL.stem}*"):
            if path.is_file():
                path.unlink()
        if STAGED_HIGH_Z_METRICS.exists():
            STAGED_HIGH_Z_METRICS.unlink()

    def _simulator_inventory(self) -> dict:
        discovered = discover_data_elements()
        data_elements = []
        for z, paths in sorted(discovered.items()):
            info = SIMULATOR_ELEMENTS.get(z, {})
            exe = self._simulator_executable(z)
            data_elements.append(
                {
                    "Z": z,
                    "symbol": info.get("symbol", f"Z{z}"),
                    "data_files": [str(path) for path in paths],
                    "program_path": str(exe),
                    "built": is_build_current(z),
                    "source_label": self._simulator_source_label(z),
                }
            )
        available_z = []
        for path in sorted(SIMULATOR_DIR.glob("Z_*/SNOP_op_Ross_single")):
            try:
                z = int(float(path.parent.name.removeprefix("Z_")))
                if is_build_current(z):
                    available_z.append(z)
            except ValueError:
                continue
        supported_elements = []
        for z, info in sorted(SIMULATOR_ELEMENTS.items()):
            exe = self._simulator_executable(z)
            supported_elements.append(
                {
                    "Z": z,
                    "symbol": info.get("symbol", f"Z{z}"),
                    "program_path": str(exe),
                    "built": is_build_current(z),
                    "source_label": self._simulator_source_label(z),
                }
            )
        supported_z = [item["Z"] for item in supported_elements]
        return {
            "data_elements": data_elements,
            "supported_elements": supported_elements,
            "supported_z": supported_z,
            "available_z": sorted(set(available_z)),
            "missing_data_z": [item["Z"] for item in data_elements if not item["built"]],
            "missing_supported_z": [item["Z"] for item in supported_elements if not item["built"]],
        }

    def _cleanup_prediction_jobs_locked(self) -> None:
        if len(self._predict_jobs) <= 40:
            return
        finished = [
            (job.get("finished_at") or job.get("started_at") or 0.0, job_id)
            for job_id, job in self._predict_jobs.items()
            if job.get("status") in {"succeeded", "failed"}
        ]
        for _, job_id in sorted(finished)[: max(0, len(self._predict_jobs) - 40)]:
            self._predict_jobs.pop(job_id, None)

    def _update_prediction_job(self, job_id: str, **updates) -> None:
        with self._predict_lock:
            job = self._predict_jobs[job_id]
            old_phase = job.get("phase")
            job.update(updates)
            if updates.get("phase") and updates.get("phase") != old_phase:
                job["phase_started_at"] = time.time()

    def _prediction_job_snapshot(self, job_id: str) -> dict:
        with self._predict_lock:
            if job_id not in self._predict_jobs:
                raise KeyError("预测任务不存在或已过期")
            job = dict(self._predict_jobs[job_id])
        now = time.time()
        started_at = job.get("started_at")
        if started_at:
            job["elapsed_ms"] = (now - float(started_at)) * 1000.0
        if job.get("status") == "running" and job.get("phase") == "simulator":
            phase_started = float(job.get("phase_started_at") or started_at or now)
            fraction = min(1.0, max(0.0, (now - phase_started) / SIMULATOR_TIMEOUT_SECONDS))
            job["progress"] = max(float(job.get("progress", 58.0)), min(92.0, 58.0 + 34.0 * fraction))
            job["message"] = "真实程序运行中，正在等待 SNOP_op_Ross_single 返回"
        return job

    def _run_prediction_job(self, job_id: str, payload: dict) -> None:
        def progress(message: str, percent: float, phase: str) -> None:
            self._update_prediction_job(
                job_id,
                status="running",
                phase=phase,
                progress=float(percent),
                message=message,
            )

        try:
            progress("预测任务已开始", 12.0, "queued")
            result = self.predict(payload, progress_callback=progress)
            self._update_prediction_job(
                job_id,
                status="succeeded",
                phase="done",
                progress=100.0,
                message="预测完成",
                result=result,
                finished_at=time.time(),
                runSimulation=bool(payload.get("runSimulation", False)),
            )
        except Exception as exc:
            self._update_prediction_job(
                job_id,
                status="failed",
                phase="failed",
                progress=100.0,
                message="预测失败",
                error=str(exc),
                finished_at=time.time(),
                runSimulation=bool(payload.get("runSimulation", False)),
            )

    def start_prediction(self, payload: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        job = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress": 2.0,
            "message": "预测任务已进入队列",
            "started_at": now,
            "phase_started_at": now,
            "finished_at": None,
            "runSimulation": bool(payload.get("runSimulation", False)),
        }
        with self._predict_lock:
            self._predict_jobs[job_id] = job
            self._cleanup_prediction_jobs_locked()
        thread = threading.Thread(target=self._run_prediction_job, args=(job_id, payload), daemon=True)
        thread.start()
        return self._prediction_job_snapshot(job_id)

    def prediction_status(self, query: dict) -> dict:
        values = query.get("id") or query.get("job_id")
        if not values:
            raise ValueError("缺少预测任务 id")
        return self._prediction_job_snapshot(str(values[0]))

    @staticmethod
    def _load_json_if_exists(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def capacitor_payload(self) -> dict:
        return {
            "title": "脉冲电容器寿命预测",
            "app_dir": str(CAPACITOR_APP_DIR),
            "port": CAPACITOR_SERVICE_PORT,
            "source": f"便携包内独立网页服务 {CAPACITOR_APP_DIR}",
            "data_note": "内置 CSV 是 Sandia/OEDI 钽电容高温老化真实测量数据，不是本项目不透明度程序计算表。",
        }

    def _flash_run_dirs(self) -> list[Path]:
        preferred = [FLASH_ROOT / name for name in FLASH_RUN_DIR_NAMES]
        discovered: list[Path] = []
        try:
            for child in FLASH_ROOT.iterdir():
                if not child.is_dir():
                    continue
                if (child / "flash.par").is_file() or (child / "flash4").is_file():
                    discovered.append(child)
        except OSError:
            discovered = []

        ordered: list[Path] = []
        seen: set[Path] = set()
        for path in preferred + sorted(discovered, key=lambda item: item.name.lower()):
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in seen or not path.is_dir():
                continue
            seen.add(key)
            ordered.append(path)
        return ordered

    def _resolve_flash_run_dir(self, value: object = None) -> Path:
        run_dirs = self._flash_run_dirs()
        if not run_dirs:
            raise FileNotFoundError(f"未找到 FLASH 运行目录: {FLASH_ROOT}")
        if value is None or str(value).strip() == "":
            return run_dirs[0]
        text = str(value).strip()
        for path in run_dirs:
            if text in {path.name, str(path)}:
                return path
        raise ValueError("FLASH 运行目录不在允许列表中")

    @staticmethod
    def _coerce_flash_value(value: str) -> object:
        text = str(value).strip()
        lowered = text.lower()
        if lowered in {".true.", "true", "t"}:
            return True
        if lowered in {".false.", "false", "f"}:
            return False
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            return text[1:-1]
        try:
            if any(ch in text.lower() for ch in (".", "e")):
                return float(text)
            return int(text)
        except ValueError:
            return text

    @staticmethod
    def _format_flash_value(value: object) -> str:
        if isinstance(value, bool):
            return ".true." if value else ".false."
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        text = str(value)
        if text.lower() in {".true.", ".false."}:
            return text.lower()
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            return text
        stripped = text.strip()
        try:
            float(stripped)
            if any(ch.isdigit() for ch in stripped):
                return stripped
        except ValueError:
            pass
        return json.dumps(text)

    @staticmethod
    def _flash_value_type(raw: str, value: object) -> str:
        lowered = str(raw).strip().lower()
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "number"
        if len(str(raw).strip()) >= 2 and str(raw).strip()[0] == '"' and str(raw).strip()[-1] == '"':
            return "string"
        if lowered in {".true.", ".false.", "true", "false", "t", "f"}:
            return "bool"
        return "string"

    @staticmethod
    def _flash_param_group(key: str) -> str:
        if key.startswith("op_"):
            return "op"
        if key.startswith("rt_"):
            return "rt"
        if key.startswith("sim_"):
            return "sim"
        if key.startswith("eos_"):
            return "eos"
        if key.startswith("diff_"):
            return "diff"
        if key.startswith("ms_"):
            return "ms"
        if key.startswith("gr_"):
            return "gr"
        if "_" in key:
            return key.split("_", 1)[0]
        match = re.match(r"[A-Za-z]+", key)
        return match.group(0) if match else "other"

    @staticmethod
    def _parse_flash_par(path: Path) -> dict:
        params = {}
        if not path.exists():
            return params
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                params[key] = {
                    "raw": value,
                    "value": FreePathWebApp._coerce_flash_value(value),
                }
        return params

    @staticmethod
    def _parse_flash_par_rows(path: Path) -> list[dict]:
        rows: list[dict] = []
        if not path.exists():
            return rows
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            line_without_comment, sep, comment = raw_line.partition("#")
            if "=" not in line_without_comment:
                continue
            key, raw_value = line_without_comment.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if not key:
                continue
            value = FreePathWebApp._coerce_flash_value(raw_value)
            value_type = FreePathWebApp._flash_value_type(raw_value, value)
            rows.append(
                {
                    "key": key,
                    "line": line_number,
                    "raw": raw_value,
                    "value": value,
                    "value_type": value_type,
                    "group": FreePathWebApp._flash_param_group(key),
                    "comment": comment.strip() if sep else "",
                    "editable": bool(FLASH_PARAM_NAME_RE.match(key)),
                }
            )
        return rows

    @staticmethod
    def _flash_param_groups(rows: list[dict]) -> list[dict]:
        counts: dict[str, int] = {}
        for row in rows:
            group = str(row.get("group") or "other")
            counts[group] = counts.get(group, 0) + 1
        return [{"name": key, "count": counts[key]} for key in sorted(counts)]

    @staticmethod
    def _tail_text(path: Path, max_bytes: int = 24000) -> str:
        if not path.exists() or not path.is_file():
            return ""
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
        return data.decode("utf-8", errors="replace")

    def _flash_pid_info(self, run_dir: Path) -> dict:
        path = run_dir / FLASH_WEB_PID
        if not path.exists():
            return {"running": False, "pid_file": str(path)}
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        pid = int(info.get("pid") or 0)
        running = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                running = True
            except OSError:
                running = False
            if running:
                try:
                    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
                    parts = stat_text.split()
                    if len(parts) > 2 and parts[2] == "Z":
                        running = False
                        info["state"] = "zombie"
                except OSError:
                    pass
        info.update({"running": running, "pid": pid, "pid_file": str(path)})
        return info

    def _flash_files(self, run_dir: Path) -> list[dict]:
        names = []
        patterns = (
            "*.log", "*.out", "*.err", "*.txt", "*.dat", "*.csv", "*.png", "*.jpg", "*.jpeg",
            "*.h5", "*plt*", "*chk*", FLASH_WEB_LOG, "flash.par"
        )
        for pattern in patterns:
            names.extend(path for path in run_dir.glob(pattern) if path.is_file())
        unique = []
        seen = set()
        for path in names:
            if path.name in seen:
                continue
            seen.add(path.name)
            unique.append(path)
        rows = []
        for path in sorted(unique, key=lambda item: item.stat().st_mtime, reverse=True)[:120]:
            stat = path.stat()
            rows.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "kind": (
                        "hdf5" if ("plt" in path.name or "chk" in path.name or path.suffix == ".h5")
                        else ("image" if path.suffix.lower() in FLASH_IMAGE_SUFFIXES else "text")
                    ),
                }
            )
        return rows

    def flash_status_payload(self, query: dict | None = None, include_log: bool = True) -> dict:
        query = query or {}
        run_value = (query.get("runDir") or query.get("run_dir") or [None])[0] if isinstance(query, dict) else None
        run_dir = self._resolve_flash_run_dir(run_value)
        par_path = run_dir / "flash.par"
        params = self._parse_flash_par(par_path)
        param_rows = self._parse_flash_par_rows(par_path)
        run_dirs = [{"name": path.name, "path": str(path)} for path in self._flash_run_dirs()]
        files = self._flash_files(run_dir)
        log_candidates = [item["name"] for item in files if item["name"].endswith(".log") or item["name"] == FLASH_WEB_LOG]
        selected_log = FLASH_WEB_LOG if (run_dir / FLASH_WEB_LOG).exists() else (log_candidates[0] if log_candidates else "")
        payload = {
            "root": str(FLASH_ROOT),
            "run_dirs": run_dirs,
            "run_dir": run_dir.name,
            "run_dir_path": str(run_dir),
            "flash4_exists": (run_dir / "flash4").exists(),
            "flash_par_exists": par_path.exists(),
            "pid": self._flash_pid_info(run_dir),
            "param_specs": FLASH_PARAM_SPECS,
            "params": params,
            "param_rows": param_rows,
            "param_groups": self._flash_param_groups(param_rows),
            "files": files,
            "logs": log_candidates,
            "selected_log": selected_log,
        }
        if include_log and selected_log:
            payload["log_tail"] = self._tail_text(run_dir / selected_log)
        return payload

    def update_flash_params(self, payload: dict) -> dict:
        run_dir = self._resolve_flash_run_dir(payload.get("runDir", payload.get("run_dir")))
        par_path = run_dir / "flash.par"
        if not par_path.exists():
            raise FileNotFoundError(f"未找到 flash.par: {par_path}")
        updates = payload.get("params") or {}
        if not isinstance(updates, dict):
            raise ValueError("params 必须是对象")
        if not updates:
            status = self.flash_status_payload({"runDir": [run_dir.name]})
            status["backup_path"] = None
            status["updated_count"] = 0
            return status
        bad_names = [key for key in updates if not FLASH_PARAM_NAME_RE.match(str(key))]
        if bad_names:
            raise ValueError("参数名只能包含字母、数字和下划线，且不能以数字开头: " + ", ".join(map(str, bad_names[:8])))
        existing_params = self._parse_flash_par(par_path)
        allow_new = bool(payload.get("allowNew", payload.get("allow_new", False)))
        unknown = sorted(set(map(str, updates)) - set(existing_params))
        if unknown and not allow_new:
            raise ValueError("这些参数不在当前 flash.par 中；如需新增请使用新增参数功能: " + ", ".join(unknown[:8]))

        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = par_path.with_name(f"flash.par.web_bak_{stamp}")
        shutil.copy2(par_path, backup_path)
        lines = par_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        remaining = {str(key): value for key, value in updates.items()}
        out_lines = []
        for raw_line in lines:
            line_without_comment, sep, comment = raw_line.partition("#")
            if "=" in line_without_comment:
                key = line_without_comment.split("=", 1)[0].strip()
                if key in remaining:
                    formatted = self._format_flash_value(remaining.pop(key))
                    suffix = f" #{comment}" if sep else ""
                    out_lines.append(f"{key} = {formatted}{suffix}")
                    continue
            out_lines.append(raw_line)
        for key, value in remaining.items():
            out_lines.append(f"{key} = {self._format_flash_value(value)}")
        par_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        status = self.flash_status_payload({"runDir": [run_dir.name]})
        status["backup_path"] = str(backup_path)
        status["updated_count"] = len(updates)
        return status

    def create_flash_project(self, payload: dict) -> dict:
        name = str(payload.get("name") or payload.get("projectName") or "").strip()
        if not name:
            raise ValueError("新项目名不能为空")
        if not FLASH_PROJECT_NAME_RE.match(name):
            raise ValueError("项目名只能用字母、数字、下划线、中划线、点，且长度不超过 80")
        source = self._resolve_flash_run_dir(payload.get("template", payload.get("sourceRunDir", payload.get("runDir"))))
        destination = (FLASH_ROOT / name).resolve()
        root = FLASH_ROOT.resolve()
        if root not in destination.parents:
            raise ValueError("新项目必须位于 FLASH 根目录内")
        if destination.exists():
            raise FileExistsError(f"项目已存在: {name}")
        copy_mode = str(payload.get("copyMode") or payload.get("copy_mode") or "light").lower()
        ignore = None if copy_mode == "full" else FLASH_PROJECT_LIGHT_IGNORE
        t0 = time.perf_counter()
        shutil.copytree(source, destination, symlinks=True, ignore=ignore)
        for cleanup in (FLASH_WEB_PID, FLASH_WEB_LOG):
            path = destination / cleanup
            if path.exists():
                path.unlink()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status = self.flash_status_payload({"runDir": [destination.name]}, include_log=False)
        status["created_project"] = {
            "name": destination.name,
            "path": str(destination),
            "template": source.name,
            "copy_mode": copy_mode,
            "elapsed_ms": elapsed_ms,
        }
        return status

    def start_flash_run(self, payload: dict) -> dict:
        run_dir = self._resolve_flash_run_dir(payload.get("runDir", payload.get("run_dir")))
        pid_info = self._flash_pid_info(run_dir)
        if pid_info.get("running"):
            raise RuntimeError(f"FLASH 已在运行，pid={pid_info.get('pid')}")
        command = str(payload.get("command") or "./flash4").strip()
        if not command:
            raise ValueError("启动命令不能为空")
        log_path = run_dir / FLASH_WEB_LOG
        with log_path.open("ab") as log_file:
            log_file.write(f"\n===== web start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{command}\n".encode("utf-8"))
            proc = subprocess.Popen(
                command,
                cwd=run_dir,
                shell=True,
                executable="/bin/bash",
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (run_dir / FLASH_WEB_PID).write_text(
            json.dumps({"pid": proc.pid, "command": command, "started_at": time.time()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        time.sleep(0.2)
        return self.flash_status_payload({"runDir": [run_dir.name]})

    def stop_flash_run(self, payload: dict) -> dict:
        run_dir = self._resolve_flash_run_dir(payload.get("runDir", payload.get("run_dir")))
        pid_info = self._flash_pid_info(run_dir)
        pid = int(pid_info.get("pid") or 0)
        if pid > 0 and pid_info.get("running"):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.time() + 4.0
            while time.time() < deadline:
                time.sleep(0.2)
                if not self._flash_pid_info(run_dir).get("running"):
                    break
            if self._flash_pid_info(run_dir).get("running"):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                time.sleep(0.3)
        if not self._flash_pid_info(run_dir).get("running"):
            try:
                (run_dir / FLASH_WEB_PID).unlink()
            except FileNotFoundError:
                pass
        return self.flash_status_payload({"runDir": [run_dir.name]})

    def run_flash_command(self, payload: dict) -> dict:
        run_dir = self._resolve_flash_run_dir(payload.get("runDir", payload.get("run_dir")))
        command = str(payload.get("command") or "").strip()
        if not command:
            raise ValueError("命令不能为空")
        timeout = float(payload.get("timeout", 120.0))
        timeout = max(1.0, min(timeout, 900.0))
        t0 = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=run_dir,
            shell=True,
            executable="/bin/bash",
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "run_dir": run_dir.name,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-24000:],
            "stderr": completed.stderr[-24000:],
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def _resolve_flash_file(self, query: dict) -> tuple[Path, Path]:
        run_dir = self._resolve_flash_run_dir((query.get("runDir") or query.get("run_dir") or [None])[0])
        values = query.get("file") or query.get("name")
        if not values:
            raise ValueError("缺少 file 参数")
        rel = Path(str(values[0]))
        path = (run_dir / rel).resolve()
        run_root = run_dir.resolve()
        flash_root = FLASH_ROOT.resolve()
        in_run_dir = run_root in path.parents or path == run_root
        in_flash_tree = flash_root in path.parents or path == flash_root
        if not in_run_dir and not in_flash_tree:
            raise ValueError("文件不在 FLASH 运行目录或 FLASH 根目录内")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        return run_dir, path

    @staticmethod
    def _flash_hdf5_variables(handle) -> list[dict]:
        import h5py  # type: ignore

        nblocks = None
        if "bounding box" in handle:
            try:
                nblocks = int(handle["bounding box"].shape[0])
            except Exception:
                nblocks = None
        variables = []
        metadata_names = {
            "block size",
            "bounding box",
            "coordinates",
            "gid",
            "node type",
            "processor number",
            "refine level",
            "unknown names",
        }
        for name, obj in handle.items():
            if name in metadata_names or not isinstance(obj, h5py.Dataset):
                continue
            dtype = getattr(obj, "dtype", None)
            shape = getattr(obj, "shape", ())
            if dtype is None or not np.issubdtype(dtype, np.number):
                continue
            if len(shape) < 3:
                continue
            if nblocks is not None and shape[0] != nblocks:
                continue
            variables.append({"name": name, "shape": list(shape), "dtype": str(dtype)})
        return sorted(variables, key=lambda item: item["name"])

    @staticmethod
    def _flash_hdf5_scalars(handle) -> dict:
        scalars: dict[str, object] = {}
        for dataset_name in ("integer scalars", "real scalars", "logical scalars", "string scalars"):
            if dataset_name not in handle:
                continue
            try:
                for item in handle[dataset_name][()]:
                    name = item["name"].decode("utf-8", errors="ignore").strip()
                    value = item["value"]
                    if hasattr(value, "decode"):
                        value = value.decode("utf-8", errors="ignore").strip()
                    elif hasattr(value, "item"):
                        value = value.item()
                    scalars[name] = value
            except Exception:
                continue
        return scalars

    @staticmethod
    def _flash_table_columns(path: Path, max_probe_lines: int = 80) -> dict:
        if path.suffix.lower() not in {".dat", ".csv", ".txt"}:
            return {"available": False, "columns": []}
        header_labels: list[str] = []
        numeric_width = 0
        delimiter = "," if path.suffix.lower() == ".csv" else None
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for _, raw_line in zip(range(max_probe_lines), handle):
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        text = line.lstrip("#").strip()
                        labels = [part.strip() for part in re.split(r"\s{2,}|,", text) if part.strip()]
                        if len(labels) > len(header_labels):
                            header_labels = labels
                        continue
                    parts = line.split(",") if delimiter == "," else line.split()
                    numeric = []
                    for part in parts:
                        try:
                            numeric.append(float(part))
                        except ValueError:
                            pass
                    if len(numeric) >= 2:
                        numeric_width = max(numeric_width, len(numeric))
                        break
        except OSError:
            return {"available": False, "columns": []}
        if numeric_width < 2:
            return {"available": False, "columns": []}
        columns = []
        for index in range(numeric_width):
            label = header_labels[index] if index < len(header_labels) else f"col{index}"
            columns.append({"index": index, "name": f"col{index}", "label": label})
        return {"available": True, "columns": columns}

    @staticmethod
    def _column_index(value: object, fallback: int) -> int:
        text = str(value if value is not None else "").strip()
        if not text:
            return fallback
        if text.lower().startswith("col"):
            text = text[3:]
        try:
            return max(0, int(float(text)))
        except ValueError:
            return fallback

    def flash_file_payload(self, query: dict) -> dict:
        _, path = self._resolve_flash_file(query)
        stat = path.stat()
        result = {
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }
        if path.suffix in FLASH_TEXT_SUFFIXES or path.name.endswith(".log"):
            result.update({"kind": "text", "content": self._tail_text(path, max_bytes=50000)})
            if path.suffix.lower() in {".dat", ".csv", ".txt"}:
                result["table"] = self._flash_table_columns(path)
            return result
        if path.suffix.lower() in FLASH_IMAGE_SUFFIXES:
            if stat.st_size > 16 * 1024 * 1024:
                result.update({"kind": "image", "note": "图片文件过大，未内嵌显示"})
                return result
            mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else f"image/{path.suffix.lower().lstrip('.')}"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            result.update({"kind": "image", "data_url": f"data:{mime};base64,{data}"})
            return result
        if "plt" in path.name or "chk" in path.name or path.suffix == ".h5":
            try:
                import h5py  # type: ignore

                datasets = []
                with h5py.File(path, "r") as handle:
                    def visitor(name, obj):
                        if len(datasets) >= 120:
                            return
                        shape = getattr(obj, "shape", None)
                        dtype = getattr(obj, "dtype", None)
                        datasets.append(
                            {
                                "name": name,
                                "type": obj.__class__.__name__,
                                "shape": list(shape) if shape is not None else None,
                                "dtype": str(dtype) if dtype is not None else None,
                            }
                        )

                    handle.visititems(visitor)
                    variables = self._flash_hdf5_variables(handle)
                    scalars = self._flash_hdf5_scalars(handle)
                result.update({"kind": "hdf5", "datasets": datasets, "variables": variables, "scalars": scalars})
            except Exception as exc:
                result.update({"kind": "binary", "note": f"HDF5 摘要读取失败: {exc}"})
            return result
        result.update({"kind": "binary", "note": "二进制文件只能在此处查看文件信息"})
        return result

    def _flash_table_plot_payload(self, path: Path, query: dict, t0: float) -> dict:
        table = self._flash_table_columns(path)
        if not table.get("available"):
            raise ValueError("这个文本结果没有识别出可绘制的数字列")
        x_index = self._column_index((query.get("xColumn") or query.get("x_column") or ["0"])[0], 0)
        y_index = self._column_index((query.get("variable") or query.get("yColumn") or query.get("y_column") or ["1"])[0], 1)
        delimiter = "," if path.suffix.lower() == ".csv" else None
        data = np.genfromtxt(path, comments="#", delimiter=delimiter, invalid_raise=False)
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("这个文本结果没有足够的数字列")
        x_index = min(x_index, data.shape[1] - 1)
        y_index = min(y_index, data.shape[1] - 1)
        x = data[:, x_index]
        y = data[:, y_index]
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 1:
            raise ValueError("选中的列没有可绘制的有限数值")
        if x.size > 20000:
            sample = np.linspace(0, x.size - 1, 20000).astype(int)
            x = x[sample]
            y = y[sample]
        scale = str((query.get("scale") or ["linear"])[0]).lower()
        if scale == "log10":
            mask = y > 0.0
            x = x[mask]
            y = np.log10(y[mask])
            if y.size < 1:
                raise ValueError("log10 尺度下没有正的 Y 值可绘制")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        columns = table["columns"]
        x_label = columns[x_index]["label"] if x_index < len(columns) else f"col{x_index}"
        y_label_raw = columns[y_index]["label"] if y_index < len(columns) else f"col{y_index}"
        y_label = f"log10({y_label_raw})" if scale == "log10" else y_label_raw
        fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=140)
        ax.plot(x, y, linewidth=1.2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{path.name} | {y_label} vs {x_label}", fontsize=9)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "kind": "plot",
            "plot_type": "table",
            "file": path.name,
            "variable": f"col{y_index}",
            "x_column": f"col{x_index}",
            "columns": columns,
            "scale": scale,
            "rows_plotted": int(x.size),
            "value_min": float(np.nanmin(y)),
            "value_max": float(np.nanmax(y)),
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "image": "data:image/png;base64," + encoded,
        }

    def flash_plot_payload(self, query: dict) -> dict:
        _, path = self._resolve_flash_file(query)
        t0 = time.perf_counter()
        if not ("plt" in path.name or "chk" in path.name or path.suffix == ".h5"):
            return self._flash_table_plot_payload(path, query, t0)
        variable = (query.get("variable") or query.get("var") or [""])[0]
        scale = str((query.get("scale") or ["linear"])[0]).lower()
        cmap = str((query.get("cmap") or ["viridis"])[0])
        range_mode = str((query.get("range") or ["robust"])[0]).lower()
        z_index = int(float((query.get("zIndex") or query.get("z_index") or [0])[0]))
        leaf_only = str((query.get("leafOnly") or query.get("leaf_only") or ["true"])[0]).lower() not in {"0", "false", "no"}
        block_edges = str((query.get("blockEdges") or query.get("block_edges") or ["false"])[0]).lower() in {"1", "true", "yes"}

        import h5py  # type: ignore
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        with h5py.File(path, "r") as handle:
            variables = self._flash_hdf5_variables(handle)
            if not variables:
                raise ValueError("这个 HDF5 文件没有可直接二维渲染的数值变量")
            if not variable:
                variable = variables[0]["name"]
            if variable not in {item["name"] for item in variables}:
                raise ValueError(f"变量不存在: {variable}")
            data = handle[variable]
            bbox = np.asarray(handle["bounding box"], dtype=float) if "bounding box" in handle else None
            if bbox is None or bbox.ndim != 3 or bbox.shape[1] < 2:
                raise ValueError("缺少 FLASH bounding box，无法把 block 拼成图")
            nblocks = int(data.shape[0])
            if data.ndim == 4:
                z_index = max(0, min(z_index, data.shape[1] - 1))
                block_data = np.asarray(data[:, z_index, :, :], dtype=float)
            elif data.ndim == 3:
                block_data = np.asarray(data[:, :, :], dtype=float)
            else:
                raise ValueError(f"暂不支持变量维度: {data.shape}")
            if "node type" in handle and leaf_only:
                node_type = np.asarray(handle["node type"])
                block_indices = [i for i in range(min(nblocks, len(node_type))) if int(node_type[i]) == 1]
            else:
                block_indices = list(range(nblocks))
            if not block_indices:
                block_indices = list(range(nblocks))

            plotted_blocks = []
            finite_values = []
            for i in block_indices:
                arr = np.asarray(block_data[i], dtype=float)
                if scale == "log10":
                    arr = np.where(arr > 0.0, np.log10(arr), np.nan)
                plotted_blocks.append((i, arr))
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    finite_values.append(finite)
            if not finite_values:
                raise ValueError("选中的变量没有可绘制的有限数值")
            finite_all = np.concatenate(finite_values)
            if range_mode == "full" or finite_all.size < 20:
                vmin = float(np.nanmin(finite_all))
                vmax = float(np.nanmax(finite_all))
            else:
                vmin, vmax = [float(v) for v in np.nanpercentile(finite_all, [2.0, 98.0])]
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                center = float(np.nanmean(finite_all)) if finite_all.size else 0.0
                vmin = center - 0.5
                vmax = center + 0.5
            scalars = self._flash_hdf5_scalars(handle)

            fig, ax = plt.subplots(figsize=(8.4, 6.6), dpi=140)
            norm = Normalize(vmin=vmin, vmax=vmax)
            image = None
            x_min = float(np.nanmin(bbox[:nblocks, 0, 0]))
            x_max = float(np.nanmax(bbox[:nblocks, 0, 1]))
            y_min = float(np.nanmin(bbox[:nblocks, 1, 0]))
            y_max = float(np.nanmax(bbox[:nblocks, 1, 1]))
            for i, arr in plotted_blocks:
                extent = [bbox[i, 0, 0], bbox[i, 0, 1], bbox[i, 1, 0], bbox[i, 1, 1]]
                image = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
                if block_edges:
                    ax.plot(
                        [extent[0], extent[1], extent[1], extent[0], extent[0]],
                        [extent[2], extent[2], extent[3], extent[3], extent[2]],
                        color="white",
                        linewidth=0.25,
                        alpha=0.65,
                    )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("x [cm]")
            ax.set_ylabel("y [cm]")
            label = f"log10({variable})" if scale == "log10" else variable
            title_bits = [path.name, label]
            if "time" in scalars:
                title_bits.append(f"t={scalars['time']:.4e}")
            if "nstep" in scalars:
                title_bits.append(f"nstep={scalars['nstep']}")
            ax.set_title(" | ".join(title_bits), fontsize=9)
            if image is not None:
                fig.colorbar(image, ax=ax, label=label)
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            plt.close(fig)

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "kind": "plot",
            "file": path.name,
            "variable": variable,
            "variables": variables,
            "scale": scale,
            "cmap": cmap,
            "range_mode": range_mode,
            "z_index": z_index,
            "leaf_only": leaf_only,
            "block_edges": block_edges,
            "blocks_plotted": len(plotted_blocks),
            "value_min": float(np.nanmin(finite_all)),
            "value_max": float(np.nanmax(finite_all)),
            "display_min": vmin,
            "display_max": vmax,
            "scalars": scalars,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "image": "data:image/png;base64," + encoded,
        }

    def zpinch_payload(self) -> dict:
        summary = self._load_json_if_exists(ZPINCH_SUMMARY) or self._load_json_if_exists(ZPINCH_LEGACY_SUMMARY) or {}
        return {
            "title": "套筒速度优先预测",
            "root": str(ZPINCH_ROOT),
            "python": str(ZPINCH_PYTHON),
            "artifact": str(ZPINCH_ARTIFACT),
            "artifact_exists": ZPINCH_ARTIFACT.exists(),
            "summary_path": str(ZPINCH_SUMMARY if ZPINCH_SUMMARY.exists() else ZPINCH_LEGACY_SUMMARY),
            "summary": summary,
            "data_note": f"当前 Z-pinch 使用 {ZPINCH_ROOT / 'ai_training_outputs'} 的优化速度优先模型；模型输出 v_cm_per_s，再由 E = v^2*m/(2e16) 反算动能。数据来自零维程序/表格结果，不属于当前不透明度自由程原始表。",
        }

    def dataset_payload(self) -> dict:
        return {
            "artifact": str(self.artifact_path),
            "standard_prediction_format": self.metadata.get("standard_prediction_format", " ".join(STANDARD_INPUT_COLUMNS)),
            "standard_training_format": self.metadata.get("standard_training_format", " ".join(STANDARD_TRAINING_COLUMNS)),
            "standard_training_data_path": self.metadata.get("standard_training_data_path", str(STANDARD_TRAINING_DATA_PATH)),
            "elements": self._element_map(),
            "simulators": self._simulator_inventory(),
            "high_z_model": {
                "available": self.high_z_model is not None,
                "metadata": self.high_z_model.metadata if self.high_z_model is not None else None,
            },
            "metric_thresholds": SIMULATION_THRESHOLDS,
            "threshold_labels": THRESHOLD_LABELS,
            "benchmark_metrics": self._benchmark_metrics_payload(),
            "data_source_audit": DATA_SOURCE_AUDIT,
            "prediction_controls": self._source_control_payload(),
            "flash_control": self.flash_control_payload(),
            "flash_console": self.flash_status_payload(include_log=False),
            "capacitor_app": self.capacitor_payload(),
            "zpinch_app": self.zpinch_payload(),
        }

    def zpinch_predict(self, payload: dict) -> dict:
        if not ZPINCH_PYTHON.exists():
            raise FileNotFoundError(f"未找到套筒速度 Python 环境: {ZPINCH_PYTHON}")
        if not ZPINCH_ARTIFACT.exists():
            raise FileNotFoundError(f"未找到套筒速度模型权重: {ZPINCH_ARTIFACT}")
        values = {
            "I": float(payload.get("I")),
            "tr": float(payload.get("tr")),
            "liner_r": float(payload.get("liner_r")),
            "foam_r": float(payload.get("foam_r")),
            "m": float(payload.get("m")),
            "Z": float(payload.get("Z", 13.0)),
        }
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError("套筒速度输入包含非有限数值")
        script = r'''
import json
import sys
from pathlib import Path

import numpy as np

root = Path("__ZPINCH_ROOT__")
sys.path.insert(0, str(root))
from train_zpinch_surrogate import V_FORMULA_COEF, calc_v_from_E_m, load_model_artifact, make_features

payload = json.loads(sys.stdin.read())
artifact = Path("__ZPINCH_ARTIFACT__")
model_payload = load_model_artifact(str(artifact))
model = model_payload["model"]
prediction_target = model_payload.get("prediction_target", "E_MJ_per_cm")
x_base = np.array([[
    float(payload["I"]),
    float(payload["tr"]),
    float(payload["liner_r"]),
    float(payload["foam_r"]),
    float(payload["m"]),
    float(payload["Z"]),
]], dtype=float)
x_feat, feature_names = make_features(x_base)
raw_pred = float(np.asarray(model.predict(x_feat)).reshape(-1)[0])
if prediction_target == "v_cm_per_s":
    v_pred = raw_pred
    E_pred = float((v_pred * v_pred) * float(payload["m"]) / V_FORMULA_COEF)
    pipeline = "velocity_first"
else:
    E_pred = float(max(raw_pred, 1e-12))
    v_pred = float(calc_v_from_E_m(np.array([E_pred]), np.array([float(payload["m"])]))[0])
    pipeline = "energy_first"
print(json.dumps({
    "model_name": model_payload.get("model_name"),
    "artifact": str(artifact),
    "doc_path": model_payload.get("doc_path"),
    "xlsx_path": model_payload.get("xlsx_path"),
    "prediction_target": prediction_target,
    "pipeline": pipeline,
    "feature_names": feature_names,
    "E_MJ_per_cm": E_pred,
    "v_cm_per_s": v_pred,
}, ensure_ascii=False))
'''
        script = script.replace("__ZPINCH_ROOT__", str(ZPINCH_ROOT)).replace("__ZPINCH_ARTIFACT__", str(ZPINCH_ARTIFACT))
        t0 = time.perf_counter()
        completed = subprocess.run(
            [str(ZPINCH_PYTHON), "-c", script],
            cwd=ZPINCH_ROOT,
            input=json.dumps(values),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "套筒速度预测失败")[-2000:])
        result = json.loads(completed.stdout)
        result.update(
            {
                "input": values,
                "elapsed_ms": elapsed_ms,
                "summary": self.zpinch_payload().get("summary", {}),
                "data_note": self.zpinch_payload().get("data_note"),
            }
        )
        return result

    def _model_prediction_result(self, z_value: float, point: np.ndarray, data_source: str) -> dict:
        t0 = time.perf_counter()
        x_model = self.model.standard_point(z_value, point)
        routed_log, route_info, base_log = self._predict_log_routed(x_model, data_source=data_source)
        log_lnu = float(routed_log[0])
        lnu = float(10.0 ** log_lnu)
        result = {
            "artifact": str(self.artifact_path),
            "Z": float(z_value),
            "mode": "unified_local_regression",
            "input": {
                "Z": float(z_value),
                "rod": float(point[0]),
                "tep": float(point[1]),
                "tgama": float(point[2]),
            },
            "model_coordinates": {
                "Z": float(x_model[0]),
                "rod": float(x_model[1]),
                "tep": float(x_model[2]),
                "tgama": float(x_model[3]),
            },
            "lnu": lnu,
            "log10_lnu": log_lnu,
            "model_route": route_info,
            "data_source": data_source,
            "timing": {"model_ms": (time.perf_counter() - t0) * 1000.0},
            "selected_source": "model",
            "selected_source_label": SELECTED_SOURCE_LABELS["model"],
        }
        if route_info.get("high_z_model_used_rows"):
            result["base_unified_prediction"] = {
                "lnu": float(10.0 ** float(base_log[0])),
                "log10_lnu": float(base_log[0]),
            }
            result["mode"] = (route_info.get("high_z_model") or {}).get("artifact_type", "high_z_specialized")
        return result

    def _table_truth_prediction_result(
        self,
        z_value: float,
        point: np.ndarray,
        table_truth: dict,
        source_mode: str,
    ) -> dict:
        true_lnu = float(table_truth["lnu"])
        true_log = float(table_truth["log10_lnu"])
        return {
            "artifact": str(self.artifact_path),
            "Z": float(z_value),
            "mode": "hybrid_table_truth" if source_mode == "hybrid" else "table_truth",
            "input": {
                "Z": float(z_value),
                "rod": float(point[0]),
                "tep": float(point[1]),
                "tgama": float(point[2]),
            },
            "model_coordinates": {
                "Z": float(z_value),
                "rod": float(point[0]),
                "tep": float(point[1]),
                "tgama": float(point[2]),
            },
            "lnu": true_lnu,
            "log10_lnu": true_log,
            "model_route": {
                "base_model": "not_called",
                "data_source": "table_truth",
                "high_z_model_available": self.high_z_model is not None,
                "high_z_model_used_rows": 0,
            },
            "data_source": "table_truth",
            "timing": {
                "model_ms": None,
                "truth_lookup_ms": table_truth.get("lookup_ms"),
            },
            "selected_source": "table_truth",
            "selected_source_label": SELECTED_SOURCE_LABELS["table_truth"],
        }

    def _attach_policy(self, result: dict, policy: dict, z_value: float) -> None:
        result["prediction_policy"] = {
            "mode": policy["mode"],
            "mode_label": policy["mode_label"],
            "model_elements": policy["model_elements"],
            "model_element_labels": policy["model_element_labels"],
            "model_allowed_for_z": self._model_allowed_for_z(z_value, policy),
            "selected_source": result.get("selected_source"),
            "selected_source_label": result.get("selected_source_label"),
        }

    def predict(self, payload: dict, progress_callback=None) -> dict:
        z_value = float(payload.get("Z", payload.get("z", payload.get("sourceId", payload.get("source_id")))))
        point = np.array(
            [float(payload["rod"]), float(payload["tep"]), float(payload["tgama"])],
            dtype=float,
        )
        data_source = self._normalize_data_source(payload.get("dataSource", payload.get("data_source")), default="current")
        policy = self._prediction_policy(payload)
        run_simulation = bool(payload.get("runSimulation", False))
        if progress_callback:
            progress_callback("正在查询训练数据命中状态", 34.0, "membership")
        truth_info = self._membership_and_truth(z_value, point)
        table_truth = truth_info["table_truth"]

        use_table_truth = policy["mode"] in {"hybrid", "truth"} and bool(table_truth.get("found"))
        need_model = policy["mode"] == "model" or (policy["mode"] == "hybrid" and not use_table_truth)
        model_allowed = self._model_allowed_for_z(z_value, policy)

        if use_table_truth:
            result = self._table_truth_prediction_result(z_value, point, table_truth, policy["mode"])
            result.update(truth_info)
        elif need_model:
            if not model_allowed:
                allowed = ", ".join(policy["model_element_labels"]) or "无"
                raise ValueError(f"Z={z_value:g} 未启用模型兜底；当前启用模型元素: {allowed}")
            if progress_callback:
                progress_callback("正在执行模型预测", 42.0, "model")
            result = self._model_prediction_result(z_value, point, data_source=data_source)
            result.update(self._membership_and_truth(z_value, point, result["lnu"], result["log10_lnu"]))
        else:
            raise ValueError("只用已计算真值模式下，该点未命中标准数据表")

        result["simulation_requested"] = run_simulation
        self._attach_policy(result, policy, z_value)

        if run_simulation:
            if progress_callback:
                progress_callback("已勾选真实程序，准备调用单点不透明度程序", 45.0, "simulator")
            result["reference_truth"] = self._run_single_point_simulator(
                z_value,
                point,
                result["lnu"],
                result["log10_lnu"],
                progress_callback=progress_callback,
            )
        if progress_callback:
            progress_callback("正在整理预测结果", 94.0, "finalize")
        warnings = self._range_warnings_for_point(z_value, point)
        result["range_warnings"] = warnings
        if warnings and result.get("selected_source") == "model":
            result["mode"] = "extrapolation"
        return result

    @staticmethod
    def _parse_prediction_points_text(text: str, default_z: object = None) -> list[dict]:
        rows = []
        default_z_value = None if str(default_z or "").strip() == "" else float(default_z)
        for lineno, raw_line in enumerate(str(text or "").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            lowered = [part.lower() for part in parts]
            if lowered[:4] == ["z", "rod", "tep", "tgama"] or lowered[:5] == [name.lower() for name in STANDARD_TRAINING_COLUMNS]:
                continue
            try:
                if len(parts) >= 5:
                    row = {
                        "line": lineno,
                        "Z": float(parts[0]),
                        "rod": float(parts[1]),
                        "tep": float(parts[2]),
                        "tgama": float(parts[3]),
                        "target_lnu": float(parts[4]),
                    }
                elif len(parts) == 4:
                    row = {
                        "line": lineno,
                        "Z": float(parts[0]),
                        "rod": float(parts[1]),
                        "tep": float(parts[2]),
                        "tgama": float(parts[3]),
                        "target_lnu": None,
                    }
                elif len(parts) == 3 and default_z_value is not None:
                    row = {
                        "line": lineno,
                        "Z": default_z_value,
                        "rod": float(parts[0]),
                        "tep": float(parts[1]),
                        "tgama": float(parts[2]),
                        "target_lnu": None,
                    }
                else:
                    raise ValueError("需要 4 列 Z rod tep tgama；或填写默认 Z 后提供 3 列 rod tep tgama")
            except ValueError as exc:
                raise ValueError(f"第 {lineno} 行无法解析为批量预测点") from exc
            rows.append(row)
            if len(rows) > BATCH_PREDICTION_MAX_ROWS:
                raise ValueError(f"批量点数超过上限 {BATCH_PREDICTION_MAX_ROWS}")
        if not rows:
            raise ValueError("没有找到有效预测点")
        return rows

    def predict_batch(self, payload: dict) -> dict:
        rows = self._parse_prediction_points_text(payload.get("text", ""), payload.get("defaultZ", payload.get("default_z")))
        policy = self._prediction_policy(payload)
        data_source = self._normalize_data_source(payload.get("dataSource", payload.get("data_source")), default="current")

        results: list[dict | None] = [None] * len(rows)
        model_indices = []
        x_model_rows = []
        model_points = []
        model_z_values = []
        table_truth_rows = 0
        error_rows = 0
        t0 = time.perf_counter()

        for index, row in enumerate(rows):
            z_value = float(row["Z"])
            point = np.array([row["rod"], row["tep"], row["tgama"]], dtype=float)
            truth_info = self._membership_and_truth(z_value, point)
            table_truth = truth_info["table_truth"]
            use_table_truth = policy["mode"] in {"hybrid", "truth"} and bool(table_truth.get("found"))
            need_model = policy["mode"] == "model" or (policy["mode"] == "hybrid" and not use_table_truth)
            if use_table_truth:
                table_truth_rows += 1
                lnu = float(table_truth["lnu"])
                log_lnu = float(table_truth["log10_lnu"])
                item = {
                    "line": row["line"],
                    "Z": z_value,
                    "input": {
                        "rod": float(point[0]),
                        "tep": float(point[1]),
                        "tgama": float(point[2]),
                    },
                    "lnu": lnu,
                    "log10_lnu": log_lnu,
                    "mode": "hybrid_table_truth" if policy["mode"] == "hybrid" else "table_truth",
                    "selected_source": "table_truth",
                    "selected_source_label": SELECTED_SOURCE_LABELS["table_truth"],
                    "data_membership": truth_info["data_membership"],
                    "table_truth": table_truth,
                    "range_warnings": self._range_warnings_for_point(z_value, point),
                    "target_lnu": row["target_lnu"],
                }
                if row["target_lnu"] is not None and float(row["target_lnu"]) > 0:
                    item["target_error"] = self._prediction_error(lnu, log_lnu, float(row["target_lnu"]))
                results[index] = item
            elif need_model:
                if not self._model_allowed_for_z(z_value, policy):
                    error_rows += 1
                    allowed = ", ".join(policy["model_element_labels"]) or "无"
                    results[index] = {
                        "line": row["line"],
                        "Z": z_value,
                        "input": {
                            "rod": float(point[0]),
                            "tep": float(point[1]),
                            "tgama": float(point[2]),
                        },
                        "selected_source": "error",
                        "selected_source_label": SELECTED_SOURCE_LABELS["error"],
                        "error": f"Z={z_value:g} 未启用模型兜底；当前启用模型元素: {allowed}",
                        "data_membership": truth_info["data_membership"],
                        "table_truth": table_truth,
                        "range_warnings": self._range_warnings_for_point(z_value, point),
                        "target_lnu": row["target_lnu"],
                    }
                    continue
                x_model_rows.append([z_value, float(point[0]), float(point[1]), float(point[2])])
                model_points.append(point)
                model_z_values.append(z_value)
                model_indices.append(index)
            else:
                error_rows += 1
                results[index] = {
                    "line": row["line"],
                    "Z": z_value,
                    "input": {
                        "rod": float(point[0]),
                        "tep": float(point[1]),
                        "tgama": float(point[2]),
                    },
                    "selected_source": "error",
                    "selected_source_label": SELECTED_SOURCE_LABELS["error"],
                    "error": "只用已计算真值模式下，该点未命中标准数据表",
                    "data_membership": truth_info["data_membership"],
                    "table_truth": table_truth,
                    "range_warnings": self._range_warnings_for_point(z_value, point),
                    "target_lnu": row["target_lnu"],
                }

        model_rows = 0
        route_info = None
        if x_model_rows:
            x_model = np.asarray(x_model_rows, dtype=float)
            pred_log, route_info, base_log = self._predict_log_routed(x_model, data_source=data_source)
            pred_log = np.asarray(pred_log, dtype=float).reshape(-1)
            model_rows = int(pred_log.size)
            for local_index, row_index in enumerate(model_indices):
                source_row = rows[row_index]
                z_value = model_z_values[local_index]
                point = model_points[local_index]
                log_lnu = float(pred_log[local_index])
                lnu = float(10.0 ** log_lnu)
                truth_info = self._membership_and_truth(z_value, point, lnu, log_lnu)
                mode = "unified_local_regression"
                if route_info and route_info.get("high_z_model_used_rows"):
                    used_z = int(round(z_value)) == 79
                    if used_z:
                        mode = (route_info.get("high_z_model") or {}).get("artifact_type", "high_z_specialized")
                item = {
                    "line": source_row["line"],
                    "Z": z_value,
                    "input": {
                        "rod": float(point[0]),
                        "tep": float(point[1]),
                        "tgama": float(point[2]),
                    },
                    "lnu": lnu,
                    "log10_lnu": log_lnu,
                    "mode": mode,
                    "selected_source": "model",
                    "selected_source_label": SELECTED_SOURCE_LABELS["model"],
                    "data_membership": truth_info["data_membership"],
                    "table_truth": truth_info["table_truth"],
                    "range_warnings": self._range_warnings_for_point(z_value, point),
                    "target_lnu": source_row["target_lnu"],
                }
                if source_row["target_lnu"] is not None and float(source_row["target_lnu"]) > 0:
                    item["target_error"] = self._prediction_error(lnu, log_lnu, float(source_row["target_lnu"]))
                results[row_index] = item

        final_rows = [item for item in results if item is not None]
        target_true = []
        target_pred = []
        for item in final_rows:
            if item.get("error") or item.get("target_lnu") is None:
                continue
            target = float(item["target_lnu"])
            if target > 0 and item.get("log10_lnu") is not None:
                target_true.append(np.log10(target))
                target_pred.append(float(item["log10_lnu"]))
        metrics = metric_dict(np.asarray(target_true), np.asarray(target_pred)) if target_true else None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "rows": final_rows,
            "summary": {
                "rows_total": len(rows),
                "rows_succeeded": len(rows) - error_rows,
                "rows_error": error_rows,
                "table_truth_rows": table_truth_rows,
                "model_rows": model_rows,
                "elapsed_ms": elapsed_ms,
                "target_metrics": metrics,
            },
            "prediction_policy": {
                "mode": policy["mode"],
                "mode_label": policy["mode_label"],
                "model_elements": policy["model_elements"],
                "model_element_labels": policy["model_element_labels"],
            },
            "model_route": route_info,
            "notes": [
                "批量预测不会逐点调用 SNOP_op_Ross_single 真实程序。",
                "混合模式只对标准数据表中精确命中的点使用表值；其余点按元素开关使用模型。",
            ],
        }

    def evaluate_uploaded(self, payload: dict) -> dict:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("empty evaluation data")
        data_source = self._normalize_data_source(payload.get("dataSource", payload.get("data_source")), default="current")
        rows = []
        nonpositive_rows = 0
        remove_nonpositive = bool(payload.get("removeNonpositive", False))
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            lowered = [part.lower() for part in parts[:5]]
            if lowered == [name.lower() for name in STANDARD_TRAINING_COLUMNS]:
                continue
            if len(parts) < 5:
                continue
            try:
                row = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            except ValueError as exc:
                raise ValueError("evaluation data contains non-numeric values") from exc
            if row[4] <= 0.0:
                nonpositive_rows += 1
                if remove_nonpositive:
                    continue
            rows.append(row)
        if not rows:
            raise ValueError("no valid five-column rows found")
        arr = np.asarray(rows, dtype=float)
        if np.any(arr[:, 4] <= 0.0):
            raise ValueError(f"evaluation target lnu must be positive; found {nonpositive_rows} rows with lnu<=0")
        self._require_trained_z(arr[:, 0])
        pred_log, route_info, _ = self._predict_log_routed(arr[:, :4], data_source=data_source)
        true_log = np.log10(arr[:, 4])
        metrics = metric_dict(true_log, pred_log)
        return {
            "metrics": metrics,
            "threshold_status": threshold_status(metrics, "element"),
            "rows_used": int(arr.shape[0]),
            "nonpositive_rows_removed": int(nonpositive_rows if remove_nonpositive else 0),
            "model_route": route_info,
            "data_source": data_source,
        }

    def benchmark_file(self, payload: dict) -> dict:
        path = Path(str(payload.get("path", ""))).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.exists():
            raise ValueError(f"benchmark file not found: {path}")
        data_source = self._infer_benchmark_data_source(path, payload.get("dataSource", payload.get("data_source")))
        arr = np.loadtxt(path, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] not in {4, 5}:
            raise ValueError("benchmark file must have 4 columns (rod tep tgama lnu) or 5 columns (Z rod tep tgama lnu)")
        if np.any(~np.isfinite(arr)):
            raise ValueError("benchmark file contains non-finite values")
        sample_rows = int(payload.get("sampleRows", payload.get("sample_rows", 3000)) or 3000)
        sample_rows = max(1, min(sample_rows, arr.shape[0]))
        if arr.shape[0] > sample_rows:
            rng = np.random.default_rng(20260618)
            idx = np.sort(rng.choice(arr.shape[0], size=sample_rows, replace=False))
            arr = arr[idx]

        if arr.shape[1] == 4:
            z_value = float(payload.get("Z", payload.get("z")))
            self._require_trained_z(z_value)
            transform = infer_coordinate_transform(arr[:, :3])
            xyz = apply_coordinate_transform(arr[:, :3], transform)
            x_model = np.column_stack([np.full(xyz.shape[0], z_value), xyz])
            lnu = arr[:, 3]
        else:
            transform = "standard"
            self._require_trained_z(arr[:, 0])
            x_model = arr[:, :4]
            lnu = arr[:, 4]

        if np.any(lnu <= 0.0):
            raise ValueError("benchmark target lnu must be positive")

        pred_log, route_info, _ = self._predict_log_routed(x_model, data_source=data_source)
        metrics = metric_dict(np.log10(lnu), pred_log)
        return {
            "path": str(path),
            "rows_used": int(arr.shape[0]),
            "coordinate_transform": transform,
            "metrics": metrics,
            "threshold_status": threshold_status(metrics, "element"),
            "model_route": route_info,
            "data_source": data_source,
        }

    def _load_staged_summary(self) -> dict | None:
        return self._load_summary_for_path(STAGED_METRICS)

    def _load_active_summary(self) -> dict | None:
        return self._load_summary_for_path(ACTIVE_METRICS)

    @staticmethod
    def _is_current_summary(summary: dict | None) -> bool:
        if not summary:
            return False
        return bool(
            summary.get("standard_prediction_format") == "Z rod tep tgama"
            and "benchmark_metrics" in summary
            and "extrapolation_benchmark_metrics" in summary
        )

    def training_status(self) -> dict:
        with self._lock:
            status = dict(self._train_job)
        staged_summary = self._load_staged_summary()
        has_current_staged = STAGED_ARTIFACT.exists() and self._is_current_summary(staged_summary)
        has_staged_high_z = STAGED_HIGH_Z_METRICS.exists()
        summary = staged_summary if has_current_staged else self._load_active_summary()
        high_z_metrics_path = STAGED_HIGH_Z_METRICS if has_current_staged and has_staged_high_z else HIGH_Z_METADATA_PATH
        summary = self._summary_with_routed_high_z_metrics(summary, high_z_metrics_path)
        status["has_staged_artifact"] = has_current_staged and has_staged_high_z
        status["has_staged_unified_model"] = has_current_staged
        status["has_staged_high_z_model"] = has_staged_high_z
        status["staged_artifact"] = str(STAGED_ARTIFACT)
        status["staged_high_z_model"] = str(STAGED_HIGH_Z_MODEL)
        status["summary"] = self._summary_for_api(summary)
        status["summary_source"] = "staged" if has_current_staged else "active"
        return status

    def start_training(self, payload: dict) -> dict:
        with self._lock:
            if self._train_job.get("status") == "running":
                raise ValueError("training is already running")
            self._train_job = {
                "status": "running",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": "",
            }

        standard_data_path = Path(str(payload.get("standardDataPath") or STANDARD_TRAINING_DATA_PATH))
        self._remove_staged_high_z_files()
        cmd = [
            sys.executable,
            str(ROOT / "train_unified_free_path_model.py"),
            "--standard-data",
            str(standard_data_path),
            "--artifact",
            str(STAGED_ARTIFACT),
            "--metrics",
            str(STAGED_METRICS),
            "--benchmark-csv",
            str(STAGED_BENCHMARK),
            "--extrapolation-csv",
            str(STAGED_EXTRAPOLATION_BENCHMARK),
            "--k-neighbors",
            "260",
            "--local-degree",
            "3",
            "--ridge-alpha",
            "0.001",
            "--distance-power",
            "1.6",
            "--z-weight",
            "2.0",
            "--au-k-neighbors",
            "140",
            "--au-local-degree",
            "2",
            "--au-ridge-alpha",
            "0.003",
            "--au-distance-power",
            "1.1",
            "--disable-au-low-boundary-extension",
            "--enable-au-low-rod-log-shift",
            "--au-low-rod-shift-trigger-below",
            "-1.35",
            "--au-low-rod-shift-reference-rod",
            "-1.87",
            "--au-low-rod-shift-power",
            "2.0",
            "--au-low-rod-shift",
            "0.125",
            "--enable-au-high-boundary-extension",
            "--au-high-boundary-nfit",
            "9",
            "--au-high-boundary-knn",
            "64",
            "--au-high-boundary-power",
            "0.0",
            "--au-high-boundary-trigger-above",
            "4.0",
            "--au-high-boundary-interp-neighbors",
            "16",
            "--au-high-boundary-interp-power",
            "1.0",
            "--au-high-boundary-blend",
            "0.75",
            "--au-high-boundary-output-shift",
            "0.0",
            "--enable-au-high-both-low-log-shift",
            "--au-high-both-low-shift-trigger-above",
            "3.9",
            "--au-high-both-low-shift-reference-rod",
            "4.477",
            "--au-high-both-low-shift-power",
            "1.0",
            "--au-high-both-low-shift",
            "1.5",
            "--enable-au-high-both-low-log-shift-refine",
            "--au-high-both-low-refine-trigger-above",
            "3.9",
            "--au-high-both-low-refine-reference-rod",
            "4.477",
            "--au-high-both-low-refine-power",
            "0.5",
            "--au-high-both-low-refine-shift",
            "1.5",
            "--enable-au-high-curve-polynomial-extension",
            "--au-high-curve-poly-degree",
            "2",
            "--au-high-curve-poly-nfit",
            "6",
            "--au-high-curve-poly-ridge",
            "0.1",
            "--au-high-curve-poly-trigger-above",
            "3.9",
            "--au-high-curve-poly-interp-neighbors",
            "1",
            "--au-high-curve-poly-interp-power",
            "2.0",
            "--au-high-curve-poly-blend",
            "0.5",
            "--enable-au-high-low-tgama-polynomial-extension",
            "--au-high-low-tgama-poly-degree",
            "2",
            "--au-high-low-tgama-poly-nfit",
            "24",
            "--au-high-low-tgama-poly-ridge",
            "0.01",
            "--au-high-low-tgama-poly-trigger-above",
            "3.9",
            "--au-high-low-tgama-poly-interp-neighbors",
            "1",
            "--au-high-low-tgama-poly-interp-power",
            "2.0",
            "--au-high-low-tgama-poly-blend",
            "0.35",
            "--enable-au-high-low-tgama-log-shift-refine",
            "--au-high-low-tgama-refine-trigger-above",
            "3.9",
            "--au-high-low-tgama-refine-reference-rod",
            "4.477",
            "--au-high-low-tgama-refine-power",
            "0.5",
            "--au-high-low-tgama-refine-shift",
            "0.02",
            "--enable-au-high-low-tep-polynomial-refine",
            "--au-high-low-tep-refine-degree",
            "2",
            "--au-high-low-tep-refine-nfit",
            "32",
            "--au-high-low-tep-refine-ridge",
            "10.0",
            "--au-high-low-tep-refine-trigger-above",
            "3.9",
            "--au-high-low-tep-refine-interp-neighbors",
            "1",
            "--au-high-low-tep-refine-interp-power",
            "2.0",
            "--au-high-low-tep-refine-blend",
            "0.2",
            "--enable-au-high-stat-cap",
            "--au-high-stat-cap-tail-min-rod",
            "2.5",
            "--au-high-stat-cap-stat",
            "q10",
            "--au-high-stat-cap-trigger-above",
            "3.9",
            "--au-high-stat-cap-interp-neighbors",
            "1",
            "--au-high-stat-cap-interp-power",
            "2.0",
            "--au-high-stat-cap-output-shift",
            "0.0",
            "--au-high-stat-cap-strength",
            "1.0",
            "--enable-au-high-low-tep-high-tgama-stat-cap",
            "--au-high-low-tep-high-tgama-cap-tail-min-rod",
            "2.5",
            "--au-high-low-tep-high-tgama-cap-stat",
            "q01",
            "--au-high-low-tep-high-tgama-cap-trigger-above",
            "3.9",
            "--au-high-low-tep-high-tgama-cap-tgama-min",
            "3.0",
            "--au-high-low-tep-high-tgama-cap-output-shift",
            "0.5",
            "--au-high-low-tep-high-tgama-cap-strength",
            "1.0",
        ]
        high_z_cmd = [
            sys.executable,
            str(ROOT / "stage_noop_high_z_model.py"),
            "--base-model",
            str(STAGED_ARTIFACT),
            "--model-path",
            str(STAGED_HIGH_Z_MODEL),
            "--metrics-path",
            str(STAGED_HIGH_Z_METRICS),
        ]

        def runner() -> None:
            try:
                unified_completed = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                high_z_completed = None
                if unified_completed.returncode == 0:
                    high_z_completed = subprocess.run(
                        high_z_cmd,
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                final_returncode = unified_completed.returncode
                if high_z_completed is not None:
                    final_returncode = high_z_completed.returncode
                stdout = unified_completed.stdout
                stderr = unified_completed.stderr
                if high_z_completed is not None:
                    stdout += "\n\n===== staged high-Z training =====\n" + high_z_completed.stdout
                    stderr += "\n\n===== staged high-Z training =====\n" + high_z_completed.stderr
                with self._lock:
                    self._train_job.update(
                        {
                            "status": "succeeded" if final_returncode == 0 else "failed",
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "returncode": final_returncode,
                            "stdout": stdout[-8000:],
                            "stderr": stderr[-8000:],
                            "error": "" if final_returncode == 0 else (stderr or stdout)[-2000:],
                        }
                    )
            except Exception as exc:
                with self._lock:
                    self._train_job.update(
                        {
                            "status": "failed",
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "returncode": -1,
                            "error": str(exc),
                        }
                    )

        threading.Thread(target=runner, daemon=True).start()
        return self.training_status()

    def confirm_staged_model(self) -> dict:
        status = self.training_status()
        if status.get("status") == "running":
            raise ValueError("training is still running")
        if not STAGED_ARTIFACT.exists():
            raise ValueError("no staged model artifact exists")
        if not STAGED_HIGH_Z_METRICS.exists():
            raise ValueError("no staged high-Z routing metadata exists")
        staged_high_z_metadata = json.loads(STAGED_HIGH_Z_METRICS.read_text(encoding="utf-8"))
        if staged_high_z_metadata.get("route_enabled") is not False and not STAGED_HIGH_Z_MODEL.exists():
            raise ValueError("enabled staged high-Z route has no model artifact")
        if not self._is_current_summary(self._load_staged_summary()):
            raise ValueError("staged model is from an older metric format; please run staged training again")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        for src in [ACTIVE_ARTIFACT, ACTIVE_METRICS, ACTIVE_BENCHMARK, ACTIVE_EXTRAPOLATION_BENCHMARK]:
            if src.exists():
                backup = src.with_name(f"{src.stem}.bak_{stamp}{src.suffix}")
                shutil.copy2(src, backup)
        for src in self._high_z_paths_from_metadata(HIGH_Z_METADATA_PATH, HIGH_Z_MODEL_PATH):
            if src.exists():
                backup = src.with_name(f"{src.stem}.bak_{stamp}{src.suffix}")
                shutil.copy2(src, backup)
        if HIGH_Z_METADATA_PATH.exists():
            backup = HIGH_Z_METADATA_PATH.with_name(f"{HIGH_Z_METADATA_PATH.stem}.bak_{stamp}{HIGH_Z_METADATA_PATH.suffix}")
            shutil.copy2(HIGH_Z_METADATA_PATH, backup)

        copy_pairs = [
            (STAGED_ARTIFACT, ACTIVE_ARTIFACT),
            (STAGED_METRICS, ACTIVE_METRICS),
            (STAGED_BENCHMARK, ACTIVE_BENCHMARK),
            (STAGED_EXTRAPOLATION_BENCHMARK, ACTIVE_EXTRAPOLATION_BENCHMARK),
        ]
        for src, dst in copy_pairs:
            if src.exists():
                shutil.copy2(src, dst)
        for src in self._high_z_paths_from_metadata(STAGED_HIGH_Z_METRICS, STAGED_HIGH_Z_MODEL):
            if src.exists():
                dst = self._staged_high_z_destination(src)
                shutil.copy2(src, dst)
        active_high_z_metadata = self._rewrite_high_z_metadata_paths(staged_high_z_metadata)
        HIGH_Z_METADATA_PATH.write_text(json.dumps(active_high_z_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        for src in [STAGED_ARTIFACT, STAGED_METRICS, STAGED_BENCHMARK, STAGED_EXTRAPOLATION_BENCHMARK]:
            try:
                src.unlink(missing_ok=True)
            except TypeError:
                if src.exists():
                    src.unlink()
        self._remove_staged_high_z_files()

        self.artifact_path = ACTIVE_ARTIFACT
        self.model = UnifiedModel.load(ACTIVE_ARTIFACT)
        self.high_z_model = HighZModel.load_if_available()
        self.metadata = self.model.metadata
        with self._truth_lock:
            self._truth_cache = None
        with self._lock:
            self._train_job.update(
                {
                    "status": "published",
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": "",
                }
            )
        return {"ok": True, "training_status": self.training_status(), "datasets": self.dataset_payload()}


def make_handler(app: FreePathWebApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FreePathWeb/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print("%s - %s" % (self.address_string(), fmt % args))

        def send_json(self, data: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_html(self) -> None:
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_training_file(self, path: Path) -> None:
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path in {"/", "/index.html"}:
                    self.send_html()
                elif path == "/api/datasets":
                    self.send_json(app.dataset_payload())
                elif path == "/api/flash/control":
                    self.send_json(app.flash_control_payload())
                elif path == "/api/flash/status":
                    self.send_json(app.flash_status_payload(parse_qs(parsed.query)))
                elif path == "/api/flash/file":
                    self.send_json(app.flash_file_payload(parse_qs(parsed.query)))
                elif path == "/api/flash/plot":
                    self.send_json(app.flash_plot_payload(parse_qs(parsed.query)))
                elif path == "/api/predict/status":
                    self.send_json(app.prediction_status(parse_qs(parsed.query)))
                elif path == "/api/train/status":
                    self.send_json(app.training_status())
                elif path == "/api/training-data/export":
                    self.send_training_file(app.export_training_data_path(parse_qs(parsed.query)))
                elif path == "/health":
                    self.send_json({"ok": True})
                elif path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
            elif path in {"/api/datasets", "/health"}:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
            elif path == "/api/training-data/export":
                try:
                    file_path = app.export_training_data_path(parse_qs(parsed.query))
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(file_path.stat().st_size))
                    self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
                    self.end_headers()
                except Exception:
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
            else:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/predict":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.predict(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/predict/start":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.start_prediction(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/predict/batch":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.predict_batch(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/flash/control":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.update_flash_control(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/flash/params":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.update_flash_params(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/flash/run":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.start_flash_run(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/flash/stop":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.stop_flash_run(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/flash/command":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.run_flash_command(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/flash/project":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.create_flash_project(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/train/start":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.start_training(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/train/confirm":
                try:
                    self.send_json(app.confirm_staged_model())
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/training-data/import":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.import_training_data(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/evaluate/upload":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.evaluate_uploaded(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/benchmark/file":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.benchmark_file(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/zpinch/predict":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw) if raw else {}
                    self.send_json(app.zpinch_predict(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the free-path prediction web app.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--model", type=Path, default=ACTIVE_ARTIFACT)
    args = parser.parse_args()

    app = FreePathWebApp(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving free-path web app at http://{args.host}:{args.port}")
    print(f"artifact: {args.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
