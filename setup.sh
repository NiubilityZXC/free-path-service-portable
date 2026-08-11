#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_COMMAND=${PYTHON_BIN:-python3}
INSTALL_FULL=0
INSTALL_TORCH=0
INSTALL_FLASH_CLI=1

usage() {
  echo "用法: ./setup.sh [--full] [--torch] [--python /path/to/python] [--no-flash-cli]"
  echo "默认安装网页、统一自由程模型和当前 Z-pinch 推理依赖。"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      INSTALL_FULL=1
      shift
      ;;
    --torch)
      INSTALL_TORCH=1
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python 缺少路径" >&2; exit 2; }
      PYTHON_COMMAND=$2
      shift 2
      ;;
    --no-flash-cli)
      INSTALL_FLASH_CLI=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v "$PYTHON_COMMAND" >/dev/null 2>&1 || {
  echo "找不到 Python: $PYTHON_COMMAND" >&2
  echo "推荐使用 Python 3.13，或直接运行 docker compose up --build -d。" >&2
  exit 1
}

"$PYTHON_COMMAND" - <<'PY'
import os
import sys

validated = (3, 13)
current = sys.version_info[:2]
if current != validated and os.environ.get("ALLOW_UNTESTED_PYTHON") != "1":
    raise SystemExit(
        f"当前 Python 是 {current[0]}.{current[1]}，模型验证环境是 3.13。"
        "请改用 Python 3.13；确需尝试可设置 ALLOW_UNTESTED_PYTHON=1。"
    )
print(f"Python {sys.version.split()[0]}")
PY

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  "$PYTHON_COMMAND" -m venv "$ROOT_DIR/.venv"
fi
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements-runtime.txt"

if [[ $INSTALL_FULL -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements-training.txt"
fi
if [[ $INSTALL_TORCH -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements-torch.txt"
fi
if [[ $INSTALL_FLASH_CLI -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install -e "$ROOT_DIR/cli-anything-flash/agent-harness"
fi

mkdir -p "$ROOT_DIR/runtime/logs"
echo "安装完成。启动: $ROOT_DIR/start_all.sh"
echo "离线模型自检: $VENV_PYTHON $ROOT_DIR/scripts/portable_smoke_test.py"
