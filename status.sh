#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_COMMAND=${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}
FREE_PATH_PORT=${FREE_PATH_PORT:-8790}

show_process() {
  local name=$1
  local pid_file=$2
  local marker=$3
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(tr -cd '0-9' < "$pid_file")
    local command_line
    command_line=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && [[ "$command_line" == *"$marker"* ]]; then
      echo "$name: 运行中 (PID $pid)"
      return
    fi
  fi
  echo "$name: 未运行或 PID 已失效"
}

show_process "自由程服务" "$ROOT_DIR/runtime/freepath.pid" "free_path_web_app.py"
show_process "电容服务" "$ROOT_DIR/runtime/capacitor.pid" "pulse_capacitor_online_eval/app.py"
if [[ -x "$PYTHON_COMMAND" ]]; then
  "$PYTHON_COMMAND" "$ROOT_DIR/scripts/wait_http.py" "http://127.0.0.1:$FREE_PATH_PORT/health" --timeout 1 --quiet \
    && echo "HTTP $FREE_PATH_PORT: 正常" \
    || echo "HTTP $FREE_PATH_PORT: 不可访问"
  "$PYTHON_COMMAND" "$ROOT_DIR/scripts/wait_http.py" "http://127.0.0.1:8890/" --timeout 1 --quiet \
    && echo "HTTP 8890: 正常" \
    || echo "HTTP 8890: 不可访问"
fi
