#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR="$ROOT_DIR/runtime"
LOG_DIR="$RUNTIME_DIR/logs"
FREE_PATH_HOST=${FREE_PATH_HOST:-0.0.0.0}
FREE_PATH_PORT=${FREE_PATH_PORT:-8790}
CAPACITOR_PORT=8890
FOREGROUND=0

if [[ ${1:-} == "--foreground" ]]; then
  FOREGROUND=1
elif [[ $# -gt 0 ]]; then
  echo "用法: ./start_all.sh [--foreground]" >&2
  exit 2
fi

if [[ -n ${PYTHON_BIN:-} ]]; then
  PYTHON_COMMAND=$PYTHON_BIN
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_COMMAND="$ROOT_DIR/.venv/bin/python"
else
  echo "未找到 .venv；请先运行 ./setup.sh，或设置 PYTHON_BIN。" >&2
  exit 1
fi
if [[ "$PYTHON_COMMAND" != */* ]]; then
  PYTHON_COMMAND=$(command -v "$PYTHON_COMMAND")
fi

mkdir -p "$LOG_DIR"

pid_is_valid() {
  local pid_file=$1
  local marker=$2
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid=$(tr -cd '0-9' < "$pid_file")
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local command_line
  command_line=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ "$command_line" == *"$marker"* ]]
}

wait_url() {
  "$PYTHON_COMMAND" "$ROOT_DIR/scripts/wait_http.py" "$1" --timeout 60
}

COMMON_ENV=(
  "FREE_PATH_BUNDLE_ROOT=$ROOT_DIR"
  "CAPACITOR_APP_DIR=$ROOT_DIR/pulse_capacitor_online_eval"
  "ZPINCH_ROOT=$ROOT_DIR/zpinch"
  "FREE_PATH_PYTHON=$PYTHON_COMMAND"
  "FLASH_ROOT=${FLASH_ROOT:-$ROOT_DIR/FLASH4.8}"
)
CAP_COMMAND=("$PYTHON_COMMAND" -u "$ROOT_DIR/pulse_capacitor_online_eval/app.py" --host "$FREE_PATH_HOST" --port "$CAPACITOR_PORT")
FREE_COMMAND=("$PYTHON_COMMAND" -u "$ROOT_DIR/freepath_service/free_path_web_app.py" --host "$FREE_PATH_HOST" --port "$FREE_PATH_PORT" --model "$ROOT_DIR/freepath_service/free_path_model_outputs/unified_free_path_model.npz")

if [[ $FOREGROUND -eq 1 ]]; then
  child_pids=()
  cleanup() {
    for pid in "${child_pids[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM
  env "${COMMON_ENV[@]}" "${CAP_COMMAND[@]}" >>"$LOG_DIR/capacitor.log" 2>&1 &
  child_pids+=("$!")
  env "${COMMON_ENV[@]}" "${FREE_COMMAND[@]}" >>"$LOG_DIR/freepath.log" 2>&1 &
  child_pids+=("$!")
  wait_url "http://127.0.0.1:$CAPACITOR_PORT/"
  wait_url "http://127.0.0.1:$FREE_PATH_PORT/health"
  echo "服务已启动: http://127.0.0.1:$FREE_PATH_PORT/"
  wait -n "${child_pids[@]}"
  exit $?
fi

new_pids=()
rollback() {
  for pid in "${new_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap rollback ERR

CAP_PID_FILE="$RUNTIME_DIR/capacitor.pid"
FREE_PID_FILE="$RUNTIME_DIR/freepath.pid"
if pid_is_valid "$CAP_PID_FILE" "pulse_capacitor_online_eval/app.py"; then
  echo "电容服务已经在运行，PID $(tr -cd '0-9' < "$CAP_PID_FILE")"
else
  rm -f "$CAP_PID_FILE"
  nohup setsid env "${COMMON_ENV[@]}" "${CAP_COMMAND[@]}" </dev/null >>"$LOG_DIR/capacitor.log" 2>&1 &
  cap_pid=$!
  echo "$cap_pid" > "$CAP_PID_FILE"
  new_pids+=("$cap_pid")
fi
wait_url "http://127.0.0.1:$CAPACITOR_PORT/"

if pid_is_valid "$FREE_PID_FILE" "free_path_web_app.py"; then
  echo "自由程服务已经在运行，PID $(tr -cd '0-9' < "$FREE_PID_FILE")"
else
  rm -f "$FREE_PID_FILE"
  nohup setsid env "${COMMON_ENV[@]}" "${FREE_COMMAND[@]}" </dev/null >>"$LOG_DIR/freepath.log" 2>&1 &
  free_pid=$!
  echo "$free_pid" > "$FREE_PID_FILE"
  new_pids+=("$free_pid")
fi
wait_url "http://127.0.0.1:$FREE_PATH_PORT/health"
trap - ERR

echo "主服务: http://127.0.0.1:$FREE_PATH_PORT/"
echo "内网访问: http://<这台电脑的IP>:$FREE_PATH_PORT/"
echo "电容服务: http://127.0.0.1:$CAPACITOR_PORT/"
echo "日志目录: $LOG_DIR"
