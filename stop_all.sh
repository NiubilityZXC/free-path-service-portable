#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR="$ROOT_DIR/runtime"

stop_one() {
  local name=$1
  local pid_file=$2
  local marker=$3
  if [[ ! -f "$pid_file" ]]; then
    echo "$name 未记录 PID"
    return
  fi
  local pid
  pid=$(tr -cd '0-9' < "$pid_file")
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "$name 已停止（清理旧 PID 文件）"
    rm -f "$pid_file"
    return
  fi
  local command_line
  command_line=$(ps -p "$pid" -o args= 2>/dev/null || true)
  if [[ "$command_line" != *"$marker"* ]]; then
    echo "$name 的 PID $pid 已被其他进程占用，不执行停止。" >&2
    return 1
  fi
  kill "$pid"
  for _ in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "$name 已停止"
      return
    fi
    sleep 0.2
  done
  echo "$name 未在 6 秒内退出；请检查 PID $pid。" >&2
  return 1
}

result=0
stop_one "自由程服务" "$RUNTIME_DIR/freepath.pid" "free_path_web_app.py" || result=1
stop_one "电容服务" "$RUNTIME_DIR/capacitor.pid" "pulse_capacitor_online_eval/app.py" || result=1
exit $result
