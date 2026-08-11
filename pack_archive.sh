#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(dirname "$ROOT_DIR")
BUNDLE_NAME=$(basename "$ROOT_DIR")
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH=${1:-$PARENT_DIR/${BUNDLE_NAME}_${STAMP}.tar.gz}

mkdir -p "$(dirname "$OUTPUT_PATH")"
tar \
  --exclude="$BUNDLE_NAME/.git" \
  --exclude="$BUNDLE_NAME/.venv" \
  --exclude="$BUNDLE_NAME/.playwright-cli" \
  --exclude="$BUNDLE_NAME/runtime/*.pid" \
  --exclude="$BUNDLE_NAME/runtime/logs/*" \
  --exclude="$BUNDLE_NAME/**/__pycache__" \
  --exclude="$BUNDLE_NAME/**/*.pyc" \
  -czf "$OUTPUT_PATH" -C "$PARENT_DIR" "$BUNDLE_NAME"
sha256sum "$OUTPUT_PATH" > "$OUTPUT_PATH.sha256"
echo "归档: $OUTPUT_PATH"
echo "校验: $OUTPUT_PATH.sha256"
