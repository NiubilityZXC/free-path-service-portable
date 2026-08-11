#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

LINK_TMP=$(mktemp "$ROOT_DIR/.SYMLINKS.txt.XXXXXX")
HASH_TMP=$(mktemp "$ROOT_DIR/.SHA256SUMS.XXXXXX")
trap 'rm -f "$LINK_TMP" "$HASH_TMP"' EXIT

find . \
  -path './.git' -prune -o \
  -path './.venv' -prune -o \
  -path './runtime' -prune -o \
  -path './.playwright-cli' -prune -o \
  -type l -printf '%P -> %l\n' \
  | LC_ALL=C sort > "$LINK_TMP"
mv "$LINK_TMP" SYMLINKS.txt

git ls-files --cached -z \
  | LC_ALL=C sort -z \
  | while IFS= read -r -d '' path; do
      if [[ "$path" == SHA256SUMS || -L "$path" ]]; then
        continue
      fi
      digest=$(git cat-file --filters --path="$path" ":$path" | sha256sum | awk '{print $1}')
      printf '%s  %s\n' "$digest" "$path"
    done > "$HASH_TMP"
mv "$HASH_TMP" SHA256SUMS

trap - EXIT
printf 'SYMLINKS.txt: %s links\n' "$(wc -l < SYMLINKS.txt)"
printf 'SHA256SUMS: %s files\n' "$(wc -l < SHA256SUMS)"
