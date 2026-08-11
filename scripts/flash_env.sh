#!/usr/bin/env bash
# Source this file from the portable bundle root before running FLASH tools.

_FLASH_BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export FLASH_ROOT=${FLASH_ROOT:-$_FLASH_BUNDLE_ROOT/FLASH4.8}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export FLASH_OFFLOAD_RUNTIME=${FLASH_OFFLOAD_RUNTIME:-$HOME/.local/gcc-offload/usr/lib/x86_64-linux-gnu}

_FLASH_HYPRE_LIB="$FLASH_ROOT/gpu/deps/install/hypre-cuda-3.1.0/lib"
_FLASH_CUDA_LIB="$CUDA_HOME/lib64"
export PATH="$FLASH_ROOT/bin:${PATH:-}"
if [[ -d "$_FLASH_HYPRE_LIB" ]]; then
  export LD_LIBRARY_PATH="$_FLASH_HYPRE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$_FLASH_CUDA_LIB" ]]; then
  export LD_LIBRARY_PATH="$_FLASH_CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

printf 'FLASH_ROOT=%s\n' "$FLASH_ROOT"
printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
unset _FLASH_BUNDLE_ROOT _FLASH_HYPRE_LIB _FLASH_CUDA_LIB
