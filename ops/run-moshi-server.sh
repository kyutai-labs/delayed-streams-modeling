#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SENTENCEPIECE_PREFIX="${SENTENCEPIECE_PREFIX:-$HOME/.local/sentencepiece}"
PKGCONFIG_DIR="${SENTENCEPIECE_PREFIX}/lib/pkgconfig"

maybe_setup_cuda_env() {
  local nvcc_bin=""
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_bin="$(command -v nvcc)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin="/usr/local/cuda/bin/nvcc"
  fi

  if [[ -n "$nvcc_bin" ]] && "$nvcc_bin" --version 2>/dev/null | grep -q "release 13.1"; then
    source "$SCRIPT_DIR/setup_env.sh"
  fi
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

maybe_free_gpu_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "nvidia-smi unavailable; skipping GPU memory check." >&2
    return 0
  fi

  echo "GPU memory usage:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r gpu_index gpu_name gpu_used gpu_total; do
    gpu_index="$(trim "$gpu_index")"
    gpu_name="$(trim "$gpu_name")"
    gpu_used="$(trim "$gpu_used")"
    gpu_total="$(trim "$gpu_total")"
    echo "  GPU ${gpu_index} (${gpu_name}): ${gpu_used} MiB / ${gpu_total} MiB"
  done

  local proc_lines
  proc_lines="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$proc_lines" ]]; then
    return 0
  fi

  echo "Processes using GPU memory:"
  echo "  PID  USER  MB  COMMAND"
  while IFS=',' read -r pid proc_name used_mb; do
    pid="$(trim "$pid")"
    proc_name="$(trim "$proc_name")"
    used_mb="$(trim "$used_mb")"
    [[ -z "$pid" ]] && continue
    local user cmd
    user="$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}' || true)"
    cmd="$(ps -o args= -p "$pid" 2>/dev/null | sed 's/^[[:space:]]*//' || true)"
    if [[ -z "$cmd" ]]; then
      cmd="$proc_name"
    fi
    echo "  ${pid}  ${user:-?}  ${used_mb}  ${cmd}"
  done <<< "$proc_lines"

  local pids
  read -r -p "Enter PIDs to terminate (space-separated) or press Enter to continue: " pids
  if [[ -n "$pids" ]]; then
    echo "Terminating: $pids"
    kill $pids || true
  fi
}

if [[ -d "$PKGCONFIG_DIR" ]]; then
  PKG_CONFIG_PATH_WITH_LOCAL="${PKGCONFIG_DIR}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  if PKG_CONFIG_PATH="$PKG_CONFIG_PATH_WITH_LOCAL" pkg-config --exists sentencepiece 2>/dev/null; then
    export PKG_CONFIG_PATH="$PKG_CONFIG_PATH_WITH_LOCAL"
    export LD_LIBRARY_PATH="${SENTENCEPIECE_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

if ! pkg-config --exists sentencepiece 2>/dev/null; then
  "$SCRIPT_DIR/setup-sentencepiece.sh"
  export PKG_CONFIG_PATH="${PKGCONFIG_DIR}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  export LD_LIBRARY_PATH="${SENTENCEPIECE_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

maybe_setup_cuda_env
maybe_free_gpu_memory

cargo install --path "${REPO_ROOT}/server/rust/moshi/moshi-server" --features cuda --verbose

moshi-server worker --config configs/stt/config-stt-en_fr-hf.toml
