#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

source "$SCRIPT_DIR/setup_env.sh"

cd "$PROJECT_ROOT"
exec cargo check -p moshi-server --features cuda "$@"
