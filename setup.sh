#!/usr/bin/env bash
set -euo pipefail

# Local setup for Q-Hawkeye (fork of Visual-RFT)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/src/virft"

# Triton/Deepspeed cache (local)
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$SCRIPT_DIR/.cache/triton"}
export DEEPSPEED_CACHE_DIR=${DEEPSPEED_CACHE_DIR:-"$SCRIPT_DIR/.cache/deepspeed"}
mkdir -p "$TRITON_CACHE_DIR" "$DEEPSPEED_CACHE_DIR"

pip install -e ".[dev]"
