#!/bin/bash
# Setup LLVM/MLIR + StableHLO dependencies for libtt's optional MLIR frontend.
#
# Usage:
#   ./scripts/setup_deps.sh [--prefix /path/to/install] [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/setup_deps_llvm.sh" "$@"

PREFIX="${PREFIX:-$HOME/.local/libtt-deps}"
prev=""
for arg in "$@"; do
    if [ "$prev" = "--prefix" ]; then PREFIX="$arg"; fi
    prev="$arg"
done

echo ""
echo "=== libtt MLIR dependencies installed ==="
echo ""
echo "To build libtt with the MLIR frontend enabled:"
echo "  export CMAKE_PREFIX_PATH=$PREFIX"
echo "  cargo build --features mlir-frontend"
