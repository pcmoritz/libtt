#!/bin/bash
# Build LLVM/MLIR, StableHLO, and XLA headers for libtt.
#
# Usage:
#   ./scripts/setup_deps_llvm.sh [--prefix /path/to/install] [--force]

source "$(dirname "$0")/setup_deps_common.sh" "$@"

# Pin to versions matching jaxlib 0.9.0 for StableHLO bytecode compatibility.
XLA_COMMIT="${XLA_COMMIT:-bb760b047bdbfeff962f0366ad5cc782c98657e0}"
STABLEHLO_COMMIT="${STABLEHLO_COMMIT:-127d2f238010589ac96f2f402a27afc9dccbb7ab}"
LLVM_COMMIT="${LLVM_COMMIT:-f6d0a512972a74ef100723b9526a6a0ddb23f894}"

echo "=== libtt LLVM/StableHLO setup ==="
echo "Prefix:       $PREFIX"
echo "Jobs:         $JOBS"
echo "XLA:          $XLA_COMMIT"
echo "StableHLO:    $STABLEHLO_COMMIT"
echo "LLVM:         $LLVM_COMMIT"
echo ""

LLVM_STAMP="$PREFIX/.llvm-versions"
LLVM_EXPECTED_STAMP="llvm=$LLVM_COMMIT stablehlo=$STABLEHLO_COMMIT xla=$XLA_COMMIT"
if [ -f "$LLVM_STAMP" ] && [ "$(cat "$LLVM_STAMP")" != "$LLVM_EXPECTED_STAMP" ]; then
    echo "=== Version mismatch detected, forcing rebuild ==="
    FORCE_REBUILD=true
fi

if [ "$FORCE_REBUILD" = true ]; then
    rm -rf "$PREFIX/lib/cmake/mlir" "$PREFIX/lib/cmake/llvm"
    rm -f "$PREFIX/lib/libStablehloOps.a"
    rm -f "$PREFIX/include/xla/pjrt/c/pjrt_c_api.h"
    rm -f "$LLVM_STAMP"
    rm -rf "$BUILD_DIR/llvm-build" "$BUILD_DIR/stablehlo-build"
fi

LLVM_DIR="$BUILD_DIR/llvm-project"
if [ ! -d "$LLVM_DIR" ]; then
    echo "=== Fetching LLVM commit $LLVM_COMMIT (minimal clone) ==="
    mkdir -p "$LLVM_DIR"
    cd "$LLVM_DIR"
    git init
    git remote add origin https://github.com/llvm/llvm-project.git
    git fetch --depth 1 origin "$LLVM_COMMIT"
    git checkout FETCH_HEAD
else
    echo "=== LLVM already cloned ==="
    cd "$LLVM_DIR"
    CURRENT_COMMIT=$(git rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$LLVM_COMMIT" ]; then
        echo "=== Fetching LLVM commit $LLVM_COMMIT ==="
        git fetch --depth 1 origin "$LLVM_COMMIT"
        git checkout FETCH_HEAD
    fi
fi

LLVM_BUILD_DIR="$BUILD_DIR/llvm-build"
if [ ! -f "$PREFIX/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "=== Building LLVM/MLIR ==="
    cmake -G Ninja -B "$LLVM_BUILD_DIR" -S "$LLVM_DIR/llvm" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_BACKTRACES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DMLIR_ENABLE_EXECUTION_ENGINE=OFF

    cmake --build "$LLVM_BUILD_DIR" -j "$JOBS"
    cmake --install "$LLVM_BUILD_DIR"
    echo "LLVM/MLIR installed to $PREFIX"
else
    echo "=== LLVM/MLIR already installed ==="
fi

install_stablehlo_headers() {
    if [ ! -d "$STABLEHLO_DIR" ]; then
        echo "WARNING: StableHLO source not available, skipping header install"
        return
    fi
    echo "=== Installing StableHLO headers ==="
    mkdir -p "$PREFIX/include/stablehlo/dialect"
    mkdir -p "$PREFIX/include/stablehlo/api"
    mkdir -p "$PREFIX/include/stablehlo/transforms"
    mkdir -p "$PREFIX/include/stablehlo/transforms/optimization"
    cp "$STABLEHLO_DIR/stablehlo/dialect/"*.h "$PREFIX/include/stablehlo/dialect/"
    cp "$STABLEHLO_DIR/stablehlo/api/"*.h "$PREFIX/include/stablehlo/api/"
    cp "$STABLEHLO_DIR/stablehlo/transforms/"*.h "$PREFIX/include/stablehlo/transforms/"
    cp "$STABLEHLO_DIR/stablehlo/transforms/optimization/"*.h \
        "$PREFIX/include/stablehlo/transforms/optimization/"
    if [ -d "$STABLEHLO_BUILD_DIR" ]; then
        cp "$STABLEHLO_BUILD_DIR/stablehlo/dialect/"*.inc "$PREFIX/include/stablehlo/dialect/" 2>/dev/null || true
        cp "$STABLEHLO_BUILD_DIR/stablehlo/transforms/"*.inc "$PREFIX/include/stablehlo/transforms/" 2>/dev/null || true
        cp "$STABLEHLO_BUILD_DIR/stablehlo/transforms/optimization/"*.inc \
            "$PREFIX/include/stablehlo/transforms/optimization/" 2>/dev/null || true
    fi
}

STABLEHLO_DIR="$BUILD_DIR/stablehlo"
if [ ! -d "$STABLEHLO_DIR" ]; then
    echo "=== Cloning StableHLO at commit $STABLEHLO_COMMIT ==="
    mkdir -p "$STABLEHLO_DIR"
    cd "$STABLEHLO_DIR"
    git init
    git remote add origin https://github.com/openxla/stablehlo.git
    git fetch --depth 1 origin "$STABLEHLO_COMMIT"
    git checkout FETCH_HEAD
else
    echo "=== Checking StableHLO commit ==="
    cd "$STABLEHLO_DIR"
    CURRENT_COMMIT=$(git rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$STABLEHLO_COMMIT" ]; then
        echo "=== Updating StableHLO to commit $STABLEHLO_COMMIT ==="
        git fetch --depth 1 origin "$STABLEHLO_COMMIT"
        git checkout FETCH_HEAD
    fi
fi

STABLEHLO_BUILD_DIR="$BUILD_DIR/stablehlo-build"
if [ ! -f "$PREFIX/lib/libStablehloOps.a" ]; then
    echo "=== Building StableHLO ==="
    cmake -G Ninja -B "$STABLEHLO_BUILD_DIR" -S "$STABLEHLO_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DMLIR_DIR="$PREFIX/lib/cmake/mlir" \
        -DLLVM_DIR="$PREFIX/lib/cmake/llvm" \
        -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
        -DSTABLEHLO_BUILD_EMBEDDED=OFF

    cmake --build "$STABLEHLO_BUILD_DIR" -j "$JOBS"
    cmake --install "$STABLEHLO_BUILD_DIR"
    install_stablehlo_headers
    echo "StableHLO installed to $PREFIX"
else
    echo "=== StableHLO already installed ==="
    install_stablehlo_headers
fi

if [ ! -f "$PREFIX/include/xla/pjrt/c/pjrt_c_api.h" ]; then
    XLA_DIR="$BUILD_DIR/xla"
    echo "=== Fetching XLA headers at commit $XLA_COMMIT ==="
    if [ ! -d "$XLA_DIR" ]; then
        mkdir -p "$XLA_DIR"
        cd "$XLA_DIR"
        git init
        git remote add origin https://github.com/openxla/xla.git
        git fetch --depth 1 origin "$XLA_COMMIT"
        git checkout FETCH_HEAD
    else
        cd "$XLA_DIR"
        CURRENT_COMMIT=$(git rev-parse HEAD)
        if [ "$CURRENT_COMMIT" != "$XLA_COMMIT" ]; then
            git fetch --depth 1 origin "$XLA_COMMIT"
            git checkout FETCH_HEAD
        fi
    fi

    mkdir -p "$PREFIX/include/xla/pjrt/c"
    cp "$XLA_DIR/xla/pjrt/c/pjrt_c_api.h" "$PREFIX/include/xla/pjrt/c/"
fi

echo "$LLVM_EXPECTED_STAMP" > "$LLVM_STAMP"

echo ""
echo "=== LLVM/StableHLO setup complete ==="
echo "Installed to: $PREFIX"
