#!/bin/bash
# Build the C++ separate FFN TP vs single benchmark.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build_cpp"
CACHE_DIR="$BUILD_DIR/clang_module_cache"
FAKE_HOME="$BUILD_DIR/fake_home"
mkdir -p "$CACHE_DIR" "$FAKE_HOME"
export CLANG_MODULE_CACHE_PATH="$CACHE_DIR"
export HOME="$FAKE_HOME"

# Only rerun CMake configure when the build directory is missing a cache.
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -B "$BUILD_DIR" -S "$ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_BENCHMARKS=ON \
    -DCMAKE_CXX_FLAGS="-fmodules-cache-path=$CACHE_DIR"
else
  echo "Found existing build at $BUILD_DIR (skipping configure)."
fi

cmake --build "$BUILD_DIR" --target separate_tp_vs_single

echo "Benchmark built at: $BUILD_DIR/benchmarks/cpp/separate_tp_vs_single"
