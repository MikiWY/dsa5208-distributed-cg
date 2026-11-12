#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CXX="${CXX:-g++}"
MPICXX="${MPICXX:-mpicxx}"
OPENMP_FLAG="${OPENMP_FLAG:--fopenmp}"
BUILD_DIR="${BUILD_DIR:-build}"

COMMIT="$(git rev-parse HEAD)"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
  -std=c++17
  -O2
  -Wall
  -Wextra
  -Wpedantic
  -Iinclude
  "-DDSA5208_GIT_COMMIT=\"${COMMIT}\""
)

echo "===== Release configuration ====="
echo "Git commit: $COMMIT"
echo "CXX: $CXX"
echo "MPICXX: $MPICXX"
echo "OpenMP: $OPENMP_FLAG"

"$CXX" --version | head -n 1
"$MPICXX" --version | head -n 1

echo "===== Build Serial CG ====="

"$CXX" "${COMMON_FLAGS[@]}" \
  src/matrix.cpp \
  src/cg.cpp \
  src/benchmark_csv.cpp \
  src/serial_benchmark_main.cpp \
  -o "$BUILD_DIR/cg_serial_benchmark"

echo "PASS: Serial benchmark compiled"

echo "===== Build MPI and Hybrid CG ====="

"$MPICXX" "${COMMON_FLAGS[@]}" \
  "$OPENMP_FLAG" \
  src/matrix.cpp \
  src/partition.cpp \
  src/distributed_spmv.cpp \
  src/distributed_vector_ops.cpp \
  src/distributed_cg.cpp \
  src/halo_exchange.cpp \
  src/halo_spmv.cpp \
  src/distributed_cg_halo.cpp \
  src/openmp_vector_ops.cpp \
  src/hybrid_vector_ops.cpp \
  src/hybrid_spmv.cpp \
  src/hybrid_cg.cpp \
  src/hybrid_halo_spmv.cpp \
  src/hybrid_halo_cg.cpp \
  src/benchmark_csv.cpp \
  src/benchmark_options.cpp \
  src/correctness_csv.cpp \
  src/benchmark_main.cpp \
  -o "$BUILD_DIR/cg_mpi_benchmark"

echo "PASS: MPI and Hybrid benchmark compiled"
echo "PASS: All CPU benchmark targets compiled"
