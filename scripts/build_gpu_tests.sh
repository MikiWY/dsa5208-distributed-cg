#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p build

echo "===== Build GPU SpMV ====="

nvcc -std=c++17 -O2 -arch=sm_86 \
  -Iinclude \
  src/matrix.cpp \
  src/gpu_spmv.cu \
  tests/test_gpu_spmv.cu \
  -o build/test_gpu_spmv

echo "===== Build GPU CG ====="

nvcc -std=c++17 -O2 -arch=sm_86 \
  -Iinclude \
  src/matrix.cpp \
  src/cg.cpp \
  src/gpu_cg.cu \
  tests/test_gpu_cg.cu \
  -o build/test_gpu_cg

echo "PASS: GPU test compilation"

echo "===== Build GPU Workspace ====="

nvcc -std=c++17 -O2 -arch=sm_86 \
  -Iinclude \
  src/matrix.cpp \
  src/gpu_benchmark.cu \
  tests/test_gpu_workspace.cu \
  -o build/test_gpu_workspace

echo "PASS: GPU Workspace compilation"

echo "===== Build GPU Benchmark ====="

COMMIT=$(git rev-parse HEAD)

nvcc -std=c++17 -O2 -arch=sm_86 \
  -Iinclude \
  "-DDSA5208_GIT_COMMIT=\"$COMMIT\"" \
  src/matrix.cpp \
  src/gpu_benchmark.cu \
  src/benchmark_csv.cpp \
  src/correctness_csv.cpp \
  src/gpu_timing_csv.cpp \
  src/gpu_benchmark_main.cu \
  -o build/gpu_benchmark

echo "PASS: GPU Benchmark compilation"
