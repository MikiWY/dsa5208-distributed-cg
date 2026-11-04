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
