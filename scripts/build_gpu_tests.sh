#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NVCC="${NVCC:-nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"

NVCC_ARGS=(
  -std=c++17
  -O2
  "-arch=${CUDA_ARCH}"
  -Iinclude
)

if [[ -n "${NVCC_HOST_CXX:-}" ]]; then
  if [[ ! -x "${NVCC_HOST_CXX}" ]]; then
    echo "ERROR: NVCC_HOST_CXX is not executable:"
    echo "${NVCC_HOST_CXX}"
    exit 1
  fi

  NVCC_ARGS+=("-ccbin=${NVCC_HOST_CXX}")
fi

echo "===== CUDA compiler ====="
"${NVCC}" --version

echo "===== Host compiler ====="

if [[ -n "${NVCC_HOST_CXX:-}" ]]; then
  "${NVCC_HOST_CXX}" --version | head -n 1
else
  echo "Using NVCC default host compiler"
  command -v g++ || true
  g++ --version | head -n 1 || true
fi

echo "===== CUDA architecture ====="
echo "${CUDA_ARCH}"

echo "===== Git commit ====="
COMMIT=$(git rev-parse HEAD)
echo "${COMMIT}"

mkdir -p build

echo "===== Build GPU SpMV ====="

"${NVCC}" "${NVCC_ARGS[@]}" \
  src/matrix.cpp \
  src/gpu_spmv.cu \
  tests/test_gpu_spmv.cu \
  -o build/test_gpu_spmv

echo "PASS: GPU SpMV compilation"

echo "===== Build GPU CG ====="

"${NVCC}" "${NVCC_ARGS[@]}" \
  src/matrix.cpp \
  src/cg.cpp \
  src/gpu_cg.cu \
  tests/test_gpu_cg.cu \
  -o build/test_gpu_cg

echo "PASS: GPU CG compilation"

echo "===== Build GPU Workspace ====="

"${NVCC}" "${NVCC_ARGS[@]}" \
  src/matrix.cpp \
  src/gpu_benchmark.cu \
  tests/test_gpu_workspace.cu \
  -o build/test_gpu_workspace

echo "PASS: GPU Workspace compilation"

echo "===== Build GPU Benchmark ====="

"${NVCC}" "${NVCC_ARGS[@]}" \
  "-DDSA5208_GIT_COMMIT=\"${COMMIT}\"" \
  src/matrix.cpp \
  src/gpu_benchmark.cu \
  src/benchmark_csv.cpp \
  src/correctness_csv.cpp \
  src/gpu_timing_csv.cpp \
  src/gpu_benchmark_main.cu \
  -o build/gpu_benchmark

echo "PASS: GPU Benchmark compilation"

echo "PASS: All GPU targets compiled"
