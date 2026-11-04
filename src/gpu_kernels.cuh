#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace gpu_detail {

inline void check_cuda(
    cudaError_t status,
    const char* operation
) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status)
        );
    }
}

__global__ inline void csr_spmv_kernel(
    int rows,
    const int* row_ptr,
    const int* col_idx,
    const double* values,
    const double* x,
    double* y
) {
    const int row =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;

    double sum = 0.0;

    for (int k = row_ptr[row];
         k < row_ptr[row + 1]; ++k) {
        sum += values[k] * x[col_idx[k]];
    }

    y[row] = sum;
}

__global__ inline void update_x_r_kernel(
    int N,
    double alpha,
    const double* p,
    const double* q,
    double* x,
    double* r
) {
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        x[i] += alpha * p[i];
        r[i] -= alpha * q[i];
    }
}

__global__ inline void update_p_kernel(
    int N,
    double beta,
    const double* r,
    double* p
) {
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        p[i] = r[i] + beta * p[i];
    }
}

} // namespace gpu_detail
