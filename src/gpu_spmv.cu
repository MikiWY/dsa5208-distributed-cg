#include "gpu_spmv.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status)
        );
    }
}

__global__ void csr_spmv_kernel(
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

    for (int p = row_ptr[row];
         p < row_ptr[row + 1]; ++p) {
        sum += values[p] * x[col_idx[p]];
    }

    y[row] = sum;
}

} // namespace

std::vector<double> gpu_csr_spmv(
    const CSRMatrix& A,
    const std::vector<double>& x
) {
    if (A.global_rows <= 0 ||
        A.global_row_start != 0 ||
        A.local_num_rows != A.global_rows ||
        x.size() != static_cast<std::size_t>(A.global_rows) ||
        A.row_ptr.size() !=
            static_cast<std::size_t>(A.local_num_rows) + 1 ||
        A.col_idx.size() != A.values.size()) {
        throw std::invalid_argument(
            "GPU SpMV requires a complete valid CSR matrix"
        );
    }

    // Validate CSR structure before transferring to the GPU.
    if (A.row_ptr.front() != 0 ||
        A.row_ptr.back() < 0 ||
        static_cast<std::size_t>(A.row_ptr.back()) !=
            A.values.size()) {
        throw std::invalid_argument(
            "Invalid GPU CSR row pointers"
        );
    }

    for (int r = 0; r < A.local_num_rows; ++r) {
        if (A.row_ptr[r] < 0 ||
            A.row_ptr[r] > A.row_ptr[r + 1]) {
            throw std::invalid_argument(
                "GPU CSR row pointers are not monotonic"
            );
        }
    }

    for (int col : A.col_idx) {
        if (col < 0 || col >= A.global_rows) {
            throw std::invalid_argument(
                "GPU CSR column index out of range"
            );
        }
    }

    const int rows = A.local_num_rows;
    const std::size_t nnz = A.values.size();

    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    double* d_values = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;

    try {
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_row_ptr),
                A.row_ptr.size() * sizeof(int)
            ),
            "cudaMalloc row_ptr"
        );

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_col_idx),
                nnz * sizeof(int)
            ),
            "cudaMalloc col_idx"
        );

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_values),
                nnz * sizeof(double)
            ),
            "cudaMalloc values"
        );

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_x),
                x.size() * sizeof(double)
            ),
            "cudaMalloc x"
        );

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_y),
                rows * sizeof(double)
            ),
            "cudaMalloc y"
        );

        cuda_check(
            cudaMemcpy(
                d_row_ptr, A.row_ptr.data(),
                A.row_ptr.size() * sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "copy row_ptr"
        );

        cuda_check(
            cudaMemcpy(
                d_col_idx, A.col_idx.data(),
                nnz * sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "copy col_idx"
        );

        cuda_check(
            cudaMemcpy(
                d_values, A.values.data(),
                nnz * sizeof(double),
                cudaMemcpyHostToDevice
            ),
            "copy values"
        );

        cuda_check(
            cudaMemcpy(
                d_x, x.data(),
                x.size() * sizeof(double),
                cudaMemcpyHostToDevice
            ),
            "copy x"
        );

        constexpr int block_size = 256;

        const int blocks =
            (rows + block_size - 1) / block_size;

        csr_spmv_kernel<<<blocks, block_size>>>(
            rows,
            d_row_ptr,
            d_col_idx,
            d_values,
            d_x,
            d_y
        );

        cuda_check(
            cudaGetLastError(),
            "CSR SpMV kernel launch"
        );

        cuda_check(
            cudaDeviceSynchronize(),
            "CSR SpMV kernel execution"
        );

        std::vector<double> y(rows);

        cuda_check(
            cudaMemcpy(
                y.data(), d_y,
                rows * sizeof(double),
                cudaMemcpyDeviceToHost
            ),
            "copy y"
        );

        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);

        return y;

    } catch (...) {
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        throw;
    }
}
