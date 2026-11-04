#include "gpu_cg.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(where) + ": " +
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    double sum = 0.0;

    for (int k = row_ptr[row];
         k < row_ptr[row + 1]; ++k) {
        sum += values[k] * x[col_idx[k]];
    }

    y[row] = sum;
}

__global__ void update_x_r_kernel(
    int N,
    double alpha,
    const double* p,
    const double* q,
    double* x,
    double* r
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        x[i] += alpha * p[i];
        r[i] -= alpha * q[i];
    }
}

__global__ void update_p_kernel(
    int N,
    double beta,
    const double* r,
    double* p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        p[i] = r[i] + beta * p[i];
    }
}

template<typename T>
T* ptr(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

template<typename T>
const T* ptr(const thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

double gpu_dot(
    const thrust::device_vector<double>& a,
    const thrust::device_vector<double>& b
) {
    return thrust::inner_product(
        a.begin(), a.end(), b.begin(), 0.0
    );
}

void validate_inputs(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const SolverOptions& options
) {
    if (A.global_rows <= 0 ||
        A.global_row_start != 0 ||
        A.local_num_rows != A.global_rows ||
        b.size() != static_cast<std::size_t>(A.global_rows) ||
        A.row_ptr.size() != b.size() + 1 ||
        A.col_idx.size() != A.values.size()) {
        throw std::invalid_argument(
            "GPU CG requires complete square CSR matrix"
        );
    }

    if (!std::isfinite(options.tolerance) ||
        options.tolerance <= 0.0 ||
        options.max_iterations < 0) {
        throw std::invalid_argument("Invalid CG options");
    }

    if (A.row_ptr.front() != 0 ||
        A.row_ptr.back() < 0 ||
        static_cast<std::size_t>(A.row_ptr.back()) !=
            A.values.size()) {
        throw std::invalid_argument("Invalid CSR row pointers");
    }

    for (int i = 0; i < A.local_num_rows; ++i) {
        if (A.row_ptr[i] < 0 ||
            A.row_ptr[i] > A.row_ptr[i + 1]) {
            throw std::invalid_argument(
                "Non-monotonic CSR row pointers"
            );
        }
    }

    for (int col : A.col_idx) {
        if (col < 0 || col >= A.global_rows) {
            throw std::invalid_argument(
                "CSR column index out of range"
            );
        }
    }

    for (double value : A.values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Non-finite CSR matrix value"
            );
        }
    }
}

} // namespace

SolverResult gpu_conjugate_gradient(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const SolverOptions& options
) {
    validate_inputs(A, b, options);

    const int N = A.global_rows;

    SolverResult result;
    result.x.assign(N, 0.0);

    // Zero RHS is an exact solution for x0 = 0.
    bool zero_rhs = true;
    for (double value : b) {
        if (!std::isfinite(value)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            return result;
        }
        if (value != 0.0) zero_rhs = false;
    }

    if (zero_rhs) {
        result.converged = true;
        result.iterations = 0;
        result.recursive_rel_residual = 0.0;
        result.termination_reason =
            TerminationReason::Converged;
        return result;
    }

    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    // Device allocations are managed automatically.
    // Matrix and vectors remain on the GPU during CG.
    thrust::device_vector<int> d_row(A.row_ptr.begin(),
                                     A.row_ptr.end());
    thrust::device_vector<int> d_col(A.col_idx.begin(),
                                     A.col_idx.end());
    thrust::device_vector<double> d_val(A.values.begin(),
                                        A.values.end());

    thrust::device_vector<double> d_b(b.begin(), b.end());
    thrust::device_vector<double> d_x(N, 0.0);
    thrust::device_vector<double> d_r = d_b;
    thrust::device_vector<double> d_p = d_b;
    thrust::device_vector<double> d_q(N, 0.0);

    const double bb = gpu_dot(d_b, d_b);

    if (!std::isfinite(bb) || bb <= 0.0) {
        result.termination_reason =
            TerminationReason::NonFiniteValue;
        return result;
    }

    const double b_norm = std::sqrt(bb);
    double rr_old = bb;

    result.recursive_rel_residual = 1.0;

    const int max_iterations =
        options.max_iterations > 0
        ? options.max_iterations
        : static_cast<int>(
            std::min(10LL * N, 100000LL)
          );

    constexpr int block_size = 256;
    const int blocks =
        (N + block_size - 1) / block_size;

    for (int k = 0; k < max_iterations; ++k) {

        // q = A*p on GPU.
        csr_spmv_kernel<<<blocks, block_size>>>(
            N,
            ptr(d_row),
            ptr(d_col),
            ptr(d_val),
            ptr(d_p),
            ptr(d_q)
        );

        check_cuda(
            cudaGetLastError(), "GPU CSR SpMV launch"
        );

        const double pq = gpu_dot(d_p, d_q);

        if (!std::isfinite(pq)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            break;
        }

        if (pq <= 0.0) {
            result.termination_reason =
                TerminationReason::NumericalBreakdown;
            break;
        }

        const double alpha = rr_old / pq;

        if (!std::isfinite(alpha)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            break;
        }

        update_x_r_kernel<<<blocks, block_size>>>(
            N, alpha,
            ptr(d_p), ptr(d_q),
            ptr(d_x), ptr(d_r)
        );

        check_cuda(
            cudaGetLastError(), "GPU x/r update launch"
        );

        result.iterations = k + 1;

        const double rr_new = gpu_dot(d_r, d_r);

        if (!std::isfinite(rr_new)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            break;
        }

        result.recursive_rel_residual =
            std::sqrt(rr_new) / b_norm;

        if (result.recursive_rel_residual <=
            options.tolerance) {
            result.converged = true;
            result.termination_reason =
                TerminationReason::Converged;
            break;
        }

        const double beta = rr_new / rr_old;

        if (!std::isfinite(beta)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            break;
        }

        update_p_kernel<<<blocks, block_size>>>(
            N, beta, ptr(d_r), ptr(d_p)
        );

        check_cuda(
            cudaGetLastError(), "GPU p update launch"
        );

        rr_old = rr_new;
    }

    // Ensure all queued GPU operations completed.
    check_cuda(
        cudaDeviceSynchronize(), "GPU CG synchronization"
    );

    // Copy solution back once, after CG.
    thrust::copy(
        d_x.begin(), d_x.end(), result.x.begin()
    );

    return result;
}
