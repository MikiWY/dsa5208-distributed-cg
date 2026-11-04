#include "gpu_benchmark.hpp"
#include "gpu_kernels.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(
        Clock::now() - start
    ).count();
}

template<typename T>
T* device_data(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

double device_dot(
    const thrust::device_vector<double>& a,
    const thrust::device_vector<double>& b
) {
    return thrust::inner_product(
        a.begin(), a.end(), b.begin(), 0.0
    );
}

void validate(
    const CSRMatrix& A,
    const std::vector<double>& rhs
) {
    if (A.global_rows <= 0 ||
        A.global_row_start != 0 ||
        A.local_num_rows != A.global_rows ||
        rhs.size() != static_cast<std::size_t>(A.global_rows) ||
        A.row_ptr.size() != rhs.size() + 1 ||
        A.col_idx.size() != A.values.size()) {
        throw std::invalid_argument(
            "GPU Workspace requires complete CSR"
        );
    }

    if (A.row_ptr.front() != 0 ||
        A.row_ptr.back() < 0 ||
        static_cast<std::size_t>(A.row_ptr.back()) !=
            A.values.size()) {
        throw std::invalid_argument(
            "Invalid CSR row pointers"
        );
    }

    for (int r = 0; r < A.local_num_rows; ++r) {
        if (A.row_ptr[r] < 0 ||
            A.row_ptr[r] > A.row_ptr[r + 1]) {
            throw std::invalid_argument(
                "Non-monotonic CSR row pointers"
            );
        }
    }

    for (std::size_t k = 0; k < A.values.size(); ++k) {
        if (A.col_idx[k] < 0 ||
            A.col_idx[k] >= A.global_rows ||
            !std::isfinite(A.values[k])) {
            throw std::invalid_argument(
                "Invalid CSR matrix entry"
            );
        }
    }
}

} // namespace

struct GpuCgWorkspace::Impl {
    int N;

    thrust::device_vector<int> row;
    thrust::device_vector<int> col;
    thrust::device_vector<double> values;

    thrust::device_vector<double> b;
    thrust::device_vector<double> x;
    thrust::device_vector<double> r;
    thrust::device_vector<double> p;
    thrust::device_vector<double> q;

    Impl(
        const CSRMatrix& A,
        const std::vector<double>& rhs,
        GpuBenchmarkTiming* timing
    ) : N(A.global_rows) {
        const auto start = Clock::now();

        validate(A, rhs);

        gpu_detail::check_cuda(
            cudaSetDevice(0), "cudaSetDevice"
        );

        row.resize(A.row_ptr.size());
        col.resize(A.col_idx.size());
        values.resize(A.values.size());

        b.resize(N);
        x.resize(N);
        r.resize(N);
        p.resize(N);
        q.resize(N);

        gpu_detail::check_cuda(
            cudaDeviceSynchronize(),
            "Workspace allocation"
        );

        const auto upload_start = Clock::now();

        thrust::copy(
            A.row_ptr.begin(), A.row_ptr.end(),
            row.begin()
        );
        thrust::copy(
            A.col_idx.begin(), A.col_idx.end(),
            col.begin()
        );
        thrust::copy(
            A.values.begin(), A.values.end(),
            values.begin()
        );
        thrust::copy(
            rhs.begin(), rhs.end(), b.begin()
        );

        gpu_detail::check_cuda(
            cudaDeviceSynchronize(),
            "Workspace upload"
        );

        if (timing) {
            timing->h2d_s = elapsed(upload_start);
            timing->preprocessing_s = elapsed(start);
        }
    }

    SolverResult solve(
        const SolverOptions& options,
        GpuBenchmarkTiming* timing
    ) {
        if (!std::isfinite(options.tolerance) ||
            options.tolerance <= 0.0 ||
            options.max_iterations < 0) {
            throw std::invalid_argument(
                "Invalid GPU CG options"
            );
        }

        if (timing) {
            timing->time_s = 0.0;
            timing->d2h_s = 0.0;
        }

        SolverResult result;
        result.x.assign(N, 0.0);

        // Reset all CG state before every run.
        thrust::fill(x.begin(), x.end(), 0.0);
        thrust::copy(b.begin(), b.end(), r.begin());
        thrust::copy(b.begin(), b.end(), p.begin());

        gpu_detail::check_cuda(
            cudaDeviceSynchronize(),
            "CG state reset"
        );

        const auto start = Clock::now();

        const double bb = device_dot(b, b);

        if (!std::isfinite(bb)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
        } else if (bb == 0.0) {
            result.converged = true;
            result.iterations = 0;
            result.recursive_rel_residual = 0.0;
            result.termination_reason =
                TerminationReason::Converged;
        } else {
            const double b_norm = std::sqrt(bb);
            double rr_old = bb;

            result.recursive_rel_residual = 1.0;

            const int limit =
                options.max_iterations > 0
                ? options.max_iterations
                : static_cast<int>(
                    std::min(10LL * N, 100000LL)
                  );

            constexpr int block_size = 256;
            const int blocks =
                (N + block_size - 1) / block_size;

            for (int k = 0; k < limit; ++k) {
                gpu_detail::csr_spmv_kernel
                    <<<blocks, block_size>>>(
                        N,
                        device_data(row),
                        device_data(col),
                        device_data(values),
                        device_data(p),
                        device_data(q)
                    );

                gpu_detail::check_cuda(
                    cudaGetLastError(),
                    "GPU SpMV launch"
                );

                const double pq = device_dot(p, q);

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

                gpu_detail::update_x_r_kernel
                    <<<blocks, block_size>>>(
                        N, alpha,
                        device_data(p),
                        device_data(q),
                        device_data(x),
                        device_data(r)
                    );

                gpu_detail::check_cuda(
                    cudaGetLastError(),
                    "GPU x/r update"
                );

                result.iterations = k + 1;

                const double rr_new = device_dot(r, r);

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

                gpu_detail::update_p_kernel
                    <<<blocks, block_size>>>(
                        N, beta,
                        device_data(r),
                        device_data(p)
                    );

                gpu_detail::check_cuda(
                    cudaGetLastError(),
                    "GPU p update"
                );

                rr_old = rr_new;
            }
        }

        gpu_detail::check_cuda(
            cudaDeviceSynchronize(),
            "GPU CG completion"
        );

        if (timing) {
            timing->time_s = elapsed(start);
        }

        const auto download_start = Clock::now();

        thrust::copy(
            x.begin(), x.end(), result.x.begin()
        );

        gpu_detail::check_cuda(
            cudaDeviceSynchronize(),
            "GPU solution download"
        );

        if (timing) {
            timing->d2h_s = elapsed(download_start);
        }

        return result;
    }
};

GpuCgWorkspace::GpuCgWorkspace(
    const CSRMatrix& A,
    const std::vector<double>& b,
    GpuBenchmarkTiming* timing
) : impl_(nullptr) {
    if (timing) {
        *timing = GpuBenchmarkTiming{};
    }

    impl_ = std::make_unique<Impl>(A, b, timing);
}

GpuCgWorkspace::~GpuCgWorkspace() = default;

GpuCgWorkspace::GpuCgWorkspace(
    GpuCgWorkspace&&
) noexcept = default;

GpuCgWorkspace& GpuCgWorkspace::operator=(
    GpuCgWorkspace&&
) noexcept = default;

SolverResult GpuCgWorkspace::solve(
    const SolverOptions& options,
    GpuBenchmarkTiming* timing
) {
    if (!impl_) {
        throw std::logic_error(
            "Cannot solve using moved GPU Workspace"
        );
    }

    return impl_->solve(options, timing);
}
