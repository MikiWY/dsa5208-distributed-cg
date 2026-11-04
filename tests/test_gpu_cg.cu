#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "cg.hpp"
#include "gpu_cg.hpp"
#include "matrix.hpp"

std::vector<double> reference_spmv(
    const CSRMatrix& A,
    const std::vector<double>& x
) {
    std::vector<double> y(A.global_rows, 0.0);

    for (int r = 0; r < A.local_num_rows; ++r) {
        for (int k = A.row_ptr[r];
             k < A.row_ptr[r + 1]; ++k) {
            y[r] += A.values[k] * x[A.col_idx[k]];
        }
    }

    return y;
}

int main() {
    int count = 0;

    if (cudaGetDeviceCount(&count) != cudaSuccess ||
        count == 0) {
        std::cerr << "FAIL: No CUDA GPU available\n";
        return 1;
    }

    cudaDeviceProp device{};

    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) {
        std::cerr << "FAIL: Cannot query CUDA device\n";
        return 1;
    }

    std::cout << "CUDA device: "
              << device.name << "\n";

    for (int n : {8, 32, 64}) {
        const int N = n * n;
        const CSRMatrix A =
            generate_poisson_matrix(n, 0, N);

        std::vector<double> x_true(N);

        for (int k = 0; k < N; ++k) {
            const int i = k % n + 1;
            const int j = k / n + 1;

            x_true[k] =
                1.0
                + 0.1 * std::sin(3.0 * i + 5.0 * j)
                + 0.05 * std::cos(7.0 * i - 2.0 * j);
        }

        const auto b = reference_spmv(A, x_true);

        SolverOptions options;

        const auto cpu =
            conjugate_gradient(A, b, options);

        const auto gpu =
            gpu_conjugate_gradient(A, b, options);

        const auto Ax = reference_spmv(A, gpu.x);

        double residual_sq = 0.0;
        double b_sq = 0.0;
        double error_sq = 0.0;
        double true_sq = 0.0;
        double diff_sq = 0.0;
        double cpu_sq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double r = b[i] - Ax[i];
            const double e = gpu.x[i] - x_true[i];
            const double d = gpu.x[i] - cpu.x[i];

            residual_sq += r * r;
            b_sq += b[i] * b[i];
            error_sq += e * e;
            true_sq += x_true[i] * x_true[i];
            diff_sq += d * d;
            cpu_sq += cpu.x[i] * cpu.x[i];
        }

        const double residual =
            std::sqrt(residual_sq / b_sq);
        const double error =
            std::sqrt(error_sq / true_sq);
        const double difference =
            std::sqrt(diff_sq / cpu_sq);

        std::cout
            << "n=" << n
            << " GPU iterations=" << gpu.iterations
            << " residual=" << residual
            << " error=" << error
            << " vs_CPU=" << difference << "\n";

        if (!cpu.converged || !gpu.converged ||
            !std::isfinite(residual) ||
            !std::isfinite(error) ||
            !std::isfinite(difference) ||
            residual > 1e-8 ||
            error > 1e-6 ||
            difference > 1e-6) {
            std::cerr << "FAIL: GPU CG correctness\n";
            return 1;
        }
    }

    const CSRMatrix A =
        generate_poisson_matrix(8, 0, 64);

    SolverOptions options;
    std::vector<double> zero(64, 0.0);

    const auto result =
        gpu_conjugate_gradient(A, zero, options);

    if (!result.converged ||
        result.iterations != 0 ||
        result.recursive_rel_residual != 0.0) {
        std::cerr << "FAIL: GPU zero RHS\n";
        return 1;
    }


    // Maximum iteration limit with a nontrivial RHS.
    {
        const int n = 8;
        const int N = n * n;

        const CSRMatrix A =
            generate_poisson_matrix(n, 0, N);

        std::vector<double> x_true(N);

        for (int k = 0; k < N; ++k) {
            const int i = k % n + 1;
            const int j = k / n + 1;

            x_true[k] =
                1.0
                + 0.1 * std::sin(3.0 * i + 5.0 * j)
                + 0.05 * std::cos(7.0 * i - 2.0 * j);
        }

        const auto b = reference_spmv(A, x_true);

        SolverOptions limited;
        limited.max_iterations = 1;

        const auto result =
            gpu_conjugate_gradient(A, b, limited);

        if (result.converged ||
            result.iterations != 1 ||
            result.termination_reason !=
                TerminationReason::MaxIterations) {
            std::cerr
                << "FAIL: GPU maximum iteration test\n";
            return 1;
        }

        std::cout
            << "PASS: GPU maximum iteration limit\n";
    }

    std::cout << "PASS: GPU CG correctness suite\n";
    return 0;
}
