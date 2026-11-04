#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "gpu_benchmark.hpp"
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
    int devices = 0;

    if (cudaGetDeviceCount(&devices) != cudaSuccess ||
        devices == 0) {
        std::cerr << "FAIL: No CUDA GPU available\n";
        return 1;
    }

    cudaDeviceProp prop{};

    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::cerr << "FAIL: Cannot query GPU\n";
        return 1;
    }

    std::cout << "CUDA device: " << prop.name << "\n";

    const int n = 32;
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

    GpuBenchmarkTiming setup;
    GpuCgWorkspace workspace(A, b, &setup);

    SolverOptions options;

    GpuBenchmarkTiming t1;
    GpuBenchmarkTiming t2;

    const auto first = workspace.solve(options, &t1);
    const auto second = workspace.solve(options, &t2);

    const auto Ax = reference_spmv(A, second.x);

    double diff_sq = 0.0;
    double norm_sq = 0.0;
    double residual_sq = 0.0;
    double b_sq = 0.0;
    double error_sq = 0.0;
    double true_sq = 0.0;

    for (int i = 0; i < N; ++i) {
        const double d = first.x[i] - second.x[i];
        const double r = b[i] - Ax[i];
        const double e = second.x[i] - x_true[i];

        diff_sq += d * d;
        norm_sq += first.x[i] * first.x[i];
        residual_sq += r * r;
        b_sq += b[i] * b[i];
        error_sq += e * e;
        true_sq += x_true[i] * x_true[i];
    }

    const double difference =
        std::sqrt(diff_sq / norm_sq);

    const double residual =
        std::sqrt(residual_sq / b_sq);

    const double error =
        std::sqrt(error_sq / true_sq);

    std::cout
        << "First iterations: " << first.iterations << "\n"
        << "Second iterations: " << second.iterations << "\n"
        << "Repeat difference: " << difference << "\n"
        << "True residual: " << residual << "\n"
        << "Solution error: " << error << "\n"
        << "Preprocessing_s: " << setup.preprocessing_s << "\n"
        << "H2D_s: " << setup.h2d_s << "\n"
        << "CG-only_s: " << t2.time_s << "\n"
        << "D2H_s: " << t2.d2h_s << "\n";

    if (!first.converged ||
        !second.converged ||
        first.iterations != second.iterations ||
        !std::isfinite(difference) ||
        !std::isfinite(residual) ||
        !std::isfinite(error) ||
        difference > 1e-12 ||
        residual > 1e-8 ||
        error > 1e-6 ||
        !std::isfinite(setup.preprocessing_s) ||
        !std::isfinite(setup.h2d_s) ||
        !std::isfinite(t1.time_s) ||
        !std::isfinite(t2.time_s) ||
        !std::isfinite(t2.d2h_s) ||
        setup.preprocessing_s < 0.0 ||
        setup.h2d_s < 0.0 ||
        t1.time_s <= 0.0 ||
        t2.time_s <= 0.0 ||
        t2.d2h_s < 0.0) {
        std::cerr << "FAIL: GPU Workspace repeated solve\n";
        return 1;
    }

    // Zero RHS on a separate workspace.
    {
        std::vector<double> zero(N, 0.0);
        GpuCgWorkspace zero_workspace(A, zero);

        const auto result = zero_workspace.solve(options);

        if (!result.converged ||
            result.iterations != 0 ||
            result.recursive_rel_residual != 0.0) {
            std::cerr << "FAIL: GPU Workspace zero RHS\n";
            return 1;
        }
    }

    // Iteration limit on the original nonzero RHS.
    {
        SolverOptions limited;
        limited.max_iterations = 1;

        const auto result = workspace.solve(limited);

        if (result.converged ||
            result.iterations != 1 ||
            result.termination_reason !=
                TerminationReason::MaxIterations) {
            std::cerr << "FAIL: GPU Workspace iteration limit\n";
            return 1;
        }
    }

    std::cout << "PASS: GPU Workspace correctness suite\n";
    return 0;
}
