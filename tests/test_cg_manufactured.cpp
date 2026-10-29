#include <cmath>
#include <iostream>
#include <vector>

#include "cg.hpp"
#include "matrix.hpp"

bool test_grid(int n) {
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // C's approved deterministic manufactured solution.
    // i and j are one-based interior grid indices.
    std::vector<double> x_true(N);

    for (int j0 = 0; j0 < n; ++j0) {
        for (int i0 = 0; i0 < n; ++i0) {
            const int k = j0 * n + i0;

            const double i = static_cast<double>(i0 + 1);
            const double j = static_cast<double>(j0 + 1);

            x_true[k] =
                1.0
                + 0.1 * std::sin(3.0 * i + 5.0 * j)
                + 0.05 * std::cos(7.0 * i - 2.0 * j);
        }
    }

    // Construct b = A * x_true.
    std::vector<double> b(N, 0.0);

    for (int r = 0; r < N; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            b[r] += A.values[p] * x_true[A.col_idx[p]];
        }
    }

    // Solve from x0 = 0.
    SolverOptions options;
    SolverResult result = conjugate_gradient(A, b, options);

    if (result.x.size() != static_cast<std::size_t>(N)) {
        std::cerr << "FAIL: Incorrect solution size\n";
        return false;
    }

    // Independent true residual and solution error.
    double residual_sq = 0.0;
    double b_sq = 0.0;
    double error_sq = 0.0;
    double true_sq = 0.0;

    for (int r = 0; r < N; ++r) {
        double Ax = 0.0;

        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            Ax += A.values[p] * result.x[A.col_idx[p]];
        }

        const double residual = b[r] - Ax;
        const double error = result.x[r] - x_true[r];

        residual_sq += residual * residual;
        b_sq += b[r] * b[r];
        error_sq += error * error;
        true_sq += x_true[r] * x_true[r];
    }

    const double true_rel_residual =
        std::sqrt(residual_sq) / std::sqrt(b_sq);

    const double rel_error =
        std::sqrt(error_sq) / std::sqrt(true_sq);

    std::cout << "n=" << n
              << ", N=" << N
              << ", iterations=" << result.iterations
              << ", recursive_residual="
              << result.recursive_rel_residual
              << ", true_residual="
              << true_rel_residual
              << ", rel_error="
              << rel_error
              << "\n";

    // True residual is the final numerical acceptance check.
    const bool passed =
        result.converged &&
        result.termination_reason ==
            TerminationReason::Converged &&
        std::isfinite(result.recursive_rel_residual) &&
        std::isfinite(true_rel_residual) &&
        std::isfinite(rel_error) &&
        result.recursive_rel_residual <= options.tolerance &&
        true_rel_residual <= options.tolerance;

    if (passed) {
        std::cout << "PASS: Manufactured solution n="
                  << n << "\n";
    } else {
        std::cerr << "FAIL: Manufactured solution n="
                  << n << "\n";
    }

    return passed;
}

int main() {
    bool all_passed = true;

    for (int n : {8, 32, 64}) {
        if (!test_grid(n)) {
            all_passed = false;
        }
    }

    return all_passed ? 0 : 1;
}
