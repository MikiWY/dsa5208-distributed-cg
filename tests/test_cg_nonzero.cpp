#include <cmath>
#include <iostream>
#include <vector>

#include "cg.hpp"
#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // Known solution, used only for testing.
    std::vector<double> x_true(N);
    for (int i = 0; i < N; ++i) {
        x_true[i] = static_cast<double>(i + 1);
    }

    // Construct b = A * x_true.
    std::vector<double> b(N, 0.0);

    for (int r = 0; r < N; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            b[r] += A.values[p] * x_true[A.col_idx[p]];
        }
    }

    // Solve A*x = b, starting from x0 = 0.
    SolverOptions options;
    SolverResult result = conjugate_gradient(A, b, options);

    // Independently compute true residual and solution error.
    double residual_squared = 0.0;
    double b_squared = 0.0;
    double error_squared = 0.0;
    double true_squared = 0.0;

    if (result.x.size() != static_cast<std::size_t>(N)) {
        std::cerr << "FAIL: Incorrect solution size\n";
        return 1;
    }

    for (int r = 0; r < N; ++r) {
        double Ax = 0.0;

        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            Ax += A.values[p] * result.x[A.col_idx[p]];
        }

        residual_squared += (b[r] - Ax) * (b[r] - Ax);
        b_squared += b[r] * b[r];

        double diff = result.x[r] - x_true[r];
        error_squared += diff * diff;
        true_squared += x_true[r] * x_true[r];
    }

    const double true_residual =
        std::sqrt(residual_squared / b_squared);

    const double relative_error =
        std::sqrt(error_squared / true_squared);

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Recursive residual: "
              << result.recursive_rel_residual << "\n";
    std::cout << "True relative residual: "
              << true_residual << "\n";
    std::cout << "Relative solution error: "
              << relative_error << "\n";

    if (!result.converged ||
        result.iterations <= 0 ||
        result.termination_reason != TerminationReason::Converged ||
        !std::isfinite(true_residual) ||
        !std::isfinite(relative_error) ||
        true_residual > 1e-8 ||
        relative_error > 1e-8) {
        std::cerr << "FAIL: Serial CG nonzero RHS test\n";
        return 1;
    }

    std::cout << "PASS: Serial CG nonzero RHS test\n";
    return 0;
}
