#include <cmath>
#include <iostream>
#include <vector>

#include "cg.hpp"
#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // Deterministic known solution for testing.
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

    SolverOptions options;
    options.tolerance = 1e-8;
    options.max_iterations = 1;

    SolverResult result = conjugate_gradient(A, b, options);

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Recursive relative residual: "
              << result.recursive_rel_residual << "\n";

    if (result.converged ||
        result.iterations != 1 ||
        result.termination_reason !=
            TerminationReason::MaxIterations ||
        !std::isfinite(result.recursive_rel_residual) ||
        result.recursive_rel_residual <= options.tolerance) {
        std::cerr << "FAIL: Maximum iteration handling\n";
        return 1;
    }

    std::cout << "PASS: Maximum iterations test\n";
    return 0;
}
