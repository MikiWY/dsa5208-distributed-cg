#include <iostream>
#include <vector>

#include "cg.hpp"
#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // Zero right-hand side.
    std::vector<double> b(N, 0.0);

    SolverOptions options;

    SolverResult result = conjugate_gradient(A, b, options);

    if (!result.converged ||
        result.iterations != 0 ||
        result.recursive_rel_residual != 0.0 ||
        result.termination_reason !=
            TerminationReason::Converged ||
        result.x.size() != static_cast<std::size_t>(N)) {
        std::cerr << "FAIL: Zero RHS solver result\n";
        return 1;
    }

    for (double value : result.x) {
        if (value != 0.0) {
            std::cerr << "FAIL: Nonzero solution entry\n";
            return 1;
        }
    }

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Recursive relative residual: "
              << result.recursive_rel_residual << "\n";
    std::cout << "PASS: Serial CG zero RHS test\n";

    return 0;
}
