#include <iostream>
#include <vector>

#include "cg.hpp"

int main() {
    // Deliberately non-SPD matrix: A = -I.
    CSRMatrix A;
    A.global_rows = 2;
    A.global_row_start = 0;
    A.local_num_rows = 2;
    A.row_ptr = {0, 1, 2};
    A.col_idx = {0, 1};
    A.values = {-1.0, -1.0};

    std::vector<double> b = {1.0, 1.0};

    SolverOptions options;
    SolverResult result = conjugate_gradient(A, b, options);

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";

    if (result.converged ||
        result.iterations != 0 ||
        result.termination_reason !=
            TerminationReason::NumericalBreakdown) {
        std::cerr << "FAIL: Numerical breakdown handling\n";
        return 1;
    }

    std::cout << "PASS: Numerical breakdown detected\n";
    return 0;
}
