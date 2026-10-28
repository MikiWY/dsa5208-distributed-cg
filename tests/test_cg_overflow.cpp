#include <iostream>
#include <vector>

#include "cg.hpp"

int main() {
    // Identity matrix A = I.
    CSRMatrix A;
    A.global_rows = 2;
    A.global_row_start = 0;
    A.local_num_rows = 2;
    A.row_ptr = {0, 1, 2};
    A.col_idx = {0, 1};
    A.values = {1.0, 1.0};

    // Finite inputs whose squared norm overflows double.
    std::vector<double> b = {1e200, 1e200};

    SolverOptions options;
    SolverResult result = conjugate_gradient(A, b, options);

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";

    if (result.converged ||
        result.termination_reason !=
            TerminationReason::NonFiniteValue) {
        std::cerr << "FAIL: Overflow handling\n";
        return 1;
    }

    std::cout << "PASS: Numerical overflow detected\n";
    return 0;
}
