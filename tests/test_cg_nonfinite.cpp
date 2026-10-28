#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "cg.hpp"

int main() {
    // Identity matrix: A = I.
    CSRMatrix A;
    A.global_rows = 2;
    A.global_row_start = 0;
    A.local_num_rows = 2;
    A.row_ptr = {0, 1, 2};
    A.col_idx = {0, 1};
    A.values = {1.0, 1.0};

    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    const double inf =
        std::numeric_limits<double>::infinity();

    const std::vector<std::vector<double>> test_cases = {
        {nan, 1.0},
        {inf, 1.0}
    };

    const char* names[] = {"NaN RHS", "Inf RHS"};

    for (int test = 0; test < 2; ++test) {
        SolverOptions options;

        SolverResult result =
            conjugate_gradient(A, test_cases[test], options);

        if (result.converged ||
            result.termination_reason !=
                TerminationReason::NonFiniteValue) {
            std::cerr << "FAIL: " << names[test] << "\n";
            return 1;
        }

        std::cout << "PASS: " << names[test]
                  << " detected\n";
    }

    std::cout << "PASS: All non-finite RHS tests\n";
    return 0;
}
