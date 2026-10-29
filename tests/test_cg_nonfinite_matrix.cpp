#include <iostream>
#include <limits>
#include <vector>

#include "cg.hpp"

int main() {
    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    const double inf =
        std::numeric_limits<double>::infinity();

    const double bad_values[] = {nan, inf};
    const char* names[] = {"NaN matrix", "Inf matrix"};

    for (int test = 0; test < 2; ++test) {
        CSRMatrix A;
        A.global_rows = 2;
        A.global_row_start = 0;
        A.local_num_rows = 2;
        A.row_ptr = {0, 1, 2};
        A.col_idx = {0, 1};
        A.values = {bad_values[test], 1.0};

        std::vector<double> b = {1.0, 1.0};

        SolverOptions options;
        SolverResult result = conjugate_gradient(A, b, options);

        if (result.converged ||
            result.termination_reason !=
                TerminationReason::NonFiniteValue) {
            std::cerr << "FAIL: " << names[test] << "\n";
            return 1;
        }

        std::cout << "PASS: " << names[test]
                  << " detected\n";
    }

    std::cout << "PASS: All non-finite matrix tests\n";
    return 0;
}
