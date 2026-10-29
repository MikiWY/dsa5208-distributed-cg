#include <iostream>
#include <stdexcept>
#include <vector>

#include "cg.hpp"
#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);
    std::vector<double> b(N, 1.0);

    int passed = 0;

    // Test 1: Negative tolerance.
    try {
        SolverOptions options;
        options.tolerance = -1e-8;
        conjugate_gradient(A, b, options);
        std::cerr << "FAIL: Negative tolerance accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {
        std::cout << "PASS: Negative tolerance rejected\n";
        ++passed;
    }

    // Test 2: Zero tolerance.
    try {
        SolverOptions options;
        options.tolerance = 0.0;
        conjugate_gradient(A, b, options);
        std::cerr << "FAIL: Zero tolerance accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {
        std::cout << "PASS: Zero tolerance rejected\n";
        ++passed;
    }

    // Test 3: Negative maximum iterations.
    try {
        SolverOptions options;
        options.max_iterations = -5;
        conjugate_gradient(A, b, options);
        std::cerr << "FAIL: Negative max iterations accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {
        std::cout << "PASS: Negative max iterations rejected\n";
        ++passed;
    }

    // Test 4: Incorrect RHS vector length.
    try {
        SolverOptions options;
        std::vector<double> wrong_b(N - 1, 1.0);
        conjugate_gradient(A, wrong_b, options);
        std::cerr << "FAIL: Incorrect RHS size accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {
        std::cout << "PASS: Incorrect RHS size rejected\n";
        ++passed;
    }

    if (passed != 4) {
        std::cerr << "FAIL: Not all invalid-input tests passed\n";
        return 1;
    }

    std::cout << "PASS: All 4 invalid-input tests\n";
    return 0;
}
