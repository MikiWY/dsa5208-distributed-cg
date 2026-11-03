#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "openmp_vector_ops.hpp"

int main() {
    const std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b = {2.0, 2.0, 2.0, 2.0};

    const double result = openmp_dot_product(a, b);

    if (!std::isfinite(result) ||
        std::abs(result - 20.0) > 1e-12) {
        std::cerr << "FAIL: OpenMP dot product value\n";
        return 1;
    }

    std::cout << "PASS: Dot product = "
              << result << "\n";

    // Empty vectors should produce zero.
    const std::vector<double> empty;

    if (openmp_dot_product(empty, empty) != 0.0) {
        std::cerr << "FAIL: Empty vector dot product\n";
        return 1;
    }

    std::cout << "PASS: Empty vectors\n";

    // Mismatched vector sizes must be rejected.
    bool caught = false;

    try {
        openmp_dot_product(
            std::vector<double>{1.0, 2.0},
            std::vector<double>{1.0}
        );
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    if (!caught) {
        std::cerr << "FAIL: Mismatched sizes accepted\n";
        return 1;
    }

    std::cout << "PASS: Mismatched sizes rejected\n";
    std::cout << "OpenMP max threads: "
              << omp_get_max_threads() << "\n";
    std::cout << "PASS: All OpenMP dot product tests\n";

    return 0;
}
