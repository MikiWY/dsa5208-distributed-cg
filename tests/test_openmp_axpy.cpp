#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "openmp_vector_ops.hpp"

bool vectors_equal(
    const std::vector<double>& actual,
    const std::vector<double>& expected
) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > 1e-12) {
            return false;
        }
    }

    return true;
}

int main() {
    // Test 1: Positive alpha.
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {4.0, 5.0, 6.0};

    openmp_axpy(2.0, x, y);

    if (!vectors_equal(y, {6.0, 9.0, 12.0})) {
        std::cerr << "FAIL: Positive alpha\n";
        return 1;
    }

    std::cout << "PASS: Positive alpha\n";

    // Test 2: Negative alpha.
    openmp_axpy(-1.0, x, y);

    if (!vectors_equal(y, {5.0, 7.0, 9.0})) {
        std::cerr << "FAIL: Negative alpha\n";
        return 1;
    }

    std::cout << "PASS: Negative alpha\n";

    // Test 3: Empty vectors.
    std::vector<double> empty_x;
    std::vector<double> empty_y;

    openmp_axpy(2.0, empty_x, empty_y);

    if (!empty_y.empty()) {
        std::cerr << "FAIL: Empty vectors\n";
        return 1;
    }

    std::cout << "PASS: Empty vectors\n";

    // Test 4: Mismatched sizes.
    bool caught = false;

    try {
        std::vector<double> wrong_y = {1.0};
        openmp_axpy(2.0, x, wrong_y);
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

    std::cout << "PASS: All OpenMP AXPY tests\n";

    return 0;
}
