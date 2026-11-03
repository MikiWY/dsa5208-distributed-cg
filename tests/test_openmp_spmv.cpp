#include <cmath>
#include <iostream>
#include <vector>

#include <omp.h>

#include "matrix.hpp"
#include "openmp_vector_ops.hpp"

int main() {
    const int n = 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    std::vector<double> x(N);

    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<double>(i + 1);
    }

    // Independent reference provided by C.
    const std::vector<double> expected = {
        -32.0, -16.0, 64.0,
        48.0, 0.0, 112.0,
        256.0, 176.0, 352.0
    };

    const std::vector<double> result =
        openmp_csr_spmv(A, x);

    double max_error = 0.0;

    for (int i = 0; i < N; ++i) {
        const double error =
            std::abs(result[i] - expected[i]);

        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "OpenMP max threads: "
              << omp_get_max_threads() << "\n";

    std::cout << "Maximum SpMV error: "
              << max_error << "\n";

    if (!std::isfinite(max_error) ||
        max_error > 1e-12) {
        std::cerr << "FAIL: OpenMP SpMV\n";
        return 1;
    }

    std::cout << "PASS: OpenMP SpMV\n";
    return 0;
}
