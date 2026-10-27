#include <cmath>
#include <iostream>
#include <vector>

#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;
    const double h = 1.0 / (n + 1.0);
    const double scale = 1.0 / (h * h);

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // Deterministic nonconstant test vector.
    std::vector<double> x(N);

    for (int k = 0; k < N; ++k) {
        x[k] = 1.0 + 0.1 * k + 0.03 * k * k;
    }

    // Method 1: Matrix-vector multiplication using CSR.
    std::vector<double> y_csr(N, 0.0);

    for (int r = 0; r < N; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            y_csr[r] += A.values[p] * x[A.col_idx[p]];
        }
    }

    // Method 2: Independent five-point stencil.
    std::vector<double> y_ref(N, 0.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int k = j * n + i;

            double result = 4.0 * x[k];

            if (i > 0) {
                result -= x[k - 1];
            }
            if (i < n - 1) {
                result -= x[k + 1];
            }
            if (j > 0) {
                result -= x[k - n];
            }
            if (j < n - 1) {
                result -= x[k + n];
            }

            y_ref[k] = scale * result;
        }
    }

    // Compare the two independent calculations.
    double max_error = 0.0;

    for (int k = 0; k < N; ++k) {
        const double error = std::abs(y_csr[k] - y_ref[k]);

        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Maximum absolute difference: "
              << max_error << "\n";

    if (!std::isfinite(max_error) || max_error > 1e-10) {
        std::cerr << "FAIL: CSR and reference stencil disagree\n";
        return 1;
    }

    std::cout << "PASS: Independent reference stencil (n=3)\n";
    return 0;
}
