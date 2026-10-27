#include <cmath>
#include <iostream>
#include <vector>

#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;
    const double scale = static_cast<double>((n + 1) * (n + 1));

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    const int positions[] = {0, 1, 4};
    const char* names[] = {"Corner", "Edge", "Interior"};

    for (int test = 0; test < 3; ++test) {
        std::vector<double> x(N, 0.0);
        x[positions[test]] = 1.0;

        std::vector<double> y_csr(N, 0.0);
        std::vector<double> y_ref(N, 0.0);

        // CSR calculation.
        for (int r = 0; r < N; ++r) {
            for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
                y_csr[r] += A.values[p] * x[A.col_idx[p]];
            }
        }

        // Independent five-point stencil.
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int k = j * n + i;
                double value = 4.0 * x[k];

                if (i > 0) value -= x[k - 1];
                if (i < n - 1) value -= x[k + 1];
                if (j > 0) value -= x[k - n];
                if (j < n - 1) value -= x[k + n];

                y_ref[k] = scale * value;
            }
        }

        double max_error = 0.0;

        for (int k = 0; k < N; ++k) {
            double error = std::abs(y_csr[k] - y_ref[k]);
            if (!std::isfinite(error)) {
                std::cerr << "FAIL: Non-finite error\n";
                return 1;
            }
            if (error > max_error) max_error = error;
        }

        std::cout << names[test]
                  << " impulse: max error = "
                  << max_error << "\n";

        if (max_error > 1e-10) {
            std::cerr << "FAIL: " << names[test] << "\n";
            return 1;
        }
    }

    std::cout << "PASS: All impulse stencil tests\n";
    return 0;
}
