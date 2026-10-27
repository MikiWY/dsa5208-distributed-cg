#include <cmath>
#include <iostream>
#include <vector>

#include "matrix.hpp"

int main() {
    const int n = 8;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // Convert CSR to dense matrix for small-grid validation.
    std::vector<std::vector<double>> dense(
        N, std::vector<double>(N, 0.0)
    );

    for (int r = 0; r < N; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            dense[r][A.col_idx[p]] = A.values[p];
        }
    }

    // Check symmetry.
    double max_asymmetry = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double diff = std::abs(dense[i][j] - dense[j][i]);

            if (diff > max_asymmetry) {
                max_asymmetry = diff;
            }
        }
    }

    std::cout << "Maximum symmetry error: "
              << max_asymmetry << "\n";

    if (!std::isfinite(max_asymmetry) ||
        max_asymmetry > 1e-10) {
        std::cerr << "FAIL: Matrix is not symmetric\n";
        return 1;
    }

    // Independent Cholesky factorization: A = L * L^T.
    std::vector<std::vector<double>> L(
        N, std::vector<double>(N, 0.0)
    );

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = dense[i][j];

            for (int k = 0; k < j; ++k) {
                sum -= L[i][k] * L[j][k];
            }

            if (i == j) {
                if (!std::isfinite(sum) || sum <= 0.0) {
                    std::cerr << "FAIL: Cholesky breakdown\n";
                    return 1;
                }

                L[i][j] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }

    std::cout << "PASS: Matrix symmetry (n=8)\n";
    std::cout << "PASS: Cholesky SPD check (n=8)\n";

    return 0;
}
