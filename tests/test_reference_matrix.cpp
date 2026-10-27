#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.hpp"

bool check_matrix(int n) {
    const int N = n * n;

    const std::string filename =
        "tests/reference/poisson_n" +
        std::to_string(n) + "_matrix.csv";

    std::ifstream file(filename);

    if (!file) {
        std::cerr << "FAIL: Cannot open " << filename << "\n";
        return false;
    }

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // Convert our CSR matrix to dense format for testing.
    std::vector<std::vector<double>> actual(
        N, std::vector<double>(N, 0.0)
    );

    for (int r = 0; r < N; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            actual[r][A.col_idx[p]] = A.values[p];
        }
    }

    double max_error = 0.0;
    std::string line;
    int row = 0;

    while (std::getline(file, line)) {
        if (row >= N) {
            std::cerr << "FAIL: Too many rows\n";
            return false;
        }

        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        while (std::getline(ss, cell, ',')) {
            if (col >= N) {
                std::cerr << "FAIL: Too many columns\n";
                return false;
            }

            double expected = std::stod(cell);
            double error = std::abs(actual[row][col] - expected);

            if (!std::isfinite(error)) {
                std::cerr << "FAIL: Non-finite error\n";
                return false;
            }

            if (error > max_error) {
                max_error = error;
            }

            ++col;
        }

        if (col != N) {
            std::cerr << "FAIL: Incorrect column count\n";
            return false;
        }

        ++row;
    }

    if (row != N) {
        std::cerr << "FAIL: Incorrect row count\n";
        return false;
    }

    std::cout << "n=" << n
              << ", maximum matrix error = "
              << max_error << "\n";

    if (max_error > 1e-10) {
        std::cerr << "FAIL: Matrix mismatch\n";
        return false;
    }

    std::cout << "PASS: Independent matrix reference n="
              << n << "\n";

    return true;
}

int main() {
    bool passed_n3 = check_matrix(3);
    bool passed_n4 = check_matrix(4);

    return (passed_n3 && passed_n4) ? 0 : 1;
}
