#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.hpp"

int main(int argc, char* argv[]) {
    const int n = (argc > 1) ? std::stoi(argv[1]) : 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    // Read C's independent reference CSV.
    std::ifstream file("tests/reference/poisson_n" + std::to_string(n) + "_spmv.csv");

    if (!file) {
        std::cerr << "FAIL: Cannot open reference CSV\n";
        return 1;
    }

    std::string line;
    std::getline(file, line); // Skip CSV header.

    std::vector<double> p(N, 0.0);
    std::vector<double> expected(N, 0.0);
    std::vector<bool> seen(N, false);

    int rows_read = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string index_str, p_str, result_str;

        if (!std::getline(ss, index_str, ',') ||
            !std::getline(ss, p_str, ',') ||
            !std::getline(ss, result_str, ',')) {
            std::cerr << "FAIL: Invalid CSV row\n";
            return 1;
        }

        int k = std::stoi(index_str);

        if (k < 0 || k >= N || seen[k]) {
            std::cerr << "FAIL: Invalid or duplicate index\n";
            return 1;
        }

        seen[k] = true;
        p[k] = std::stod(p_str);
        expected[k] = std::stod(result_str);
        ++rows_read;
    }

    if (rows_read != N) {
        std::cerr << "FAIL: Missing reference rows\n";
        return 1;
    }

    // Compute A*p using the generated CSR matrix.
    std::vector<double> actual(N, 0.0);

    for (int r = 0; r < N; ++r) {
        for (int q = A.row_ptr[r]; q < A.row_ptr[r + 1]; ++q) {
            actual[r] += A.values[q] * p[A.col_idx[q]];
        }
    }

    double max_error = 0.0;

    for (int k = 0; k < N; ++k) {
        double error = std::abs(actual[k] - expected[k]);

        if (!std::isfinite(error)) {
            std::cerr << "FAIL: Non-finite error\n";
            return 1;
        }

        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Grid n = " << n << "\n";
    std::cout << "Maximum SpMV error = " << max_error << "\n";

    if (max_error > 1e-10) {
        std::cerr << "FAIL: Reference SpMV mismatch\n";
        return 1;
    }

    std::cout << "PASS: C's independent SpMV reference (n=" << n << ")\n";
    return 0;
}
