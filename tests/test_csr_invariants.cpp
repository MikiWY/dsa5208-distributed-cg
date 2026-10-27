#include <iostream>
#include <stdexcept>

#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    bool passed = true;

    // I1: Row pointer array must contain local_rows + 1 entries.
    if (A.row_ptr.size() !=
        static_cast<std::size_t>(A.local_num_rows + 1)) {
        std::cerr << "FAIL: row_ptr size\n";
        passed = false;
    }

    // I2: Row pointers must start at zero.
    if (A.row_ptr.empty() || A.row_ptr[0] != 0) {
        std::cerr << "FAIL: row_ptr[0]\n";
        passed = false;
    }

    // I3: CSR arrays must agree on nonzero count.
    if (A.row_ptr.empty() ||
        A.row_ptr.back() < 0 ||
        static_cast<std::size_t>(A.row_ptr.back()) != A.values.size() ||
        A.values.size() != A.col_idx.size()) {
        std::cerr << "FAIL: CSR nonzero counts\n";
        passed = false;
    }

    // I4: Row pointers must be non-decreasing.
    for (std::size_t i = 1; i < A.row_ptr.size(); ++i) {
        if (A.row_ptr[i] < A.row_ptr[i - 1]) {
            std::cerr << "FAIL: row_ptr order\n";
            passed = false;
        }
    }

    // I5: Every column index must be valid.
    for (int col : A.col_idx) {
        if (col < 0 || col >= A.global_rows) {
            std::cerr << "FAIL: column index\n";
            passed = false;
        }
    }

    // Check the expected matrix dimensions and nnz.
    if (A.global_rows != 9 ||
        A.global_row_start != 0 ||
        A.local_num_rows != 9 ||
        A.values.size() != 33) {
        std::cerr << "FAIL: matrix metadata\n";
        passed = false;
    }

    if (!passed) {
        return 1;
    }

    std::cout << "PASS: CSR invariants (n=3)\n";
    return 0;
}
