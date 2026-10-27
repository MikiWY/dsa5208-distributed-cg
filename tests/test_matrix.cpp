#include <iostream>
#include <iomanip>
#include <vector>

#include "matrix.hpp"

int main() {
    const int n = 3;
    const int N = n * n;

    CSRMatrix A = generate_poisson_matrix(n, 0, N);

    std::vector<std::vector<double>> dense(
        N, std::vector<double>(N, 0.0)
    );

    for (int r = 0; r < A.local_num_rows; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            dense[r][A.col_idx[p]] = A.values[p];
        }
    }

    std::cout << "Poisson matrix (n=3, N=9):\n";

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(5) << dense[i][j];
        }
        std::cout << "\n";
    }

    std::cout << "\nCSR information:\n";
    std::cout << "Global rows: " << A.global_rows << "\n";
    std::cout << "Local rows: " << A.local_num_rows << "\n";
    std::cout << "Nonzero entries: " << A.values.size() << "\n";

    return 0;
}
