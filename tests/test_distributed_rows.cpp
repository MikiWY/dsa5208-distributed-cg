#include <iostream>
#include <vector>

#include "matrix.hpp"

int main() {
    const int n = 4;
    const int N = n * n;

    CSRMatrix serial = generate_poisson_matrix(n, 0, N);

    // Simulated MPI ownership:
    // Rank 0: global rows 0-7
    // Rank 1: global rows 8-11
    // Rank 2: global rows 12-15
    const std::vector<int> starts = {0, 8, 12};
    const std::vector<int> counts = {8, 4, 4};

    int total_owned = 0;

    for (std::size_t rank = 0; rank < starts.size(); ++rank) {
        CSRMatrix local = generate_poisson_matrix(
            n, starts[rank], counts[rank]
        );

        if (local.global_rows != N ||
            local.global_row_start != starts[rank] ||
            local.local_num_rows != counts[rank]) {
            std::cerr << "FAIL: Ownership metadata\n";
            return 1;
        }

        for (int r = 0; r < local.local_num_rows; ++r) {
            const int global_r = local.global_row_start + r;

            const int local_begin = local.row_ptr[r];
            const int local_end = local.row_ptr[r + 1];

            const int serial_begin = serial.row_ptr[global_r];
            const int serial_end = serial.row_ptr[global_r + 1];

            if (local_end - local_begin !=
                serial_end - serial_begin) {
                std::cerr << "FAIL: Row nonzero count\n";
                return 1;
            }

            for (int p = 0; p < local_end - local_begin; ++p) {
                if (local.col_idx[local_begin + p] !=
                        serial.col_idx[serial_begin + p] ||
                    local.values[local_begin + p] !=
                        serial.values[serial_begin + p]) {
                    std::cerr << "FAIL: Row mismatch at "
                              << global_r << "\n";
                    return 1;
                }
            }
        }

        total_owned += local.local_num_rows;

        std::cout << "Simulated rank " << rank
                  << ": rows " << starts[rank]
                  << "-" << starts[rank] + counts[rank] - 1
                  << " PASS\n";
    }

    if (total_owned != N) {
        std::cerr << "FAIL: Incomplete ownership\n";
        return 1;
    }

    std::cout << "PASS: Distributed rows match serial CSR\n";
    return 0;
}
