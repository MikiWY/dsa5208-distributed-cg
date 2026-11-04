#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include "halo_exchange.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 3;
    const int N = n * n;

    const RowPartition part =
        partition_rows(N, rank, size);

    const CSRMatrix A =
        generate_poisson_matrix(
            n,
            part.global_row_start,
            part.local_num_rows
        );

    const HaloExchangePlan plan =
        build_halo_exchange_plan(A, MPI_COMM_WORLD);

    bool local_pass =
        plan.vector_positions.size() == A.col_idx.size();

    // Independently construct the expected ghost positions.
    std::unordered_map<int, int> expected_ghost;

    for (std::size_t i = 0;
         i < plan.recv_global_indices.size(); ++i) {

        if (i >= static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            local_pass = false;
            break;
        }

        const int global_index =
            plan.recv_global_indices[i];

        const auto inserted = expected_ghost.emplace(
            global_index, static_cast<int>(i)
        );

        if (!inserted.second) {
            local_pass = false;
        }
    }

    const int local_start = A.global_row_start;
    const int local_end =
        local_start + A.local_num_rows;

    if (local_pass) {
        for (std::size_t k = 0; k < A.col_idx.size(); ++k) {
            const int col = A.col_idx[k];

            int expected = 0;

            if (col >= local_start && col < local_end) {
                expected = col - local_start;
            } else {
                const auto it = expected_ghost.find(col);

                if (it == expected_ghost.end()) {
                    local_pass = false;
                    break;
                }

                expected = -(it->second + 1);
            }

            if (plan.vector_positions[k] != expected) {
                local_pass = false;
                break;
            }
        }
    }

    const int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size << "\n";

        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": Halo CSR vector-position cache\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
