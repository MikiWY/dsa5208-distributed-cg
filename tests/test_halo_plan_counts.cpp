#include <mpi.h>

#include <iostream>
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

    if (size != 3) {
        if (rank == 0) {
            std::cerr << "Test requires exactly 3 ranks\n";
        }
        MPI_Finalize();
        return 1;
    }

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

    const std::vector<std::vector<int>> expected_counts = {
        {0, 3, 0},
        {3, 0, 3},
        {0, 3, 0}
    };

    const std::vector<std::vector<int>> expected_displs = {
        {0, 0, 3},
        {0, 3, 3},
        {0, 0, 3}
    };

    bool local_pass =
        plan.recv_counts == expected_counts[rank] &&
        plan.send_counts == expected_counts[rank] &&
        plan.recv_displs == expected_displs[rank] &&
        plan.send_displs == expected_displs[rank];

    // Verify the requested global indices.
    const std::vector<std::vector<int>> expected_indices = {
        {3, 4, 5},
        {0, 1, 2, 6, 7, 8},
        {3, 4, 5}
    };

    if (plan.recv_global_indices != expected_indices[rank]) {
        local_pass = false;
    }

    int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok,
        &global_ok,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout
            << (global_ok ? "PASS" : "FAIL")
            << ": Halo Plan counts and displacements\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
