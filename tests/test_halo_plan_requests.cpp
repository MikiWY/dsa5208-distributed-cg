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
            std::cerr << "Test requires exactly 3 MPI ranks\n";
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

    const std::vector<std::vector<int>> expected_send = {
        {0, 1, 2},
        {0, 1, 2, 0, 1, 2},
        {0, 1, 2}
    };

    const std::vector<std::vector<int>> expected_recv = {
        {3, 4, 5},
        {0, 1, 2, 6, 7, 8},
        {3, 4, 5}
    };

    bool local_pass =
        plan.send_local_indices == expected_send[rank] &&
        plan.recv_global_indices == expected_recv[rank];

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
            << ": Halo index request exchange\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
