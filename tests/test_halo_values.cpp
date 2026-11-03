#include <mpi.h>

#include <cmath>
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

    // Build the communication plan once.
    const HaloExchangePlan plan =
        build_halo_exchange_plan(
            A, MPI_COMM_WORLD
        );

    // Each rank stores only its owned vector values.
    std::vector<double> x_local(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        const int global_index =
            part.global_row_start + i;

        x_local[i] =
            10.0 * (global_index + 1);
    }

    // Exchange actual vector values.
    const std::vector<double> ghost_values =
        exchange_halo_values_blocking(
            plan, x_local, MPI_COMM_WORLD
        );

    bool local_pass =
        ghost_values.size() ==
        plan.recv_global_indices.size();

    if (local_pass) {
        for (std::size_t i = 0;
             i < ghost_values.size(); ++i) {

            const int global_index =
                plan.recv_global_indices[i];

            const double expected =
                10.0 * (global_index + 1);

            if (!std::isfinite(ghost_values[i]) ||
                std::abs(ghost_values[i] - expected)
                    > 1e-12) {
                local_pass = false;
                break;
            }
        }
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
            << "Grid n=" << n
            << ", MPI ranks=" << size << "\n";

        std::cout
            << (global_ok ? "PASS" : "FAIL")
            << ": Blocking Halo Value Exchange\n";
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}
