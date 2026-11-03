#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "distributed_spmv.hpp"
#include "halo_exchange.hpp"
#include "halo_spmv.hpp"
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

    std::vector<int> counts(size);
    std::vector<int> displs(size);

    for (int r = 0; r < size; ++r) {
        const RowPartition p =
            partition_rows(N, r, size);

        counts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    // Each rank stores only locally owned vector entries.
    std::vector<double> x_local(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        const int global_index =
            part.global_row_start + i;

        x_local[i] =
            static_cast<double>(global_index + 1);
    }

    // Reference: MPI_Allgatherv SpMV.
    const std::vector<double> y_allgatherv =
        distributed_spmv_allgatherv(
            A, x_local, counts, displs, MPI_COMM_WORLD
        );

    // Build Halo communication plan once.
    const HaloExchangePlan plan =
        build_halo_exchange_plan(A, MPI_COMM_WORLD);

    // New implementation: Blocking Halo SpMV.
    const std::vector<double> y_halo =
        distributed_spmv_halo(
            A, x_local, plan, MPI_COMM_WORLD
        );

    double local_error = 0.0;

    for (int i = 0; i < part.local_num_rows; ++i) {
        local_error = std::max(
            local_error,
            std::abs(y_halo[i] - y_allgatherv[i])
        );
    }

    double global_error = 0.0;

    MPI_Allreduce(
        &local_error,
        &global_error,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD
    );

    const int local_ok =
        std::isfinite(global_error) &&
        global_error <= 1e-12 ? 1 : 0;

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
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size << "\n";

        std::cout << "Maximum Halo-vs-Allgatherv error: "
                  << global_error << "\n";

        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": Halo vs Allgatherv SpMV\n";
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}
