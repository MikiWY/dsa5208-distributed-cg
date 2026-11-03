#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "distributed_spmv.hpp"
#include "hybrid_spmv.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;

    MPI_Init_thread(
        &argc, &argv,
        MPI_THREAD_FUNNELED, &provided
    );

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 8;
    const int N = n * n;

    RowPartition part = partition_rows(N, rank, size);

    CSRMatrix A = generate_poisson_matrix(
        n, part.global_row_start, part.local_num_rows
    );

    std::vector<int> counts(size);
    std::vector<int> displs(size);

    for (int r = 0; r < size; ++r) {
        RowPartition p = partition_rows(N, r, size);
        counts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    std::vector<double> x_local(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        int k = part.global_row_start + i;
        x_local[i] = 1.0 + 0.01 * k;
    }

    // MPI-only reference.
    std::vector<double> y_mpi =
        distributed_spmv_allgatherv(
            A, x_local, counts, displs, MPI_COMM_WORLD
        );

    // Hybrid implementation.
    std::vector<double> y_hybrid =
        hybrid_spmv_allgatherv(
            A, x_local, counts, displs, MPI_COMM_WORLD
        );

    double local_error = 0.0;

    for (int i = 0; i < part.local_num_rows; ++i) {
        local_error = std::max(
            local_error,
            std::abs(y_mpi[i] - y_hybrid[i])
        );
    }

    double global_error = 0.0;

    MPI_Allreduce(
        &local_error, &global_error,
        1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD
    );

    int local_ok =
        (provided >= MPI_THREAD_FUNNELED &&
         std::isfinite(global_error) &&
         global_error <= 1e-12) ? 1 : 0;

    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size
                  << ", OpenMP threads=" << omp_get_max_threads()
                  << "\n";

        std::cout << "Maximum MPI-vs-Hybrid SpMV error: "
                  << global_error << "\n";

        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": Hybrid Allgatherv SpMV\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
