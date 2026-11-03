#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "halo_exchange.hpp"
#include "halo_spmv.hpp"
#include "hybrid_halo_spmv.hpp"
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

    std::vector<double> x_local(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        const int global_index =
            part.global_row_start + i;

        x_local[i] =
            static_cast<double>(global_index + 1);
    }

    const HaloExchangePlan plan =
        build_halo_exchange_plan(A, MPI_COMM_WORLD);

    // V2: MPI-only Halo SpMV.
    const std::vector<double> y_mpi =
        distributed_spmv_halo(
            A, x_local, plan, MPI_COMM_WORLD
        );

    // V4: Hybrid MPI + OpenMP Halo SpMV.
    const std::vector<double> y_hybrid =
        hybrid_spmv_halo(
            A, x_local, plan, MPI_COMM_WORLD
        );

    double local_error = 0.0;

    bool local_pass =
        provided >= MPI_THREAD_FUNNELED &&
        y_mpi.size() == y_hybrid.size();

    if (local_pass) {
        for (std::size_t i = 0; i < y_mpi.size(); ++i) {
            if (!std::isfinite(y_mpi[i]) ||
                !std::isfinite(y_hybrid[i])) {
                local_pass = false;
                break;
            }

            local_error = std::max(
                local_error,
                std::abs(y_mpi[i] - y_hybrid[i])
            );
        }
    }

    const int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    double global_error = 0.0;

    MPI_Allreduce(
        &local_error, &global_error,
        1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD
    );

    const bool passed =
        global_ok &&
        std::isfinite(global_error) &&
        global_error <= 1e-12;

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size
                  << ", OpenMP threads="
                  << omp_get_max_threads() << "\n";

        std::cout << "Maximum Hybrid-vs-MPI Halo SpMV error: "
                  << global_error << "\n";

        std::cout << (passed ? "PASS" : "FAIL")
                  << ": Hybrid Halo SpMV consistency\n";
    }

    MPI_Finalize();

    return passed ? 0 : 1;
}
