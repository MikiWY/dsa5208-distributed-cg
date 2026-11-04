#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "distributed_cg_halo.hpp"
#include "hybrid_halo_cg.hpp"
#include "halo_exchange.hpp"
#include "distributed_spmv.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &provided
    );

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 8;
    const int N = n * n;

    const RowPartition part =
        partition_rows(N, rank, size);

    const CSRMatrix A = generate_poisson_matrix(
        n, part.global_row_start, part.local_num_rows
    );

    std::vector<int> counts(size), displs(size);

    for (int r = 0; r < size; ++r) {
        const RowPartition p = partition_rows(N, r, size);
        counts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    std::vector<double> x_true(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        const int k = part.global_row_start + i;
        const int gi = k % n + 1;
        const int gj = k / n + 1;

        x_true[i] =
            1.0
            + 0.1 * std::sin(3.0 * gi + 5.0 * gj)
            + 0.05 * std::cos(7.0 * gi - 2.0 * gj);
    }

    const std::vector<double> b_local =
        distributed_spmv_allgatherv(
            A, x_true, counts, displs, MPI_COMM_WORLD
        );

    SolverOptions options;

    // Preprocessing: outside solver calls.
    const HaloExchangePlan plan =
        build_halo_exchange_plan(A, MPI_COMM_WORLD);

    // Compare original and prebuilt-plan interfaces.
    const SolverResult v2_original =
        distributed_conjugate_gradient_halo(
            A, b_local, options, MPI_COMM_WORLD
        );

    const SolverResult v2_prebuilt =
        distributed_conjugate_gradient_halo(
            A, b_local, options, plan, MPI_COMM_WORLD
        );

    const SolverResult v4_original =
        hybrid_conjugate_gradient_halo(
            A, b_local, options, MPI_COMM_WORLD
        );

    const SolverResult v4_prebuilt =
        hybrid_conjugate_gradient_halo(
            A, b_local, options, plan, MPI_COMM_WORLD
        );

    double local_error = 0.0;
    bool local_pass =
        provided >= MPI_THREAD_FUNNELED &&
        v2_original.converged &&
        v2_prebuilt.converged &&
        v4_original.converged &&
        v4_prebuilt.converged &&
        v2_original.iterations == v2_prebuilt.iterations &&
        v4_original.iterations == v4_prebuilt.iterations;

    for (int i = 0; i < part.local_num_rows; ++i) {
        local_error = std::max(
            local_error,
            std::abs(
                v2_original.x[i] - v2_prebuilt.x[i]
            )
        );

        local_error = std::max(
            local_error,
            std::abs(
                v4_original.x[i] - v4_prebuilt.x[i]
            )
        );
    }

    double global_error = 0.0;

    MPI_Allreduce(
        &local_error, &global_error,
        1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD
    );

    const int local_ok =
        local_pass &&
        std::isfinite(global_error) &&
        global_error <= 1e-12 ? 1 : 0;

    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size
                  << ", OpenMP threads="
                  << omp_get_max_threads() << "\n";

        std::cout << "Maximum original-vs-prebuilt error: "
                  << global_error << "\n";

        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": Prebuilt Halo CG interfaces\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
