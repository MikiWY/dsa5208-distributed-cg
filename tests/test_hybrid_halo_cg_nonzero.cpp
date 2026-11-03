#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "distributed_cg.hpp"
#include "distributed_spmv.hpp"
#include "hybrid_halo_cg.hpp"
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

    // Optional grid size argument (default n=8).
    int n = 8;

    if (argc > 1) {
        try {
            std::size_t parsed = 0;
            const std::string argument(argv[1]);
            n = std::stoi(argument, &parsed);

            if (parsed != argument.size() ||
                n <= 0 || n > 10000) {
                throw std::invalid_argument("Invalid grid size");
            }
        } catch (const std::exception&) {
            if (rank == 0) {
                std::cerr << "Invalid grid size\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    const int N = n * n;

    RowPartition part = partition_rows(N, rank, size);

    CSRMatrix A = generate_poisson_matrix(
        n, part.global_row_start, part.local_num_rows
    );

    std::vector<int> counts(size), displs(size);

    for (int r = 0; r < size; ++r) {
        RowPartition p = partition_rows(N, r, size);
        counts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    // C's approved manufactured solution.
    std::vector<double> x_true(part.local_num_rows);

    for (int r = 0; r < part.local_num_rows; ++r) {
        const int k = part.global_row_start + r;
        const int i = k % n + 1;
        const int j = k / n + 1;

        x_true[r] =
            1.0
            + 0.1 * std::sin(3.0 * i + 5.0 * j)
            + 0.05 * std::cos(7.0 * i - 2.0 * j);
    }

    // Generate the same RHS for both solvers.
    const std::vector<double> b_local =
        distributed_spmv_allgatherv(
            A, x_true, counts, displs, MPI_COMM_WORLD
        );

    SolverOptions options;

    const SolverResult mpi_result =
        distributed_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        );

    const SolverResult hybrid_result =
        hybrid_conjugate_gradient_halo(
            A, b_local, options, MPI_COMM_WORLD
        );

    // Independently recompute Hybrid true residual.
    const std::vector<double> Ax =
        distributed_spmv_allgatherv(
            A, hybrid_result.x,
            counts, displs, MPI_COMM_WORLD
        );

    double local_sums[5] = {};

    for (int i = 0; i < part.local_num_rows; ++i) {
        const double residual = b_local[i] - Ax[i];
        const double error = hybrid_result.x[i] - x_true[i];
        const double difference =
            hybrid_result.x[i] - mpi_result.x[i];

        local_sums[0] += residual * residual;
        local_sums[1] += b_local[i] * b_local[i];
        local_sums[2] += error * error;
        local_sums[3] += x_true[i] * x_true[i];
        local_sums[4] += difference * difference;
    }

    double global_sums[5] = {};

    MPI_Allreduce(
        local_sums, global_sums, 5,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
    );

    const double true_residual =
        std::sqrt(global_sums[0] / global_sums[1]);

    const double relative_error =
        std::sqrt(global_sums[2] / global_sums[3]);

    // Compute relative Hybrid-vs-MPI solution difference.
    double local_mpi_norm_sq = 0.0;

    for (double value : mpi_result.x) {
        local_mpi_norm_sq += value * value;
    }

    double global_mpi_norm_sq = 0.0;

    MPI_Allreduce(
        &local_mpi_norm_sq, &global_mpi_norm_sq,
        1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
    );

    const double relative_difference =
        std::sqrt(global_sums[4] / global_mpi_norm_sq);

    const bool local_pass =
        provided >= MPI_THREAD_FUNNELED &&
        mpi_result.converged &&
        hybrid_result.converged &&
        hybrid_result.termination_reason ==
            TerminationReason::Converged &&
        std::isfinite(true_residual) &&
        std::isfinite(relative_error) &&
        std::isfinite(relative_difference) &&
        true_residual <= options.tolerance &&
        relative_difference <= 1e-8;

    int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok, 1,
        MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size
                  << ", OpenMP threads="
                  << omp_get_max_threads() << "\n";

        std::cout << "MPI iterations: "
                  << mpi_result.iterations << "\n";
        std::cout << "Hybrid iterations: "
                  << hybrid_result.iterations << "\n";
        std::cout << "Hybrid true relative residual: "
                  << true_residual << "\n";
        std::cout << "Hybrid relative solution error: "
                  << relative_error << "\n";
        std::cout << "Hybrid-vs-MPI relative difference: "
                  << relative_difference << "\n";

        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": Hybrid Halo CG manufactured solution\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
