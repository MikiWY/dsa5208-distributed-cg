#include <mpi.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "distributed_cg.hpp"
#include "distributed_cg_halo.hpp"
#include "distributed_spmv.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8;
    bool valid_n = true;

    if (argc > 2) {
        valid_n = false;
    }

    if (argc == 2) {
        try {
            const std::string argument(argv[1]);
            std::size_t parsed = 0;
            const long long requested =
                std::stoll(argument, &parsed);

            if (parsed != argument.size() ||
                requested < 1 || requested > 512) {
                valid_n = false;
            } else {
                n = static_cast<int>(requested);
            }
        } catch (const std::exception&) {
            valid_n = false;
        }
    }

    int local_valid = valid_n ? 1 : 0;
    int global_valid = 0;

    MPI_Allreduce(
        &local_valid, &global_valid,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    int min_n = 0;
    int max_n = 0;

    MPI_Allreduce(
        &n, &min_n, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    if (!global_valid || min_n != max_n) {
        if (rank == 0) {
            std::cerr << "Invalid or inconsistent grid size\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int N = n * n;

    const RowPartition part =
        partition_rows(N, rank, size);

    const CSRMatrix A =
        generate_poisson_matrix(
            n, part.global_row_start, part.local_num_rows
        );

    std::vector<int> counts(size);
    std::vector<int> displs(size);

    for (int r = 0; r < size; ++r) {
        const RowPartition p = partition_rows(N, r, size);
        counts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    // C's approved manufactured solution.
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
    options.tolerance = 1e-8;

    const SolverResult v1 =
        distributed_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        );

    const SolverResult v2 =
        distributed_conjugate_gradient_halo(
            A, b_local, options, MPI_COMM_WORLD
        );

    // Independently recompute the true residual of V2.
    const std::vector<double> Ax_v2 =
        distributed_spmv_allgatherv(
            A, v2.x, counts, displs, MPI_COMM_WORLD
        );

    double local_sums[5] = {};

    for (int i = 0; i < part.local_num_rows; ++i) {
        const double difference = v2.x[i] - v1.x[i];
        const double residual = b_local[i] - Ax_v2[i];
        const double error = v2.x[i] - x_true[i];

        local_sums[0] += difference * difference;
        local_sums[1] += v1.x[i] * v1.x[i];
        local_sums[2] += residual * residual;
        local_sums[3] += b_local[i] * b_local[i];
        local_sums[4] += error * error;
    }

    double global_sums[5] = {};

    MPI_Allreduce(
        local_sums, global_sums, 5,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
    );

    double local_true_norm_sq = 0.0;

    for (double value : x_true) {
        local_true_norm_sq += value * value;
    }

    double global_true_norm_sq = 0.0;

    MPI_Allreduce(
        &local_true_norm_sq,
        &global_true_norm_sq,
        1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
    );

    const double relative_difference =
        global_sums[1] > 0.0
            ? std::sqrt(global_sums[0] / global_sums[1])
            : std::numeric_limits<double>::infinity();

    const double true_residual =
        global_sums[3] > 0.0
            ? std::sqrt(global_sums[2] / global_sums[3])
            : std::numeric_limits<double>::infinity();

    const double relative_error =
        global_true_norm_sq > 0.0
            ? std::sqrt(
                global_sums[4] / global_true_norm_sq
              )
            : std::numeric_limits<double>::infinity();

    const bool local_pass =
        v1.converged &&
        v2.converged &&
        v1.termination_reason ==
            TerminationReason::Converged &&
        v2.termination_reason ==
            TerminationReason::Converged &&
        v1.iterations == v2.iterations &&
        std::isfinite(relative_difference) &&
        std::isfinite(true_residual) &&
        std::isfinite(relative_error) &&
        relative_difference <= 1e-8 &&
        true_residual <= 1e-8 &&
        relative_error <= 1e-6;

    const int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size << "\n";

        std::cout << "V1 iterations: "
                  << v1.iterations << "\n";

        std::cout << "V2 iterations: "
                  << v2.iterations << "\n";

        std::cout << "V1-vs-V2 relative difference: "
                  << relative_difference << "\n";

        std::cout << "V2 true relative residual: "
                  << true_residual << "\n";

        std::cout << "V2 relative solution error: "
                  << relative_error << "\n";

        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": V1 vs V2 CG consistency\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
