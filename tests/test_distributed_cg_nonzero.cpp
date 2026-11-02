#include <mpi.h>

#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "distributed_cg.hpp"
#include "distributed_spmv.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Grid size can be supplied as a command-line argument.
    // Default: n = 8.
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
                std::cerr
                    << "Invalid grid size. Usage: "
                    << argv[0] << " [n]\n";
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

    // Generate RHS using the distributed matrix.
    std::vector<double> b_local =
        distributed_spmv_allgatherv(
            A, x_true, counts, displs, MPI_COMM_WORLD
        );

    SolverOptions options;

    SolverResult result =
        distributed_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        );

    // Check global consistency of solver state.
    int local_state[3] = {
        result.iterations,
        result.converged ? 1 : 0,
        static_cast<int>(result.termination_reason)
    };

    int min_state[3] = {};
    int max_state[3] = {};

    MPI_Allreduce(
        local_state, min_state, 3,
        MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        local_state, max_state, 3,
        MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    double min_residual = 0.0;
    double max_residual = 0.0;

    MPI_Allreduce(
        &result.recursive_rel_residual,
        &min_residual, 1,
        MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &result.recursive_rel_residual,
        &max_residual, 1,
        MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD
    );

    bool state_consistent = true;

    for (int i = 0; i < 3; ++i) {
        if (min_state[i] != max_state[i]) {
            state_consistent = false;
        }
    }

    if (!std::isfinite(min_residual) ||
        !std::isfinite(max_residual) ||
        min_residual != max_residual) {
        state_consistent = false;
    }

    // Independently recompute the true residual.
    std::vector<double> Ax_local =
        distributed_spmv_allgatherv(
            A, result.x, counts, displs, MPI_COMM_WORLD
        );

    double local_residual_sq = 0.0;
    double local_b_sq = 0.0;
    double local_error_sq = 0.0;
    double local_true_sq = 0.0;

    for (int i = 0; i < part.local_num_rows; ++i) {
        const double residual = b_local[i] - Ax_local[i];
        const double error = result.x[i] - x_true[i];

        local_residual_sq += residual * residual;
        local_b_sq += b_local[i] * b_local[i];
        local_error_sq += error * error;
        local_true_sq += x_true[i] * x_true[i];
    }

    double local_sums[4] = {
        local_residual_sq, local_b_sq,
        local_error_sq, local_true_sq
    };
    double global_sums[4] = {};

    MPI_Allreduce(
        local_sums, global_sums, 4,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
    );

    const double true_residual =
        std::sqrt(global_sums[0] / global_sums[1]);

    const double rel_error =
        std::sqrt(global_sums[2] / global_sums[3]);

    bool local_pass =
        state_consistent &&
        result.converged &&
        result.termination_reason ==
            TerminationReason::Converged &&
        result.iterations > 0 &&
        result.x.size() ==
            static_cast<std::size_t>(part.local_num_rows) &&
        std::isfinite(true_residual) &&
        std::isfinite(rel_error) &&
        true_residual <= options.tolerance;

    int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok, 1,
        MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", N=" << N
                  << ", MPI ranks=" << size << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Recursive relative residual: "
                  << result.recursive_rel_residual << "\n";
        std::cout << "True relative residual: "
                  << true_residual << "\n";
        std::cout << "Relative solution error: "
                  << rel_error << "\n";
        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": Distributed CG nonzero RHS\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
