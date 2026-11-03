#include <mpi.h>
#include <omp.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "distributed_cg.hpp"
#include "distributed_cg_halo.hpp"
#include "distributed_spmv.hpp"
#include "hybrid_cg.hpp"
#include "hybrid_halo_cg.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;

    MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &provided
    );

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8;
    bool valid = argc <= 2;

    if (argc == 2) {
        try {
            const std::string arg(argv[1]);
            std::size_t pos = 0;
            const long long requested = std::stoll(arg, &pos);

            if (pos != arg.size() ||
                requested < 1 || requested > 512) {
                valid = false;
            } else {
                n = static_cast<int>(requested);
            }
        } catch (const std::exception&) {
            valid = false;
        }
    }

    int local_valid = valid ? 1 : 0;
    int global_valid = 0;
    int min_n = 0;
    int max_n = 0;

    MPI_Allreduce(
        &local_valid, &global_valid,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &n, &min_n, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    if (!global_valid || min_n != max_n ||
        provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "Invalid test configuration\n";
        }
        MPI_Finalize();
        return 1;
    }

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

    // All ranks invoke solvers in the same order.
    const std::array<SolverResult, 4> results = {
        distributed_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        ),
        distributed_conjugate_gradient_halo(
            A, b_local, options, MPI_COMM_WORLD
        ),
        hybrid_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        ),
        hybrid_conjugate_gradient_halo(
            A, b_local, options, MPI_COMM_WORLD
        )
    };

    const std::array<std::string, 4> names = {
        "V1 MPI Allgatherv",
        "V2 MPI Halo",
        "V3 Hybrid Allgatherv",
        "V4 Hybrid Halo"
    };

    const double inf =
        std::numeric_limits<double>::infinity();

    std::array<double, 4> residuals{};
    std::array<double, 4> solution_errors{};
    std::array<double, 4> differences{};

    bool local_pass = true;

    for (int v = 0; v < 4; ++v) {
        const SolverResult& result = results[v];

        // Independent true residual calculation.
        const std::vector<double> Ax =
            distributed_spmv_allgatherv(
                A, result.x, counts, displs, MPI_COMM_WORLD
            );

        double local_sums[6] = {};

        for (int i = 0; i < part.local_num_rows; ++i) {
            const double residual = b_local[i] - Ax[i];
            const double error = result.x[i] - x_true[i];
            const double difference =
                result.x[i] - results[0].x[i];

            local_sums[0] += residual * residual;
            local_sums[1] += b_local[i] * b_local[i];
            local_sums[2] += error * error;
            local_sums[3] += x_true[i] * x_true[i];
            local_sums[4] += difference * difference;
            local_sums[5] += results[0].x[i] *
                             results[0].x[i];
        }

        double sums[6] = {};

        MPI_Allreduce(
            local_sums, sums, 6,
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
        );

        residuals[v] = sums[1] > 0.0
            ? std::sqrt(sums[0] / sums[1]) : inf;

        solution_errors[v] = sums[3] > 0.0
            ? std::sqrt(sums[2] / sums[3]) : inf;

        differences[v] = sums[5] > 0.0
            ? std::sqrt(sums[4] / sums[5]) : inf;

        local_pass =
            local_pass &&
            result.converged &&
            result.termination_reason ==
                TerminationReason::Converged &&
            result.iterations == results[0].iterations &&
            result.x.size() ==
                static_cast<std::size_t>(part.local_num_rows) &&
            std::isfinite(result.recursive_rel_residual) &&
            std::isfinite(residuals[v]) &&
            std::isfinite(solution_errors[v]) &&
            std::isfinite(differences[v]) &&
            residuals[v] <= 1e-8 &&
            solution_errors[v] <= 1e-6 &&
            differences[v] <= 1e-8;
    }

    const int local_ok = local_pass ? 1 : 0;
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

        for (int v = 0; v < 4; ++v) {
            std::cout
                << names[v]
                << ": iterations=" << results[v].iterations
                << ", true_residual=" << residuals[v]
                << ", solution_error=" << solution_errors[v]
                << ", vs_V1=" << differences[v]
                << "\n";
        }

        std::cout
            << (global_ok ? "PASS" : "FAIL")
            << ": All four CG configurations\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
