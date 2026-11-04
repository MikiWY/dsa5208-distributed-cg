#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <exception>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "benchmark_csv.hpp"
#include "benchmark_version.hpp"
#include "solver_timing.hpp"
#include "distributed_cg.hpp"
#include "distributed_cg_halo.hpp"
#include "hybrid_cg.hpp"
#include "hybrid_halo_cg.hpp"
#include "halo_exchange.hpp"
#include "distributed_spmv.hpp"
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

    int is_main_thread = 0;
    MPI_Is_thread_main(&is_main_thread);

    const int local_thread_ok =
        (provided >= MPI_THREAD_FUNNELED &&
         is_main_thread != 0) ? 1 : 0;

    int global_thread_ok = 0;

    MPI_Allreduce(
        &local_thread_ok, &global_thread_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (!global_thread_ok) {
        if (rank == 0) {
            std::cerr
                << "MPI_THREAD_FUNNELED support required\n";
        }

        MPI_Finalize();
        return 1;
    }

    // Grid size: default n=8, optionally specified by --n.
    int n = 8;
    std::string run_id = "manual";
    std::string timing_output;

    // Solver and communication configuration.
    std::string solver = "mpi";
    std::string communication = "allgatherv";

    bool arguments_valid = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--n" && i + 1 < argc) {
            try {
                const std::string value(argv[++i]);
                std::size_t parsed = 0;
                const long long requested =
                    std::stoll(value, &parsed);

                if (parsed != value.size() ||
                    requested < 1 ||
                    requested > 4096) {
                    arguments_valid = false;
                } else {
                    n = static_cast<int>(requested);
                }
            } catch (const std::exception&) {
                arguments_valid = false;
            }
        } else if (arg == "--solver" && i + 1 < argc) {
            solver = argv[++i];

            if (solver != "mpi" && solver != "hybrid") {
                arguments_valid = false;
            }

        } else if (arg == "--comm" && i + 1 < argc) {
            communication = argv[++i];

            if (communication != "allgatherv" &&
                communication != "halo") {
                arguments_valid = false;
            }

        } else if (arg == "--run-id" && i + 1 < argc) {
            run_id = argv[++i];
            if (run_id.empty()) {
                arguments_valid = false;
            }
        } else if (arg == "--timing-output" && i + 1 < argc) {
            timing_output = argv[++i];
            if (timing_output.empty()) {
                arguments_valid = false;
            }
        } else {
            arguments_valid = false;
        }
    }

    // Check all ranks before continuing.
    const int local_ok = arguments_valid ? 1 : 0;
    int global_arguments_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_arguments_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (!global_arguments_ok) {
        if (rank == 0) {
            std::cerr << "Usage: "
                      << argv[0]
                      << " [--n GRID_SIZE]"
                      << " [--solver mpi|hybrid]"
                      << " [--comm allgatherv|halo]"
                      << " [--run-id ID]"
                      << " [--timing-output PATH]\n";
        }
        MPI_Finalize();
        return 1;
    }

    // All MPI ranks must use the same grid size.
    int min_n = 0;
    int max_n = 0;

    MPI_Allreduce(
        &n, &min_n, 1,
        MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &n, &max_n, 1,
        MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    if (min_n != max_n) {
        if (rank == 0) {
            std::cerr
                << "Inconsistent grid size across MPI ranks\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Encode solver and communication choices.
    const int solver_code =
        (solver == "mpi") ? 0 : 1;

    const int communication_code =
        (communication == "allgatherv") ? 0 : 1;

    // Ensure every MPI rank selects the same configuration.
    int min_solver = 0;
    int max_solver = 0;
    int min_comm = 0;
    int max_comm = 0;

    MPI_Allreduce(
        &solver_code, &min_solver,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &solver_code, &max_solver,
        1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &communication_code, &min_comm,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &communication_code, &max_comm,
        1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
    );

    if (min_solver != max_solver ||
        min_comm != max_comm) {
        if (rank == 0) {
            std::cerr
                << "Inconsistent solver or communication "
                << "configuration across MPI ranks\n";
        }

        MPI_Finalize();
        return 1;
    }

    const int N = n * n;

    const RowPartition part =
        partition_rows(N, rank, size);

    // Matrix generation is outside the timed region.
    const CSRMatrix A = generate_poisson_matrix(
        n,
        part.global_row_start,
        part.local_num_rows
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

    // RHS construction is outside the timed region.
    const std::vector<double> b_local =
        distributed_spmv_allgatherv(
            A, x_true, counts, displs, MPI_COMM_WORLD
        );

    SolverOptions options;
    options.tolerance = 1e-8;

    // Halo preprocessing is excluded from CG-only timing.
    HaloExchangePlan halo_plan;

    if (communication == "halo") {
        halo_plan = build_halo_exchange_plan(
            A, MPI_COMM_WORLD
        );
    }

    // Synchronize ranks before timing.
    MPI_Barrier(MPI_COMM_WORLD);

    const double start = MPI_Wtime();

    SolverResult result;
    SolverTiming component_timing;

    const bool timing_available = true;

    if (solver == "mpi" && communication == "allgatherv") {
        result = distributed_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD,
            &component_timing
        );
    } else if (solver == "mpi" && communication == "halo") {
        result = distributed_conjugate_gradient_halo(
            A, b_local, options, halo_plan, MPI_COMM_WORLD,
            &component_timing
        );
    } else if (solver == "hybrid" &&
               communication == "allgatherv") {
        result = hybrid_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD,
            &component_timing
        );
    } else {
        result = hybrid_conjugate_gradient_halo(
            A, b_local, options, halo_plan, MPI_COMM_WORLD,
            &component_timing
        );
    }

    const double local_time = MPI_Wtime() - start;

    // Maximum CG time across MPI ranks.
    double time_s = 0.0;

    MPI_Allreduce(
        &local_time,
        &time_s,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD
    );

    // Aggregate measured component times after CG-only timing.
    // Each category uses the maximum accumulated rank time.
    double component_local[3] = {
        component_timing.compute_s,
        component_timing.vector_comm_s,
        component_timing.reduction_s
    };

    double component_max[3] = {};

    if (timing_available) {
        MPI_Allreduce(
            component_local, component_max,
            3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD
        );
    }

    // Independent true residual validation (not timed).
    const std::vector<double> Ax =
        distributed_spmv_allgatherv(
            A, result.x, counts, displs, MPI_COMM_WORLD
        );

    double local_sums[4] = {};

    for (int i = 0; i < part.local_num_rows; ++i) {
        const double residual = b_local[i] - Ax[i];
        const double error = result.x[i] - x_true[i];

        local_sums[0] += residual * residual;
        local_sums[1] += b_local[i] * b_local[i];
        local_sums[2] += error * error;
        local_sums[3] += x_true[i] * x_true[i];
    }

    double global_sums[4] = {};

    MPI_Allreduce(
        local_sums,
        global_sums,
        4,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    const double true_residual =
        std::sqrt(global_sums[0] / global_sums[1]);

    const double relative_error =
        std::sqrt(global_sums[2] / global_sums[3]);

    const bool local_pass =
        result.converged &&
        result.termination_reason ==
            TerminationReason::Converged &&
        std::isfinite(time_s) &&
        time_s >= 0.0 &&
        std::isfinite(true_residual) &&
        std::isfinite(relative_error) &&
        true_residual <= 1e-8 &&
        relative_error <= 1e-6;

    const int validation_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &validation_ok,
        &global_ok,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << std::setprecision(12);

        std::cout << "Solver: " << solver << "\n";
        std::cout << "Communication: " << communication << "\n";
        std::cout << "Threads: "
                  << ((solver == "hybrid")
                      ? omp_get_max_threads() : 1)
                  << "\n";

        std::cout << "Grid n: " << n << "\n";
        std::cout << "MPI ranks: " << size << "\n";
        std::cout << "Iterations: "
                  << result.iterations << "\n";
        std::cout << "CG-only time_s: "
                  << time_s << "\n";
        std::cout << "Recursive relative residual: "
                  << result.recursive_rel_residual << "\n";
        std::cout << "True relative residual: "
                  << true_residual << "\n";
        std::cout << "Relative solution error: "
                  << relative_error << "\n";
        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": CG benchmark smoke test\n";
    }

    // Write one timing record from rank 0 only.
    // CSV I/O is outside the CG timing interval.
    int write_ok = 1;

    if (rank == 0 && !timing_output.empty()) {
        try {
            BenchmarkTimingRecord record;

            record.run_id = run_id;
            record.dim = 2;
            record.n = n;
            record.comm = communication;
            record.ranks = size;
            record.threads =
                (solver == "hybrid") ? omp_get_max_threads() : 1;
            record.iterations = result.iterations;
            record.converged = result.converged;
            record.time_s = time_s;
            record.relative_residual = true_residual;
            record.git_commit = benchmark_git_commit();

            // V1/V3 component timing is measured.
            // V2/V4 fields remain empty until instrumented.
            if (timing_available) {
                auto format_time = [](double seconds) {
                    std::ostringstream out;
                    out << std::setprecision(17) << seconds;
                    return out.str();
                };

                record.compute_s =
                    format_time(component_max[0]);
                record.vector_comm_s =
                    format_time(component_max[1]);
                record.reduction_s =
                    format_time(component_max[2]);
            }

            append_benchmark_timing_csv(
                timing_output, record
            );
        } catch (const std::exception& e) {
            std::cerr << "Timing CSV error: "
                      << e.what() << "\n";
            write_ok = 0;
        }
    }

    // Ensure all MPI ranks agree on the output status.
    MPI_Bcast(
        &write_ok, 1, MPI_INT, 0, MPI_COMM_WORLD
    );

    MPI_Finalize();
    return (global_ok && write_ok) ? 0 : 1;
}
