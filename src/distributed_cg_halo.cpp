#include "distributed_cg_halo.hpp"
#include "timing_utils.hpp"

#include <algorithm>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "distributed_vector_ops.hpp"
#include "halo_exchange.hpp"
#include "halo_spmv.hpp"
#include "partition.hpp"

// MPI-only CG using blocking Halo Exchange.
SolverResult distributed_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm,
    SolverTiming* timing
) {
    if (timing) {
        *timing = SolverTiming{};
    }

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Check global dimension consistency before partitioning.
    int min_global_rows = 0;
    int max_global_rows = 0;

    MPI_Allreduce(
        &A.global_rows, &min_global_rows,
        1, MPI_INT, MPI_MIN, comm
    );

    MPI_Allreduce(
        &A.global_rows, &max_global_rows,
        1, MPI_INT, MPI_MAX, comm
    );

    const bool dimension_valid =
        min_global_rows > 0 &&
        min_global_rows == max_global_rows;

    // Only compute the expected partition for valid dimensions.
    RowPartition expected;

    if (dimension_valid) {
        expected = partition_rows(A.global_rows, rank, size);
    }

    const bool local_valid =
        dimension_valid &&
        A.global_row_start == expected.global_row_start &&
        A.local_num_rows == expected.local_num_rows &&
        b_local.size() ==
            static_cast<std::size_t>(A.local_num_rows) &&
        std::isfinite(options.tolerance) &&
        options.tolerance > 0.0 &&
        options.max_iterations >= 0;

    // Check solver-option consistency across all ranks.
    double min_tolerance = 0.0;
    double max_tolerance = 0.0;

    MPI_Allreduce(
        &options.tolerance, &min_tolerance,
        1, MPI_DOUBLE, MPI_MIN, comm
    );

    MPI_Allreduce(
        &options.tolerance, &max_tolerance,
        1, MPI_DOUBLE, MPI_MAX, comm
    );

    int min_iterations = 0;
    int max_iterations_setting = 0;

    MPI_Allreduce(
        &options.max_iterations, &min_iterations,
        1, MPI_INT, MPI_MIN, comm
    );

    MPI_Allreduce(
        &options.max_iterations, &max_iterations_setting,
        1, MPI_INT, MPI_MAX, comm
    );

    const bool options_consistent =
        min_tolerance == max_tolerance &&
        min_iterations == max_iterations_setting;

    // All ranks must agree before proceeding.
    const int local_ok =
        (local_valid && options_consistent) ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok,
        &global_ok,
        1,
        MPI_INT,
        MPI_MIN,
        comm
    );

    if (global_ok == 0) {
        throw std::invalid_argument(
            "Invalid distributed CG inputs"
        );
    }

    const int local_N = A.local_num_rows;

    SolverResult result;
    result.x.assign(local_N, 0.0);

    // Initial guess x0 = 0.
    // Therefore r0 = b and p0 = r0.
    std::vector<double> r = b_local;
    std::vector<double> p = r;

    // Global squared norm of the RHS.
    const double b_norm_squared =
        distributed_dot_product(b_local, b_local, comm, timing);

    if (!std::isfinite(b_norm_squared)) {
        result.termination_reason =
            TerminationReason::NonFiniteValue;
        return result;
    }

    // Global zero RHS: x0 = 0 is already the solution.
    if (b_norm_squared == 0.0) {
        result.converged = true;
        result.iterations = 0;
        result.recursive_rel_residual = 0.0;
        result.termination_reason =
            TerminationReason::Converged;
        return result;
    }

    // Determine the global maximum iteration count.
    const int max_iterations =
        (options.max_iterations > 0)
            ? options.max_iterations
            : static_cast<int>(
                std::min(10LL * A.global_rows, 100000LL)
              );

    const double b_norm = std::sqrt(b_norm_squared);
    double rr_old = b_norm_squared;

    result.recursive_rel_residual = 1.0;

    for (int k = 0; k < max_iterations; ++k) {
        // Step 1: Distributed sparse matrix-vector product.
        const std::vector<double> q =
            distributed_spmv_halo(
                A, p, halo_plan, comm, timing
            );

        // Step 2: Global dot product p^T * A * p.
        const double pq =
            distributed_dot_product(p, q, comm, timing);

        if (!std::isfinite(pq)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            return result;
        }

        if (pq <= 0.0) {
            result.termination_reason =
                TerminationReason::NumericalBreakdown;
            return result;
        }

        const double alpha = rr_old / pq;

        if (!std::isfinite(alpha)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            return result;
        }

        // Step 3: Update locally owned solution and residual.
        const double update_start = timing_now(timing);

        for (int i = 0; i < local_N; ++i) {
            result.x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        timing_add_compute(timing, update_start);

        result.iterations = k + 1;

        // Step 4: Global squared residual norm.
        const double rr_new =
            distributed_dot_product(r, r, comm, timing);

        if (!std::isfinite(rr_new)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            return result;
        }

        result.recursive_rel_residual =
            std::sqrt(rr_new) / b_norm;

        if (result.recursive_rel_residual <=
            options.tolerance) {
            result.converged = true;
            result.termination_reason =
                TerminationReason::Converged;
            return result;
        }

        // Step 5: Update local search direction.
        const double beta = rr_new / rr_old;

        if (!std::isfinite(beta)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            return result;
        }

        const double direction_start = timing_now(timing);

        for (int i = 0; i < local_N; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        timing_add_compute(timing, direction_start);

        rr_old = rr_new;
    }

    result.termination_reason =
        TerminationReason::MaxIterations;

    return result;
}

// Compatibility interface: construct the Halo Plan before solving.
SolverResult distributed_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    MPI_Comm comm
) {
    const HaloExchangePlan halo_plan =
        build_halo_exchange_plan(A, comm);

    return distributed_conjugate_gradient_halo(
        A, b_local, options, halo_plan, comm
    );
}

// Prebuilt-plan interface without component timing.
SolverResult distributed_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm
) {
    return distributed_conjugate_gradient_halo(
        A, b_local, options, halo_plan, comm, nullptr
    );
}
