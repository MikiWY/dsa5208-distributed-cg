#include "hybrid_halo_cg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "hybrid_halo_spmv.hpp"
#include "halo_exchange.hpp"
#include "hybrid_vector_ops.hpp"
#include "openmp_vector_ops.hpp"
#include "partition.hpp"

// Hybrid MPI + OpenMP CG using blocking Halo Exchange.
// Stage 1: initialization and global zero-RHS handling.
SolverResult hybrid_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm
) {
    int rank = 0;
    int size = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // MPI_THREAD_FUNNELED requires MPI calls from
    // the thread that initialized MPI.
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);

    int is_main_thread = 0;
    MPI_Is_thread_main(&is_main_thread);

    const bool thread_valid =
        provided >= MPI_THREAD_FUNNELED &&
        is_main_thread != 0;

    // Check global matrix dimensions.
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

    RowPartition expected;

    if (dimension_valid) {
        expected = partition_rows(A.global_rows, rank, size);
    }

    // Validate solver options across ranks.
    double min_tolerance = 0.0;
    double max_tolerance = 0.0;
    int min_iterations = 0;
    int max_iterations_setting = 0;

    MPI_Allreduce(
        &options.tolerance, &min_tolerance,
        1, MPI_DOUBLE, MPI_MIN, comm
    );

    MPI_Allreduce(
        &options.tolerance, &max_tolerance,
        1, MPI_DOUBLE, MPI_MAX, comm
    );

    MPI_Allreduce(
        &options.max_iterations, &min_iterations,
        1, MPI_INT, MPI_MIN, comm
    );

    MPI_Allreduce(
        &options.max_iterations, &max_iterations_setting,
        1, MPI_INT, MPI_MAX, comm
    );

    const bool options_valid =
        std::isfinite(options.tolerance) &&
        options.tolerance > 0.0 &&
        min_tolerance == max_tolerance &&
        options.max_iterations >= 0 &&
        min_iterations == max_iterations_setting;

    // Validate local CSR invariants before entering CG collectives.
    // Do not access row_ptr elements until its size is verified.
    bool csr_valid =
        dimension_valid &&
        A.local_num_rows >= 0 &&
        A.row_ptr.size() ==
            static_cast<std::size_t>(A.local_num_rows) + 1 &&
        A.col_idx.size() == A.values.size();

    if (csr_valid) {
        csr_valid =
            A.row_ptr.front() == 0 &&
            A.row_ptr.back() >= 0 &&
            static_cast<std::size_t>(A.row_ptr.back()) ==
                A.values.size();
    }

    if (csr_valid) {
        const int nnz = A.row_ptr.back();

        for (int r = 0; r < A.local_num_rows; ++r) {
            if (A.row_ptr[r] < 0 ||
                A.row_ptr[r] > A.row_ptr[r + 1] ||
                A.row_ptr[r + 1] > nnz) {
                csr_valid = false;
                break;
            }
        }
    }

    if (csr_valid) {
        for (std::size_t k = 0; k < A.values.size(); ++k) {
            if (A.col_idx[k] < 0 ||
                A.col_idx[k] >= A.global_rows ||
                !std::isfinite(A.values[k])) {
                csr_valid = false;
                break;
            }
        }
    }

    // Check the local CSR metadata and RHS dimensions.
    const bool local_valid =
        thread_valid &&
        dimension_valid &&
        A.global_row_start == expected.global_row_start &&
        A.local_num_rows == expected.local_num_rows &&
        b_local.size() ==
            static_cast<std::size_t>(A.local_num_rows) &&
        csr_valid &&
        options_valid;

    const int local_ok = local_valid ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, comm
    );

    if (!global_ok) {
        throw std::invalid_argument(
            "Invalid Hybrid CG inputs or MPI thread support"
        );
    }

    SolverResult result;
    result.x.assign(A.local_num_rows, 0.0);

    std::vector<double> r = b_local;
    std::vector<double> p = r;

    // Global RHS norm: OpenMP local dot + MPI_Allreduce.
    const double b_norm_squared =
        hybrid_global_dot_product(b_local, b_local, comm);

    if (!std::isfinite(b_norm_squared)) {
        result.termination_reason =
            TerminationReason::NonFiniteValue;
        return result;
    }

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
        // Step 1: Hybrid SpMV.
        // MPI communication is outside OpenMP parallel regions.
        const std::vector<double> q =
            hybrid_spmv_halo(
                A, p, halo_plan, comm
            );

        // Step 2: Global p^T * A * p.
        const double pq =
            hybrid_global_dot_product(p, q, comm);

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
        openmp_axpy(alpha, p, result.x);
        openmp_axpy(-alpha, q, r);

        result.iterations = k + 1;

        // Step 4: Global residual norm.
        const double rr_new =
            hybrid_global_dot_product(r, r, comm);

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

        // Step 5: Update the search direction.
        const double beta = rr_new / rr_old;

        if (!std::isfinite(beta)) {
            result.termination_reason =
                TerminationReason::NonFiniteValue;
            return result;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < A.local_num_rows; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rr_old = rr_new;
    }

    result.termination_reason =
        TerminationReason::MaxIterations;

    return result;
}

// Compatibility interface: build the Halo Plan before solving.
// Existing correctness tests continue using this interface.
SolverResult hybrid_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    MPI_Comm comm
) {
    const HaloExchangePlan halo_plan =
        build_halo_exchange_plan(A, comm);

    return hybrid_conjugate_gradient_halo(
        A, b_local, options, halo_plan, comm
    );
}
