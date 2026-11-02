#include "cg.hpp"

#include <algorithm>

#include <cmath>

#include <stdexcept>
#include <vector>

namespace {

// Compute the dot product of two vectors.
double dot_product(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Dot product vector sizes do not match"
        );
    }

    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// Serial CSR matrix-vector multiplication.
std::vector<double> csr_spmv(
    const CSRMatrix& A,
    const std::vector<double>& x
) {
    if (A.global_row_start != 0 ||
        A.local_num_rows != A.global_rows ||
        x.size() != static_cast<std::size_t>(A.global_rows)) {
        throw std::invalid_argument(
            "Serial SpMV requires a complete matrix and vector"
        );
    }

    std::vector<double> y(A.global_rows, 0.0);

    for (int r = 0; r < A.local_num_rows; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            y[r] += A.values[p] * x[A.col_idx[p]];
        }
    }

    return y;
}

} // namespace

// Serial Conjugate Gradient solver.
// Serial Conjugate Gradient solver with convergence and breakdown handling.
SolverResult conjugate_gradient(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const SolverOptions& options
) {
    // Validate solver inputs.
    if (A.global_rows <= 0 ||
        A.global_row_start != 0 ||
        A.local_num_rows != A.global_rows ||
        b.size() != static_cast<std::size_t>(A.global_rows)) {
        throw std::invalid_argument(
            "Serial CG requires a complete square matrix and RHS"
        );
    }

    if (!std::isfinite(options.tolerance) ||
        options.tolerance <= 0.0 ||
        options.max_iterations < 0) {
        throw std::invalid_argument(
            "Invalid solver options"
        );
    }

    const int N = A.global_rows;

    SolverResult result;
    result.x.assign(N, 0.0);

    // Initial residual: r0 = b - A*x0 = b.
    std::vector<double> r = b;

    // Initial search direction: p0 = r0.
    std::vector<double> p = r;

    const double b_norm_squared = dot_product(b, b);

    if (!std::isfinite(b_norm_squared)) {
        result.termination_reason =
            TerminationReason::NonFiniteValue;
        return result;
    }

    // Zero RHS: x0 = 0 is already the exact solution.
    if (b_norm_squared == 0.0) {
        result.converged = true;
        result.iterations = 0;
        result.recursive_rel_residual = 0.0;
        result.termination_reason =
            TerminationReason::Converged;
        return result;
    }

    // Determine the maximum iteration count.
    const int max_iterations =
        (options.max_iterations > 0)
            ? options.max_iterations
            : static_cast<int>(
                std::min(10LL * N, 100000LL)
              );

    const double b_norm = std::sqrt(b_norm_squared);
    double rr_old = b_norm_squared;

    result.recursive_rel_residual = 1.0;

    for (int k = 0; k < max_iterations; ++k) {
        // Step 1: Sparse matrix-vector multiplication.
        const std::vector<double> q = csr_spmv(A, p);

        // Step 2: Compute the CG step size.
        const double pq = dot_product(p, q);

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

        // Step 3: Update solution and residual.
        for (int i = 0; i < N; ++i) {
            result.x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        result.iterations = k + 1;

        // Step 4: Compute the new residual norm.
        const double rr_new = dot_product(r, r);

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

        for (int i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rr_old = rr_new;
    }

    result.termination_reason =
        TerminationReason::MaxIterations;

    return result;
}
