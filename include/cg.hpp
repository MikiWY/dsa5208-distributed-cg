#pragma once

#include <vector>

#include "common.hpp"

// Configuration shared by serial and distributed CG.
//
// max_iterations:
//   0  = use min(10 * global_N, 100000)
//   >0 = user-specified iteration limit
//   <0 = invalid
struct SolverOptions {
    double tolerance = 1e-8;
    int max_iterations = 0;
};

// Reason why the solver stopped.
enum class TerminationReason {
    Converged,
    MaxIterations,
    NumericalBreakdown,
    NonFiniteValue
};

// Result shared by serial and distributed CG.
//
// Serial:
//   x contains the full solution vector.
//
// Distributed:
//   x contains only locally owned entries.
//   Iteration count, convergence and residual are global.
struct SolverResult {
    std::vector<double> x;

    int iterations = 0;
    bool converged = false;

    double recursive_rel_residual = 0.0;

    TerminationReason termination_reason =
        TerminationReason::MaxIterations;
};

// Solve A*x = b using serial Conjugate Gradient.
//
// Requirements:
//   A is symmetric positive definite.
//   A contains all global rows.
//   b.size() == A.global_rows.
//   Initial guess x0 = 0.
//
// This function performs no MPI communication.
SolverResult conjugate_gradient(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const SolverOptions& options
);
