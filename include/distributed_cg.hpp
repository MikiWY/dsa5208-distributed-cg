#pragma once

#include <mpi.h>
#include <vector>

#include "cg.hpp"
#include "common.hpp"

// Distributed Conjugate Gradient solver using MPI_Allgatherv.
//
// Each rank owns a contiguous block of matrix rows.
//
// A:
//   Contains only locally owned CSR rows.
//   Column indices remain global.
//
// b_local:
//   Contains only locally owned RHS entries.
//
// options:
//   Shared solver configuration.
//   max_iterations = 0 uses min(10 * global_N, 100000).
//
// comm:
//   MPI communicator containing all participating ranks.
//
// Returns:
//   SolverResult.x contains only locally owned solution entries.
//
//   iterations, converged, recursive_rel_residual,
//   and termination_reason represent the global solver state
//   and must be consistent across all ranks.
//
// Requirements:
//   A is part of a globally symmetric positive definite matrix.
//   All ranks participate in the same collective operations.
//   Initial guess x0 = 0.
//
// Solver timing:
//   Includes initial residual computation, CG iterations,
//   vector communication and global reductions.
//   Excludes matrix construction, preprocessing,
//   independent validation and CSV output.

SolverResult distributed_conjugate_gradient_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    MPI_Comm comm
);
