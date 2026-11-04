#pragma once

#include "solver_timing.hpp"

#include <mpi.h>
#include <vector>

#include "cg.hpp"
#include "common.hpp"

// Hybrid MPI + OpenMP Conjugate Gradient solver.
//
// Communication strategy:
//   MPI_Allgatherv.
//
// Local computation:
//   OpenMP CSR SpMV, dot products and vector updates.
//
// MPI threading:
//   Requires MPI_THREAD_FUNNELED or higher.
//   All MPI calls must be made by the thread that
//   initialized MPI.
//
// A contains only locally owned CSR rows.
// CSR column indices are global.
//
// b_local contains only locally owned RHS entries.
//
// SolverResult.x contains only locally owned solution entries.
//
// iterations, converged, recursive_rel_residual and
// termination_reason must be globally consistent.
//
// The initial guess is x0 = 0.
//
// This solver shares SolverOptions and SolverResult
// with the Serial and MPI-only implementations.
//
// Timing instrumentation will be added separately.

SolverResult hybrid_conjugate_gradient_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    MPI_Comm comm
);

// CG with optional component timing.
SolverResult hybrid_conjugate_gradient_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    MPI_Comm comm,
    SolverTiming* timing
);
