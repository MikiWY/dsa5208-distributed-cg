#pragma once

#include "solver_timing.hpp"

#include <mpi.h>
#include <vector>

#include "cg.hpp"
#include "common.hpp"
#include "halo_exchange.hpp"

// Hybrid MPI + OpenMP Conjugate Gradient solver
// using blocking Halo Exchange.
//
// A contains locally owned CSR rows.
// CSR column indices remain global.
//
// b_local contains only locally owned RHS entries.
//
// Communication:
//   Blocking Halo Exchange for distributed SpMV.
//   MPI_Allreduce for global CG dot products.
//
// Computation:
//   OpenMP CSR SpMV, local dot products and vector updates.
//
// MPI threading:
//   Requires MPI_THREAD_FUNNELED or higher.
//   All MPI calls must be made by the thread
//   that initialized MPI, outside OpenMP parallel regions.
//
// The initial guess is x0 = 0.
//
// SolverOptions and SolverResult are shared with
// Serial, MPI-only and Hybrid Allgatherv CG.
//
// SolverResult.x contains locally owned solution entries.
//
// iterations, converged, recursive_rel_residual and
// termination_reason must be globally consistent.
//
// Timing:
//   This initial correctness implementation constructs
//   the HaloExchangePlan inside the solver call.
//   A later benchmark interface will separate preprocessing
//   from CG-only timing.

SolverResult hybrid_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    MPI_Comm comm
);

// Hybrid Halo CG using a preconstructed communication plan.
//
// The plan must correspond to A and the same MPI communicator.
// MPI communication must be performed by the initializing thread,
// outside OpenMP parallel regions.
//
// Intended for CG-only timing without Halo Plan preprocessing.
//
// The supplied plan is reused and is not modified.

SolverResult hybrid_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm
);

// Prebuilt Halo Plan with optional component timing.
SolverResult hybrid_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm,
    SolverTiming* timing
);
