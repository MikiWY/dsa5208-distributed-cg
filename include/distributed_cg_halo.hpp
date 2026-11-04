#pragma once

#include "solver_timing.hpp"

#include <mpi.h>
#include <vector>

#include "cg.hpp"
#include "common.hpp"
#include "halo_exchange.hpp"

// MPI-only Conjugate Gradient solver using blocking Halo Exchange.
//
// A contains only locally owned CSR rows.
// CSR column indices remain global.
//
// b_local contains only locally owned RHS entries.
//
// options uses the same tolerance and iteration-limit semantics
// as the Serial and MPI Allgatherv CG solvers.
//
// Communication:
//   A reusable HaloExchangePlan is constructed before
//   the CG iteration loop.
//
//   Each SpMV exchanges only the remotely owned vector
//   entries required by local CSR rows.
//
//   Global dot products still use MPI_Allreduce.
//
// Returns:
//   SolverResult.x contains only locally owned solution entries.
//
//   iterations, converged, recursive_rel_residual,
//   and termination_reason represent global solver state.
//
// Requirements:
//   A belongs to a globally symmetric positive definite matrix.
//   All ranks participate in the same MPI collectives.
//   Initial guess x0 = 0.
//
// Timing:
//   This initial correctness implementation includes Halo Plan
//   construction inside the solver call.
//   A later benchmark interface will separate preprocessing
//   from CG-only timing.

SolverResult distributed_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    MPI_Comm comm
);

// Solve using a HaloExchangePlan built before this call.
//
// The plan must correspond to A and the same MPI communicator.
// All ranks must participate in the same collective operations.
//
// Intended for benchmarks that exclude Halo Plan construction
// from the CG-only timed region.
//
// The supplied plan is reused and is not modified.

SolverResult distributed_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm
);

// Prebuilt Halo Plan with optional component timing.
SolverResult distributed_conjugate_gradient_halo(
    const CSRMatrix& A,
    const std::vector<double>& b_local,
    const SolverOptions& options,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm,
    SolverTiming* timing
);
