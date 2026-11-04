#pragma once

#include "solver_timing.hpp"

#include <mpi.h>
#include <vector>

#include "common.hpp"
#include "halo_exchange.hpp"

// Distributed CSR SpMV using blocking Halo Exchange.
//
// A contains locally owned CSR rows.
// CSR column indices remain global.
//
// x_local contains only locally owned vector entries.
//
// plan is constructed once using build_halo_exchange_plan()
// and reused across SpMV calls.
//
// Remote vector values are obtained using
// exchange_halo_values_blocking().
//
// Returns only locally owned output entries.
//
// MPI calls must occur outside OpenMP parallel regions.
//
// Timing:
//   Halo communication belongs to vector_comm_s.
//   Local CSR multiplication belongs to compute_s.

std::vector<double> distributed_spmv_halo(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const HaloExchangePlan& plan,
    MPI_Comm comm
);

// Optional component timing interface.
std::vector<double> distributed_spmv_halo(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const HaloExchangePlan& plan,
    MPI_Comm comm,
    SolverTiming* timing
);
