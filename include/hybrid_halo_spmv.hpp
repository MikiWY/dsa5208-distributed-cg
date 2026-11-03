#pragma once

#include <mpi.h>
#include <vector>

#include "common.hpp"
#include "halo_exchange.hpp"

// Hybrid MPI + OpenMP CSR SpMV using blocking Halo Exchange.
//
// A contains locally owned CSR rows.
// CSR column indices remain global.
//
// x_local contains only locally owned vector entries.
// plan must be built collectively before this function is called.
//
// Phase 1:
//   Blocking Halo Exchange is performed by the MPI main thread,
//   outside any OpenMP parallel region.
//
// Phase 2:
//   OpenMP threads compute independent local CSR rows.
//
// Returns locally owned output entries.
//
// MPI threading:
//   Requires MPI_THREAD_FUNNELED or higher.
//   Must be called by the thread that initialized MPI.
//
// Timing:
//   Halo communication contributes to vector_comm_s.
//   Local parallel CSR multiplication contributes to compute_s.
//
// This is the initial correctness implementation.
// Ghost lookup and buffer reuse will be optimized later.

std::vector<double> hybrid_spmv_halo(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const HaloExchangePlan& plan,
    MPI_Comm comm
);
