#pragma once

#include "solver_timing.hpp"

#include <mpi.h>
#include <vector>

#include "common.hpp"

// Distributed CSR SpMV using MPI_Allgatherv.
//
// Each rank owns a contiguous block of global matrix rows.
// The local input vector contains only entries owned by this rank.
// The output vector contains only locally owned result entries.
//
// The CSR matrix stores global column indices.
//
// This function performs MPI communication during each call.
// Communication time must be included in solver timing.

std::vector<double> distributed_spmv_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const std::vector<int>& recvcounts,
    const std::vector<int>& displs,
    MPI_Comm comm
);

// Optional component timing overload.
std::vector<double> distributed_spmv_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const std::vector<int>& recvcounts,
    const std::vector<int>& displs,
    MPI_Comm comm,
    SolverTiming* timing
);
