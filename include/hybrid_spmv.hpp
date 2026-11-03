#pragma once

#include <mpi.h>
#include <vector>

#include "common.hpp"

// Hybrid MPI + OpenMP CSR SpMV using MPI_Allgatherv.
//
// Each MPI rank owns a contiguous block of CSR rows.
// x_local contains locally owned input-vector entries.
// CSR column indices remain global.
//
// MPI_Allgatherv is called outside OpenMP parallel regions
// by the thread that initialized MPI.
//
// OpenMP computes locally owned output rows.
//
// recvcounts and displs describe the global row partition.

std::vector<double> hybrid_spmv_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const std::vector<int>& recvcounts,
    const std::vector<int>& displs,
    MPI_Comm comm
);
