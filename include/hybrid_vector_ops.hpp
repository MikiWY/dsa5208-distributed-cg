#pragma once

#include <mpi.h>
#include <vector>

// Hybrid global dot product.
//
// Each rank computes its local dot product using OpenMP.
// MPI_Allreduce(MPI_SUM) combines local contributions.
//
// Must be called by the thread that initialized MPI,
// outside an OpenMP parallel region.
//
// The input vectors must have equal local lengths.
// All ranks must participate in the collective operation.
//
// Input dimensions must be validated collectively by the
// calling solver before this function is used.

double hybrid_global_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm
);
